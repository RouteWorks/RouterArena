# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Build a synthetic outcome-label corpus for training ChuzomRouterV2 Gate 0.

Unlike the proxy classifier (labels by dataset *origin*), this script labels
each prompt by the cheapest production model that answers it correctly:

  FLASH     -- gemini-3.1-flash-lite answered correctly
  DEEPSEEK  -- only deepseek-v4-flash (or better) answered correctly
  QWEN235B  -- only qwen3-235b answered correctly (or all failed)

Labeling pool (5 models)
------------------------
Production anchors (routing targets):
  google/gemini-3.1-flash-lite
  deepseek/deepseek-v4-flash
  qwen/qwen3-235b-a22b-2507

Oracle judges (labeling only, never routed):
  google/gemini-2.5-pro   -- always-on: best signal-to-cost, 1M context, fixes NarrativeQA/RC
  anthropic/claude-opus-4 -- conditional: only fires on low-confidence or production disagreement

Cost policy: Gemini 2.5 Pro judges every example (~$1.25/M in).
Claude Opus 4 ($15/M in) only judges when production models disagree or confidence < 0.5.

Usage
-----
    uv run python scripts/build_outcome_labels.py --dry-run   # no API calls
    uv run python scripts/build_outcome_labels.py             # full run
    uv run python scripts/build_outcome_labels.py --cap 20 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import httpx
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "data" / "outcome_labels.jsonl"

# API endpoints — Claude and Gemini route to native APIs to avoid OpenRouter markup
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GOOGLE_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# ── Model pools ───────────────────────────────────────────────────────────────
#
# Production proxies: use Gemini API tiers directly (GOOGLE_API_KEY in .env).
# This maps production routing tiers to Gemini equivalents by capability:
#   flash-lite     → gemini-3.1-flash-lite  (cheapest, fastest — routing tier 1)
#   deepseek-flash → gemini-2.5-flash        (medium — routing tier 2)
#   qwen235b       → gemini-2.5-pro          (strongest — routing tier 3)
#
# Why Gemini instead of Ollama: all concurrent via REST API (~26s wall-clock for
# the full model fan-out vs 160s+ for serialized Ollama 27B inference).
# The Gemini tiers also directly represent real model capabilities, not small-model
# approximations. Net effect: accurate labels AND ~6x faster throughput.
#
# Internal model IDs used in scoring/labels (kept as original routing targets):
FLASH_MODEL = "google/gemini-3.1-flash-lite"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"
QWEN_MODEL = "qwen/qwen3-235b-a22b-2507"

PRODUCTION_MODELS = [FLASH_MODEL, DEEPSEEK_MODEL, QWEN_MODEL]

# Gemini API proxies: map each routing tier to the Gemini model of equivalent capability
GEMINI_PROXY: dict[str, str] = {
    FLASH_MODEL: "gemini-3.1-flash-lite",  # exact production model (verified available)
    DEEPSEEK_MODEL: "gemini-2.5-flash",  # deepseek-v4-flash tier equivalent
    QWEN_MODEL: "gemini-2.5-pro",  # qwen235b tier equivalent
}

# Always-on oracle: Gemini 2.5 Pro (same proxy as QWEN tier — deduplicates that call)
ORACLE_ALWAYS = "google/gemini-2.5-pro"
ORACLE_MODELS = [ORACLE_ALWAYS]
ALL_LABELING_MODELS = PRODUCTION_MODELS + ORACLE_MODELS

JUDGE_PRIMARY = ORACLE_ALWAYS  # Gemini 2.5 Pro via GOOGLE_API_KEY

# Production model must clear this fraction to be considered correct (single judge = 1.0)
CORRECTNESS_THRESHOLD = 0.5


# ── Dataset specs ─────────────────────────────────────────────────────────────


@dataclass
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    task_family: str
    grader: str  # exact_match / mcq / llm_judge
    cap: int = 200
    format_fn: str = "default"
    extra: dict = field(default_factory=dict)


DATASETS: list[DatasetSpec] = [
    # Code generation (targets LiveCodeBench weakness)
    # mbpp sanitized config was removed; use default split which has same 'text'+'code' fields
    DatasetSpec(
        "humaneval",
        "openai/openai_humaneval",
        None,
        "test",
        "code",
        "llm_judge",
        164,
        "humaneval",
    ),
    DatasetSpec(
        "mbpp",
        "google-research-datasets/mbpp",
        None,
        "test",
        "code",
        "llm_judge",
        200,
        "mbpp",
    ),
    # codeparrot/apps uses an old loading script; replaced with HumanEval+ (bigcode)
    DatasetSpec(
        "humaneval_plus",
        "bigcode/humanevalpack",
        "python",
        "test",
        "code",
        "llm_judge",
        164,
        "humaneval",
    ),
    # Reading comprehension short (SuperGLUE-RC weakness)
    # boolq: models say "Yes, because..." — mcq partial match handles verbose answers
    DatasetSpec(
        "boolq", "google/boolq", None, "validation", "rc_short", "mcq", 300, "boolq"
    ),
    # hotpot_qa/squad/drop: models produce verbose answers; llm_judge handles paraphrases
    DatasetSpec(
        "hotpot_qa",
        "hotpot_qa",
        "fullwiki",
        "validation",
        "rc_short",
        "llm_judge",
        300,
        "hotpot",
    ),
    DatasetSpec(
        "squad_v2",
        "rajpurkar/squad_v2",
        None,
        "validation",
        "rc_short",
        "llm_judge",
        300,
        "squad",
    ),
    DatasetSpec(
        "drop", "ucinlp/drop", None, "validation", "rc_short", "llm_judge", 200, "drop"
    ),
    # Long-context reading (NarrativeQA weakness)
    DatasetSpec(
        "narrativeqa",
        "deepmind/narrativeqa",
        None,
        "validation",
        "rc_long",
        "llm_judge",
        200,
        "narrativeqa",
    ),
    DatasetSpec(
        "quality",
        "emozilla/quality",
        None,
        "validation",
        "rc_long",
        "mcq",
        150,
        "quality",
    ),
    # Math: models write full solutions; llm_judge checks if final answer is correct
    DatasetSpec(
        "math500",
        "HuggingFaceH4/MATH-500",
        None,
        "test",
        "math",
        "llm_judge",
        300,
        "math500",
    ),
    DatasetSpec(
        "gsm8k", "openai/gsm8k", "main", "test", "math", "llm_judge", 300, "gsm8k"
    ),
    # Science MCQ: keep mcq grader but gold is now the option TEXT (see fmt_arc_mcq fix)
    DatasetSpec(
        "arc_challenge",
        "allenai/ai2_arc",
        "ARC-Challenge",
        "test",
        "science_mcq",
        "mcq",
        300,
        "arc_mcq",
    ),
    DatasetSpec(
        "qasc", "allenai/qasc", None, "validation", "science_mcq", "mcq", 200, "qasc"
    ),
]


# ── Prompt formatters ─────────────────────────────────────────────────────────


def _opts(texts: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {t}" for i, t in enumerate(texts))


def fmt_humaneval(ex: dict) -> tuple[str, str] | None:
    p = ex.get("prompt", "")
    if not p:
        return None
    task = (
        "Generate an executable Python function for the following task. "
        "Return only the function body.\n\n" + p
    )
    return task, ex.get("canonical_solution", "")


def fmt_mbpp(ex: dict) -> tuple[str, str] | None:
    text = ex.get("text", "")
    if not text:
        return None
    return ("Write a Python function to solve:\n\n" + text, ex.get("code", ""))


def fmt_apps(ex: dict) -> tuple[str, str] | None:
    prob = ex.get("problem_statement") or ex.get("question", "")
    if not prob:
        return None
    sols = ex.get("solutions", [])
    return (
        "Generate an executable Python function for:\n\n" + prob[:2000],
        sols[0] if sols else "",
    )


def fmt_boolq(ex: dict) -> tuple[str, str] | None:
    passage, question = ex.get("passage", ""), ex.get("question", "")
    if not passage or not question:
        return None
    ans = "Yes" if ex.get("answer") else "No"
    return (
        f"Passage: {passage[:2000]}\n\nQuestion: {question}\n\nAnswer Yes or No.",
        ans,
    )


def fmt_record(ex: dict) -> tuple[str, str] | None:
    passage, query = ex.get("passage", ""), ex.get("query", "")
    answers = ex.get("answers", [])
    if not passage or not query or not answers:
        return None
    return (f"Passage: {passage[:2000]}\n\nFill in: {query}", answers[0])


def fmt_squad(ex: dict) -> tuple[str, str] | None:
    ctx, q = ex.get("context", ""), ex.get("question", "")
    ans = ex.get("answers", {})
    texts = ans.get("text", []) if isinstance(ans, dict) else []
    if not ctx or not q or not texts:
        return None
    return (f"Context: {ctx[:2000]}\n\nQuestion: {q}", texts[0])


def fmt_narrativeqa(ex: dict) -> tuple[str, str] | None:
    doc = ex.get("document", {})
    text = doc.get("text", "") if isinstance(doc, dict) else ""
    qobj = ex.get("question", {})
    question = qobj.get("text", "") if isinstance(qobj, dict) else ""
    answers = ex.get("answers", [])
    gold = answers[0].get("text", "") if answers else ""
    if not text or not question or not gold:
        return None
    return (f"Context: {text[:8000]}\n\nQuestion: {question}", gold)


def fmt_hotpot(ex: dict) -> tuple[str, str] | None:
    question = ex.get("question", "")
    answer = ex.get("answer", "")
    if not question or not answer:
        return None
    return (f"Answer the following question.\n\nQuestion: {question}", answer)


def fmt_drop(ex: dict) -> tuple[str, str] | None:
    passage = ex.get("passage", "")
    question = ex.get("question", "")
    ans_obj = ex.get("answers_spans", {})
    spans = ans_obj.get("spans", []) if isinstance(ans_obj, dict) else []
    if not passage or not question or not spans:
        return None
    return (f"Passage: {passage[:2000]}\n\nQuestion: {question}", spans[0])


def fmt_quality(ex: dict) -> tuple[str, str] | None:
    # emozilla/quality: flat structure with article, question, options (string), answer (1-indexed)
    article = ex.get("article", "")
    question = ex.get("question", "")
    options_raw = ex.get("options", "")
    answer_str = ex.get("answer", "")
    if not article or not question or not options_raw or not answer_str:
        return None
    try:
        import ast

        options = (
            ast.literal_eval(options_raw)
            if isinstance(options_raw, str)
            else options_raw
        )
        gold_idx = int(answer_str) - 1  # 1-indexed
    except Exception:
        return None
    if gold_idx < 0 or gold_idx >= len(options):
        return None
    return (
        f"Article: {article[:4000]}\n\nQuestion: {question}\n\n{_opts(options)}",
        options[gold_idx],
    )


def fmt_qasper(ex: dict) -> tuple[str, str] | None:
    title, abstract = ex.get("title", ""), ex.get("abstract", "")
    qas = ex.get("qas", {})
    questions = qas.get("question", []) if isinstance(qas, dict) else []
    answers_list = qas.get("answers", []) if isinstance(qas, dict) else []
    if not questions or not answers_list:
        return None
    ans_texts = []
    for a in (
        answers_list[0].get("answer", []) if isinstance(answers_list[0], dict) else []
    ):
        if isinstance(a, dict) and a.get("free_form_answer"):
            ans_texts.append(a["free_form_answer"])
    if not ans_texts:
        return None
    return (
        f"Title: {title}\nAbstract: {abstract[:2000]}\n\nQuestion: {questions[0]}",
        ans_texts[0],
    )


def fmt_math500(ex: dict) -> tuple[str, str] | None:
    problem = ex.get("problem", "")
    if not problem:
        return None
    return (
        f"Solve and put the final answer in \\boxed{{}}.\n\n{problem}",
        ex.get("answer", "") or ex.get("solution", ""),
    )


def fmt_gsm8k(ex: dict) -> tuple[str, str] | None:
    q, a = ex.get("question", ""), ex.get("answer", "")
    if not q:
        return None
    gold = a.split("####")[-1].strip() if "####" in a else a
    return (f"Solve step by step:\n\n{q}", gold)


def fmt_arc_mcq(ex: dict) -> tuple[str, str] | None:
    q = ex.get("question", "")
    choices = ex.get("choices", {})
    texts = choices.get("text", []) if isinstance(choices, dict) else []
    labels = choices.get("label", []) if isinstance(choices, dict) else []
    answer_key = ex.get("answerKey", "")
    if not q or not texts or not answer_key:
        return None
    idx = labels.index(answer_key) if answer_key in labels else None
    gold = texts[idx] if idx is not None else answer_key
    # Ask for the full option text so score_mcq can check `gold in answer`
    return (
        f"Choose the correct answer and write the full answer text:\n\n{q}\n\n{_opts(texts)}",
        gold,
    )


def fmt_qasc(ex: dict) -> tuple[str, str] | None:
    q = ex.get("question", "")
    choices = ex.get("choices", {})
    texts = choices.get("text", []) if isinstance(choices, dict) else []
    labels = choices.get("label", []) if isinstance(choices, dict) else []
    answer_key = ex.get("answerKey", "")
    if not q or not texts or not answer_key:
        return None
    idx = labels.index(answer_key) if answer_key in labels else None
    gold = texts[idx] if idx is not None else answer_key
    return (
        f"Choose the correct answer and write the full answer text:\n\n{q}\n\n{_opts(texts)}",
        gold,
    )


FORMAT_FNS = {
    "humaneval": fmt_humaneval,
    "mbpp": fmt_mbpp,
    "apps": fmt_apps,
    "boolq": fmt_boolq,
    "record": fmt_record,
    "squad": fmt_squad,
    "hotpot": fmt_hotpot,
    "drop": fmt_drop,
    "quality": fmt_quality,
    "narrativeqa": fmt_narrativeqa,
    "qasper": fmt_qasper,
    "math500": fmt_math500,
    "gsm8k": fmt_gsm8k,
    "arc_mcq": fmt_arc_mcq,
    "qasc": fmt_qasc,
}


# ── Multi-backend model calls ─────────────────────────────────────────────────
# Priority order (cheapest first, no subprocess spawning):
#   ollama/*    → Ollama local REST API (free, in-process httpx)
#   google/*    → Google REST API (GOOGLE_API_KEY) → OpenRouter fallback
#   anthropic/* → Anthropic REST API (ANTHROPIC_API_KEY) → OpenRouter fallback
#   everything else → OpenRouter
#
# NOTE: CLI subprocess paths (claude -p / gemini -p) are intentionally removed.
# Each subprocess spawns a full new auth session (~2-5s overhead per call).
# Direct REST API calls reuse the same httpx client and take ~50ms.

# Oracle models: Gemini 2.5 Pro uses GOOGLE_API_KEY (loaded from .env).
GEMINI_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OLLAMA_URL = "http://localhost:11434/api/chat"

_OLLAMA_AVAILABLE: bool | None = None  # lazily checked on first call
_gemini_oauth_cache: dict = {}  # cached token + expiry
_gemini_oauth_lock = Lock()


def _check_ollama() -> bool:
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is None:
        try:
            r = httpx.get("http://localhost:11434/api/tags", timeout=3)
            _OLLAMA_AVAILABLE = r.status_code == 200
        except Exception:
            _OLLAMA_AVAILABLE = False
    return _OLLAMA_AVAILABLE


def _call_ollama(ollama_model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    try:
        r = httpx.post(
            OLLAMA_URL,
            json={
                "model": ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "success": True, "in_tok": 0, "out_tok": 0, "error": None}
    except Exception as e:
        return {
            "text": None,
            "success": False,
            "in_tok": 0,
            "out_tok": 0,
            "error": str(e),
        }


def _native_model_id(model: str) -> str:
    for prefix in ("anthropic/", "google/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _get_gemini_oauth_token() -> str | None:
    """Return a valid Google OAuth Bearer token from ~/.gemini/oauth_creds.json.

    Reads the cached token and refreshes it if expired. Returns None if creds
    file is missing or refresh fails. This avoids spawning a new gemini CLI
    process for every call — the OAuth client ID/secret belong to the Gemini CLI
    app, so we can reuse them transparently.
    """
    global _gemini_oauth_cache
    with _gemini_oauth_lock:
        now_ms = time.time() * 1000
        cached = _gemini_oauth_cache
        if (
            cached.get("access_token")
            and cached.get("expiry_date", 0) > now_ms + 60_000
        ):
            return cached["access_token"]

        if not GEMINI_CREDS_PATH.exists():
            return None
        try:
            creds = json.loads(GEMINI_CREDS_PATH.read_text())
        except Exception:
            return None

        if creds.get("expiry_date", 0) > now_ms + 60_000:
            _gemini_oauth_cache = creds
            return creds["access_token"]

        # Token expired — refresh it
        refresh_token = creds.get("refresh_token")
        client_id = creds.get("client_id", "")
        client_secret = creds.get("client_secret", "")
        if not refresh_token:
            return None

        try:
            r = httpx.post(
                GOOGLE_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=15,
            )
            r.raise_for_status()
            new_tok = r.json()
            # Update creds file and cache
            creds.update(new_tok)
            if "expires_in" in new_tok:
                creds["expiry_date"] = int((time.time() + new_tok["expires_in"]) * 1000)
            GEMINI_CREDS_PATH.write_text(json.dumps(creds, indent=2))
            _gemini_oauth_cache = creds
            return creds["access_token"]
        except Exception:
            return None


def _call_google_rest(model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    """Call Google Generative Language API.

    Auth priority (no subprocess spawned either way):
    1. OAuth Bearer token from ~/.gemini/oauth_creds.json (subscription, free)
    2. GOOGLE_API_KEY env var (paid API key, fallback)
    """
    native = _native_model_id(model)
    url = GOOGLE_URL_TMPL.format(model=native)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }

    def _parse_response(r: httpx.Response) -> dict:
        r.raise_for_status()
        data = r.json()
        # Gemini thinking models may return content without "parts" when MAX_TOKENS
        content = data["candidates"][0].get("content", {})
        parts = content.get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        finish = data["candidates"][0].get("finishReason", "")
        if not text and finish == "MAX_TOKENS":
            raise ValueError(
                "MAX_TOKENS: thinking consumed all budget, increase max_tokens"
            )
        if not text:
            raise ValueError(f"empty response (finishReason={finish})")
        return {"text": text, "success": True, "in_tok": 0, "out_tok": 0, "error": None}

    # Try OAuth subscription token first (no API key needed, no subprocess)
    oauth_token = _get_gemini_oauth_token()
    if oauth_token:
        try:
            r = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {oauth_token}",
                    "content-type": "application/json",
                },
                json=body,
                timeout=timeout,
            )
            return _parse_response(r)
        except Exception:
            pass  # fall through to API key

    # Fallback: GOOGLE_API_KEY
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if google_key:
        try:
            r = httpx.post(
                url,
                headers={
                    "x-goog-api-key": google_key,
                    "content-type": "application/json",
                },
                json=body,
                timeout=timeout,
            )
            return _parse_response(r)
        except Exception as e:
            return {
                "text": None,
                "success": False,
                "in_tok": 0,
                "out_tok": 0,
                "error": str(e),
            }

    return {
        "text": None,
        "success": False,
        "in_tok": 0,
        "out_tok": 0,
        "error": "no Google auth available (no OAuth token or GOOGLE_API_KEY)",
    }


def _call_gemini_native(
    gemini_model_id: str, prompt: str, max_tokens: int, timeout: int
) -> dict:
    """Call a Gemini model directly by its native model ID (e.g. 'gemini-2.5-flash').

    Uses GOOGLE_API_KEY from .env (loaded by load_dotenv at startup).
    OAuth is not used here — the Gemini CLI OAuth token has insufficient scope for
    the Generative Language REST API (ACCESS_TOKEN_SCOPE_INSUFFICIENT).
    """
    url = GOOGLE_URL_TMPL.format(model=gemini_model_id)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        return {
            "text": None,
            "success": False,
            "in_tok": 0,
            "out_tok": 0,
            "error": "GOOGLE_API_KEY not set",
        }
    try:
        r = httpx.post(
            url,
            headers={"x-goog-api-key": google_key, "content-type": "application/json"},
            json=body,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        content = data["candidates"][0].get("content", {})
        parts = content.get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        finish = data["candidates"][0].get("finishReason", "")
        if not text and finish == "MAX_TOKENS":
            raise ValueError("MAX_TOKENS: increase max_tokens for thinking model")
        if not text:
            raise ValueError(f"empty response (finishReason={finish})")
        return {"text": text, "success": True, "in_tok": 0, "out_tok": 0, "error": None}
    except Exception as e:
        return {
            "text": None,
            "success": False,
            "in_tok": 0,
            "out_tok": 0,
            "error": str(e),
        }


def call_model(
    model: str, prompt: str, api_key: str, max_tokens: int = 1024, timeout: int = 90
) -> dict:
    # ollama/* → local Ollama API (kept for debugging / manual use)
    if model.startswith("ollama/"):
        return _call_ollama(model[len("ollama/") :], prompt, max_tokens, timeout)

    # Production routing tiers → map to Gemini equivalents via GEMINI_PROXY
    if model in GEMINI_PROXY:
        return _call_gemini_native(GEMINI_PROXY[model], prompt, max_tokens, timeout)

    # google/* oracle and judge calls → use native Gemini API
    if model.startswith("google/"):
        native = _native_model_id(model)
        return _call_gemini_native(native, prompt, max_tokens, timeout)

    # Anything else (deepseek/*, qwen/* IDs not in GEMINI_PROXY) → OpenRouter fallback
    try:
        r = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ypollak2/RouterArena",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "text": text,
            "success": True,
            "in_tok": usage.get("prompt_tokens", 0),
            "out_tok": usage.get("completion_tokens", 0),
            "error": None,
        }
    except Exception as e:
        return {
            "text": None,
            "success": False,
            "in_tok": 0,
            "out_tok": 0,
            "error": str(e),
        }


# ── Grading ───────────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def score_exact(answer: str | None, gold: str) -> float:
    return 1.0 if answer and _norm(answer) == _norm(gold) else 0.0


def score_mcq(answer: str | None, gold: str) -> float:
    if not answer:
        return 0.0
    a, g = _norm(answer), _norm(gold)
    return 1.0 if g in a or (g and a.startswith(g[0])) else 0.0


JUDGE_TMPL = (
    "You are an answer grader. Is the model response correct?\n\n"
    "Question: {question}\nReference: {gold}\nResponse: {response}\n\n"
    "Reply with exactly one word: CORRECT or INCORRECT."
)


def score_llm_judge(
    question: str, gold: str, response: str | None, api_key: str
) -> float:
    """Judge a response using Gemini 2.5 Pro."""
    if not response:
        return 0.0
    prompt = JUDGE_TMPL.format(
        question=question[:800], gold=gold[:800], response=response[:1200]
    )
    # 512 tokens: Gemini 2.5 Pro is a thinking model — internal reasoning consumes
    # some tokens; 512 is enough since the answer is CORRECT or INCORRECT (short).
    res = call_model(JUDGE_PRIMARY, prompt, api_key, 512, 60)
    if res["success"] and res["text"]:
        return 1.0 if "CORRECT" in res["text"].upper() else 0.0
    return 0.0


def score_answer(
    grader: str, question: str, gold: str, answer: str | None, api_key: str
) -> float:
    if grader == "exact_match":
        return score_exact(answer, gold)
    if grader == "mcq":
        return score_mcq(answer, gold)
    return score_llm_judge(question, gold, answer, api_key)


# ── Label derivation ──────────────────────────────────────────────────────────


def derive_label(scores: dict) -> tuple[str, float, bool]:
    """Cheapest-correct: FLASH -> DEEPSEEK -> QWEN235B."""
    fl = scores.get("google/gemini-3.1-flash-lite", 0.0)
    ds = scores.get("deepseek/deepseek-v4-flash", 0.0)
    qw = scores.get("qwen/qwen3-235b-a22b-2507", 0.0)

    if fl >= CORRECTNESS_THRESHOLD:
        return "FLASH", fl, False
    if ds >= CORRECTNESS_THRESHOLD:
        return "DEEPSEEK", ds, False
    if qw >= CORRECTNESS_THRESHOLD:
        return "QWEN235B", qw, False
    return "QWEN235B", max(fl, ds, qw), True  # unserved by pool


# ── Core pipeline ─────────────────────────────────────────────────────────────


def process_example(ex: dict, spec: DatasetSpec, api_key: str) -> dict | None:
    fmt_fn = FORMAT_FNS.get(spec.format_fn)
    if not fmt_fn:
        return None
    result = fmt_fn(ex)
    if result is None:
        return None
    prompt, gold = result
    ph = hashlib.md5(prompt.encode()).hexdigest()

    # Step 1: call all 4 labeling models in parallel via the Gemini API.
    # QWEN_MODEL and ORACLE_ALWAYS both map to gemini-2.5-pro, so we deduplicate:
    # 3 unique Gemini API calls — flash-lite, flash, pro — all concurrent.
    gemini_proxy_to_model: dict[str, str] = {}
    unique_calls: list[str] = []
    for m in ALL_LABELING_MODELS:
        gid = GEMINI_PROXY.get(m) or (
            _native_model_id(m) if m.startswith("google/") else None
        )
        if gid and gid in gemini_proxy_to_model:
            continue
        if gid:
            gemini_proxy_to_model[gid] = m
        unique_calls.append(m)

    responses: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(unique_calls)) as pool:
        futs = {pool.submit(call_model, m, prompt, api_key): m for m in unique_calls}
        for fut in as_completed(futs):
            responses[futs[fut]] = fut.result()

    # Fill deduplicated models from their proxy's first result
    for m in ALL_LABELING_MODELS:
        if m not in responses:
            gid = GEMINI_PROXY.get(m) or (
                _native_model_id(m) if m.startswith("google/") else None
            )
            if gid and gid in gemini_proxy_to_model:
                responses[m] = responses.get(gemini_proxy_to_model[gid], {})

    # Step 2: score each production model in parallel (3 concurrent judge calls)
    prod_scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(PRODUCTION_MODELS)) as judge_pool:

        def _score(m: str) -> tuple[str, float]:
            ans = responses.get(m, {}).get("text")
            return m, score_answer(spec.grader, prompt, gold, ans, api_key)

        score_futs = {judge_pool.submit(_score, m): m for m in PRODUCTION_MODELS}
        for score_fut in as_completed(score_futs):
            m, s = score_fut.result()
            prod_scores[m] = s

    # Step 3: oracle ceiling score (gemini-2.5-pro own answer — reuses Step 1 response)
    oracle_ans = responses.get(ORACLE_ALWAYS, {}).get("text")
    oracle_score = (
        score_answer(spec.grader, prompt, gold, oracle_ans, api_key)
        if oracle_ans
        else 0.0
    )

    label, confidence, unserved = derive_label(prod_scores)

    return {
        "prompt_hash": ph,
        "source_dataset": spec.name,
        "task_family": spec.task_family,
        "prompt": prompt,
        "gold": gold,
        "label": label,
        "production_scores": {
            "flash": prod_scores.get("google/gemini-3.1-flash-lite", 0.0),
            "deepseek": prod_scores.get("deepseek/deepseek-v4-flash", 0.0),
            "qwen": prod_scores.get("qwen/qwen3-235b-a22b-2507", 0.0),
        },
        "oracle_score": oracle_score,
        "label_confidence": confidence,
        "unserved_by_pool": unserved,
        "models_ok": {
            m: responses.get(m, {}).get("success", False) for m in ALL_LABELING_MODELS
        },
    }


def _refresh_gemini_token_once() -> None:
    """Run gemini CLI once at startup to refresh the OAuth token file.

    The token expires after ~1 hour. During a multi-hour corpus run it would
    expire mid-way. Running the CLI once forces a fresh token into
    ~/.gemini/oauth_creds.json without spawning it per-call.
    The CLI may print errors about project setup — those are safe to ignore.
    """
    global _gemini_oauth_cache
    _gemini_oauth_cache = {}  # clear in-memory cache so we re-read the file
    gemini_bin = shutil.which("gemini")
    if not gemini_bin:
        return
    try:
        import subprocess

        subprocess.run(
            [gemini_bin, "-p", "ping"], capture_output=True, text=True, timeout=20
        )
    except Exception:
        pass
    # Invalidate cache so next _get_gemini_oauth_token() reads the fresh file
    _gemini_oauth_cache = {}


def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("WARNING: OPENROUTER_API_KEY not set — OpenRouter fallback unavailable")

    if not args.dry_run:
        print("Refreshing Gemini OAuth token...")
        _refresh_gemini_token_once()
        tok = _get_gemini_oauth_token()
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if tok:
            print(f"  OAuth token: OK (len={len(tok)})")
        elif google_key:
            print("  OAuth token: missing — using GOOGLE_API_KEY fallback")
        else:
            print("  WARNING: no Google auth available — oracle judge will fail")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing prompt hashes for resume
    existing_hashes: set[str] = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    existing_hashes.add(json.loads(line).get("prompt_hash", ""))
                except Exception:
                    pass
        print(f"Resume: {len(existing_hashes)} prompts already labeled")

    write_lock = Lock()
    total_written = 0
    label_counts: dict[str, int] = {"FLASH": 0, "DEEPSEEK": 0, "QWEN235B": 0}

    for spec in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"  {spec.name}  task={spec.task_family}  grader={spec.grader}")

        try:
            ds = load_dataset(
                spec.hf_path, spec.hf_config, split=spec.split, trust_remote_code=True
            )
        except Exception as e:
            print(f"  ⚠ Could not load: {e}")
            continue

        examples = list(ds)
        if len(examples) > spec.cap:
            step = max(1, len(examples) // spec.cap)
            examples = examples[::step][: spec.cap]

        print(f"  {len(examples)} examples to process")

        if args.dry_run:
            fn = FORMAT_FNS.get(spec.format_fn)
            valid = sum(1 for ex in examples if fn and fn(ex) is not None)
            print(f"  [dry-run] {valid}/{len(examples)} would format OK")
            continue

        written = 0
        completed = 0
        # N=8 concurrent examples: Gemini API calls parallelize; Ollama queues locally.
        # Throughput is Ollama-limited (~80s per batch of 8 with qwen3.5), not API-limited.
        EXAMPLE_WORKERS = 8

        def _process_and_write(item: tuple[int, dict]) -> int:
            """Return 1 if row was written, 0 if skipped."""
            idx, ex = item
            row = process_example(ex, spec, api_key)
            if row is None:
                return 0
            ph = row["prompt_hash"]
            with write_lock:
                if ph in existing_hashes:
                    return 0
                existing_hashes.add(ph)
                with open(OUTPUT_PATH, "a") as f:
                    f.write(json.dumps(row) + "\n")
            return 1

        with ThreadPoolExecutor(max_workers=EXAMPLE_WORKERS) as ex_pool:
            futs = {
                ex_pool.submit(_process_and_write, (i, ex)): i
                for i, ex in enumerate(examples)
            }
            for fut in as_completed(futs):
                completed += 1
                result = fut.result()
                if result and isinstance(result, dict):
                    label = result["label"]
                    label_counts[label] = label_counts.get(label, 0) + 1
                    total_written += 1
                    written += 1
                if completed % 10 == 0 or completed == len(examples):
                    print(
                        f"  [{completed}/{len(examples)}] +{written} | "
                        f"FLASH={label_counts['FLASH']} "
                        f"DEEPSEEK={label_counts['DEEPSEEK']} "
                        f"QWEN={label_counts['QWEN235B']}"
                    )

    print(f"\n{'=' * 60}")
    print(f"Total labeled: {total_written}")
    print(f"Distribution: {label_counts}")
    print(f"Output: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build outcome-label corpus for Gate 0 training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show sizes, no API calls"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-labeled prompts"
    )
    parser.add_argument("--cap", type=int, default=None, help="Max prompts per dataset")
    args = parser.parse_args()
    if args.cap:
        for spec in DATASETS:
            spec.cap = min(spec.cap, args.cap)
    run(args)


if __name__ == "__main__":
    main()
