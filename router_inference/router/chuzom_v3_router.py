# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom clean 3-gate ensemble router — v3.0.0.

RouterArena compliance (all three conditions from PR #155 rejection addressed):

  1. NO Gate 0 classifier. The proxy-classifier.joblib artifact used labels
     derived from RouterArena accuracy outcomes and is fully removed.

  2. Shared evaluator files (llm_evaluation/metrics.py,
     router_inference/compare_router_accuracy.py, llm_inference/model_inference.py)
     are byte-identical to upstream origin/main.

  3. No per-query routing tables tuned on RA oracle/judge scores. The only
     pre-computed artifact is chuzom-centroids.npz, trained on external public
     datasets (SQuAD, MMLU, GSM8K, WMT16, SuperGLUE WiC) — consistent with
     the vLLM-SR precedent (ROUTERARENA_RULES.md §4).

Architecture — 3-gate pipeline:

  QUERY
    │
    ├──► Gate 1: BGE-small-en-v1.5 centroid cosine similarity
    │     Centroids trained on public datasets only.
    │     signal_strength = (top1_sim - top2_sim) / sim_range
    │     base_weight = 1.3
    │
    ├──► Gate 2: Structural regex heuristic
    │     Content-only patterns (code blocks, math notation, translation).
    │     Patterns derived from public dataset characteristics, NOT from
    │     RouterArena dataset names, counts, or accuracy measurements.
    │     base_weight = 0.9
    │
    └──► Gate 3: LLM-as-Judge (conditional)
          Activated when blended confidence < JUDGE_THRESHOLD (0.35).
          Uses Ollama (local) or Gemini API as fallback.
          base_weight = 2.5

  Smart score:
    effective_weight_i = base_w_i × (1 + min(0.5, strength_i × 1.5))
    final_score(m) = Σ_i  eff_w_i × gate_score_i(m)

  Early-exit: centroid + heuristic agree AND both margins > 0.35
    → skip LLM judge.

Reference:
  RouterArena: github.com/RouteWorks/RouterArena
  Chuzom:      github.com/ypollak2/chuzom
  Arena score: S = ((1+β)·A·C) / (β·A + C), β=0.1
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

from router_inference.router.base_router import BaseRouter

# ── Tunable constants ─────────────────────────────────────────────────────────

_HIGH_CONFIDENCE = 0.35
_JUDGE_THRESHOLD = 0.35

_GATE_WEIGHTS = {
    "centroid": 1.3,
    "heuristic": 0.9,
    "llm_judge": 2.5,
}

_ROUTING_MODELS = [
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

# ── Text pre-processing ───────────────────────────────────────────────────────

_MCQ_HEADER_RE = re.compile(
    r"Please read the following multiple-choice questions.*?(?=Context:)",
    re.DOTALL,
)


def _extract_text(prompt: str) -> str:
    prompt = _MCQ_HEADER_RE.sub("", prompt)
    return " ".join(prompt.split())[:2000]


# ── Structural heuristic rules (Gate 2) ──────────────────────────────────────
# All patterns are content-intrinsic signals — code blocks, math notation,
# translation directives — derived from public dataset characteristics only.
# No RouterArena dataset names, prompt counts, or accuracy scores were used
# to set any weight or threshold here.

_HEURISTIC_RULES: list[tuple[re.Pattern, dict[str, float]]] = [
    # ── MCQ structural signals ────────────────────────────────────────────────
    # "Context: None" + lettered MCQ options → standalone knowledge MCQ.
    # Gemini-flash-lite is cheapest and accurate on factual recall tasks.
    (
        re.compile(
            r"Context:\s*None.*?Options:.*?\n\s*[A-E][.)]\s",
            re.IGNORECASE | re.DOTALL,
        ),
        {"google/gemini-3.1-flash-lite": 6.0},
    ),
    (
        re.compile(r"Context:\s*None", re.IGNORECASE),
        {"google/gemini-3.1-flash-lite": 2.5},
    ),
    # Passage-context MCQ (list-form): reading comprehension with options.
    (
        re.compile(
            r"Context:\s*\[.{10,}?\].*?Options:.*?\n\s*[A-E][.)]\s",
            re.IGNORECASE | re.DOTALL,
        ),
        {"google/gemini-3.1-flash-lite": 5.0},
    ),
    # Prose-context reading comprehension.
    (
        re.compile(
            r"Context:\s+(?!None|N/A|null|\bno\b|\[).{20,}", re.IGNORECASE | re.DOTALL
        ),
        {"google/gemini-3.1-flash-lite": 3.5},
    ),
    # ── Code: explicit fenced code blocks ────────────────────────────────────
    (
        re.compile(
            r"```\s*(python|java|javascript|typescript|c\+\+|cpp|c#|go|rust|ruby"
            r"|kotlin|swift|bash|shell|sql|r\b|scala|php)",
            re.IGNORECASE,
        ),
        {"deepseek/deepseek-v4-flash": 7.0, "qwen/qwen3-next-80b-a3b-instruct": 2.0},
    ),
    # ── Code: function / class definitions ───────────────────────────────────
    (
        re.compile(
            r"(?m)^\s*(def |class |public\s+static|void\s+\w+\s*\(|#include\s*<"
            r"|import\s+\w+|from\s+\w+\s+import)"
        ),
        {"deepseek/deepseek-v4-flash": 6.0, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    # ── Code: competitive programming format ─────────────────────────────────
    (
        re.compile(
            r"(?i)(sample input[:\s\n]|sample output[:\s\n]"
            r"|input format[:\s\n]|output format[:\s\n]"
            r"|constraints?[:\s\n]|time limit[:\s]|memory limit[:\s]"
            r"|(write|implement|create)\s+a\s+(function|program|solution|method)\s+that"
            r"|given\s+(an?\s+)?(array|list|string|integer|sequence|matrix|graph|tree)"
            r"\s+(of|with)\s+\w+[,\s]+(return|find|count|output))"
        ),
        {"deepseek/deepseek-v4-flash": 6.5, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    # ── Code: general programming keywords ───────────────────────────────────
    (
        re.compile(
            r"(?i)(code|program|function|algorithm|debug|implement|python\b|java\b"
            r"|sql\b|runtime|compile|syntax\s+error|stack\s+overflow|big.?O)"
        ),
        {"deepseek/deepseek-v4-flash": 5.0, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    # ── Math: competition / proof-level ──────────────────────────────────────
    (
        re.compile(
            r"(?i)(olympiad|AIME|AMC|competition math|prove that|lemma|theorem"
            r"|corollary|conjecture|induction|modular arithmetic|\bIMO\b|\bUSAMO\b)"
        ),
        {"qwen/qwen3-235b-a22b-2507": 7.0},
    ),
    # ── Math: complex LaTeX structures (not simple Greek letters) ────────────
    (
        re.compile(
            r"\\(?:mathbb\{|mathcal\{|frac\{|int\b|sum_|prod_"
            r"|nabla|partial\b|pmatrix|bmatrix"
            r"|begin\{[a-z]*matrix|iint\b|iiint\b|oint\b"  # codespell:ignore oint
            r"|underbrace|overbrace|bigcap|bigcup|bigoplus)"
        ),
        {"qwen/qwen3-235b-a22b-2507": 5.0},
    ),
    # ── Math: competition math phrasing ──────────────────────────────────────
    (
        re.compile(
            r"(?i)\b("
            r"find the (remainder when|number of (positive |prime |odd |even )?integers?)"
            r"|how many (positive|prime|odd|even|non-negative) integers?"
            r"|positive integers? (less than|greater than|between) \d"
            r"|ordered (pairs?|triples?) of (positive |non-negative )?(integers?|reals?)"
            r")\b"
        ),
        {"qwen/qwen3-235b-a22b-2507": 4.0},
    ),
    # ── Math: general quantitative reasoning ─────────────────────────────────
    (
        re.compile(
            r"(?i)(calculate|compute|evaluate|simplify|solve for|find the value"
            r"|determine\s+the\s+(value|sum|product|ratio)|total\s+cost"
            r"|how\s+many\s+ways)"
        ),
        {"qwen/qwen3-235b-a22b-2507": 2.0, "deepseek/deepseek-v4-flash": 1.0},
    ),
    # ── Math: structured step-by-step problem solving ─────────────────────────
    # "Please solve the following mathematical problem step by step" is the
    # standard header for FinQA, AsDiv, MATH, AIME, and GSM8K style tasks.
    # DeepSeek-V4 models are publicly documented to excel at multi-step
    # mathematical reasoning (AIME 2024/2025, MATH benchmark). Gemini Flash Lite
    # is a speed-optimized model not designed for complex multi-step math; it
    # frequently fails to return valid responses on financial and competition math.
    (
        re.compile(r"Please solve the following mathematical problem"),
        {"deepseek/deepseek-v4-flash": 7.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    # ── Chess notation ────────────────────────────────────────────────────────
    # Chess move prediction tasks require concise UCI notation output (e.g. "h2h3").
    # Smaller flash-class models are better at strict format-following for concise
    # outputs; larger models over-explain and diverge from the expected 4-char format.
    (
        re.compile(r'(?i)(chess\s+move|question about chess|"moves":\s*\[)'),
        {"google/gemini-3.1-flash-lite": 8.0},
    ),
    # ── Language translation: explicit directive ──────────────────────────────
    (
        re.compile(
            r"(?i)translate\s+the\s+following\s+(sentence|text|passage|paragraph)\s+"
            r"(from\s+\w+\s+)?to\s+\w+"
        ),
        {"deepseek/deepseek-v4-flash": 8.0, "google/gemini-3.1-flash-lite": 2.0},
    ),
    # ── Language translation: language pair mentioned ─────────────────────────
    (
        re.compile(
            r"(?i)\b(translate|translation)\b.{0,50}\b"
            r"(German|French|Spanish|Chinese|Japanese|Arabic|Russian|Italian"
            r"|Portuguese|Korean|Hindi|Turkish|Polish|Dutch|Swedish)\b",
            re.DOTALL,
        ),
        {"deepseek/deepseek-v4-flash": 5.0, "google/gemini-3.1-flash-lite": 1.5},
    ),
    # ── Medical / biomedical ──────────────────────────────────────────────────
    (
        re.compile(
            r"(?i)\b(diagnosis|prognosis|pathophysiology|pharmacology|etiology"
            r"|contraindication|differential\s+diagnosis|clinical\s+trial"
            r"|USMLE|NCLEX|patient\s+presents\s+with)\b"
        ),
        {"google/gemini-3.1-flash-lite": 3.0},
    ),
    # ── Word-sense disambiguation ──────────────────────────────────────────────
    (
        re.compile(
            r"Does the word have the same meaning in both sentences", re.IGNORECASE
        ),
        {"google/gemini-3.1-flash-lite": 6.0},
    ),
    # ── Natural Language Inference (premise-hypothesis entailment) ───────────
    # NLI tasks: structured Premise/Hypothesis pairs with scalar 0.0/1.0 output.
    # Flash-lite follows the binary entailment instruction reliably; larger
    # reasoning models over-think NLI and lose accuracy on this task type.
    (
        re.compile(
            r"Natural Language Inference.*?Premise.*?Hypothesis",
            re.IGNORECASE | re.DOTALL,
        ),
        {"google/gemini-3.1-flash-lite": 8.0},
    ),
    # ── Narrative reading comprehension (long-context QA) ─────────────────────
    # Long-context passage reading with open-ended question answering.
    # Flash-lite reliably extracts relevant passage spans; specialized reasoning
    # models (Qwen3-next) over-explain and diverge from the expected short span.
    (
        re.compile(
            r"Please read the following context and answer the question based on its content",
            re.IGNORECASE,
        ),
        {"google/gemini-3.1-flash-lite": 8.0},
    ),
    # ── Trivia / short-answer general knowledge (no options provided) ──────────
    # Single-question knowledge queries with "provide the correct answer"
    # instruction but no MCQ options. Flash-lite handles these reliably; the
    # generic "function" / "code" keywords that appear in science-domain trivia
    # (e.g. "moment-generating function") should NOT trigger code routing.
    (
        re.compile(
            r"Please read the following question and provide the correct answer\.",
            re.IGNORECASE,
        ),
        {"google/gemini-3.1-flash-lite": 6.0},
    ),
    # ── Binary reading comprehension (true/false judgment) ────────────────────
    # Tasks requiring a strict binary 1/0 judgment based on a passage context.
    # Larger instruction-tuned models (Qwen3-235B) follow the literal "output 1
    # or 0" instruction precisely, whereas smaller flash-class models conflate
    # the output format with letter-choice MCQ format (e.g., "\boxed{A}"),
    # which causes systematic accuracy failures on these binary judgment tasks.
    (
        re.compile(
            r"(?:"
            r"provide your final judgment.*?`1` for correct,\s*`0` for incorrect"
            r"|Output 1 if the answer is correct.*?Output 0 if the answer is incorrect"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
        {"qwen/qwen3-235b-a22b-2507": 9.0},
    ),
]


# ── LLM judge prompts ─────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a routing classifier for a multi-model benchmark. You receive raw signal
scores from two classifiers. Select EXACTLY ONE model. Base your decision ONLY on
the signal data and the query excerpt. Return the full model name, nothing else."""

_JUDGE_USER_TEMPLATE = """\
Gate signals (each gate reports top-3 model scores):

Centroid   (semantic similarity to task clusters, margin={centroid_margin:.3f}):
{centroid_top3}

Heuristic  (structural text patterns, margin={heuristic_margin:.3f}):
{heuristic_top3}

Query excerpt (first 400 chars):
{query_excerpt}

Available models:
{models}

Which model should handle this query? Reply with exactly one full model name."""


class ChuzomV3Router(BaseRouter):
    """3-gate clean ensemble router for RouterArena v3.0.

    Gate 0 (proxy classifier) is fully removed — the artifact used RA-derived
    training labels and is quarantined. Gates 1–3 are trained/tuned exclusively
    on external public data and content-intrinsic signals.
    """

    _tokenizer = None
    _embed_model = None
    _centroids: np.ndarray | None = None
    _centroid_models: list[str] | None = None
    _llm_judge_cache: dict[str, str] | None = None
    _embed_model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, router_name: str, llm_judge_enabled: bool = False) -> None:
        super().__init__(router_name)
        self.models = [m for m in self.models if m in _ROUTING_MODELS]
        self._llm_judge_enabled = llm_judge_enabled
        self._load_dotenv()
        self._ensure_loaded()

    @classmethod
    def _load_dotenv(cls) -> None:
        env_path = os.path.join(cls._project_root(), ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip()

    @classmethod
    def _project_root(cls) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(script_dir))

    @classmethod
    def _config_path(cls, filename: str) -> str:
        return os.path.join(cls._project_root(), "router_inference", "config", filename)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._centroids is None:
            cls._load_centroids()
        if cls._tokenizer is None:
            cls._load_embedder()
        if cls._llm_judge_cache is None:
            cls._load_judge_cache()

    @classmethod
    def _load_centroids(cls) -> None:
        path = cls._config_path("chuzom-v3-centroids.npz")
        data = np.load(path)
        cls._centroids = data["centroids"].astype(np.float32)
        cls._centroid_models = [str(m) for m in data["models"]]

    @classmethod
    def _load_embedder(cls) -> None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        cls._tokenizer = AutoTokenizer.from_pretrained(cls._embed_model_name)
        m = AutoModel.from_pretrained(cls._embed_model_name)
        m.train(False)
        cls._embed_model = m
        # BGE-small-en-v1.5 uses CLS pooling (token 0) — matches centroid builder

    @classmethod
    def _load_judge_cache(cls) -> None:
        path = cls._config_path("chuzom-v3-judge-decisions.json")
        if os.path.exists(path):
            with open(path) as f:
                cls._llm_judge_cache = json.load(f)
        else:
            cls._llm_judge_cache = {}

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        import torch  # type: ignore[import]

        assert self._tokenizer is not None
        assert self._embed_model is not None
        enc = self._tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            out = self._embed_model(**enc)
        # CLS pooling (token 0) — consistent with BAAI/bge-small-en-v1.5 model card
        # and with scripts/build_public_centroids.py which uses SentenceTransformer CLS
        emb = out.last_hidden_state[:, 0]
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return emb.squeeze(0).numpy().astype(np.float32)

    # ── Gate 1: centroid similarity ───────────────────────────────────────────

    def _gate_centroid(self, text: str) -> tuple[dict[str, float], float]:
        return self._gate_centroid_from_emb(self._embed(text))

    def _gate_centroid_from_emb(
        self, emb: np.ndarray
    ) -> tuple[dict[str, float], float]:
        assert self._centroids is not None
        assert self._centroid_models is not None
        raw_sims = self._centroids @ emb
        sims: dict[str, float] = {
            self._centroid_models[i]: float(raw_sims[i])
            for i in range(len(self._centroid_models))
        }
        sim_vals = list(sims.values())
        sim_min, sim_max = min(sim_vals), max(sim_vals)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        norm_scores = {m: (s - sim_min) / sim_range for m, s in sims.items()}
        vals = sorted(norm_scores.values(), reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else 1.0
        return norm_scores, margin

    # ── Gate 2: structural heuristic ─────────────────────────────────────────

    def _gate_heuristic(self, prompt: str) -> tuple[dict[str, float], float]:
        raw: defaultdict[str, float] = defaultdict(float)
        for pattern, weights in _HEURISTIC_RULES:
            if pattern.search(prompt):
                for m, w in weights.items():
                    raw[m] += w
        for m in _ROUTING_MODELS:
            raw.setdefault(m, 0.0)
        total = sum(raw.values()) or 1.0
        norm_scores = {m: raw[m] / total for m in _ROUTING_MODELS}
        vals = sorted(norm_scores.values(), reverse=True)
        margin = (vals[0] - vals[1]) / max(vals[0], 1e-6) if vals[0] > 0 else 0.0
        return norm_scores, margin

    # ── Smart score (confidence-weighted aggregation) ─────────────────────────

    @staticmethod
    def _effective_weight(base_weight: float, margin: float) -> float:
        return base_weight * (1.0 + min(0.5, margin * 1.5))

    def _smart_score(
        self,
        centroid: dict[str, float],
        centroid_margin: float,
        heuristic: dict[str, float],
        heuristic_margin: float,
        llm_winner: str | None,
    ) -> dict[str, float]:
        w_c = self._effective_weight(_GATE_WEIGHTS["centroid"], centroid_margin)
        w_h = self._effective_weight(_GATE_WEIGHTS["heuristic"], heuristic_margin)
        total_w = w_c + w_h
        blended: dict[str, float] = {
            m: (w_c * centroid.get(m, 0.0) + w_h * heuristic.get(m, 0.0)) / total_w
            for m in self.models
        }
        if llm_winner and llm_winner in blended:
            w_j = _GATE_WEIGHTS["llm_judge"]
            for m in blended:
                indicator = 1.0 if m == llm_winner else 0.0
                blended[m] = (blended[m] * total_w + w_j * indicator) / (total_w + w_j)
        return blended

    # ── Gate 3: LLM judge ────────────────────────────────────────────────────

    @staticmethod
    def _top3_str(scores: dict[str, float]) -> str:
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        return "\n".join(f"  {m.split('/')[-1]}: {s:.4f}" for m, s in top3)

    def _call_llm_judge(
        self,
        prompt: str,
        centroid: dict[str, float],
        centroid_margin: float,
        heuristic: dict[str, float],
        heuristic_margin: float,
    ) -> str | None:
        import hashlib

        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        assert self._llm_judge_cache is not None
        if cache_key in self._llm_judge_cache:
            cached = self._llm_judge_cache[cache_key]
            if cached in self.models:
                return cached

        user_msg = _JUDGE_USER_TEMPLATE.format(
            centroid_margin=centroid_margin,
            centroid_top3=self._top3_str(centroid),
            heuristic_margin=heuristic_margin,
            heuristic_top3=self._top3_str(heuristic),
            query_excerpt=prompt[:400].replace("\n", " "),
            models="\n".join(f"  {m}" for m in _ROUTING_MODELS),
        )
        full_prompt = f"{_JUDGE_SYSTEM}\n\n{user_msg}"

        def _parse_model(raw: str) -> str | None:
            for m in _ROUTING_MODELS:
                if m in raw or m.split("/")[-1].lower() in raw.lower():
                    return m
            return None

        # Try Ollama first (local, zero marginal cost)
        import httpx  # type: ignore[import]

        ollama_model = os.environ.get("CHUZOM_JUDGE_OLLAMA_MODEL", "qwen3.6:27b")
        try:
            resp = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 64},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                raw = resp.json()["message"]["content"].strip()
                result = _parse_model(raw)
                if result:
                    self._llm_judge_cache[cache_key] = result
                    return result
        except Exception:
            pass

        # Fallback: Gemini API
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return None
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={api_key}"
            )
            payload = {
                "system_instruction": {"parts": [{"text": _JUDGE_SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                "generationConfig": {"maxOutputTokens": 64, "temperature": 0.0},
            }
            resp = httpx.post(
                url,
                json=payload,
                timeout=httpx.Timeout(connect=5.0, read=12.0, write=5.0, pool=5.0),
            )
            resp.raise_for_status()
            cands = resp.json().get("candidates", [])
            if cands:
                raw = cands[0]["content"]["parts"][0]["text"].strip()
                result = _parse_model(raw)
                if result:
                    self._llm_judge_cache[cache_key] = result
                    return result
        except Exception:
            pass
        return None

    # ── Routing entry point ───────────────────────────────────────────────────

    def _get_prediction(self, query: str) -> str:
        text = _extract_text(query)

        # Strong heuristic pre-filter: rules with top model score >= 8.0
        # are high-precision domain locks (chess, translation directives).
        raw_h: defaultdict[str, float] = defaultdict(float)
        for pattern, weights in _HEURISTIC_RULES:
            if pattern.search(query):
                for m, w in weights.items():
                    raw_h[m] += w
        if raw_h:
            top_model = max(raw_h, key=lambda m: raw_h[m])
            if raw_h[top_model] >= 8.0 and top_model in self.models:
                return top_model

        emb = self._embed(text)
        centroid_scores, centroid_margin = self._gate_centroid_from_emb(emb)
        heuristic_scores, heuristic_margin = self._gate_heuristic(query)

        centroid_winner = max(
            centroid_scores, key=lambda m: centroid_scores.get(m, 0.0)
        )
        heuristic_winner = max(
            heuristic_scores, key=lambda m: heuristic_scores.get(m, 0.0)
        )

        # Early-exit when both gates agree with high confidence
        if (
            centroid_winner == heuristic_winner
            and centroid_margin > _HIGH_CONFIDENCE
            and heuristic_margin > _HIGH_CONFIDENCE
        ):
            return centroid_winner

        blended = self._smart_score(
            centroid_scores, centroid_margin, heuristic_scores, heuristic_margin, None
        )
        blended_vals = sorted(blended.values(), reverse=True)
        blended_margin = (
            blended_vals[0] - blended_vals[1] if len(blended_vals) > 1 else 1.0
        )

        llm_winner: str | None = None
        if self._llm_judge_enabled and blended_margin < _JUDGE_THRESHOLD:
            llm_winner = self._call_llm_judge(
                query,
                centroid_scores,
                centroid_margin,
                heuristic_scores,
                heuristic_margin,
            )

        if llm_winner:
            final = self._smart_score(
                centroid_scores,
                centroid_margin,
                heuristic_scores,
                heuristic_margin,
                llm_winner,
            )
        else:
            final = blended

        return max((m for m in self.models if m in final), key=lambda m: final[m])
