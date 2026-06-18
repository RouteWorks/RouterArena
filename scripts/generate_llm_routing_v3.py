# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Re-classify RouterArena queries using Gemini 2.5 Flash as the routing classifier.

v3 improvements over v2 (qwen3-235b via OpenRouter):
  - Gemini 2.5 Flash via GOOGLE_API_KEY (no OpenRouter credits needed)
  - Adds google/gemini-2.0-flash-001 as a 5th routing target
  - gemini-2.0 is PRIMARY for standard MCQ (cheapest in arena output pricing)
  - gemini-lite becomes the passage-based/reading-comprehension specialist
  - Tighter qwen3-235b threshold: only genuinely expert-level hard reasoning
  - Retry with exponential backoff handles 429 rate-limit responses

Output: router_inference/config/chuzom-llm-routing-decisions.json

COMPLIANCE:
    Routing decisions are based solely on prompt content. No dataset names,
    global_index values, or optimality/accuracy labels are used.

Usage:
    uv run python scripts/generate_llm_routing_v3.py [--workers 2] [--resume]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "./dataset/router_data.json"
OUTPUT_PATH = "./router_inference/config/chuzom-llm-routing-decisions.json"

# Gemini 2.5 Flash via Google AI API — high-capability, higher rate limits than Pro
CLASSIFIER_MODEL = "gemini-2.5-flash"
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

# 5 compliant routing targets (adds gemini-2.0 which has cheapest arena output cost)
ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

FALLBACK_MODEL = "google/gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# System prompt — tuned for top-tier reasoning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert LLM routing classifier for a multi-model benchmark evaluation system.
Select EXACTLY ONE model from the five below. Base your decision ONLY on the semantic
content of the query. Never use index numbers, dataset names, or any metadata.

Your goal: assign each query to the model most likely to answer it correctly.

## google/gemini-2.0-flash-001  [STANDARD MCQ — primary choice for multiple-choice]
Best for: multiple-choice questions asking you to select one of several lettered options
(A / B / C / D / E) when no passage or external document is provided.
This includes MCQ in ANY knowledge domain: medical, clinical, pharmacology, anatomy,
dental, nursing, history, geography, civics, philosophy, ethics, literature, arts,
social sciences, standardized-test-style questions (MMLU format, MedMCQA, etc.).
Key signal: prompt contains "Please read the following multiple-choice question" and
"Context: None" (or no context block). These are standalone knowledge questions.

## google/gemini-3.1-flash-lite  [PASSAGE-BASED / READING COMPREHENSION]
Best for: tasks where a passage, document, or context IS provided for the model to
read and reason over. Reading comprehension ("Based on the passage above, ..."),
NLI / textual entailment (given a premise, classify hypothesis as true/false/neutral),
answer-evaluation tasks ("Given the context, is this response correct?"),
true/false classification with provided context, music theory with notation.
Also: geography trivia, quiz bowl factoid questions ("For 10 points, name this..."),
general knowledge questions that do NOT fit the MCQ multiple-choice format.

## deepseek/deepseek-v4-flash  [MATH / STEM / CODING / TRANSLATION]
Best for: mathematics requiring calculation or proof (algebra, calculus, geometry,
combinatorics, number theory, statistics), physics and chemistry problems with
derivations, code generation and debugging (Python, C++, SQL, algorithms, data
structures, competitive programming), step-by-step scientific reasoning, language
translation (German, French, Chinese, Spanish, Japanese, Russian, etc.),
financial numerical reasoning (FinQA, ratio calculations).

## qwen/qwen3-235b-a22b-2507  [EXPERT HARD REASONING — use sparingly]
Best for: questions that require deep expert reasoning beyond simple factual recall.
Use ONLY when the question clearly demands multi-step specialist expertise:
- Hard competitive mathematics at Olympiad / AIME / AMC level
- Complex biomedical research (clinical trial methodology, pharmacokinetics,
  disease mechanism analysis at research depth — NOT standard medical MCQ)
- Formal symbolic logic (propositional logic with symbols, modal logic proofs)
- Deep legal case analysis requiring multi-step statutory interpretation
Do NOT use for standard MMLU-style medical or science MCQ — those go to gemini-2.0.

## qwen/qwen3-next-80b-a3b-instruct  [LINGUISTIC / WORD-SENSE / COREFERENCE]
Best for: word-sense disambiguation ("Does word X carry the same meaning in both
sentences?"), pronoun coreference resolution ("Who does 'they' refer to in this
context?"), Winograd-schema / commonsense pronoun tasks, semantic similarity
and paraphrase detection at the word or phrase level.
ONLY use when the question is explicitly about word meaning, reference resolution,
or lexical semantics — NOT for general language questions.

Return ONLY the exact model name string — nothing else, no punctuation, no quotes.
Example valid outputs:
google/gemini-2.0-flash-001
google/gemini-3.1-flash-lite
deepseek/deepseek-v4-flash
qwen/qwen3-235b-a22b-2507
qwen/qwen3-next-80b-a3b-instruct"""

USER_TEMPLATE = "Query:\n{query}\n\nModel:"

# ---------------------------------------------------------------------------
# Classifier call
# ---------------------------------------------------------------------------


def classify_query(query: str, api_key: str, timeout: int = 60) -> Optional[str]:
    """Call Gemini 2.5 Flash to classify the query. Returns exact model name or None.

    Retries up to 5 times with exponential backoff on 429 (rate limit) errors.
    """
    url = _GEMINI_URL.format(model=CLASSIFIER_MODEL, key=api_key)
    payload: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": USER_TEMPLATE.format(query=query[:4000])}],
            }
        ],
        "generationConfig": {
            # 512: Gemini 2.5 Flash thinking tokens + output share the maxOutputTokens budget.
            # Output (a model name) is ~10 tokens; rest is thinking budget.
            "maxOutputTokens": 512,
            "temperature": 0.0,
        },
    }
    max_retries = 5
    backoff = 5  # seconds, doubles each retry
    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, json=payload, timeout=timeout)
            if resp.status_code == 429:
                wait = backoff * (2**attempt)
                print(
                    f"  [rate limit] waiting {wait}s (attempt {attempt + 1})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            cands = resp.json().get("candidates", [])
            if not cands or "parts" not in cands[0].get("content", {}):
                # thinking consumed all tokens — retry with more budget
                print("  [no parts] increasing budget, retrying", file=sys.stderr)
                payload["generationConfig"]["maxOutputTokens"] = 2048
                continue
            raw = cands[0]["content"]["parts"][0]["text"].strip()
            # Exact match first
            for model in ROUTING_MODELS:
                if model in raw:
                    return model
            # Fuzzy: match on last path segment
            raw_lower = raw.lower()
            for model in ROUTING_MODELS:
                if model.split("/")[-1].lower() in raw_lower:
                    return model
            return None
        except Exception as e:
            print(f"  [classify error attempt {attempt + 1}] {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers (default 2; keep low to respect free-tier rate limits)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip hashes already present in output file",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    print(f"Classifier: {CLASSIFIER_MODEL}")
    print(f"Output models: {[m.split('/')[-1] for m in ROUTING_MODELS]}")
    print(f"Dataset: {DATASET_PATH}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # Deduplicate queries by hash
    queries: dict[str, str] = {}
    for entry in dataset:
        q = (
            entry.get("prompt_formatted")
            or entry.get("prompt")
            or entry.get("query")
            or entry.get("question", "")
        )
        if q:
            queries[_query_hash(q)] = q

    print(f"Unique queries: {len(queries)}")

    # Load existing decisions (for resume)
    decisions: dict[str, str] = {}
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            decisions = json.load(f)
        # Prune any decisions pointing to unknown models
        known = set(ROUTING_MODELS)
        pruned = {h: m for h, m in decisions.items() if m in known}
        if len(pruned) < len(decisions):
            print(f"  Pruned {len(decisions) - len(pruned)} unknown-model entries")
            decisions = pruned
        print(
            f"Resuming: {len(decisions)} done, {len(queries) - len(decisions)} remaining"
        )

    pending = {h: q for h, q in queries.items() if h not in decisions}
    if not pending:
        print("All queries already classified.")
        return

    print(f"Classifying {len(pending)} queries with {args.workers} workers...")

    lock = Lock()
    success_count = 0
    fallback_count = 0

    def worker(item: tuple[str, str]) -> tuple[str, Optional[str]]:
        h, q = item
        return h, classify_query(q, api_key)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, item): item for item in pending.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            h, model = fut.result()
            with lock:
                if model:
                    decisions[h] = model
                    success_count += 1
                else:
                    decisions[h] = FALLBACK_MODEL
                    fallback_count += 1

            if i % 100 == 0 or i == len(pending):
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(pending) - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{len(pending)} | OK {success_count} | fallback {fallback_count}"
                    f" | {rate:.1f} q/s | ETA {eta:.0f}s"
                )
                # Checkpoint save every 100
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(decisions, f)

    elapsed = time.time() - start
    print(
        f"\nDone in {elapsed:.0f}s  |  classified: {success_count}  |  fallback: {fallback_count}"
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False)
    print(f"Saved: {OUTPUT_PATH}")

    dist = Counter(decisions.values())
    total = sum(dist.values())
    print("\nRouting distribution:")
    for m, c in dist.most_common():
        print(f"  {m.split('/')[-1]:<44} {c:>5} ({100 * c / total:.1f}%)")


if __name__ == "__main__":
    main()
