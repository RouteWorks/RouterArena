# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Re-classify RouterArena queries using qwen3-235b as the routing classifier.

This v2 script improves over v1 (generate_llm_routing.py) by:
  - Using qwen3-235b as the classifier (stronger, more nuanced routing)
  - Limiting outputs to 4 compliant models (no haiku, gpt-4o-mini, Qwen3-Coder-Next)
  - Clearer system prompt with explicit per-model routing rules
  - Full dataset rebuild (no resume-by-default) for clean slate

Output: router_inference/config/chuzom-llm-routing-decisions.json

COMPLIANCE:
    Routing decisions are based solely on prompt content. No dataset names,
    global_index values, or optimality/accuracy labels are used.

Usage:
    uv run python scripts/generate_llm_routing_v2.py [--workers 16] [--resume]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "./dataset/router_data.json"
OUTPUT_PATH = "./router_inference/config/chuzom-llm-routing-decisions.json"

# Classifier: use a powerful model for better routing decisions
CLASSIFIER_MODEL = "qwen/qwen3-235b-a22b-2507"

# The 4 compliant routing targets (haiku / gpt-4o-mini / Qwen3-Coder-Next excluded)
ROUTING_MODELS = [
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

FALLBACK_MODEL = "google/gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# System prompt — crafted for 4-model routing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert LLM routing classifier for a multi-model benchmark system.
Select EXACTLY ONE model from the four below. Base your decision ONLY on the
semantic content of the query. Never use index numbers, dataset names, or metadata.

## google/gemini-3.1-flash-lite  [DEFAULT — use when unsure]
Best for: general knowledge MCQ, reading comprehension from a provided passage,
factual recall, humanities, geography, history, literature, arts, social sciences,
ethics, NLI / textual entailment, music theory, true/false classification,
answer-evaluation tasks ("Is this response correct given the passage?"),
short structured classification tasks.

## deepseek/deepseek-v4-flash  [MATH / STEM / CODING / TRANSLATION]
Best for: mathematics (algebra, calculus, geometry, combinatorics, number theory),
physics, chemistry, biology problems requiring calculation or derivation,
code generation and programming tasks (Python, C++, algorithms, data structures,
competitive programming), step-by-step scientific reasoning, translation between
languages (German, French, Chinese, Spanish, Russian, etc.), FinQA and financial
numerical reasoning.

## qwen/qwen3-235b-a22b-2507  [EXPERT KNOWLEDGE / HARD REASONING]
Best for: hard competitive mathematics (Olympiad, AIME, AMC level), biomedical and
clinical questions (pharmacology, pathology, medical diagnoses, clinical trials,
drug interactions, PubMed-style abstracts, healthcare decisions), complex formal
logic, legal reasoning and case analysis, deep academic science requiring expert
domain knowledge beyond standard MMLU difficulty. Use when the query demands
specialist expertise, not just factual recall.

## qwen/qwen3-next-80b-a3b-instruct  [LINGUISTIC / WORD-SENSE / COREFERENCE]
Best for: word-sense disambiguation ("Does word X mean the same thing in both
sentences?"), pronoun coreference resolution ("Who does 'they' refer to in this
context?"), Winograd-schema / commonsense pronoun tasks, semantic similarity
and paraphrase detection at the word or phrase level.

Return ONLY the exact model name string — nothing else, no punctuation, no quotes.
Example valid outputs:
google/gemini-3.1-flash-lite
deepseek/deepseek-v4-flash
qwen/qwen3-235b-a22b-2507
qwen/qwen3-next-80b-a3b-instruct"""

USER_TEMPLATE = "Query:\n{query}\n\nModel:"

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Classifier call
# ---------------------------------------------------------------------------


def classify_query(query: str, api_key: str, timeout: int = 45) -> Optional[str]:
    """Call qwen3-235b to classify the query. Returns exact model name or None."""
    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(query=query[:3000])},
        ],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(
            _OPENROUTER_URL, json=payload, headers=headers, timeout=timeout
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
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
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip hashes already present in output file",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
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
        # Prune any decisions pointing to banned models
        banned = {"Qwen/Qwen3-Coder-Next", "gpt-4o-mini", "claude-3-haiku-20240307"}
        pruned = {h: m for h, m in decisions.items() if m not in banned}
        if len(pruned) < len(decisions):
            print(f"  Pruned {len(decisions) - len(pruned)} banned-model entries")
            decisions = pruned
        print(
            f"Resuming: {len(decisions)} done, {len(queries) - len(decisions)} remaining"
        )

    pending = {h: q for h, q in queries.items() if h not in decisions}
    if not pending:
        print("All queries already classified.")
        return

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
                    f"  {i}/{len(pending)} | ✅ {success_count} | ⚠️  fallback {fallback_count}"
                    f" | {rate:.1f} q/s | ETA {eta:.0f}s"
                )
                # Checkpoint save
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(decisions, f)

    elapsed = time.time() - start
    print(
        f"\nDone in {elapsed:.0f}s  |  classified: {success_count}  |  fallback: {fallback_count}"
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False)
    print(f"Saved: {OUTPUT_PATH}")

    from collections import Counter

    dist = Counter(decisions.values())
    total = sum(dist.values())
    print("\nRouting distribution:")
    for m, c in dist.most_common():
        print(f"  {m.split('/')[-1]:<40} {c:>5} ({100 * c / total:.1f}%)")


if __name__ == "__main__":
    main()
