# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Rebuild predictions file from updated routing decisions.

After running generate_llm_routing_v2.py, use this script to apply
the new routing decisions to the predictions file. For each base entry:
  1. Look up the new model via SHA256(query) in routing-decisions.json
  2. Find the cached result for that model
  3. If not in cache, fall through to next best cached model
  4. Update generated_result in the predictions file

Usage:
    uv run python scripts/rebuild_predictions_from_routing.py
"""

import hashlib
import json
import os
from collections import Counter
from typing import Optional

PREDICTIONS_PATH = "./router_inference/predictions/chuzom-llm-router.json"
DECISIONS_PATH = "./router_inference/config/chuzom-llm-routing-decisions.json"

MODEL_TO_CACHE_FILE = {
    "google/gemini-3.1-flash-lite": "google_gemini-3.1-flash-lite.jsonl",
    "deepseek/deepseek-v4-flash": "deepseek_deepseek-v4-flash.jsonl",
    "qwen/qwen3-235b-a22b-2507": "qwen_qwen3-235b-a22b-2507.jsonl",
    "qwen/qwen3-next-80b-a3b-instruct": "qwen_qwen3-next-80b-a3b-instruct.jsonl",
}

# Cache fallback order when preferred model not in cache
FALLBACK_PRIORITY = [
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.1-flash-lite",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

PRICING = {
    "google/gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    "deepseek/deepseek-v4-flash": {"input": 0.07, "output": 0.28},
    "qwen/qwen3-235b-a22b-2507": {"input": 0.14, "output": 0.60},
    "qwen/qwen3-next-80b-a3b-instruct": {"input": 0.30, "output": 0.90},
}


def load_cache(model: str) -> dict:
    fname = MODEL_TO_CACHE_FILE.get(model)
    if not fname:
        return {}
    path = os.path.join("./cached_results", fname)
    if not os.path.exists(path):
        return {}
    cache: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    gidx = e.get("global_index")
                    if gidx and e.get("generated_answer"):
                        cache[gidx] = e
                except json.JSONDecodeError:
                    pass
    return cache


def build_result(cached: dict) -> dict:
    return {
        "generated_answer": cached.get("generated_answer"),
        "success": cached.get("success", False),
        "token_usage": cached.get(
            "token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        ),
        "provider": cached.get("provider", "cached"),
        "error": cached.get("error"),
    }


def estimate_cost(entry: dict) -> float:
    gr = entry.get("generated_result") or {}
    tu = gr.get("token_usage") or {}
    model = entry["prediction"]
    p = PRICING.get(model, {"input": 0.20, "output": 0.80})
    return (
        tu.get("input_tokens", 0) * p["input"]
        + tu.get("output_tokens", 0) * p["output"]
    ) / 1_000_000


def main() -> None:
    print(f"Loading predictions: {PREDICTIONS_PATH}")
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loading routing decisions: {DECISIONS_PATH}")
    with open(DECISIONS_PATH, encoding="utf-8") as f:
        decisions = json.load(f)

    print("Loading caches:")
    caches: dict[str, dict] = {}
    for model in MODEL_TO_CACHE_FILE:
        caches[model] = load_cache(model)
        print(f"  {model.split('/')[-1]}: {len(caches[model])} valid entries")

    base_entries = [e for e in predictions if not e.get("for_optimality")]
    print(f"\nBase entries to reroute: {len(base_entries)}")

    old_cost = sum(estimate_cost(e) for e in base_entries)

    applied = 0
    fallback_used = 0
    unchanged = 0
    null_issues = 0

    for entry in base_entries:
        query = entry.get("prompt", "")
        gidx = entry.get("global index", "")
        h = hashlib.sha256(query.encode()).hexdigest()

        new_model: Optional[str] = decisions.get(h)
        if not new_model:
            unchanged += 1
            continue

        # Try preferred model first, then fallback order
        models_to_try = [new_model] + [m for m in FALLBACK_PRIORITY if m != new_model]

        placed = False
        for model in models_to_try:
            cached = caches.get(model, {}).get(gidx)
            if cached:
                entry["prediction"] = model
                entry["generated_result"] = build_result(cached)
                if model == new_model:
                    applied += 1
                else:
                    fallback_used += 1
                placed = True
                break

        if not placed:
            null_issues += 1
            print(f"  WARNING: no cache for {gidx} in any model")

    new_cost = sum(estimate_cost(e) for e in base_entries)

    print(f"\nApplied preferred model: {applied}")
    print(f"Used fallback cache: {fallback_used}")
    print(f"No routing decision: {unchanged}")
    print(f"Cache miss (all models): {null_issues}")
    print(
        f"\nCost: ${old_cost:.4f} → ${new_cost:.4f} ({100 * (old_cost - new_cost) / max(old_cost, 1e-9):.1f}% change)"
    )
    print(f"Cost/1K: ${old_cost / 8.4:.4f} → ${new_cost / 8.4:.4f}")

    dist = Counter(e["prediction"] for e in base_entries)
    total = sum(dist.values())
    print("\nNew distribution:")
    for m, c in dist.most_common():
        print(f"  {m.split('/')[-1]:<40} {c:>5} ({100 * c / total:.1f}%)")

    null_count = sum(
        1
        for e in predictions
        if not e.get("for_optimality")
        and (
            not e.get("generated_result")
            or not e["generated_result"].get("generated_answer")
        )
    )
    print(f"\nNull entries: {null_count}")

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
