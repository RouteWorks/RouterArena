#!/usr/bin/env python3
"""Fill prediction file answers from cached_results/ JSONL files.

Matches each prediction entry (by global_index + predicted model) to its
cached result and writes the generated_result field. Entries that aren't
in the cache are left as-is (null generated_result).

Usage:
    uv run python scripts/fill_predictions_from_cache.py chuzom-llm-router
"""

import json
import os
import sys
from collections import defaultdict

CACHED_RESULTS_DIR = "./cached_results"
PREDICTIONS_DIR = "./router_inference/predictions"

MODEL_TO_CACHE_FILE = {
    "google/gemini-3.1-flash-lite": "google_gemini-3.1-flash-lite.jsonl",
    "deepseek/deepseek-v4-flash": "deepseek_deepseek-v4-flash.jsonl",
    "qwen/qwen3-235b-a22b-2507": "qwen_qwen3-235b-a22b-2507.jsonl",
    "Qwen/Qwen3-Coder-Next": "Qwen_Qwen3-Coder-Next.jsonl",
    "gpt-4o-mini": "gpt-4o-mini.jsonl",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307.jsonl",
    "qwen/qwen3-next-80b-a3b-instruct": "qwen_qwen3-next-80b-a3b-instruct.jsonl",
}


def load_cache(model_name: str) -> dict:
    """Load cached results for a model, indexed by global_index."""
    cache_file = MODEL_TO_CACHE_FILE.get(model_name)
    if not cache_file:
        print(f"  WARNING: No cache file mapping for model {model_name}")
        return {}

    path = os.path.join(CACHED_RESULTS_DIR, cache_file)
    if not os.path.exists(path):
        print(f"  WARNING: Cache file not found: {path}")
        return {}

    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                gidx = entry.get("global_index")
                if gidx:
                    cache[gidx] = entry
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(cache)} cached entries for {model_name}")
    return cache


def build_generated_result(cached: dict) -> dict:
    """Convert a cached result entry to the generated_result format."""
    return {
        "generated_answer": cached.get("generated_answer"),
        "success": cached.get("success", False),
        "token_usage": cached.get(
            "token_usage",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        ),
        "provider": cached.get("provider", "cached"),
        "error": cached.get("error"),
    }


def main(router_name: str):
    pred_path = os.path.join(PREDICTIONS_DIR, f"{router_name}.json")
    if not os.path.exists(pred_path):
        print(f"ERROR: Prediction file not found: {pred_path}")
        sys.exit(1)

    print(f"Loading prediction file: {pred_path}")
    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Total entries: {len(predictions)}")

    # Discover which models are used
    models_used = set(e["prediction"] for e in predictions)
    print(f"Models used: {sorted(models_used)}")

    # Load all needed caches
    print("\nLoading caches:")
    caches: dict[str, dict] = {}
    for model in models_used:
        caches[model] = load_cache(model)

    # Fill in generated_result
    filled = 0
    missing = 0
    already_filled = 0

    for entry in predictions:
        if entry.get("generated_result") and entry["generated_result"].get(
            "generated_answer"
        ):
            already_filled += 1
            continue

        model = entry["prediction"]
        gidx = entry["global index"]
        cache = caches.get(model, {})
        cached = cache.get(gidx)

        if cached:
            entry["generated_result"] = build_generated_result(cached)
            filled += 1
        else:
            missing += 1

    print("\nResults:")
    print(f"  Already filled: {already_filled}")
    print(f"  Filled from cache: {filled}")
    print(f"  Missing (no cache hit): {missing}")

    # Save updated predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"\nSaved updated predictions to: {pred_path}")

    # Sanity check
    null_count = sum(
        1
        for e in predictions
        if not e.get("generated_result")
        or not e.get("generated_result", {}).get("generated_answer")
    )
    print(f"Entries still missing answers: {null_count}")
    if null_count > 0:
        models_missing: dict[str, int] = defaultdict(int)
        for e in predictions:
            if not e.get("generated_result") or not e.get("generated_result", {}).get(
                "generated_answer"
            ):
                models_missing[e["prediction"]] += 1
        print("  Breakdown by model:")
        for m, c in sorted(models_missing.items(), key=lambda x: -x[1]):
            print(f"    {m}: {c}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fill_predictions_from_cache.py <router_name>")
        sys.exit(1)
    main(sys.argv[1])
