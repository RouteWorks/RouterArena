#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Fill generated_result in a prediction file from local cached inference results.

Avoids API costs for local evaluation by using cached model outputs.

Usage:
    .venv/bin/python3 scripts/fill_predictions_from_cache.py \
        --prediction-file router_inference/predictions/chuzom-v3.json \
        --output router_inference/predictions/chuzom-v3-filled.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Map from RouterArena model name → cache filename stem
MODEL_TO_CACHE: dict[str, str] = {
    "google/gemini-3.1-flash-lite": "google_gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash": "deepseek_deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507": "qwen_qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct": "qwen_qwen3-next-80b-a3b-instruct",
    "google/gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
}


def load_cache(model: str) -> dict[str, dict]:
    stem = MODEL_TO_CACHE.get(model)
    if not stem:
        return {}
    path = ROOT / "cached_results" / f"{stem}.jsonl"
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                gi = r.get("global_index", "")
                if gi:
                    cache[gi] = {
                        "generated_answer": r.get("generated_answer", ""),
                        "success": r.get("success", False),
                        "token_usage": r.get("token_usage", {}),
                        "provider": r.get("provider", ""),
                        "error": r.get("error"),
                    }
            except json.JSONDecodeError:
                continue
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading predictions from {args.prediction_file}...")
    with open(args.prediction_file) as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} entries")

    # Find which models are needed
    models_needed = set(p["prediction"] for p in predictions)
    print(f"\nModels in predictions: {models_needed}")

    # Load caches
    caches: dict[str, dict[str, dict]] = {}
    for model in models_needed:
        cache = load_cache(model)
        caches[model] = cache
        short = model.split("/")[-1]
        coverage = len(cache)
        print(f"  Cache for {short}: {coverage} entries")

    # Fill predictions
    filled = 0
    missing = 0
    skipped_already = 0
    stats: defaultdict[str, int] = defaultdict(int)

    out_predictions = []
    for pred in predictions:
        gi = pred.get("global index", "")
        model = pred.get("prediction", "")

        if pred.get("generated_result") and isinstance(pred["generated_result"], dict):
            skipped_already += 1
            out_predictions.append(pred)
            continue

        cache = caches.get(model, {})
        result = cache.get(gi)
        if result:
            out_pred = dict(pred)
            out_pred["generated_result"] = result
            out_predictions.append(out_pred)
            filled += 1
            stats[model.split("/")[-1]] += 1
        else:
            out_predictions.append(pred)
            missing += 1

    print(f"\nFilled: {filled}")
    print(f"Already had results: {skipped_already}")
    print(f"Missing from cache: {missing}")
    print("\nFilled by model:")
    for m, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {m}")

    print(f"\nSaving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(out_predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out_predictions)} entries.")


if __name__ == "__main__":
    main()
