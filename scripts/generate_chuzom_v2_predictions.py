# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Generate RouterArena prediction files for ChuzomRouterV2.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. No dataset names,
  test-set indices, global_index values, or optimality metadata are used.

Produces:
  router_inference/predictions/chuzom-router-v2.json       (full 8400 entries)
  router_inference/predictions/chuzom-router-v2-robustness.json  (420 entries)

Usage:
    # Full dataset (8400 prompts + optimality entries):
    uv run python scripts/generate_chuzom_v2_predictions.py

    # Robustness split only (420 paraphrase prompts):
    uv run python scripts/generate_chuzom_v2_predictions.py --split robustness

    # Quick sanity check on first 10:
    uv run python scripts/generate_chuzom_v2_predictions.py --split sub_10

Notes:
  - Pre-cache LLM judge decisions first (scripts/generate_llm_judge_decisions.py)
    so Gate 4 fires from cache and doesn't make live API calls.
  - If no judge cache exists, Gate 4 is disabled (llm_judge_enabled=False).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ROUTER_NAME = "chuzom-router-v2"
OUTPUT_DIR = Path("router_inference/predictions")

DATASET_PATHS = {
    "sub_10": "./dataset/router_data_10.json",
    "full": "./dataset/router_data.json",
    "robustness": "./dataset/router_robustness.json",
}

OUTPUT_NAMES = {
    "sub_10": "chuzom-router-v2-sub10.json",
    "full": "chuzom-router-v2.json",
    "robustness": "chuzom-router-v2-robustness.json",
}

ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

# Model answers — pulled from the full RouterArena prediction file (which has
# pre-computed generated_answers for all models). We need these to compute accuracy.
_REFERENCE_PREDICTIONS_FILE = Path("router_inference/predictions/chuzom-llm-router.json")


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_reference_answers(split: str) -> dict[str, dict]:
    """Load pre-computed model answers indexed by global_index."""
    if not _REFERENCE_PREDICTIONS_FILE.exists():
        print(
            f"WARNING: reference predictions not found at {_REFERENCE_PREDICTIONS_FILE}",
            file=sys.stderr,
        )
        return {}
    with open(_REFERENCE_PREDICTIONS_FILE) as f:
        data = json.load(f)
    return {entry["global index"]: entry for entry in data if "global index" in entry}


def load_all_model_answers(global_index: str, ref_by_idx: dict) -> dict[str, dict]:
    """Get the full per-model answer dict for a given global_index."""
    ref = ref_by_idx.get(global_index, {})
    return ref.get("all_model_results", ref.get("model_results", {}))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["sub_10", "full", "robustness"],
        default="full",
        help="Dataset split to generate predictions for",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge (Gate 4) even if cache exists",
    )
    args = parser.parse_args()

    # Setup path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from router_inference.router.chuzom_router_v2 import ChuzomRouterV2

    judge_cache_path = Path(
        "router_inference/config/chuzom-llm-judge-decisions.json"
    )
    judge_enabled = not args.no_judge and judge_cache_path.exists()
    if judge_enabled:
        print(f"LLM judge cache found — Gate 4 enabled", file=sys.stderr)
    else:
        print(
            f"LLM judge cache {'disabled (--no-judge)' if args.no_judge else 'not found'} — Gate 4 disabled",
            file=sys.stderr,
        )

    print(f"Loading ChuzomRouterV2({ROUTER_NAME})...", file=sys.stderr)
    router = ChuzomRouterV2(ROUTER_NAME, llm_judge_enabled=judge_enabled)

    dataset_path = Path(DATASET_PATHS[args.split])
    print(f"Loading dataset from {dataset_path}...", file=sys.stderr)
    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Load reference answers to embed generated_result in predictions
    print("Loading reference answers...", file=sys.stderr)
    ref_by_idx = load_reference_answers(args.split)

    routing_entries = [e for e in dataset if not e.get("for_optimality")]
    print(f"  {len(routing_entries)} routing entries", file=sys.stderr)

    predictions = []
    model_counter: Counter = Counter()
    missed_refs = 0

    for i, entry in enumerate(routing_entries):
        prompt = entry["prompt"]
        global_idx = entry.get("global_index", entry.get("global index", f"unknown_{i}"))

        predicted_model = router.get_prediction(prompt)
        model_counter[predicted_model] += 1

        # Find the generated answer for the chosen model
        ref = ref_by_idx.get(global_idx, {})
        all_model_results = ref.get("all_model_results", {})

        # Get answer from the chosen model's result
        generated_result = {}
        if predicted_model in all_model_results:
            model_result = all_model_results[predicted_model]
            generated_result = {
                "generated_answer": model_result.get(
                    "generated_answer",
                    model_result.get("response", ""),
                ),
                "cost": model_result.get("cost", 0.0),
            }
        elif ref:
            # Fallback: use whatever the reference has for this index
            generated_result = {
                "generated_answer": ref.get("generated_answer", ""),
                "cost": ref.get("cost", 0.0),
            }
        else:
            missed_refs += 1

        prediction_entry = {
            "global index": global_idx,
            "prompt": prompt,
            "prediction": predicted_model,
            "generated_result": generated_result,
            "dataset": entry.get("dataset", ""),
            "source": entry.get("source", ""),
            "correct_answer": entry.get("correct_answer", entry.get("answer", "")),
        }

        predictions.append(prediction_entry)

        if i % 500 == 0:
            print(f"  {i}/{len(routing_entries)} predictions generated", file=sys.stderr)

    # Add optimality entries (full split only)
    if args.split == "full":
        optimality_entries = [e for e in dataset if e.get("for_optimality")]
        print(f"  Adding {len(optimality_entries)} optimality entries...", file=sys.stderr)

        for entry in optimality_entries:
            global_idx = entry.get("global_index", entry.get("global index", ""))
            # Optimality entries use a specific model from the dataset
            opt_model = entry.get("model", entry.get("for_model", ROUTING_MODELS[0]))

            ref = ref_by_idx.get(global_idx, {})
            all_model_results = ref.get("all_model_results", {})
            model_result = all_model_results.get(opt_model, {})

            predictions.append({
                "global index": global_idx,
                "prompt": entry["prompt"],
                "prediction": opt_model,
                "generated_result": {
                    "generated_answer": model_result.get("generated_answer", ""),
                    "cost": model_result.get("cost", 0.0),
                },
                "dataset": entry.get("dataset", ""),
                "for_optimality": True,
                "correct_answer": entry.get("correct_answer", entry.get("answer", "")),
            })

    # Save
    output_path = OUTPUT_DIR / OUTPUT_NAMES[args.split]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved {len(predictions)} entries to {output_path}", file=sys.stderr)
    if missed_refs:
        print(
            f"  WARNING: {missed_refs} entries had no reference answer (may affect accuracy reporting)",
            file=sys.stderr,
        )

    print("\nRouting distribution:", file=sys.stderr)
    total = sum(model_counter.values())
    for m in ROUTING_MODELS:
        n = model_counter[m]
        print(f"  {m}: {n} ({n / max(total, 1) * 100:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
