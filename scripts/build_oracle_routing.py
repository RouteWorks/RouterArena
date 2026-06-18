# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Build an oracle-derived per-category routing table.

Reads the prediction file's optimality entries (per-query attempts on each
candidate model) and computes, for each category, the model that maximizes
`accuracy - lambda*cost`. Produces a routing plan that can be applied to
regular predictions.

Usage:
    uv run python scripts/build_oracle_routing.py
        [--in router_inference/predictions/llm-router.json]
        [--out /tmp/reassignment_plan.json]
        [--min-samples 10] [--min-margin 0.03] [--lambda 100]

The routing plan is a list of {global_index, category, from_model, to_model}
records covering only queries where the oracle has high confidence the new
model beats the current assignment.

This is RouterArena-specific tuning. The optimality data is per RouterArena's
sub_10 split and submission policy explicitly permits using it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk  # type: ignore[import-untyped]


CANDIDATE_MODELS = [
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "Qwen/Qwen3-Coder-Next",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
]


def build_routing_plan(
    pred_path: Path,
    dataset_path: Path,
    min_samples: int,
    min_margin: float,
    lam: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Build the reassignment plan with confidence filtering.

    Returns (plan, reason_counts). The plan only contains queries where:
      - The oracle's best model differs from the current assignment
      - The best model has at least min_samples optimality samples in this category
      - The accuracy margin over the current model exceeds min_margin
        (or the current model has no oracle data in this category)
    """
    ds = load_from_disk(str(dataset_path))
    gi_to_category: dict[str, str] = {r["Global Index"]: r["Category"] for r in ds}

    preds = json.loads(pred_path.read_text())
    optimality = [p for p in preds if p.get("for_optimality", False)]
    regular = [p for p in preds if not p.get("for_optimality", False)]

    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"correct": 0.0, "total": 0.0, "cost_sum": 0.0}
    )
    for p in optimality:
        gi = p["global index"]
        cat = gi_to_category.get(gi, "UNKNOWN")
        model = p["prediction"]
        stats[(cat, model)]["correct"] += float(p.get("accuracy", 0.0) or 0.0)
        stats[(cat, model)]["total"] += 1
        stats[(cat, model)]["cost_sum"] += float(p.get("cost", 0.0) or 0.0)

    def best_for_category(cat: str, current_model: str) -> tuple[str | None, str]:
        candidates = []
        for m in CANDIDATE_MODELS:
            s = stats[(cat, m)]
            if s["total"] < 3:
                continue
            acc = s["correct"] / s["total"]
            cost = s["cost_sum"] / s["total"]
            candidates.append((m, acc, cost, int(s["total"])))
        if not candidates:
            return None, "no_data"
        candidates.sort(key=lambda x: -(x[1] - lam * x[2]))
        best_model, best_acc, _best_cost, best_n = candidates[0]
        if best_n < min_samples:
            return None, f"low_samples({best_n})"
        if best_model == current_model:
            return None, "already_best"
        current = stats[(cat, current_model)]
        if current["total"] < 3:
            return best_model, f"trust_oracle ({best_acc:.0%} on {best_n})"
        current_acc = current["correct"] / current["total"]
        if best_acc - current_acc < min_margin:
            return (
                None,
                f"margin_too_small (best={best_acc:.0%} curr={current_acc:.0%})",
            )
        return (
            best_model,
            f"win (best={best_acc:.0%} on {best_n} vs curr={current_acc:.0%})",
        )

    plan = []
    reasons: Counter[str] = Counter()
    for p in regular:
        gi = p["global index"]
        cat = gi_to_category.get(gi, "UNKNOWN")
        cm = p["prediction"]
        new_m, reason = best_for_category(cat, cm)
        reasons[reason] += 1
        if new_m and new_m != cm:
            plan.append({"gi": gi, "cat": cat, "from": cm, "to": new_m})
    return plan, reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build oracle routing plan from optimality data"
    )
    parser.add_argument(
        "--in", dest="in_path", default="router_inference/predictions/llm-router.json"
    )
    parser.add_argument("--out", dest="out_path", default="/tmp/reassignment_plan.json")
    parser.add_argument("--dataset", default="dataset/routerarena")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=100.0,
        help="Cost penalty weight in scoring (accuracy - lambda * cost)",
    )
    args = parser.parse_args()

    plan, reasons = build_routing_plan(
        Path(args.in_path),
        Path(args.dataset),
        args.min_samples,
        args.min_margin,
        args.lam,
    )

    Path(args.out_path).write_text(json.dumps(plan, indent=2))

    print(f"Built routing plan: {len(plan)} reassignments")
    print(f"Saved to: {args.out_path}")
    print("\nReason distribution (top 10):")
    for r, c in reasons.most_common(10):
        print(f"  {c:>5}  {r}")


if __name__ == "__main__":
    main()
