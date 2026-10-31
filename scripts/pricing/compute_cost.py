# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Compute a prediction file's cost the way the evaluator does, router calls included.

Every scored row is billed for its answer (token_usage at the answering model's price) plus
its router calls (``router_usage``, see global_utils/router_usage.py), using the price table
in model_cost/. Models without a price profile are charged the table's highest price and
listed at the end. No answer is graded, so this runs offline in seconds.

Usage:
    uv run python scripts/pricing/compute_cost.py your-router [--split full] [--per-row out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "llm_evaluation"))

from evaluate_models import ModelEvaluator  # noqa: E402
from run import compute_row_cost, load_predictions_file  # noqa: E402
from universal_model_names import ModelNameManager  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("router_name")
    parser.add_argument("--split", default="full", choices=["sub_10", "full"])
    parser.add_argument("--per-row", help="write one CSV line per scored row")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    evaluator = ModelEvaluator(cached_results_dir="./cached_results")
    rows = [
        p
        for p in load_predictions_file(args.router_name, split=args.split)
        if not p.get("for_optimality")
    ]

    answer_total = router_total = 0.0
    per_row = []
    for prediction in rows:
        model = prediction.get("prediction", "")
        try:
            universal = ModelNameManager.get_universal_name(model)
        except Exception:
            universal = model
        answer, router = compute_row_cost(prediction, evaluator, universal)
        answer_total += answer
        router_total += router
        per_row.append((prediction.get("global index"), model, answer, router))

    total = answer_total + router_total
    n = max(1, len(rows))
    print(f"Router: {args.router_name} ({len(rows)} scored rows, split {args.split})")
    print(f"  answers:      ${answer_total:.6f}")
    print(f"  router calls: ${router_total:.6f}")
    print(f"  total:        ${total:.6f}  (${1000 * total / n:.4f} per 1K queries)")
    if evaluator.unpriced_models:
        print("Charged the maximum price (no price profile):")
        for name, count in sorted(evaluator.unpriced_models.items()):
            print(f"  - {name}: {count} call(s)")

    if args.per_row:
        with open(args.per_row, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["global_index", "model", "answer_cost", "router_cost"])
            writer.writerows(per_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
