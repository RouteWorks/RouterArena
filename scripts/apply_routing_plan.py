# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Apply a routing plan to a predictions file.

Reads a plan (list of {gi, from, to} dicts) and rewrites the predictions
file so each affected query's `prediction` field points to the new model.
Clears `generated_result`, `accuracy`, and `cost` for reassigned entries
so the inference + eval pipelines re-process them.

Always creates a `.bak.before-apply` backup of the predictions file.

Usage:
    uv run python scripts/apply_routing_plan.py
        [--plan /tmp/reassignment_plan.json]
        [--predictions router_inference/predictions/llm-router.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a routing plan to predictions")
    parser.add_argument("--plan", default="/tmp/reassignment_plan.json")
    parser.add_argument(
        "--predictions",
        default="router_inference/predictions/llm-router.json",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    backup_path = pred_path.with_suffix(pred_path.suffix + ".bak.before-apply")
    shutil.copy(pred_path, backup_path)

    plan = json.loads(Path(args.plan).read_text())
    plan_map = {a["gi"]: a["to"] for a in plan}

    preds = json.loads(pred_path.read_text())
    applied = 0
    for p in preds:
        if p.get("for_optimality", False):
            continue
        gi = p["global index"]
        if gi in plan_map and p["prediction"] != plan_map[gi]:
            p["prediction"] = plan_map[gi]
            p["generated_result"] = {
                "generated_answer": "",
                "success": False,
                "token_usage": {},
                "provider": "pending",
                "error": None,
            }
            p["accuracy"] = None
            p["cost"] = None
            applied += 1

    pred_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    print(f"Applied {applied} reassignments")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
