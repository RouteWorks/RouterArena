# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Re-price routers at the current price table, without re-grading.

Every row the evaluator costed is re-costed from the token counts already in the
prediction file, at the prices in model_cost/model_cost.json, exactly as the evaluator
bills it (answer plus router calls, see ``compute_row_cost`` in llm_evaluation/run.py).
Metrics are then recomputed with the evaluator's own ``router_metrics``. No answer is
graded and no submitted code runs, so this is quick and safe to run offline.

Use it after a price sync, to bring the leaderboard to current prices, or on a new
submission whose file was graded with an older price table.

Usage:
    python scripts/pricing/reprice.py ROUTER [ROUTER ...]   # report only
    python scripts/pricing/reprice.py --all                   # every router in the manifest
    python scripts/pricing/reprice.py --all --write           # also rewrite costs and the README
    python scripts/pricing/reprice.py --all --report out.md   # save the before/after table

With ``--write``, each prediction file keeps its JSON formatting and only cost values
change, and the README rows of the re-priced routers are updated and the table
re-ranked. README rows without a prediction file (external baselines) keep their numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "llm_evaluation"))

from evaluate_models import ModelEvaluator  # noqa: E402
from run import compute_row_cost, router_metrics  # noqa: E402
from universal_model_names import ModelNameManager  # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "router_inference" / "predictions"
MANIFEST_PATH = REPO_ROOT / "leaderboard_manifest.yaml"
README_PATH = REPO_ROOT / "README.md"


def reprice_rows(predictions: List[Dict[str, Any]], evaluator: ModelEvaluator) -> int:
    """Re-cost, in place, every row that has a numeric cost (the rows the evaluator
    costed). Returns the number of rows re-costed."""
    count = 0
    for prediction in predictions:
        cost = prediction.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            continue
        model = prediction.get("prediction", "")
        try:
            universal = ModelNameManager.get_universal_name(model)
        except Exception:
            universal = model
        answer_cost, router_cost = compute_row_cost(prediction, evaluator, universal)
        prediction["cost"] = answer_cost + router_cost
        if router_cost:
            prediction["router_cost"] = router_cost
        else:
            prediction.pop("router_cost", None)
        count += 1
    return count


def dump_like(original: str, data: Any) -> str:
    """Serialize ``data`` in the JSON style of ``original`` (indent, separators, ASCII
    escaping, trailing newline), so a diff shows only the values that changed."""
    parsed = json.loads(original)
    indents: List[Union[None, int, str]] = [None, 1, 2, 4, "\t"]
    for indent in indents:
        for ensure_ascii in (False, True):
            for separators in ((", ", ": "), (",", ":"), (",", ": ")):
                if indent is not None and separators == (", ", ": "):
                    continue  # with an indent, json writes "," between items
                text = json.dumps(
                    parsed,
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                    separators=separators,
                )
                if text == original.rstrip("\n"):
                    rendered = json.dumps(
                        data,
                        indent=indent,
                        ensure_ascii=ensure_ascii,
                        separators=separators,
                    )
                    return rendered + original[len(text) :]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def leaderboard_routers() -> List[Tuple[str, str]]:
    """(prediction file name, README name) for every router in the manifest."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [
        (r["prediction"], r["readme_name"])
        for r in manifest["routers"]
        if r.get("prediction") and r.get("readme_name")
    ]


def _norm(name: str) -> str:
    return re.sub(r"[\s\-‑\\]", "", name).lower()


def _row_name(line: str) -> str:
    cell = line.split(" | ")[1]
    match = re.match(r"\[([^\]]*)\]", cell)
    return match.group(1) if match else cell


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def update_readme(
    text: str, metrics: Dict[str, Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """Update the leaderboard rows named in ``metrics`` (README name -> metrics.json
    dict) and re-rank the table by Arena score. Returns (new text, names updated)."""
    lines = text.split("\n")
    header = next(
        i for i, line in enumerate(lines) if line.startswith("| Rank | Router |")
    )
    start = header + 2  # skip the header and separator rows
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    by_name = {_norm(name): name for name in metrics}
    rows, updated = [], []
    for line in lines[start:end]:
        name = by_name.get(_norm(_row_name(line)))
        if name is not None:
            m = metrics[name]
            cells = line.split(" | ")
            cells[3] = _pct(m["arena_score"])
            cells[4] = _pct(m["accuracy"])
            cells[5] = f"${m['avg_cost_per_1000']:.2f}"
            opt = m.get("optimality")
            if opt:
                cells[6:9] = [
                    _pct(opt["opt_sel"]),
                    _pct(opt["opt_cost"]),
                    _pct(opt["opt_acc"]),
                ]
            line = " | ".join(cells)
            updated.append(name)
        rows.append(line)
    rows.sort(key=lambda line: float(line.split(" | ")[3]), reverse=True)  # stable
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rows = [
        f"| {medals.get(n, n)} | " + line.split(" | ", 1)[1]
        for n, line in enumerate(rows, 1)
    ]
    lines[start:end] = rows
    return "\n".join(lines), updated


def report_table(results: List[Dict[str, Any]]) -> str:
    out = [
        "| Router | Arena | Cost / 1K queries | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda r: -r["after"]["arena_score"]):
        b, a = r["before"], r["after"]
        out.append(
            f"| {r['router']} | {_pct(b['arena_score'])} → {_pct(a['arena_score'])} | "
            f"${b['avg_cost_per_1000']:.4f} → ${a['avg_cost_per_1000']:.4f} | "
            f"{_pct(a['accuracy'])} |"
        )
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("routers", nargs="*", help="prediction file names")
    parser.add_argument(
        "--all", action="store_true", help="every router in the manifest"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the prediction files' costs and the README rows",
    )
    parser.add_argument("--report", help="also write the before/after table here")
    args = parser.parse_args(argv)

    readme_of = dict(leaderboard_routers())
    routers = list(readme_of) if args.all else args.routers
    if not routers:
        parser.error("name at least one router, or pass --all")

    os.chdir(REPO_ROOT)  # the optimality metrics read ./dataset/router_data_10.json
    logging.getLogger().setLevel(logging.WARNING)
    evaluator = ModelEvaluator(cached_results_dir="./cached_results")

    results, readme_metrics = [], {}
    for router in routers:
        path = PREDICTIONS_DIR / f"{router}.json"
        original = path.read_text(encoding="utf-8")
        predictions = json.loads(original)
        before, _ = router_metrics(predictions, router)
        recosted = reprice_rows(predictions, evaluator)
        after, _ = router_metrics(predictions, router)
        results.append(
            {"router": router, "before": before, "after": after, "rows": recosted}
        )
        if args.write:
            path.write_text(dump_like(original, predictions), encoding="utf-8")
            if router in readme_of:
                readme_metrics[readme_of[router]] = after

    table = report_table(results)
    print(table)
    if evaluator.unpriced_models:
        print("Charged the maximum price (no price profile):")
        for name, count in sorted(evaluator.unpriced_models.items()):
            print(f"  - {name}: {count} row(s)")
    if args.report:
        Path(args.report).write_text(table, encoding="utf-8")

    if args.write and readme_metrics:
        text, updated = update_readme(
            README_PATH.read_text(encoding="utf-8"), readme_metrics
        )
        README_PATH.write_text(text, encoding="utf-8")
        print(f"Updated {len(updated)} README row(s) and re-ranked the table.")
        missing = set(readme_metrics) - set(updated)
        if missing:
            print("No README row found for: " + ", ".join(sorted(missing)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
