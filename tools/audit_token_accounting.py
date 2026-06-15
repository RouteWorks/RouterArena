# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Audit token accounting in router prediction files.

Surfaces the two cost-accounting integrity issues raised in issue #135:

  Problem 1 (reasoning tokens): rows where ``total_tokens`` exceeds
    ``input_tokens + output_tokens``. The excess is billable reasoning/"thinking"
    output that the evaluator now charges at the model's output rate. This audit
    reports the magnitude so reviewers can sanity-check a submission's cost.

  Problem 2 (failed inference): rows that report ``success: True`` with a
    non-empty ``generated_answer`` but NO usable token usage (missing/empty
    ``token_usage`` or ``output_tokens <= 0``). These cannot be costed and are
    scored as wrong by the evaluator. The per-router / per-model counts here are
    what maintainers forward to the router's authors.

Usage:
    python tools/audit_token_accounting.py                 # audit all routers
    python tools/audit_token_accounting.py vllm-sr nadir-cascade-v2
    python tools/audit_token_accounting.py --strict        # exit 1 if Problem 2 found

Only "main" rows (for_optimality == False, the rows that feed the RouterArena
score) are counted, matching the leaderboard.
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, List

PREDICTIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "router_inference",
    "predictions",
)


def _has_usable_token_usage(token_usage: Any) -> bool:
    """True if token_usage reports a positive output_tokens count."""
    if not isinstance(token_usage, dict):
        return False
    output_tokens = token_usage.get("output_tokens")
    return (
        isinstance(output_tokens, (int, float))
        and not isinstance(output_tokens, bool)
        and output_tokens > 0
    )


def audit_file(path: str) -> Dict[str, Any]:
    """Audit a single prediction file; return a summary dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [r for r in data if not r.get("for_optimality", False)]

    reasoning_rows = 0
    reasoning_extra_tokens = 0
    failed_rows = 0
    failed_by_model: Counter = Counter()
    failed_indices: List[str] = []

    for r in rows:
        gen = r.get("generated_result") or {}
        token_usage = gen.get("token_usage")
        answer = gen.get("generated_answer")
        model = r.get("prediction")

        if isinstance(token_usage, dict):
            n_in = token_usage.get("input_tokens") or 0
            n_out = token_usage.get("output_tokens") or 0
            n_total = token_usage.get("total_tokens") or 0
            if n_total > n_in + n_out:
                reasoning_rows += 1
                reasoning_extra_tokens += n_total - n_in - n_out

        non_empty = bool(answer) and str(answer).strip() != ""
        if (
            gen.get("success") is True
            and non_empty
            and not _has_usable_token_usage(token_usage)
        ):
            failed_rows += 1
            failed_by_model[model] += 1
            failed_indices.append(str(r.get("global index") or r.get("global_index")))

    return {
        "router": os.path.basename(path).replace(".json", ""),
        "main_rows": len(rows),
        "reasoning_rows": reasoning_rows,
        "reasoning_extra_tokens": reasoning_extra_tokens,
        "failed_rows": failed_rows,
        "failed_by_model": dict(failed_by_model),
        "failed_indices": failed_indices,
    }


def _resolve_paths(names: List[str]) -> List[str]:
    if names:
        paths = []
        for name in names:
            name = name.replace(".json", "")
            path = os.path.join(PREDICTIONS_DIR, f"{name}.json")
            if not os.path.exists(path):
                raise SystemExit(f"Prediction file not found: {path}")
            paths.append(path)
        return paths
    paths = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.json")))
    return [p for p in paths if "robustness" not in os.path.basename(p)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "routers",
        nargs="*",
        help="Router prediction file name(s) without .json (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any router has failed-inference (Problem 2) rows.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Print the global indices of failed-inference rows.",
    )
    args = parser.parse_args()

    summaries = [audit_file(p) for p in _resolve_paths(args.routers)]

    header = (
        f"{'router':28} {'mainRows':>8} {'reasonRows':>10} "
        f"{'extraTokens':>12} {'failedInfer':>11}"
    )
    print(header)
    print("-" * len(header))
    any_failed = False
    for s in summaries:
        any_failed = any_failed or s["failed_rows"] > 0
        print(
            f"{s['router']:28} {s['main_rows']:>8} {s['reasoning_rows']:>10} "
            f"{s['reasoning_extra_tokens']:>12,} {s['failed_rows']:>11}"
        )

    failed = [s for s in summaries if s["failed_rows"]]
    if failed:
        print(
            "\nFailed-inference rows by model (forward these counts to the router authors):"
        )
        for s in failed:
            print(f"  {s['router']}: {s['failed_by_model']}")
            if args.show_indices:
                print(f"    indices: {', '.join(s['failed_indices'])}")

    if args.strict and any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
