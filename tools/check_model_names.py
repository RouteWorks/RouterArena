# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Check that every model a submission names can be priced.

Addresses issue #193. A model named by a prediction file is priced by looking
its name up in ``model_cost/model_cost.json``. That lookup is exact, while the
name is written by the submitter -- so a model that IS in the price table can
still fail to resolve on letter case or on the vendor separator, and an
unpriceable row is then scored around rather than charged.

This audit reports three things:

  Problem 1 (registry hygiene): price-table keys that differ only by case or by
    ``_`` vs ``/``. Two spellings of one model are two entries that can drift to
    two different prices, which is the relabeling hole issue #193 is about.

  Problem 2 (ambiguous universal names): entries in ``universal_names`` /
    ``mapping`` that fold onto the same key while naming different models. These
    make the canonical fallback ambiguous and should be resolved by hand.

  Problem 3 (unpriceable rows): models named by a prediction file that have no
    price entry. Split into names the canonical fold repairs (a spelling
    mismatch -- the fix is to normalise the lookup) and names nothing repairs
    (genuinely absent from the price table -- the fix is to register a price,
    per the pricing policy in issue #193).

Usage:
    python tools/check_model_names.py                    # audit all routers
    python tools/check_model_names.py llm-router lynkr   # audit named routers
    python tools/check_model_names.py --strict           # exit 1 on Problem 3

Only "main" rows (for_optimality == False, the rows that feed the RouterArena
score) are counted, matching the leaderboard.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from universal_model_names import (  # noqa: E402
    canonical_collisions,
    canonical_key,
    resolve_universal_name,
)

PREDICTIONS_DIR = os.path.join(REPO_ROOT, "router_inference", "predictions")
COST_TABLE = os.path.join(REPO_ROOT, "model_cost", "model_cost.json")


def _load_cost_table() -> Dict[str, Any]:
    with open(COST_TABLE, "r", encoding="utf-8") as f:
        return json.load(f)


def _price_of(entry: Dict[str, Any]) -> Tuple[Any, Any]:
    return (
        entry.get("input_token_price_per_million"),
        entry.get("output_token_price_per_million"),
    )


def audit_cost_table(cost: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Problem 1: price-table keys that fold together."""
    folded: Dict[str, List[str]] = {}
    for key in cost:
        folded.setdefault(canonical_key(key), []).append(key)

    duplicates = []
    for fold, keys in sorted(folded.items()):
        if len(keys) < 2:
            continue
        prices = {_price_of(cost[k]) for k in keys}
        duplicates.append(
            {
                "canonical": fold,
                "keys": sorted(keys),
                "prices": sorted(prices),
                "conflict": len(prices) > 1,
            }
        )
    return duplicates


def resolve_price_key(
    model_name: str, cost: Dict[str, Any]
) -> Tuple[str, Optional[str]]:
    """Resolve a model name to a price-table key.

    Returns (status, key) where status is one of:
      "ok"           -- the name as written has a price entry
      "fold-repairs" -- no exact entry, but a case/separator fold finds one
      "missing"      -- nothing in the price table names this model
    """
    if model_name in cost:
        return "ok", model_name

    # A submission may name a model by an alias the pipeline already knows.
    universal = resolve_universal_name(model_name)
    if universal is not None and universal in cost:
        return "ok", universal

    folded = {canonical_key(k): k for k in cost}
    for candidate in (model_name, universal):
        if candidate is None:
            continue
        hit = folded.get(canonical_key(candidate))
        if hit is not None:
            return "fold-repairs", hit

    return "missing", None


def audit_predictions(path: str, cost: Dict[str, Any]) -> Dict[str, Any]:
    """Problem 3: models named by one prediction file that cannot be priced."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        r for r in data if isinstance(r, dict) and not r.get("for_optimality", False)
    ]
    named: Counter = Counter(r.get("prediction") for r in rows)

    unpriceable = []
    for model, count in named.items():
        if model is None:
            continue
        status, key = resolve_price_key(model, cost)
        if status != "ok":
            unpriceable.append(
                {
                    "model": model,
                    "rows": count,
                    "share": count / len(rows) if rows else 0.0,
                    "status": status,
                    "resolves_to": key,
                }
            )

    return {
        "router": os.path.basename(path)[: -len(".json")],
        "main_rows": len(rows),
        "unpriceable": sorted(unpriceable, key=lambda u: -u["rows"]),
    }


def _resolve_paths(routers: List[str]) -> List[str]:
    if not routers:
        return sorted(
            p
            for p in glob.glob(os.path.join(PREDICTIONS_DIR, "*.json"))
            if not p.endswith("-robustness.json")
        )
    return [os.path.join(PREDICTIONS_DIR, f"{name}.json") for name in routers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "routers",
        nargs="*",
        help="Router names to audit (default: every non-robustness prediction file).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any named model has no price entry at all (Problem 3).",
    )
    args = parser.parse_args()

    cost = _load_cost_table()

    print("Problem 1 - price-table keys that differ only by case or separator")
    duplicates = audit_cost_table(cost)
    if not duplicates:
        print("  none")
    for dup in duplicates:
        mark = "PRICE CONFLICT" if dup["conflict"] else "duplicate"
        print(f"  [{mark}] {dup['canonical']}")
        print(f"      keys:   {dup['keys']}")
        print(f"      prices: {dup['prices']}")

    print("\nProblem 2 - universal names that fold together ambiguously")
    if not canonical_collisions:
        print("  none")
    for key, names in sorted(canonical_collisions.items()):
        print(f"  {key} <- {names}")

    print("\nProblem 3 - models named by a submission that cannot be priced")
    summaries = [audit_predictions(p, cost) for p in _resolve_paths(args.routers)]
    header = f"{'router':26} {'model named':40} {'rows':>6} {'share':>7}  resolution"
    print(header)
    print("-" * len(header))

    any_missing = False
    any_row = False
    for summary in summaries:
        for item in summary["unpriceable"]:
            any_row = True
            if item["status"] == "missing":
                any_missing = True
                resolution = "NO PRICE ENTRY - register one (issue #193 policy)"
            else:
                resolution = f"spelling only - folds to {item['resolves_to']}"
            print(
                f"{summary['router']:26} {item['model']:40} "
                f"{item['rows']:>6} {item['share'] * 100:>6.1f}%  {resolution}"
            )
    if not any_row:
        print("  every model named by every audited submission has a price entry")

    if args.strict and any_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
