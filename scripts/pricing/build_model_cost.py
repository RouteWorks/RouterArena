# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Generate ``model_cost/model_cost.json`` from ``model_cost/model_profiles.yaml``.

``model_profiles.yaml`` is the source of truth for model prices: one profile per
model, with optional aliases (other spellings of the same model). The evaluator
and the submission checks still read ``model_cost.json``, so this script expands
every profile into one JSON entry for its id and one for each alias, all at the
profile's price.

Usage:
    python scripts/pricing/build_model_cost.py          # rewrite model_cost.json
    python scripts/pricing/build_model_cost.py --check  # exit 1 if it is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = REPO_ROOT / "model_cost" / "model_profiles.yaml"
COST_PATH = REPO_ROOT / "model_cost" / "model_cost.json"


def load_profiles(path: Path = PROFILES_PATH) -> Dict[str, Dict[str, Any]]:
    """Load and validate the profile file. Raises ValueError on a bad entry."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path}: expected a non-empty 'models' mapping")

    seen: Dict[str, str] = {}
    errors: List[str] = []
    for model_id, profile in models.items():
        if not isinstance(profile, dict):
            errors.append(f"{model_id}: profile must be a mapping")
            continue
        price = profile.get("price") or {}
        for side in ("input", "output"):
            value = price.get(side)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                errors.append(f"{model_id}: price.{side} must be a number >= 0")
        aliases = profile.get("aliases") or []
        if not isinstance(aliases, list) or not all(
            isinstance(a, str) for a in aliases
        ):
            errors.append(f"{model_id}: aliases must be a list of strings")
            aliases = []
        for name in [model_id, *aliases]:
            if name in seen:
                errors.append(
                    f"{name!r} is used by both {seen[name]!r} and {model_id!r}"
                )
            seen[name] = model_id
    if errors:
        raise ValueError(f"{path}: invalid profiles:\n  " + "\n  ".join(errors))
    return models


def build_cost_table(models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Expand profiles into the flat ``model_cost.json`` mapping."""
    table: Dict[str, Dict[str, Any]] = {}
    for model_id, profile in models.items():
        entry = {
            "input_token_price_per_million": profile["price"]["input"],
            "output_token_price_per_million": profile["price"]["output"],
        }
        for name in [model_id, *(profile.get("aliases") or [])]:
            table[name] = dict(entry)
    return table


def render(table: Dict[str, Dict[str, Any]]) -> str:
    return json.dumps(table, indent=2, ensure_ascii=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if model_cost.json differs from the profiles.",
    )
    args = parser.parse_args(argv)

    try:
        expected = render(build_cost_table(load_profiles()))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    current = COST_PATH.read_text(encoding="utf-8") if COST_PATH.exists() else ""
    if args.check:
        if current != expected:
            print(
                "model_cost/model_cost.json is out of date with model_profiles.yaml. "
                "Run: python scripts/pricing/build_model_cost.py",
                file=sys.stderr,
            )
            return 1
        print("model_cost.json is up to date.")
        return 0

    COST_PATH.write_text(expected, encoding="utf-8")
    print(
        f"Wrote {COST_PATH.relative_to(REPO_ROOT)} ({len(json.loads(expected))} entries)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
