# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Price models that a router runs itself, with the rule in model_cost/local_pricing.yaml.

A profile in model_cost/model_profiles.yaml marks a local model with a ``local`` field:

    "lemonade/Qwen3.8-27B-GGUF-UD-Q4_K_XL":
      price: {input: 0.156, output: 0.488}
      local: {active_params_b: 27, bits: 4}       # no listed model with these weights
      sync: false
    "lemonade/deepseek-v4-flash":
      price: {input: 0.112, output: 0.224}
      local: {base: "deepseek/deepseek-v4-flash", bits: 2}   # a listed model's weights
      sync: false

``base`` names a profile (id or alias) whose price the local model takes; otherwise
``active_params_b`` (active parameters in billions; for a mixture-of-experts model, the
active ones) prices it from the size curve. ``bits`` is the precision of the served
weights. Both are multiplied by the rule's quantization factor.

Usage:
    python scripts/pricing/local_price.py --active-params-b 27 --bits 4   # price a model
    python scripts/pricing/local_price.py --base deepseek/deepseek-v4-flash --bits 2
    python scripts/pricing/local_price.py --check   # exit 1 if a local profile is out of date
    python scripts/pricing/local_price.py --write   # update local profiles, regenerate JSON
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_model_cost  # noqa: E402

RULE_PATH = build_model_cost.REPO_ROOT / "model_cost" / "local_pricing.yaml"

Price = Tuple[float, float]


def load_rule(path: Optional[Path] = None) -> Dict[str, Any]:
    with open(path or RULE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _round(value: float) -> float:
    return float(f"{value:.3g}")


def quantization_factor(bits: float, rule: Dict[str, Any]) -> float:
    for step in rule["quantization"]:
        if bits >= step["min_bits"]:
            return float(step["factor"])
    raise ValueError(f"no quantization factor for {bits} bits")


def size_curve_price(active_params_b: float, rule: Dict[str, Any]) -> Price:
    if active_params_b <= 0:
        raise ValueError("active_params_b must be positive")
    curve = rule["size_curve"]
    return (
        curve["input"]["coef"] * active_params_b ** curve["input"]["exponent"],
        curve["output"]["coef"] * active_params_b ** curve["output"]["exponent"],
    )


def find_profile(
    name: str, profiles: Dict[str, Dict[str, Any]]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if name in profiles:
        return name, profiles[name]
    for model_id, profile in profiles.items():
        if name in (profile.get("aliases") or []):
            return model_id, profile
    return None


def local_price(
    spec: Dict[str, Any], profiles: Dict[str, Dict[str, Any]], rule: Dict[str, Any]
) -> Tuple[Price, str]:
    """Return ((input, output), basis) for a ``local`` spec."""
    bits = spec.get("bits")
    if not isinstance(bits, (int, float)) or isinstance(bits, bool) or bits <= 0:
        raise ValueError("local.bits must be a positive number")
    factor = quantization_factor(bits, rule)
    if spec.get("base"):
        found = find_profile(spec["base"], profiles)
        if found is None:
            raise ValueError(f"local.base {spec['base']!r} has no profile")
        base_id, base = found
        if base.get("local"):
            raise ValueError(f"local.base {base_id!r} is itself a local model")
        price = (float(base["price"]["input"]), float(base["price"]["output"]))
        basis = f"price of {base_id}"
    else:
        active = spec.get("active_params_b")
        if not isinstance(active, (int, float)) or isinstance(active, bool):
            raise ValueError("local needs base or active_params_b")
        price = size_curve_price(float(active), rule)
        basis = f"size curve at {active:g}B active parameters"
    basis += f", x{factor:g} for {bits:g}-bit weights"
    return (_round(price[0] * factor), _round(price[1] * factor)), basis


def plan(
    profiles: Dict[str, Dict[str, Any]], rule: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Every local profile whose price differs from the rule's."""
    changes = []
    for model_id, profile in profiles.items():
        spec = profile.get("local")
        if not spec:
            continue
        new, basis = local_price(spec, profiles, rule)
        old = (float(profile["price"]["input"]), float(profile["price"]["output"]))
        if abs(old[0] - new[0]) > 1e-12 or abs(old[1] - new[1]) > 1e-12:
            changes.append({"model": model_id, "old": old, "new": new, "basis": basis})
    return changes


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def apply(text: str, changes: List[Dict[str, Any]]) -> str:
    """Rewrite the price line of each changed profile, leaving other lines alone."""
    lines = text.split("\n")
    for change in changes:
        header = re.compile(r'^  ("?)' + re.escape(change["model"]) + r"\1:\s*$")
        start = next((i for i, line in enumerate(lines) if header.match(line)), None)
        if start is None:
            raise ValueError(f"profile {change['model']!r} not found in the YAML file")
        new_in, new_out = change["new"]
        for j in range(start + 1, len(lines)):
            if not lines[j].startswith("    "):
                break
            if lines[j].startswith("    price:"):
                lines[j] = (
                    f"    price: {{input: {_fmt(new_in)}, output: {_fmt(new_out)}}}"
                )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="profile whose weights the local model serves")
    parser.add_argument("--active-params-b", type=float)
    parser.add_argument("--bits", type=float, default=16)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if a local profile is stale"
    )
    parser.add_argument(
        "--write", action="store_true", help="update local profiles and model_cost.json"
    )
    args = parser.parse_args(argv)
    rule = load_rule()
    profiles = build_model_cost.load_profiles(build_model_cost.PROFILES_PATH)

    if args.base or args.active_params_b:
        spec = {
            "base": args.base,
            "active_params_b": args.active_params_b,
            "bits": args.bits,
        }
        (p_in, p_out), basis = local_price(spec, profiles, rule)
        print(f"input {_fmt(p_in)} / output {_fmt(p_out)} USD per 1M tokens ({basis})")
        return 0

    changes = plan(profiles, rule)
    for c in changes:
        print(
            f"{c['model']}: {_fmt(c['old'][0])} / {_fmt(c['old'][1])} -> "
            f"{_fmt(c['new'][0])} / {_fmt(c['new'][1])} ({c['basis']})"
        )
    if args.check:
        if changes:
            print(
                "Local model prices are out of date. Run: "
                "python scripts/pricing/local_price.py --write",
                file=sys.stderr,
            )
            return 1
        print("Local model prices are up to date.")
        return 0
    if args.write and changes:
        path = build_model_cost.PROFILES_PATH
        path.write_text(
            apply(path.read_text(encoding="utf-8"), changes), encoding="utf-8"
        )
        updated = build_model_cost.load_profiles(path)
        build_model_cost.COST_PATH.write_text(
            build_model_cost.render(build_model_cost.build_cost_table(updated)),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
