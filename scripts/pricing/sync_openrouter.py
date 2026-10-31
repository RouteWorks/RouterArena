# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Sync model prices in ``model_cost/model_profiles.yaml`` with OpenRouter's list prices.

For every profile that resolves to an OpenRouter model, the profile's price is set to
OpenRouter's current list price (prompt/completion, converted to USD per 1M tokens), and
``as_of`` and ``source`` are updated. ``model_cost.json`` is then regenerated.

A profile resolves through its ``openrouter_id`` field, or else through an exact match of
its id or one of its aliases against OpenRouter's model ids. Set ``sync: false`` on a
profile to keep its price fixed (a vendor's standard price, a local model, ...).

When a price changes, the run also writes ``model_cost/price_snapshots/<date>.json`` with
the OpenRouter prices of every resolved model, so each change point is kept as history
(``--always-snapshot`` writes one even when nothing changed).

Syncing changes the price table only. Leaderboard costs change when a maintainer re-prices
the routers, as a separate step.

Usage:
    python scripts/pricing/sync_openrouter.py                    # fetch, update, write
    python scripts/pricing/sync_openrouter.py --dry-run          # report only
    python scripts/pricing/sync_openrouter.py --models-json f    # use a saved API response
    python scripts/pricing/sync_openrouter.py --report out.md    # also write a Markdown report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_model_cost  # noqa: E402

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
SNAPSHOT_DIR = build_model_cost.REPO_ROOT / "model_cost" / "price_snapshots"

Price = Tuple[float, float]


def per_million(per_token: Any) -> Optional[float]:
    """OpenRouter prices are USD per token, as strings. Return USD per 1M tokens."""
    try:
        value = float(per_token)
    except (TypeError, ValueError):
        return None
    if value < 0:  # OpenRouter uses -1 for "varies" (e.g. routers like openrouter/auto)
        return None
    return round(value * 1_000_000, 6)


def load_openrouter_prices(source: Optional[str] = None) -> Dict[str, Price]:
    """Map OpenRouter model id -> (input, output) USD per 1M tokens.

    ``source`` is a saved API response (JSON file); by default the live API is fetched."""
    if source:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(
            OPENROUTER_MODELS_URL, headers={"User-Agent": "routerarena-price-sync"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    prices: Dict[str, Price] = {}
    for model in payload.get("data", []):
        pricing = model.get("pricing") or {}
        prompt = per_million(pricing.get("prompt"))
        completion = per_million(pricing.get("completion"))
        if prompt is None or completion is None:
            continue
        prices[model["id"]] = (prompt, completion)
    return prices


def resolve_openrouter_id(
    model_id: str, profile: Dict[str, Any], openrouter: Dict[str, Price]
) -> Optional[str]:
    """The OpenRouter id a profile syncs from, or None. No fuzzy matching."""
    explicit = profile.get("openrouter_id")
    if explicit:
        return explicit if explicit in openrouter else None
    for name in [model_id, *(profile.get("aliases") or [])]:
        if name in openrouter:
            return name
    return None


def plan_updates(
    profiles: Dict[str, Dict[str, Any]], openrouter: Dict[str, Price]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """Return (changes, resolved, unresolved).

    ``changes`` lists profiles whose price differs from OpenRouter's; ``resolved`` maps
    profile id -> OpenRouter id for every synced profile; ``unresolved`` lists profiles
    that sync but match no OpenRouter model (or whose ``openrouter_id`` is unknown)."""
    changes: List[Dict[str, Any]] = []
    resolved: Dict[str, str] = {}
    unresolved: List[str] = []
    for model_id, profile in profiles.items():
        if profile.get("sync") is False:
            continue
        or_id = resolve_openrouter_id(model_id, profile, openrouter)
        if or_id is None:
            unresolved.append(model_id)
            continue
        resolved[model_id] = or_id
        new = openrouter[or_id]
        old = (float(profile["price"]["input"]), float(profile["price"]["output"]))
        if abs(old[0] - new[0]) > 1e-9 or abs(old[1] - new[1]) > 1e-9:
            changes.append(
                {"model": model_id, "openrouter_id": or_id, "old": old, "new": new}
            )
    return changes, resolved, unresolved


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def apply_updates(text: str, changes: List[Dict[str, Any]], today: str) -> str:
    """Rewrite the price, as_of and source lines of each changed profile in place,
    leaving every other line (comments, aliases, notes) untouched."""
    lines = text.split("\n")
    for change in changes:
        key = change["model"]
        header = re.compile(r'^  ("?)' + re.escape(key) + r"\1:\s*$")
        start = next((i for i, line in enumerate(lines) if header.match(line)), None)
        if start is None:
            raise ValueError(f"profile {key!r} not found as a block in the YAML file")
        end = start + 1
        while end < len(lines) and lines[end].startswith("    "):
            end += 1
        block = lines[start + 1 : end]
        new_in, new_out = change["new"]
        source = f'"https://openrouter.ai/{change["openrouter_id"]}"'
        fields = {
            "price": f"    price: {{input: {_fmt(new_in)}, output: {_fmt(new_out)}}}",
            "as_of": f'    as_of: "{today}"',
            "source": f"    source: {source}",
        }
        seen = set()
        for j, line in enumerate(block):
            for name, replacement in fields.items():
                if line.startswith(f"    {name}:"):
                    block[j] = replacement
                    seen.add(name)
        block.extend(
            fields[name] for name in ("price", "as_of", "source") if name not in seen
        )
        lines[start + 1 : end] = block
    return "\n".join(lines)


def write_snapshot(
    resolved: Dict[str, str], openrouter: Dict[str, Price], today: str
) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{today}.json"
    prices = {
        or_id: {"input": openrouter[or_id][0], "output": openrouter[or_id][1]}
        for or_id in sorted(set(resolved.values()))
    }
    payload = {"date": today, "source": OPENROUTER_MODELS_URL, "prices": prices}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def report(changes: List[Dict[str, Any]], unresolved: List[str], today: str) -> str:
    out = [f"## OpenRouter price sync, {today}", ""]
    if changes:
        out += [
            f"{len(changes)} price(s) changed (USD per 1M tokens, input / output):",
            "",
            "| Model | OpenRouter id | Old | New |",
            "| --- | --- | --- | --- |",
        ]
        for c in changes:
            out.append(
                f"| `{c['model']}` | `{c['openrouter_id']}` | "
                f"{_fmt(c['old'][0])} / {_fmt(c['old'][1])} | "
                f"{_fmt(c['new'][0])} / {_fmt(c['new'][1])} |"
            )
    else:
        out.append("No price changed.")
    if unresolved:
        out += [
            "",
            f"{len(unresolved)} profile(s) match no OpenRouter model and kept their price "
            "(add `openrouter_id`, or `sync: false` to silence): "
            + ", ".join(f"`{m}`" for m in unresolved),
        ]
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-json", help="saved OpenRouter /models response")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--dry-run", action="store_true", help="report only; write nothing"
    )
    parser.add_argument("--report", help="also write the Markdown report to this path")
    parser.add_argument(
        "--always-snapshot",
        action="store_true",
        help="write the price snapshot even when no price changed",
    )
    args = parser.parse_args(argv)

    profiles = build_model_cost.load_profiles(build_model_cost.PROFILES_PATH)
    openrouter = load_openrouter_prices(args.models_json)
    changes, resolved, unresolved = plan_updates(profiles, openrouter)
    text = report(changes, unresolved, args.date)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    if args.dry_run:
        return 0

    if changes or args.always_snapshot:
        write_snapshot(resolved, openrouter, args.date)
    if changes:
        path = build_model_cost.PROFILES_PATH
        path.write_text(
            apply_updates(path.read_text(encoding="utf-8"), changes, args.date),
            encoding="utf-8",
        )
        updated = build_model_cost.load_profiles(path)  # validates the edited file
        build_model_cost.COST_PATH.write_text(
            build_model_cost.render(build_model_cost.build_cost_table(updated)),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
