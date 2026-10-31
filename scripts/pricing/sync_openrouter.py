# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Sync model prices in ``model_cost/model_profiles.yaml`` with each model's standard price on OpenRouter.

A model is priced at its standard published price, not at a promotion and not at the
cheapest host of the day. For every profile that resolves to an OpenRouter model, the
price is read from OpenRouter's per-provider endpoints:

* only standard-tier endpoints count (no Flex, Priority or batch tiers), and only active
  ones when any are;
* a provider's promotional discount is undone (OpenRouter lists ``discount: 0.5`` next to
  a halved price, so the standard price is ``price / (1 - discount)``);
* when the model's own lab sells it on OpenRouter (Google for Gemini, Z.AI for GLM, ...),
  the lab's lowest standard price is used;
* otherwise the median over hosts is used, leaving out 4-bit quantizations when
  higher-precision hosts exist.

The profile's ``price``, ``as_of`` and ``source`` are updated and ``model_cost.json`` is
regenerated. A profile resolves through its ``openrouter_id`` field, or else through an
exact match of its id or one of its aliases against OpenRouter's model ids. Set
``sync: false`` on a profile to keep its price fixed (a retired model, a model OpenRouter
does not list, ...). A model whose endpoints cannot be read keeps its price and is listed
in the report.

When a price changes, the run also writes ``model_cost/price_snapshots/<date>.json`` with
the price and its basis for every resolved model, so each change point is kept as history
(``--always-snapshot`` writes one even when nothing changed).

Syncing changes the price table only. Leaderboard costs change when a maintainer re-prices
the routers, as a separate step (scripts/pricing/reprice.py).

Usage:
    python scripts/pricing/sync_openrouter.py                      # fetch, update, write
    python scripts/pricing/sync_openrouter.py --dry-run            # report only
    python scripts/pricing/sync_openrouter.py --models-json f      # use a saved /models response
    python scripts/pricing/sync_openrouter.py --endpoints-dir d    # use saved endpoint responses
    python scripts/pricing/sync_openrouter.py --report out.md      # also write a Markdown report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_model_cost  # noqa: E402

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{}/endpoints"
SNAPSHOT_DIR = build_model_cost.REPO_ROOT / "model_cost" / "price_snapshots"

# The lab's own provider name(s) on OpenRouter, by the model id's vendor prefix.
FIRST_PARTY = {
    "anthropic": {"Anthropic"},
    "deepseek": {"DeepSeek"},
    "google": {"Google", "Google AI Studio"},
    "inclusionai": {"inclusionAI"},
    "minimax": {"Minimax", "MiniMax"},
    "mistralai": {"Mistral"},
    "moonshotai": {"Moonshot AI"},
    "nvidia": {"NVIDIA"},
    "openai": {"OpenAI"},
    "qwen": {"Alibaba"},
    "upstage": {"Upstage"},
    "x-ai": {"xAI"},
    "xiaomi": {"Xiaomi"},
    "z-ai": {"Z.AI"},
}
# Quantizations below 8 bits: a cheaper, degraded variant, not the model's standard serving.
LOW_BIT = {"fp4", "mxfp4", "nvfp4", "int4"}

Price = Tuple[float, float]
Endpoints = Optional[List[Dict[str, Any]]]


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
    """Map OpenRouter model id -> (input, output) list price, USD per 1M tokens.

    Used to resolve profiles to OpenRouter ids; the price itself comes from
    ``standard_price``. ``source`` is a saved API response (JSON file); by default the
    live API is fetched."""
    if source:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = _get_json(OPENROUTER_MODELS_URL)
    prices: Dict[str, Price] = {}
    for model in payload.get("data", []):
        pricing = model.get("pricing") or {}
        prompt = per_million(pricing.get("prompt"))
        completion = per_million(pricing.get("completion"))
        if prompt is None or completion is None:
            continue
        prices[model["id"]] = (prompt, completion)
    return prices


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "routerarena-price-sync"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def endpoints_loader(endpoints_dir: Optional[str]) -> Callable[[str], Endpoints]:
    """Return a function that gives an OpenRouter id's endpoint list, or None if it
    can't be read. Reads ``<dir>/<vendor>__<model>.json`` when a directory is given,
    else the live API (with retries)."""

    def from_dir(or_id: str) -> Endpoints:
        assert endpoints_dir is not None
        path = Path(endpoints_dir) / f"{or_id.replace('/', '__')}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (payload.get("data") or {}).get("endpoints")

    def from_api(or_id: str) -> Endpoints:
        for attempt in range(3):
            try:
                payload = _get_json(OPENROUTER_ENDPOINTS_URL.format(or_id))
                return (payload.get("data") or {}).get("endpoints")
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    return from_dir if endpoints_dir else from_api


def standard_price(or_id: str, endpoints: Endpoints) -> Optional[Dict[str, Any]]:
    """The model's standard price from its OpenRouter endpoints (see the module
    docstring), as {"price": (input, output), "basis": str, "discount_removed": float},
    or None if no standard-tier endpoint has a price."""
    standard = [
        e
        for e in endpoints or []
        if e.get("service_tier") in (None, "standard")
        and per_million((e.get("pricing") or {}).get("prompt")) is not None
        and per_million((e.get("pricing") or {}).get("completion")) is not None
    ]
    active = [e for e in standard if (e.get("status") or 0) >= 0]
    candidates = active or standard
    if not candidates:
        return None

    def undiscounted(endpoint: Dict[str, Any]) -> Tuple[float, float, float]:
        pricing = endpoint["pricing"]
        discount = float(pricing.get("discount") or 0)
        factor = 1 / (1 - discount) if 0 < discount < 1 else 1.0
        return (
            round(float(pricing["prompt"]) * 1e6 * factor, 6),
            round(float(pricing["completion"]) * 1e6 * factor, 6),
            discount,
        )

    lab = FIRST_PARTY.get(or_id.split("/")[0], set())
    own = [e for e in candidates if e.get("provider_name") in lab]
    if own:
        best = min(own, key=lambda e: sum(undiscounted(e)[:2]))
        i, o, d = undiscounted(best)
        return {
            "price": (i, o),
            "basis": f"{best['provider_name']} (the lab's own price)",
            "discount_removed": d,
        }
    hosts = [
        e for e in candidates if str(e.get("quantization")) not in LOW_BIT
    ] or candidates
    prices = [undiscounted(e) for e in hosts]
    return {
        "price": (
            round(statistics.median(p[0] for p in prices), 6),
            round(statistics.median(p[1] for p in prices), 6),
        ),
        "basis": f"median of {len(prices)} host(s)",
        "discount_removed": max(p[2] for p in prices),
    }


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
    profiles: Dict[str, Dict[str, Any]],
    openrouter: Dict[str, Price],
    get_endpoints: Callable[[str], Endpoints],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str], List[str]]:
    """Return (changes, resolved, unresolved, no_data).

    ``changes`` lists profiles whose price differs from the standard price;
    ``resolved`` maps profile id -> {"openrouter_id", "price", "basis",
    "discount_removed"} for every synced profile; ``unresolved`` lists profiles that sync
    but match no OpenRouter model; ``no_data`` lists profiles whose endpoints could not
    be read (they keep their price)."""
    changes: List[Dict[str, Any]] = []
    resolved: Dict[str, Dict[str, Any]] = {}
    unresolved: List[str] = []
    no_data: List[str] = []
    for model_id, profile in profiles.items():
        if profile.get("sync") is False:
            continue
        or_id = resolve_openrouter_id(model_id, profile, openrouter)
        if or_id is None:
            unresolved.append(model_id)
            continue
        found = standard_price(or_id, get_endpoints(or_id))
        if found is None:
            no_data.append(model_id)
            continue
        resolved[model_id] = {"openrouter_id": or_id, **found}
        new = found["price"]
        old = (float(profile["price"]["input"]), float(profile["price"]["output"]))
        if abs(old[0] - new[0]) > 1e-9 or abs(old[1] - new[1]) > 1e-9:
            changes.append(
                {
                    "model": model_id,
                    "openrouter_id": or_id,
                    "old": old,
                    "new": new,
                    "basis": found["basis"],
                    "discount_removed": found["discount_removed"],
                }
            )
    return changes, resolved, unresolved, no_data


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


def write_snapshot(resolved: Dict[str, Dict[str, Any]], today: str) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{today}.json"
    prices = {}
    for entry in sorted(resolved.values(), key=lambda e: e["openrouter_id"]):
        prices[entry["openrouter_id"]] = {
            "input": entry["price"][0],
            "output": entry["price"][1],
            "basis": entry["basis"],
            "discount_removed": entry["discount_removed"],
        }
    payload = {
        "date": today,
        "source": OPENROUTER_ENDPOINTS_URL.format("<id>"),
        "rule": "standard price: the lab's own standard-tier price, else the median "
        "over hosts (no 4-bit); promotional discounts undone",
        "prices": prices,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def report(
    changes: List[Dict[str, Any]],
    unresolved: List[str],
    no_data: List[str],
    today: str,
) -> str:
    out = [f"## OpenRouter price sync, {today}", ""]
    if changes:
        out += [
            f"{len(changes)} price(s) changed (USD per 1M tokens, input / output):",
            "",
            "| Model | OpenRouter id | Old | New | Basis |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in changes:
            basis = c["basis"]
            if c["discount_removed"]:
                basis += f", {round(100 * c['discount_removed'])}% promotion undone"
            ratio = max(
                (c["new"][k] / c["old"][k]) if c["old"][k] else 0 for k in (0, 1)
            )
            flag = " ⚠️" if ratio >= 2 or ratio <= 0.5 else ""
            out.append(
                f"| `{c['model']}`{flag} | `{c['openrouter_id']}` | "
                f"{_fmt(c['old'][0])} / {_fmt(c['old'][1])} | "
                f"{_fmt(c['new'][0])} / {_fmt(c['new'][1])} | {basis} |"
            )
        out += ["", "⚠️ marks a price that at least doubled or halved."]
    else:
        out.append("No price changed.")
    if no_data:
        out += [
            "",
            f"{len(no_data)} profile(s) kept their price because OpenRouter's endpoints "
            "could not be read: " + ", ".join(f"`{m}`" for m in no_data),
        ]
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
    parser.add_argument(
        "--endpoints-dir",
        help="saved /models/<id>/endpoints responses, named <vendor>__<model>.json",
    )
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
    changes, resolved, unresolved, no_data = plan_updates(
        profiles, openrouter, endpoints_loader(args.endpoints_dir)
    )
    text = report(changes, unresolved, no_data, args.date)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    if args.dry_run:
        return 0

    if changes or args.always_snapshot:
        write_snapshot(resolved, args.date)
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
