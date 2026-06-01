# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Generate the derived data files consumed by the RouteWorks website.

Single source of truth: ``leaderboard_manifest.yaml`` (display names / links /
which routers are shown) plus the *evaluated* prediction files under
``router_inference/predictions/``. From these this script emits, into an output
directory mirroring the website's ``src/data`` layout:

* ``routerMetrics/leaderboard.json`` - headline metrics per router.
* ``routerMetrics/category_scores.json`` - per-difficulty accuracy/cost/robustness.
* ``flip_labels/flip_labels_<key>.json`` - per-query robustness flip labels.

The prediction files must already carry per-query ``accuracy``/``cost`` (i.e. they
have been through ``llm_evaluation/run.py``); this script does not re-run the
judge, so it is cheap enough to run in CI.

Usage:
    python scripts/website/build_site_data.py --out ./_site_data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

import re  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from global_utils.robustness import (  # noqa: E402
    compute_flip_labels,
    compute_robustness_score,
)

PRED_DIR = REPO_ROOT / "router_inference" / "predictions"
MANIFEST = REPO_ROOT / "leaderboard_manifest.yaml"
README = REPO_ROOT / "README.md"
DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


def _num(cell: str) -> Optional[float]:
    """Parse a leaderboard cell into a float, or None for '—'/blank."""
    cell = cell.strip().lstrip("$")
    if not cell or cell == "—":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _norm_name(name: str) -> str:
    """Normalize a router display name for matching (dash variants, spacing)."""
    return re.sub(r"[‐-―\-]", "-", name).strip().lower()


def parse_readme_leaderboard() -> dict[str, dict[str, Any]]:
    """Parse the README leaderboard table -> {display_name: official metrics}.

    The README rows are the authoritative, already-evaluated numbers, so the
    website's headline figures are sourced here to guarantee zero drift.
    Columns: Rank | Router | Affiliation | Arena | Accuracy | Cost/1K |
             OptSel | OptCost | OptAcc | Latency | Robustness
    """
    out: dict[str, dict[str, Any]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 11 or cells[0] in ("Rank", "------"):
            continue
        name_match = re.search(r"\[([^\]]+)\]", cells[1])
        if not name_match:
            continue
        out[_norm_name(name_match.group(1))] = {
            "arena": _num(cells[3]),
            "accuracy": _num(cells[4]),
            "cost_per_1k": _num(cells[5]),
            "opt_sel": _num(cells[6]),
            "opt_cost": _num(cells[7]),
            "opt_acc": _num(cells[8]),
            "latency": _num(cells[9]),
            "robustness": _num(cells[10]),
        }
    return out


def compute_arena_score(
    cost: float,
    accuracy: float,
    beta: float = 0.1,
    c_max: float = 200,
    c_min: float = 0.0044,
) -> float:
    """RouterArena score (mirrors llm_evaluation/run.py)."""
    cost = max(c_min, min(cost, c_max))
    c_i = (math.log2(c_max) - math.log2(cost)) / (math.log2(c_max) - math.log2(c_min))
    return ((1 + beta) * accuracy * c_i) / (beta * accuracy + c_i)


def load_predictions(
    router: str, robustness: bool = False
) -> Optional[list[dict[str, Any]]]:
    suffix = "-robustness" if robustness else ""
    path = PRED_DIR / f"{router}{suffix}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_config_models(router: str) -> Optional[list[str]]:
    """Read a router's model pool from its config (for routers.json metadata)."""
    path = REPO_ROOT / "router_inference" / "config" / f"{router}.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    params = cfg.get("pipeline_params", cfg)
    models = params.get("models")
    return models if isinstance(models, list) else None


def load_dataset_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (difficulty, domain, category) maps keyed by global index.

    Domain/category keys are stripped of the leading ``"<n> "`` index to match
    the website's ``categories``/``subcategories`` keys
    (e.g. ``"5 Science"`` -> ``"Science"``).
    """
    from datasets import load_from_disk

    dataset_dir = os.environ.get("ROUTERARENA_DATASET_DIR", str(REPO_ROOT / "dataset"))
    ds = load_from_disk(str(Path(dataset_dir) / "routerarena"))
    difficulty, domain, category = {}, {}, {}
    for row in ds:
        gi = row["Global Index"]
        difficulty[gi] = str(row["Difficulty"]).lower()
        domain[gi] = re.sub(r"^\d+\s+", "", str(row["Domain"])).strip()
        category[gi] = re.sub(r"^\d+\s+", "", str(row["Category"])).strip()
    return difficulty, domain, category


def _regular(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in predictions if not p.get("for_optimality", False)]


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def compute_headline_metrics(
    predictions: list[dict[str, Any]],
    robustness_predictions: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Arena / accuracy / cost / robustness from evaluated predictions."""
    regular = _regular(predictions)
    n = len(regular)
    accuracies = [_numeric(p.get("accuracy")) or 0.0 for p in regular]
    costs = [c for p in regular if (c := _numeric(p.get("cost"))) and c > 0]
    avg_acc = sum(accuracies) / n if n else 0.0
    cost_per_1k = (sum(costs) / n * 1000) if n else 0.0
    arena = compute_arena_score(cost_per_1k, avg_acc) if cost_per_1k > 0 else None
    robustness = None
    if robustness_predictions:
        robustness = compute_robustness_score(predictions, robustness_predictions)
    return {
        "arena": round(arena * 100, 2) if arena is not None else None,
        "accuracy": round(avg_acc * 100, 2),
        "cost_per_1k": round(cost_per_1k, 2),
        "robustness": round(robustness * 100, 2) if robustness is not None else None,
    }


def _difficulty_metrics(
    entries: list[dict[str, Any]],
    flip_map: dict[str, int],
    difficulty: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """{easy/medium/hard/all: {accuracy, cost, robustness}} over the given entries."""
    buckets: dict[str, dict[str, list[float]]] = {
        b: {"acc": [], "cost": []} for b in (*DIFFICULTY_BUCKETS, "all")
    }
    flip_buckets: dict[str, list[int]] = {b: [] for b in (*DIFFICULTY_BUCKETS, "all")}

    for p in entries:
        gi = p.get("global index")
        acc = _numeric(p.get("accuracy")) or 0.0
        cost = _numeric(p.get("cost")) or 0.0
        diff = difficulty.get(gi) if isinstance(gi, str) else None
        targets = ["all"] + ([diff] if diff in DIFFICULTY_BUCKETS else [])
        for t in targets:
            buckets[t]["acc"].append(acc)
            buckets[t]["cost"].append(cost)
        if isinstance(gi, str) and gi in flip_map:
            for t in targets:
                flip_buckets[t].append(flip_map[gi])

    out: dict[str, dict[str, Any]] = {}
    for b in (*DIFFICULTY_BUCKETS, "all"):
        accs, costs, flips = buckets[b]["acc"], buckets[b]["cost"], flip_buckets[b]
        # Empty buckets are reported as 0 (matches the website's convention).
        out[b] = {
            "accuracy": round(sum(accs) / len(accs) * 100, 1) if accs else 0,
            "cost": round(sum(costs) / len(costs), 4) if costs else 0,
            "robustness": round((1 - sum(flips) / len(flips)) * 100, 1) if flips else 0,
        }
    return out


def compute_category_scores(
    predictions: list[dict[str, Any]],
    flip_labels: list[dict[str, Any]],
    difficulty: dict[str, str],
    domain: dict[str, str],
    category: dict[str, str],
) -> dict[str, Any]:
    """Website per-router entry: overall + per-domain + per-subcategory metrics.

    Mirrors the website schema:
        {"metrics": {<difficulty>},
         "categories": {<domain>: {"metrics": {<difficulty>},
                                   "subcategories": {<category>: {"metrics": {<difficulty>}}}}}}
    """
    flip_map = {item["global index"]: item["flip"] for item in flip_labels}
    regular = _regular(predictions)

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for p in regular:
        gi = p.get("global index")
        dom = domain.get(gi) if isinstance(gi, str) else None
        if dom:
            by_domain.setdefault(dom, []).append(p)

    categories: dict[str, Any] = {}
    for dom, dom_entries in sorted(by_domain.items()):
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for p in dom_entries:
            gi = p.get("global index")
            cat = category.get(gi) if isinstance(gi, str) else None
            if cat:
                by_cat.setdefault(cat, []).append(p)
        categories[dom] = {
            "metrics": _difficulty_metrics(dom_entries, flip_map, difficulty),
            "subcategories": {
                cat: {"metrics": _difficulty_metrics(entries, flip_map, difficulty)}
                for cat, entries in sorted(by_cat.items())
            },
        }

    return {
        "metrics": _difficulty_metrics(regular, flip_map, difficulty),
        "categories": categories,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, help="Output directory (website src/data layout)."
    )
    parser.add_argument(
        "--skip-category",
        action="store_true",
        help="Skip category_scores (avoids loading the dataset).",
    )
    args = parser.parse_args(argv)

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    routers = [r for r in manifest["routers"] if r.get("on_leaderboard", True)]
    official = parse_readme_leaderboard()

    out = Path(args.out)
    lb_path = out / "routerMetrics" / "leaderboard.json"
    cs_path = out / "routerMetrics" / "category_scores.json"
    (out / "routerMetrics").mkdir(parents=True, exist_ok=True)
    (out / "flip_labels").mkdir(parents=True, exist_ok=True)

    # MERGE into existing website data: start from whatever is already there so
    # externally pre-computed baselines are preserved, then update/insert ours.
    leaderboard = (
        json.loads(lb_path.read_text(encoding="utf-8")) if lb_path.exists() else []
    )
    by_name = {_norm_name(r.get("Router Name", "")): r for r in leaderboard}
    category_scores = (
        json.loads(cs_path.read_text(encoding="utf-8")) if cs_path.exists() else {}
    )
    # routers.json holds hand-curated router metadata (affiliation, type, model
    # pool). Merge our manifest `website` fields in, preserving everything else.
    rj_path = out / "routers.json"
    routers_meta = (
        json.loads(rj_path.read_text(encoding="utf-8")) if rj_path.exists() else {}
    )

    difficulty, domain, category = (
        ({}, {}, {}) if args.skip_category else load_dataset_maps()
    )
    updated, regenerated, missing = [], [], []

    for meta in routers:
        readme_name = meta["readme_name"]
        website_name = meta["website_name"]
        metrics = official.get(_norm_name(readme_name))
        if metrics is None:
            missing.append(readme_name)
            continue
        # Update/insert the leaderboard row, matched by website name.
        row = by_name.get(_norm_name(website_name), {"Router Name": website_name})
        row.update(
            {
                "Router Name": website_name,
                "Arena Score": metrics["arena"],
                "Optimal Selection Score": metrics["opt_sel"],
                "Optimal Cost Score": metrics["opt_cost"],
                "Optimal Acc. Score": metrics["opt_acc"],
                "Robustness Score": metrics["robustness"],
                "Latency Score": metrics["latency"],
                "Accuracy": metrics["accuracy"],
                "Cost per 1k": metrics["cost_per_1k"],
            }
        )
        if _norm_name(website_name) not in by_name:
            leaderboard.append(row)
            by_name[_norm_name(website_name)] = row
        updated.append(website_name)

        # Router metadata (affiliation / github / type) for the website.
        web = meta.get("website")
        if web:
            entry = routers_meta.get(website_name, {})
            entry.setdefault("name", website_name)
            if "affiliation" in web:
                entry["affiliation"] = web["affiliation"]
            if "github_url" in web:
                entry["githubUrl"] = web["github_url"]
            if "type" in web:
                entry["type"] = web["type"]
            entry.setdefault("type", "open-source")
            entry.setdefault(
                "description", f"Submitted by {web.get('affiliation', website_name)}."
            )
            if "modelPool" not in entry and meta.get("prediction"):
                models = load_config_models(meta["prediction"])
                if models:
                    entry["modelPool"] = models
            routers_meta[website_name] = entry

        # Derived data: regenerate only when RouterArena has the inputs.
        prediction = meta.get("prediction")
        if not prediction:
            continue
        preds = load_predictions(prediction)
        if preds is None:
            continue
        rob = load_predictions(prediction, robustness=True)
        flip_key = meta.get("flip_key")
        if rob and flip_key:
            flips = compute_flip_labels(preds, rob)
            if flips:
                (out / "flip_labels" / f"flip_labels_{flip_key}.json").write_text(
                    json.dumps(flips, indent=2) + "\n", encoding="utf-8"
                )
        else:
            flips = []
        cat_key = meta.get("category_key")
        if (
            not args.skip_category
            and cat_key
            and any(_numeric(p.get("accuracy")) is not None for p in _regular(preds))
        ):
            category_scores[cat_key] = compute_category_scores(
                preds, flips, difficulty, domain, category
            )
            regenerated.append(prediction)

    leaderboard.sort(
        key=lambda r: (r.get("Arena Score") is not None, r.get("Arena Score") or 0),
        reverse=True,
    )
    lb_path.write_text(json.dumps(leaderboard, indent=2) + "\n", encoding="utf-8")
    if not args.skip_category:
        cs_path.write_text(
            json.dumps(category_scores, indent=2) + "\n", encoding="utf-8"
        )
    if rj_path.exists() or routers_meta:
        rj_path.write_text(json.dumps(routers_meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"Leaderboard rows: {len(leaderboard)} | updated/inserted: {len(updated)} | "
        f"derived regenerated: {len(regenerated)}"
    )
    if missing:
        print(f"WARNING: not found in README: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
