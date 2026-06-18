# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Regenerate chuzom-llm-routing-decisions.json with the clean v0.9.0 router.

v0.9.0 dropped TF-IDF (trained on RouterArena data).
This script re-runs ChuzomRouter (centroid + heuristic only) on all 8400
prompts and writes a fresh sha256 → model_name lookup table.

Usage:
    uv run python scripts/regenerate_llm_routing_decisions.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from router_inference.router.chuzom_router import ChuzomRouter

    print("Loading ChuzomRouter v0.9.0 (centroid + heuristic, no TF-IDF)...")
    router = ChuzomRouter("chuzom-llm-router")
    print("Router loaded.\n")

    # Load all routing entries from the predictions file
    pred_path = Path("router_inference/predictions/chuzom-llm-router.json")
    with open(pred_path, encoding="utf-8") as f:
        data = json.load(f)

    routing_entries = [e for e in data if not e.get("for_optimality")]
    print(f"Routing entries: {len(routing_entries)}")

    if args.dry_run:
        sample = routing_entries[:5]
        for e in sample:
            h = hashlib.sha256(e["prompt"].encode()).hexdigest()
            model = router._get_prediction(e["prompt"])
            print(f"  hash={h[:12]}... → {model}")
        print(f"\n[dry-run] Would write {len(routing_entries)} decisions.")
        return

    decisions: dict[str, str] = {}
    for i, e in enumerate(routing_entries, 1):
        prompt = e["prompt"]
        h = hashlib.sha256(prompt.encode()).hexdigest()
        model = router._get_prediction(prompt)
        decisions[h] = model
        if i % 500 == 0 or i == len(routing_entries):
            print(f"  {i}/{len(routing_entries)} done")

    out_path = Path("router_inference/config/chuzom-llm-routing-decisions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=None)
    print(f"\nWrote {len(decisions)} decisions → {out_path}")


if __name__ == "__main__":
    main()
