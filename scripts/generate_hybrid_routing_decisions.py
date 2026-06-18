# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""
Generate new routing decisions using the hybrid router (TF-IDF + external centroids).

This script routes all 8400 test prompts through ChuzomRouter._get_prediction()
directly. The resulting decisions file replaces the v4 ensemble decisions.

Key benefit for robustness:
  - Old flow: original -> hash hit (LLM decision) ; paraphrase -> hash miss -> fallback
  - New flow: original -> hash hit (hybrid decision) ; paraphrase -> hash miss -> SAME hybrid
  - Since both use the same hybrid router, paraphrase-invariance holds end-to-end.

Usage:
    uv run python scripts/generate_hybrid_routing_decisions.py
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PREDICTIONS_FILE = Path("router_inference/predictions/chuzom-llm-router.json")
OUTPUT_FILE = Path("router_inference/config/chuzom-llm-routing-decisions.json")


def main() -> None:
    print("Loading predictions...", file=sys.stderr)
    with open(PREDICTIONS_FILE) as f:
        predictions = json.load(f)

    routing_entries = [p for p in predictions if not p.get("for_optimality")]
    print(f"Routing entries: {len(routing_entries)}", file=sys.stderr)

    # Bootstrap the hybrid router (loads BGE-small + TF-IDF + centroids)
    print("Loading hybrid router...", file=sys.stderr)
    from router_inference.router.chuzom_router import ChuzomRouter

    router = ChuzomRouter("chuzom-llm-router")
    print("Router ready.", file=sys.stderr)

    decisions = {}
    model_counts: dict[str, int] = defaultdict(int)

    print("Classifying prompts...", file=sys.stderr)
    for i, entry in enumerate(routing_entries):
        prompt = entry["prompt"]
        h = hashlib.sha256(prompt.encode()).hexdigest()
        model = router._get_prediction(prompt)
        decisions[h] = model
        model_counts[model] += 1

        if i % 500 == 0:
            print(f"  {i}/{len(routing_entries)}", file=sys.stderr)

    print("\nDistribution:", file=sys.stderr)
    total = len(decisions)
    for m, n in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(
            f"  {m.split('/')[-1]:<40} {n:>5} ({100 * n / total:.1f}%)", file=sys.stderr
        )

    print(f"\nSaving {len(decisions)} decisions to {OUTPUT_FILE}...", file=sys.stderr)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(decisions, f, indent=2)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
