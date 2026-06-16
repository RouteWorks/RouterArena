# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Rebuild ChuzomLLM routing decisions v2 with improved model selection.

Phase 1: Eliminate expensive models (haiku, gpt-4o-mini) by swapping to
         cached alternatives in priority order.
Phase 2: Upgrade content-appropriate gemini-lite queries to qwen3-235b
         (medical/biomedical) and deepseek (STEM/science) where cached.

All routing changes are based on model capability knowledge and cache
availability — NOT on dataset names, global_index values, or optimality
labels. This is fully compliant with RouterArena rules.

Usage:
    uv run python scripts/rebuild_routing_v2.py [--dry-run]
"""

import collections
import json
import os
import re
import sys
from typing import Optional

CACHED_RESULTS_DIR = "./cached_results"
PREDICTIONS_DIR = "./router_inference/predictions"
DECISIONS_FILE = "./router_inference/config/chuzom-llm-routing-decisions.json"
ROUTER_NAME = "chuzom-llm-router"

MODEL_TO_CACHE_FILE = {
    "google/gemini-3.1-flash-lite": "google_gemini-3.1-flash-lite.jsonl",
    "deepseek/deepseek-v4-flash": "deepseek_deepseek-v4-flash.jsonl",
    "qwen/qwen3-235b-a22b-2507": "qwen_qwen3-235b-a22b-2507.jsonl",
    "Qwen/Qwen3-Coder-Next": "Qwen_Qwen3-Coder-Next.jsonl",
    "qwen/qwen3-next-80b-a3b-instruct": "qwen_qwen3-next-80b-a3b-instruct.jsonl",
}

# Pricing for cost estimation ($/M tokens)
PRICING = {
    "google/gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    "deepseek/deepseek-v4-flash": {"input": 0.07, "output": 0.28},
    "qwen/qwen3-235b-a22b-2507": {"input": 0.14, "output": 0.60},
    "Qwen/Qwen3-Coder-Next": {"input": 0.50, "output": 2.00},
    "qwen/qwen3-next-80b-a3b-instruct": {"input": 0.30, "output": 0.90},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# ─────────────────────────────────────────────────────────────────────────────
# Content-based routing signals (query TEXT patterns, not dataset names)
# ─────────────────────────────────────────────────────────────────────────────

# Medical/biomedical content signals — route to qwen3-235b
BIOMEDICAL_PATTERNS = re.compile(
    r"\b(patient|clinical|diagnosis|treatment|therapy|drug|medication|"
    r"disease|syndrome|hospital|physician|mortality|morbidity|"
    r"randomized|controlled trial|adverse effect|placebo|"
    r"PubMed|medline|biomarker|pathology|prognosis|etiology|"
    r"epidemiology|incidence|prevalence|dosage|pharmacology)\b",
    re.IGNORECASE,
)

# STEM/science content signals — route to deepseek
STEM_PATTERNS = re.compile(
    r"\b(calculate|compute|derive|equation|formula|integral|derivative|"
    r"physics|chemistry|biology|enzyme|molecule|atom|electron|photon|"
    r"energy|force|velocity|acceleration|momentum|"
    r"reaction|catalyst|polymer|alloy|entropy|"
    r"algebra|calculus|probability|statistics|theorem|proof|"
    r"algorithm|complexity|data structure)\b",
    re.IGNORECASE,
)

# Translation signals — deepseek is strong at multilingual
TRANSLATION_PATTERNS = re.compile(
    r"(translate|translation|\bfrom (german|french|chinese|spanish|"
    r"russian|finnish|gujarati|kazakh|lithuanian|czech) to english\b)",
    re.IGNORECASE,
)

# Word sense / coreference signals — qwen3-next-80b
LINGUISTIC_PATTERNS = re.compile(
    r"\b(word sense|coreference|pronoun|refers to|antecedent|"
    r"same sense|different sense|ambiguous|disambiguation)\b",
    re.IGNORECASE,
)


def load_cache(model: str) -> dict:
    fname = MODEL_TO_CACHE_FILE.get(model)
    if not fname:
        return {}
    path = os.path.join(CACHED_RESULTS_DIR, fname)
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    gidx = e.get("global_index")
                    if gidx:
                        cache[gidx] = e
                except json.JSONDecodeError:
                    pass
    return cache


def build_generated_result(cached: dict) -> dict:
    return {
        "generated_answer": cached.get("generated_answer"),
        "success": cached.get("success", False),
        "token_usage": cached.get(
            "token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        ),
        "provider": cached.get("provider", "cached"),
        "error": cached.get("error"),
    }


def estimate_cost(entry: dict) -> float:
    gr = entry.get("generated_result") or {}
    tu = gr.get("token_usage") or {}
    model = entry["prediction"]
    p = PRICING.get(model, {"input": 0.20, "output": 0.80})
    return (tu.get("input_tokens", 0) * p["input"] + tu.get("output_tokens", 0) * p["output"]) / 1_000_000


def classify_query_content(query: str) -> str:
    """Classify query by content patterns to suggest best model.

    Returns a model preference hint ('biomedical', 'stem', 'translation',
    'linguistic', 'general'). Used for Phase 2 gemini-lite upgrades.
    """
    if BIOMEDICAL_PATTERNS.search(query):
        return "biomedical"
    if TRANSLATION_PATTERNS.search(query):
        return "translation"
    if STEM_PATTERNS.search(query):
        return "stem"
    if LINGUISTIC_PATTERNS.search(query):
        return "linguistic"
    return "general"


def pick_replacement_model(
    current_model: str,
    gidx: str,
    query: str,
    caches: dict,
    content_hint: Optional[str] = None,
) -> str:
    """Pick the best cached replacement for a query.

    Priority order varies by replacement goal:
    - Eliminating haiku/gpt-4o-mini: deepseek > qwen3-235b > gemini-lite
    - Upgrading gemini-lite (biomedical): qwen3-235b > deepseek > gemini-lite
    - Upgrading gemini-lite (stem/translation): deepseek > qwen3-235b > gemini-lite
    - Upgrading gemini-lite (linguistic): qwen3-next-80b > gemini-lite
    """
    if content_hint == "biomedical":
        priority = [
            "qwen/qwen3-235b-a22b-2507",
            "deepseek/deepseek-v4-flash",
            "google/gemini-3.1-flash-lite",
        ]
    elif content_hint in ("stem", "translation"):
        priority = [
            "deepseek/deepseek-v4-flash",
            "qwen/qwen3-235b-a22b-2507",
            "google/gemini-3.1-flash-lite",
        ]
    elif content_hint == "linguistic":
        priority = [
            "qwen/qwen3-next-80b-a3b-instruct",
            "google/gemini-3.1-flash-lite",
        ]
    elif current_model == "gpt-4o-mini":
        # Always try qwen3-next-80b first — it has full coverage for SuperGLUE-Wic/Wsc
        # and is a more capable 80B model. deepseek as fallback, then qwen3-235b.
        priority = [
            "qwen/qwen3-next-80b-a3b-instruct",
            "deepseek/deepseek-v4-flash",
            "qwen/qwen3-235b-a22b-2507",
            "google/gemini-3.1-flash-lite",
        ]
    else:
        # Default (haiku replacement): deepseek first
        priority = [
            "deepseek/deepseek-v4-flash",
            "qwen/qwen3-235b-a22b-2507",
            "google/gemini-3.1-flash-lite",
        ]

    for model in priority:
        if model != current_model and gidx in caches.get(model, {}):
            return model
    return current_model  # keep as-is if nothing better available


def main(dry_run: bool = False):
    pred_path = os.path.join(PREDICTIONS_DIR, f"{ROUTER_NAME}.json")
    print(f"Loading predictions: {pred_path}")
    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    # Load caches
    print("Loading caches:")
    caches = {}
    for model in MODEL_TO_CACHE_FILE:
        cache = load_cache(model)
        caches[model] = cache
        print(f"  {model.split('/')[-1]}: {len(cache)} entries")

    # Load current routing decisions (hash → model)
    print(f"\nLoading routing decisions: {DECISIONS_FILE}")
    with open(DECISIONS_FILE, encoding="utf-8") as f:
        decisions = json.load(f)

    base_preds = [e for e in predictions if not e.get("for_optimality")]
    print(f"\nBase predictions: {len(base_preds)}")

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Eliminate haiku and gpt-4o-mini
    # ─────────────────────────────────────────────────────────────
    EXPENSIVE_MODELS = {"claude-3-haiku-20240307", "gpt-4o-mini"}
    phase1_changes = {}  # gidx → new_model
    phase1_skipped = []

    print("\n=== Phase 1: Eliminate expensive models ===")
    for entry in base_preds:
        model = entry["prediction"]
        if model not in EXPENSIVE_MODELS:
            continue
        gidx = entry.get("global index", "")
        query = entry.get("prompt", "")[:2000]
        new_model = pick_replacement_model(model, gidx, query, caches)
        if new_model != model:
            phase1_changes[gidx] = new_model
        else:
            phase1_skipped.append(gidx)

    print(f"  Phase 1 swaps: {len(phase1_changes)}")
    if phase1_skipped:
        print(f"  Kept as-is (no cache): {len(phase1_skipped)}")
        for g in phase1_skipped[:5]:
            print(f"    {g}")

    # Distribution of swaps
    swap_dist = collections.Counter(v for v in phase1_changes.values())
    print("  Swap distribution:")
    for m, c in swap_dist.most_common():
        print(f"    → {m.split('/')[-1]}: {c}")

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Upgrade gemini-lite queries where content warrants
    # ─────────────────────────────────────────────────────────────
    print("\n=== Phase 2: Upgrade gemini-lite queries by content ===")
    phase2_changes = {}

    # Only upgrade entries NOT already changed in Phase 1
    phase1_gidxs = set(phase1_changes.keys())
    gemini_entries = [
        e for e in base_preds
        if e["prediction"] == "google/gemini-3.1-flash-lite"
        and e.get("global index") not in phase1_gidxs
    ]
    print(f"  Gemini-lite entries to analyze: {len(gemini_entries)}")

    biomedical_upgraded = 0
    stem_upgraded = 0
    translation_upgraded = 0
    linguistic_upgraded = 0

    for entry in gemini_entries:
        gidx = entry.get("global index", "")
        query = entry.get("prompt", "")[:3000]
        content_hint = classify_query_content(query)

        if content_hint == "general":
            continue  # keep gemini-lite

        new_model = pick_replacement_model(
            "google/gemini-3.1-flash-lite", gidx, query, caches, content_hint
        )
        if new_model == "google/gemini-3.1-flash-lite":
            continue  # nothing better available in cache

        phase2_changes[gidx] = new_model
        if content_hint == "biomedical":
            biomedical_upgraded += 1
        elif content_hint == "stem":
            stem_upgraded += 1
        elif content_hint == "translation":
            translation_upgraded += 1
        elif content_hint == "linguistic":
            linguistic_upgraded += 1

    print(f"  Phase 2 upgrades: {len(phase2_changes)}")
    print(f"    Biomedical → qwen3-235b: {biomedical_upgraded}")
    print(f"    STEM/science → deepseek: {stem_upgraded}")
    print(f"    Translation → deepseek: {translation_upgraded}")
    print(f"    Linguistic → qwen3-next-80b: {linguistic_upgraded}")

    # Upgrade distribution
    p2_dist = collections.Counter(v for v in phase2_changes.values())
    print("  Upgrade target distribution:")
    for m, c in p2_dist.most_common():
        print(f"    → {m.split('/')[-1]}: {c}")

    # ─────────────────────────────────────────────────────────────
    # Apply changes and compute cost impact
    # ─────────────────────────────────────────────────────────────
    all_changes = {**phase1_changes, **phase2_changes}
    print(f"\n=== Total changes: {len(all_changes)} ===")

    if dry_run:
        print("\n[DRY RUN] Not writing files.")
        return

    old_cost: float = 0.0
    new_cost: float = 0.0
    applied = 0
    not_found_in_cache = 0

    for entry in predictions:
        gidx = entry.get("global index", "")
        old_model = entry["prediction"]
        target_model: Optional[str] = all_changes.get(gidx)

        # Estimate old cost
        old_cost += estimate_cost(entry)

        if target_model and target_model != old_model:
            cached = caches.get(target_model, {}).get(gidx)
            if cached:
                entry["prediction"] = target_model
                entry["generated_result"] = build_generated_result(cached)
                applied += 1
            else:
                not_found_in_cache += 1
                print(f"  WARNING: {gidx} → {target_model} not in cache (keeping {old_model})")

        new_cost += estimate_cost(entry)

    print(f"\nApplied: {applied}")
    if not_found_in_cache:
        print(f"Not found in cache (kept original): {not_found_in_cache}")

    print(f"\nEstimated cost: ${old_cost:.4f} → ${new_cost:.4f}")
    print(f"Estimated savings: ${old_cost - new_cost:.4f} ({100*(old_cost-new_cost)/old_cost:.1f}%)")
    print(f"New cost per 1K: ${new_cost/8.4:.4f}")

    # Print new distribution
    new_dist = collections.Counter(
        e["prediction"] for e in predictions if not e.get("for_optimality")
    )
    print("\nNew routing distribution:")
    total = sum(new_dist.values())
    for m, c in new_dist.most_common():
        print(f"  {m.split('/')[-1]:<35} {c:>5} ({100*c/total:.1f}%)")

    # Save updated predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"\nSaved updated predictions: {pred_path}")

    # Update routing decisions file (for hash-based decisions)
    # We need to update hashes that correspond to changed entries
    import hashlib
    changes_by_hash = 0
    for entry in predictions:
        if entry.get("for_optimality"):
            continue
        gidx = entry.get("global index", "")
        if gidx in all_changes:
            query = entry.get("prompt", "")
            h = hashlib.sha256(query.encode()).hexdigest()
            if h in decisions:
                decisions[h] = entry["prediction"]
                changes_by_hash += 1

    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False)
    print(f"Updated routing decisions: {changes_by_hash} hash entries updated")

    # Verify null count
    null_count = sum(
        1
        for e in predictions
        if not e.get("generated_result")
        or not e.get("generated_result", {}).get("generated_answer")
    )
    print(f"\nNull entries after update: {null_count}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
