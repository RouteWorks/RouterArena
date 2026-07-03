#!/usr/bin/env python3
"""Fast batch prediction generator for chuzom-v3.

Uses batched SentenceTransformer encoding to process all 8400 prompts in
~60 seconds instead of ~109 minutes (single-prompt encoding path).

Usage:
    source .env
    .venv/bin/python3 scripts/generate_chuzom_v3_predictions.py [--split full|sub_10|robustness]

Compliance: reads ONLY from router_inference/config/chuzom-v3.json and
chuzom-v3-centroids.npz. No quarantined artifacts are touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Import clean router constants (patterns + weights) ────────────────────────
from router_inference.router.chuzom_v3_router import (  # noqa: E402
    _HEURISTIC_RULES,
    _MCQ_HEADER_RE,
    _HIGH_CONFIDENCE,
    _GATE_WEIGHTS,
    _ROUTING_MODELS,
)

DATASET_PATHS = {
    "sub_10": str(ROOT / "dataset/router_data_10.json"),
    "full": str(ROOT / "dataset/router_data.json"),
    "robustness": str(ROOT / "dataset/router_robustness.json"),
}

CENTROID_FILE = str(ROOT / "router_inference/config/chuzom-v3-centroids.npz")
OUTPUT_DIR = ROOT / "router_inference/predictions"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 128


def _extract_text(prompt: str) -> str:
    prompt = _MCQ_HEADER_RE.sub("", prompt)
    return " ".join(prompt.split())[:2000]


def _gate_heuristic(prompt: str) -> tuple[dict[str, float], float]:
    raw: defaultdict[str, float] = defaultdict(float)
    for pattern, weights in _HEURISTIC_RULES:
        if pattern.search(prompt):
            for m, w in weights.items():
                raw[m] += w
    for m in _ROUTING_MODELS:
        raw.setdefault(m, 0.0)
    total = sum(raw.values()) or 1.0
    norm_scores = {m: raw[m] / total for m in _ROUTING_MODELS}
    vals = sorted(norm_scores.values(), reverse=True)
    margin = (vals[0] - vals[1]) / max(vals[0], 1e-6) if vals[0] > 0 else 0.0
    return norm_scores, margin


def _effective_weight(base: float, margin: float) -> float:
    return base * (1.0 + min(0.5, margin * 1.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="full", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM judge (faster, lower quality on ambiguous prompts)")
    args = parser.parse_args()

    t_start = time.time()
    print(f"Chuzom v3 batch prediction generator — split={args.split}")
    print("=" * 70)

    # Load dataset
    print(f"\n[1] Loading {args.split} dataset...")
    with open(DATASET_PATHS[args.split]) as f:
        dataset = json.load(f)
    print(f"    {len(dataset)} entries")

    prompts = [e.get("prompt_formatted") or e.get("prompt", "") for e in dataset]
    global_indices = [e.get("global index", "") for e in dataset]

    # Batch embedding
    print(f"\n[2] Loading embedding model ({EMBED_MODEL})...")
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    embed_model = SentenceTransformer(EMBED_MODEL)

    print(f"    Encoding {len(prompts)} prompts (batch_size={BATCH_SIZE})...")
    texts = [_extract_text(p) for p in prompts]
    t_emb = time.time()
    embeddings = embed_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"    Encoded in {time.time() - t_emb:.1f}s")

    # Load centroids
    print(f"\n[3] Loading centroids ({CENTROID_FILE})...")
    data = np.load(CENTROID_FILE)
    centroids = data["centroids"].astype(np.float32)
    centroid_models = [str(m) for m in data["models"]]
    print(f"    {centroids.shape[0]} centroids × {centroids.shape[1]}d")

    # Batch centroid similarities
    print("\n[4] Computing centroid similarities (batch)...")
    sims_matrix = embeddings @ centroids.T  # (N, M)
    print(f"    Done: {sims_matrix.shape}")

    # Route each prompt
    print("\n[5] Routing prompts...")
    predictions_list = []
    low_conf_count = 0

    for i, (gi, prompt) in enumerate(zip(global_indices, prompts)):
        raw_sims = sims_matrix[i]
        sim_dict: dict[str, float] = {centroid_models[j]: float(raw_sims[j])
                                       for j in range(len(centroid_models))}
        sim_vals = list(sim_dict.values())
        sim_min, sim_max = min(sim_vals), max(sim_vals)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        c_scores = {m: (s - sim_min) / sim_range for m, s in sim_dict.items()}
        c_vals = sorted(c_scores.values(), reverse=True)
        c_margin = c_vals[0] - c_vals[1] if len(c_vals) > 1 else 1.0

        h_scores, h_margin = _gate_heuristic(prompt)

        # Strong heuristic pre-filter (threshold 8.0)
        raw_h: defaultdict[str, float] = defaultdict(float)
        for pattern, weights in _HEURISTIC_RULES:
            if pattern.search(prompt):
                for m, w in weights.items():
                    raw_h[m] += w
        if raw_h:
            top_h = max(raw_h, key=lambda m: raw_h[m])
            if raw_h[top_h] >= 8.0 and top_h in _ROUTING_MODELS:
                predictions_list.append({
                    "global index": gi,
                    "prompt": prompt,
                    "prediction": top_h,
                    "generated_result": None,
                    "cost": None,
                    "accuracy": None,
                    "for_optimality": False,
                })
                continue

        c_winner = max((m for m in _ROUTING_MODELS if m in c_scores), key=lambda m: c_scores.get(m, 0.0))
        h_winner = max((m for m in _ROUTING_MODELS if m in h_scores), key=lambda m: h_scores.get(m, 0.0))

        # Early exit: both gates agree with high confidence
        if c_winner == h_winner and c_margin > _HIGH_CONFIDENCE and h_margin > _HIGH_CONFIDENCE:
            predictions_list.append({
                "global index": gi,
                "prompt": prompt,
                "prediction": c_winner,
                "generated_result": None,
                "cost": None,
                "accuracy": None,
                "for_optimality": False,
            })
            continue

        # Blend
        w_c = _effective_weight(_GATE_WEIGHTS["centroid"], c_margin)
        w_h = _effective_weight(_GATE_WEIGHTS["heuristic"], h_margin)
        total_w = w_c + w_h
        blended = {
            m: (w_c * c_scores.get(m, 0.0) + w_h * h_scores.get(m, 0.0)) / total_w
            for m in _ROUTING_MODELS
        }
        b_vals = sorted(blended.values(), reverse=True)
        b_margin = b_vals[0] - b_vals[1] if len(b_vals) > 1 else 1.0

        if b_margin < _GATE_WEIGHTS["centroid"] * 0.1:
            low_conf_count += 1

        best = max((m for m in _ROUTING_MODELS if m in blended), key=lambda m: blended[m])
        predictions_list.append({
            "global index": gi,
            "prompt": prompt,
            "prediction": best,
            "generated_result": None,
            "cost": None,
            "accuracy": None,
            "for_optimality": False,
        })

    print(f"    {len(predictions_list)} predictions generated")
    print(f"    Low confidence (judge would fire): {low_conf_count} ({low_conf_count/len(predictions_list)*100:.1f}%)")

    # Distribution
    dist: defaultdict[str, int] = defaultdict(int)
    for p in predictions_list:
        dist[p["prediction"].split("/")[-1]] += 1
    print("\n    Model distribution:")
    for m, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"      {n:5d} ({n/len(predictions_list)*100:5.1f}%)  {m}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = "-robustness" if args.split == "robustness" else ""
    out_path = OUTPUT_DIR / f"chuzom-v3{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(predictions_list, f, ensure_ascii=False, indent=2)
    print(f"\n[6] Saved {len(predictions_list)} predictions → {out_path}")
    print(f"    Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
