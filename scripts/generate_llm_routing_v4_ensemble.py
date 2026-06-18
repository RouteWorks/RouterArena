# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""
RouterArena v4: 3-Layer Parallel Ensemble with Weighted Borda Voting.

Layer 1 — Semantic Centroid (BGE-small embeddings): paraphrase-invariant,
           anchors robustness. Prompts cluster by task type; paraphrases
           land within ~0.05 cosine distance of each other.
Layer 2 — Structural Heuristic: Context: None / passage / math / code /
           word-sense signals extracted from prompt text deterministically.
Layer 3 — TF-IDF + Logistic Regression: fast classifier trained on the
           existing routing decisions as pseudo-labels.

All three layers run in parallel, each producing a ranked list of models.
Weighted Borda Count aggregates votes; Layer 1 gets highest base weight
(1.3) for its robustness contribution.

Output: router_inference/config/chuzom-llm-routing-decisions.json
"""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ── Configuration ─────────────────────────────────────────────────────────────

PREDICTIONS_FILE = Path("router_inference/predictions/chuzom-llm-router.json")
OUTPUT_FILE = Path("router_inference/config/chuzom-llm-routing-decisions.json")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

# Borda base weights: semantic centroid gets highest weight (robustness anchor)
LAYER_WEIGHTS = {
    "semantic": 1.3,
    "heuristic": 0.9,
    "classifier": 1.0,
}

# Borda points for rank position (5 models → 4,3,2,1,0)
BORDA_POINTS = list(range(len(ROUTING_MODELS) - 1, -1, -1))


# ── Data Loading ───────────────────────────────────────────────────────────────


def load_data():
    preds = json.load(open(PREDICTIONS_FILE))
    routing = [p for p in preds if not p.get("for_optimality")]
    return routing


def extract_prompt_text(prompt: str) -> str:
    """Normalise prompt for embedding — keep semantic content, drop boilerplate."""
    # Remove the standard MCQ instruction header (same across all prompts)
    prompt = re.sub(
        r"Please read the following multiple-choice questions.*?(?=Context:)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    # Collapse whitespace
    return " ".join(prompt.split())[:2000]


# ── Layer 1: Semantic Centroid ─────────────────────────────────────────────────


def build_centroids(routing_entries, model, embeddings):
    """Compute per-model centroid from existing routing pseudo-labels."""

    centroids = {}
    for m in ROUTING_MODELS:
        indices = [i for i, e in enumerate(routing_entries) if e["prediction"] == m]
        if not indices:
            # Fallback: use mean of all embeddings (shouldn't happen)
            centroids[m] = embeddings.mean(axis=0)
        else:
            centroids[m] = embeddings[indices].mean(axis=0)
    # L2-normalise centroids
    centroid_matrix = np.array([centroids[m] for m in ROUTING_MODELS])
    centroid_matrix = centroid_matrix / (
        np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9
    )
    return {m: centroid_matrix[i] for i, m in enumerate(ROUTING_MODELS)}


def semantic_rank(embedding: np.ndarray, centroids: dict) -> tuple[list[str], float]:
    """Rank models by cosine similarity to centroids. Returns [model, ...] best→worst."""
    sims = {m: float(np.dot(embedding, centroids[m])) for m in ROUTING_MODELS}
    confidence_top = sorted(sims.values(), reverse=True)
    margin = confidence_top[0] - confidence_top[1] if len(confidence_top) > 1 else 0.0
    ranking = sorted(ROUTING_MODELS, key=lambda m: sims[m], reverse=True)
    return ranking, margin


# ── Layer 2: Structural Heuristic ─────────────────────────────────────────────

# Keyword signals mapped to model preference scores
_HEURISTIC_RULES = [
    # (regex_pattern, {model: score})
    (
        r"Context:\s*None",
        {
            "google/gemini-2.0-flash-001": 3,
            "deepseek/deepseek-v4-flash": 2,
            "qwen/qwen3-235b-a22b-2507": 1,
        },
    ),
    (
        r"Context:\s*(?!None).{20,}",
        {
            "google/gemini-3.1-flash-lite": 4,
            "google/gemini-2.0-flash-001": 1,
        },
    ),
    (
        r"(?i)(translate|translation|spanish|french|chinese|german|japanese|arabic)",
        {
            "deepseek/deepseek-v4-flash": 3,
            "qwen/qwen3-235b-a22b-2507": 2,
        },
    ),
    (
        r"(?i)(calcul|integral|deriv|equation|mathemat|algebra|geometry|trigonometr|probability|statistics)",
        {
            "deepseek/deepseek-v4-flash": 3,
            "qwen/qwen3-235b-a22b-2507": 2,
        },
    ),
    (
        r"(?i)(code|program|function|algorithm|debug|implement|software|python|java\b)",
        {
            "deepseek/deepseek-v4-flash": 4,
            "qwen/qwen3-next-80b-a3b-instruct": 2,
        },
    ),
    (
        r"(?i)(word sense|coreference|disambigu|homograph|polysemy|synonym)",
        {
            "qwen/qwen3-next-80b-a3b-instruct": 5,
            "qwen/qwen3-235b-a22b-2507": 2,
        },
    ),
    (
        r"(?i)(medical|clinical|diagnosis|pharmacol|biochem|anatomy|physiology)",
        {
            "qwen/qwen3-235b-a22b-2507": 3,
            "deepseek/deepseek-v4-flash": 2,
        },
    ),
    (
        r"(?i)(legal|law|jurisdiction|statute|contract|constitution|court)",
        {
            "qwen/qwen3-235b-a22b-2507": 3,
            "google/gemini-2.0-flash-001": 2,
        },
    ),
    (
        r"(?i)(read(ing)? comprehension|passage|paragraph|according to the (text|passage|article))",
        {
            "google/gemini-3.1-flash-lite": 4,
            "google/gemini-2.0-flash-001": 1,
        },
    ),
]

_compiled_rules = [(re.compile(pat), scores) for pat, scores in _HEURISTIC_RULES]


def heuristic_rank(prompt: str) -> tuple[list[str], float]:
    """Score models via structural heuristics. Returns (ranking, margin)."""
    scores: defaultdict[str, float] = defaultdict(float)
    for pattern, model_scores in _compiled_rules:
        if pattern.search(prompt):
            for model, score in model_scores.items():
                scores[model] += score
    # Ensure all models have a score
    for m in ROUTING_MODELS:
        if m not in scores:
            scores[m] = 0.0
    vals = sorted(scores.values(), reverse=True)
    margin = (vals[0] - vals[1]) / max(vals[0], 1) if vals[0] > 0 else 0.0
    ranking = sorted(ROUTING_MODELS, key=lambda m: scores[m], reverse=True)
    return ranking, margin


# ── Layer 3: TF-IDF + Logistic Regression ─────────────────────────────────────


def train_classifier(routing_entries, texts):
    labels = [e["prediction"] for e in routing_entries]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=50_000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return vec, clf, le


def classifier_rank(prompt_tfidf, clf, le) -> tuple:
    proba = clf.predict_proba(prompt_tfidf)[0]
    model_probs = {le.classes_[i]: proba[i] for i in range(len(le.classes_))}
    # Map back to full model names — le.classes_ are indices into ROUTING_MODELS
    vals = sorted(model_probs.values(), reverse=True)
    margin = (vals[0] - vals[1]) if len(vals) > 1 else 0.0
    ranking = sorted(model_probs.keys(), key=lambda m: model_probs[m], reverse=True)
    # Ensure all ROUTING_MODELS are in ranking
    for m in ROUTING_MODELS:
        if m not in ranking:
            ranking.append(m)
    return ranking, margin


# ── Weighted Borda Count ───────────────────────────────────────────────────────


def weighted_borda(votes: list) -> str:
    """
    votes: list of (layer_name, ranking, margin)
    Returns the winning model.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for layer_name, ranking, margin in votes:
        base_weight = LAYER_WEIGHTS[layer_name]
        # Boost weight by confidence margin (capped at 0.5 bonus)
        certainty_bonus = min(0.5, margin * 1.5)
        weight = base_weight * (1.0 + certainty_bonus)
        for rank_pos, model in enumerate(ranking):
            borda_pts = BORDA_POINTS[rank_pos] if rank_pos < len(BORDA_POINTS) else 0
            scores[model] += weight * borda_pts
    # Tie-break: prefer semantic layer's top choice
    semantic_top = next(v[1][0] for v in votes if v[0] == "semantic")
    return max(
        ROUTING_MODELS,
        key=lambda m: (scores[m], m == semantic_top),
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print("Loading data...", file=sys.stderr)
    routing_entries = load_data()
    texts = [extract_prompt_text(e["prompt"]) for e in routing_entries]
    prompts_raw = [e["prompt"] for e in routing_entries]

    # ── Layer 1 setup: embed all prompts ──────────────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL}", file=sys.stderr)
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(texts)} prompts...", file=sys.stderr)
    embeddings = embed_model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print("Building centroids...", file=sys.stderr)
    centroids = build_centroids(routing_entries, embed_model, embeddings)

    # ── Layer 3 setup: TF-IDF classifier ─────────────────────────────────────
    print("Training TF-IDF classifier...", file=sys.stderr)
    vec, clf, le = train_classifier(routing_entries, texts)
    tfidf_matrix = vec.transform(texts)

    # ── Run all 3 layers + Borda vote for each prompt ─────────────────────────
    print("Voting on routing decisions...", file=sys.stderr)
    decisions = {}

    model_counts: defaultdict[str, int] = defaultdict(int)
    for i, entry in enumerate(routing_entries):
        # Key must match rebuild_predictions_from_routing.py: sha256(raw prompt)
        prompt_key = hashlib.sha256(entry["prompt"].encode()).hexdigest()

        # Layer 1: semantic centroid
        sem_rank, sem_margin = semantic_rank(embeddings[i], centroids)

        # Layer 2: heuristic
        heur_rank, heur_margin = heuristic_rank(prompts_raw[i])

        # Layer 3: TF-IDF classifier
        clf_rank, clf_margin = classifier_rank(tfidf_matrix[i], clf, le)

        # Borda vote
        votes = [
            ("semantic", sem_rank, sem_margin),
            ("heuristic", heur_rank, heur_margin),
            ("classifier", clf_rank, clf_margin),
        ]
        winner = weighted_borda(votes)
        decisions[prompt_key] = winner
        model_counts[winner] += 1

        if i % 500 == 0:
            print(f"  {i}/{len(routing_entries)}", file=sys.stderr)

    # ── Report distribution ───────────────────────────────────────────────────
    print("\nRouting distribution:", file=sys.stderr)
    total = len(decisions)
    for model in ROUTING_MODELS:
        n = model_counts[model]
        print(f"  {model}: {n} ({n / total * 100:.1f}%)", file=sys.stderr)

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_FILE}...", file=sys.stderr)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(decisions, f, indent=2)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
