# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""
Train TF-IDF + LogisticRegression classifier from routing decisions and save.

The classifier is trained on the 8400 routing prompts with the v4 ensemble
decisions as pseudo-labels. Used by chuzom_router.py as the fallback for
unseen paraphrased prompts.

Usage:
    uv run python scripts/save_classifier.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

DECISIONS_PATH = Path("router_inference/config/chuzom-llm-routing-decisions.json")
PREDICTIONS_PATH = Path("router_inference/predictions/chuzom-llm-router.json")
OUTPUT_PATH = Path("router_inference/config/chuzom-classifier.joblib")

ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]


def extract_prompt_text(prompt: str) -> str:
    prompt = re.sub(
        r"Please read the following multiple-choice questions.*?(?=Context:)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    return " ".join(prompt.split())[:2000]


def main() -> None:
    print("Loading routing decisions...", file=sys.stderr)
    with open(DECISIONS_PATH) as f:
        decisions = json.load(f)

    print("Loading predictions...", file=sys.stderr)
    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)

    routing_entries = []
    for entry in predictions:
        if entry.get("for_optimality"):
            continue
        prompt = entry.get("prompt", "")
        h = hashlib.sha256(prompt.encode()).hexdigest()
        model = decisions.get(h)
        if model and model in ROUTING_MODELS:
            routing_entries.append({"prompt": prompt, "model": model})

    print(f"Training on {len(routing_entries)} entries...", file=sys.stderr)

    texts = [extract_prompt_text(e["prompt"]) for e in routing_entries]
    labels = [e["model"] for e in routing_entries]

    le = LabelEncoder()
    le.fit(ROUTING_MODELS)
    y = le.transform(labels)

    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=60_000,
        sublinear_tf=True,
        min_df=2,
    )
    X = vec.fit_transform(texts)

    print("Training LR classifier...", file=sys.stderr)
    clf = LogisticRegression(
        max_iter=2000,
        C=2.0,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Quick self-eval
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    print(f"Train accuracy: {acc:.4f}", file=sys.stderr)

    dist = {}
    for i, m in enumerate(le.classes_):
        dist[m] = int(np.sum(y == i))
    print("Label distribution:", file=sys.stderr)
    for m, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {m}: {n}", file=sys.stderr)

    print(f"Saving to {OUTPUT_PATH}...", file=sys.stderr)
    joblib.dump(
        {"vectorizer": vec, "classifier": clf, "label_encoder": le},
        OUTPUT_PATH,
        compress=3,
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
