#!/usr/bin/env python3
"""Phase 2, step 3: train the per-model capability predictor.

Embeds each corpus prompt once (local MiniLM) and fits one logistic-regression
head per pool model mapping embedding -> P(correct). Reports cross-validated AUC
per model, then saves a single artifact the router loads at decision time.

Output: phase2/data/predictor.pkl
Run:    uv run python phase2/train_predictor.py
Free:   local only; no API calls.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS = "phase2/data/corpus.jsonl"
LABELS = "phase2/data/labels.jsonl"
OUT = "phase2/data/predictor.pkl"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    corpus = {r["id"]: r for r in (json.loads(x) for x in open(CORPUS))}
    labels = [json.loads(x) for x in open(LABELS)]

    # per-model: id -> correct
    by_model = {}
    for r in labels:
        by_model.setdefault(r["model"], {})[r["id"]] = int(r["correct"])
    print(f"[train] models with labels: {list(by_model)}")

    ids = list(corpus)
    embedder = SentenceTransformer(EMBEDDER)
    X_all = embedder.encode(
        [corpus[i]["prompt"] for i in ids],
        normalize_embeddings=True, batch_size=64, show_progress_bar=False,
    )
    idx = {i: k for k, i in enumerate(ids)}

    heads = {}
    for m, lab in by_model.items():
        mids = [i for i in ids if i in lab]
        if len(mids) < 50:
            print(f"[train] {m}: only {len(mids)} labels, skipping")
            continue
        X = np.array([X_all[idx[i]] for i in mids])
        y = np.array([lab[i] for i in mids])
        base = y.mean()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        try:
            auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
        except Exception:
            auc = float("nan")
        clf.fit(X, y)
        heads[m] = clf
        print(f"[train] {m:34} n={len(mids)} base_acc={base:.3f} cv_auc={auc:.3f}")

    with open(OUT, "wb") as f:
        pickle.dump({"embedder": EMBEDDER, "heads": heads}, f)
    print(f"[train] saved {len(heads)} heads -> {OUT}")


if __name__ == "__main__":
    main()
