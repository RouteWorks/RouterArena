# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom router for RouterArena -- v0.8.0.

Self-contained hybrid semantic router for paraphrase-invariant routing.
RouterArena's evaluation environment only needs this file and the config
files in router_inference/config/; the full chuzom-router PyPI package is
NOT required.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. This router does
  not inspect dataset names, test-set indices, global_index values, or
  optimality metadata.

v0.8.0 changes vs v0.7.0:
  - Replaced heuristic regex routing with a two-signal hybrid:
      Signal A -- TF-IDF + LogisticRegression classifier (60k features,
                  1-3 grams) trained on 8400 routing decisions.
      Signal B -- BAAI/bge-small-en-v1.5 semantic centroid lookup.
  - Combined score: 0.6 * tfidf_prob + 0.4 * centroid_sim.
  - Paraphrase-invariant: lexical overlap (TF-IDF) + semantic embedding
    (centroids) together handle both wording and meaning changes.
  - One-time startup: loads sklearn model (~5ms) + BGE-small (~500ms).
  - Per-query cost: TF-IDF transform (<1ms) + embed forward pass (~30ms).

Architecture:
  1. Load chuzom-classifier.joblib (TfidfVectorizer + LogisticRegression).
  2. Load BAAI/bge-small-en-v1.5 (33.4M params, 384-dim).
  3. Load chuzom-centroids.npz (5 L2-normalised centroid vectors).
  4. For each query:
     a. Extract normalised TF-IDF features -> LR probability vector.
     b. Embed prompt -> cosine similarity to each centroid.
     c. Weighted blend (0.6 * tfidf + 0.4 * centroid) -> best model.

Reference:
  RouterArena  : github.com/RouteWorks/RouterArena
  Chuzom v0.8.0: github.com/ypollak2/chuzom
  Arena formula: S = ((1+beta)*acc*C) / (beta*acc + C), beta=0.1
"""

from __future__ import annotations

import os
import re

import numpy as np

from router_inference.router.base_router import BaseRouter


# Weight of TF-IDF signal in the hybrid score (centroid gets 1 - TFIDF_WEIGHT)
_TFIDF_WEIGHT = 0.6

# Ordered list matching centroid rows in chuzom-centroids.npz
_ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

_MCQ_HEADER_RE = re.compile(
    r"Please read the following multiple-choice questions.*?(?=Context:)",
    re.DOTALL,
)


def _extract_text(prompt: str) -> str:
    prompt = _MCQ_HEADER_RE.sub("", prompt)
    return " ".join(prompt.split())[:2000]


class ChuzomRouter(BaseRouter):
    """v0.8.0 hybrid semantic router (TF-IDF + centroid).

    Uses a TF-IDF + LR classifier and BGE-small semantic centroids in
    combination for paraphrase-invariant routing. Class-level singletons
    ensure models load once per process.
    """

    # Class-level singletons -- loaded once, reused across all instances/calls
    _tfidf_vec = None
    _lr_clf = None
    _lr_le = None
    _tokenizer = None
    _embed_model = None
    _centroids: np.ndarray | None = None
    _centroid_models: list[str] | None = None
    _embed_model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
        self._ensure_loaded()

    @classmethod
    def _project_root(cls) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(script_dir))

    @classmethod
    def _config_path(cls, filename: str) -> str:
        return os.path.join(cls._project_root(), "router_inference", "config", filename)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._tfidf_vec is None:
            cls._load_classifier()
        if cls._centroids is None:
            cls._load_centroids()
        if cls._tokenizer is None:
            cls._load_embedder()

    @classmethod
    def _load_classifier(cls) -> None:
        import joblib  # type: ignore[import]

        path = cls._config_path("chuzom-classifier.joblib")
        bundle = joblib.load(path)
        cls._tfidf_vec = bundle["vectorizer"]
        cls._lr_clf = bundle["classifier"]
        cls._lr_le = bundle["label_encoder"]

    @classmethod
    def _load_centroids(cls) -> None:
        path = cls._config_path("chuzom-centroids.npz")
        data = np.load(path)
        cls._centroids = data["centroids"].astype(np.float32)
        cls._centroid_models = [str(m) for m in data["models"]]

    @classmethod
    def _load_embedder(cls) -> None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        cls._tokenizer = AutoTokenizer.from_pretrained(cls._embed_model_name)
        m = AutoModel.from_pretrained(cls._embed_model_name)
        m.train(False)
        cls._embed_model = m

    def _embed(self, text: str) -> np.ndarray:
        import torch  # type: ignore[import]

        encoded = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._embed_model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return emb.squeeze(0).numpy().astype(np.float32)

    def _get_prediction(self, query: str) -> str:
        text = _extract_text(query)

        # Signal A: TF-IDF + LR probability over ROUTING_MODELS
        tfidf_feat = self._tfidf_vec.transform([text])
        lr_proba = self._lr_clf.predict_proba(tfidf_feat)[0]
        # lr_le.classes_ are in the same order as LR classes
        tfidf_scores: dict[str, float] = {
            self._lr_le.classes_[i]: float(lr_proba[i])
            for i in range(len(self._lr_le.classes_))
        }
        # Ensure all routing models have a score (zero for absent)
        for m in _ROUTING_MODELS:
            tfidf_scores.setdefault(m, 0.0)

        # Signal B: cosine similarity to per-model centroids
        embedding = self._embed(text)
        raw_sims = self._centroids @ embedding  # (n_models,)
        centroid_scores: dict[str, float] = {
            self._centroid_models[i]: float(raw_sims[i])
            for i in range(len(self._centroid_models))
        }

        # Normalise centroid scores to [0, 1] for fair weighting
        sim_vals = list(centroid_scores.values())
        sim_min, sim_max = min(sim_vals), max(sim_vals)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        centroid_norm: dict[str, float] = {
            m: (s - sim_min) / sim_range for m, s in centroid_scores.items()
        }

        # Blended score: models must be available in this RouterArena config
        best_model = None
        best_score = -1.0
        for m in self.models:
            tfidf = tfidf_scores.get(m, 0.0)
            centroid = centroid_norm.get(m, 0.0)
            score = _TFIDF_WEIGHT * tfidf + (1.0 - _TFIDF_WEIGHT) * centroid
            if score > best_score:
                best_score = score
                best_model = m

        return best_model if best_model else self.models[0]
