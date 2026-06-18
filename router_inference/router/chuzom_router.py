# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom router for RouterArena -- v0.9.0.

Self-contained hybrid semantic router for paraphrase-invariant routing.
RouterArena's evaluation environment only needs this file and the config
files in router_inference/config/; the full chuzom-router PyPI package is
NOT required.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. This router does
  not inspect dataset names, test-set indices, global_index values, or
  optimality metadata. NO component is trained or fit on RouterArena data.

v0.9.0 changes vs v0.8.0:
  - Removed TF-IDF + LogisticRegression (was trained on RouterArena
    prompts — violates arena rules).
  - Now uses a two-signal blend:
      Signal A -- BAAI/bge-small-en-v1.5 semantic centroid lookup.
                  Centroids built from external public datasets only:
                  SQuAD, MMLU, GSM8K, WMT16, SuperGLUE WiC.
      Signal B -- Regex heuristics: hand-crafted domain patterns that
                  fire on structural features of the prompt content.
  - Combined score: 0.7 * centroid_norm + 0.3 * heuristic_norm.
  - chuzom-classifier.joblib is no longer loaded or required.

Architecture:
  1. Load BAAI/bge-small-en-v1.5 (33.4M params, 384-dim).
  2. Load chuzom-centroids.npz (5 L2-normalised centroid vectors, external data).
  3. For each query:
     a. Embed prompt -> cosine similarity to each centroid (signal A).
     b. Apply regex heuristics -> score per model (signal B).
     c. Weighted blend (0.7 * centroid + 0.3 * heuristic) -> best model.

Reference:
  RouterArena  : github.com/RouteWorks/RouterArena
  Chuzom v0.9.0: github.com/ypollak2/chuzom
  Arena formula: S = ((1+beta)*acc*C) / (beta*acc + C), beta=0.1
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np

from router_inference.router.base_router import BaseRouter

# Weight of centroid signal (heuristic gets 1 - CENTROID_WEIGHT)
_CENTROID_WEIGHT = 0.7

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

_HEURISTIC_RULES: list[tuple[re.Pattern, dict[str, float]]] = [
    (
        re.compile(r"Context:\s*None", re.IGNORECASE),
        {"google/gemini-2.0-flash-001": 3.0, "google/gemini-3.1-flash-lite": 1.0},
    ),
    (
        re.compile(
            r"Context:\s*(?!None|N/A|\bNone\b).{20,}", re.IGNORECASE | re.DOTALL
        ),
        {"google/gemini-3.1-flash-lite": 4.0, "google/gemini-2.0-flash-001": 1.0},
    ),
    (
        re.compile(
            r"(?i)(translat|spanish|french|chinese|german|japanese|arabic|russian)"
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(
            r"(?i)(calcul|integral|deriv|equation|mathemat|algebra|geometry"
            r"|trigonometr|probability|statistic|combinatoric|number theory)"
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(
            r"(?i)(code|program|function|algorithm|debug|implement|python\b|java\b|sql\b)"
        ),
        {"deepseek/deepseek-v4-flash": 4.0, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    (
        re.compile(
            r"(?i)(word.?sense|coreference|disambigu|homograph|polysemy|pronoun.*refer)"
        ),
        {"qwen/qwen3-next-80b-a3b-instruct": 5.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(r"(?i)(medical|clinical|diagnosis|pharmacol|biochem|anatomy)"),
        {"qwen/qwen3-235b-a22b-2507": 3.0, "deepseek/deepseek-v4-flash": 1.5},
    ),
    (
        re.compile(
            r"(?i)(olympiad|AIME|AMC|competition math|prove that|lemma|theorem)"
        ),
        {"qwen/qwen3-235b-a22b-2507": 4.0, "deepseek/deepseek-v4-flash": 2.0},
    ),
]


def _extract_text(prompt: str) -> str:
    prompt = _MCQ_HEADER_RE.sub("", prompt)
    return " ".join(prompt.split())[:2000]


def _heuristic_scores(prompt: str) -> dict[str, float]:
    raw: defaultdict[str, float] = defaultdict(float)
    for pattern, weights in _HEURISTIC_RULES:
        if pattern.search(prompt):
            for m, w in weights.items():
                raw[m] += w
    for m in _ROUTING_MODELS:
        raw.setdefault(m, 0.0)
    total = sum(raw.values()) or 1.0
    return {m: raw[m] / total for m in _ROUTING_MODELS}


class ChuzomRouter(BaseRouter):
    """v0.9.0 semantic + heuristic router (centroid + regex, no TF-IDF).

    Uses BGE-small semantic centroids (trained on external data only) and
    hand-crafted heuristic rules for paraphrase-invariant routing. Class-level
    singletons ensure models load once per process.
    """

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
        if cls._centroids is None:
            cls._load_centroids()
        if cls._tokenizer is None:
            cls._load_embedder()

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

        assert self._tokenizer is not None
        assert self._embed_model is not None
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

        assert self._centroids is not None
        assert self._centroid_models is not None

        # Signal A: cosine similarity to per-model centroids
        embedding = self._embed(text)
        raw_sims = self._centroids @ embedding
        centroid_scores: dict[str, float] = {
            self._centroid_models[i]: float(raw_sims[i])
            for i in range(len(self._centroid_models))
        }

        # Normalise centroid scores to [0, 1]
        sim_vals = list(centroid_scores.values())
        sim_min, sim_max = min(sim_vals), max(sim_vals)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        centroid_norm: dict[str, float] = {
            m: (s - sim_min) / sim_range for m, s in centroid_scores.items()
        }

        # Signal B: heuristic regex scores (normalised)
        heuristic = _heuristic_scores(query)

        # Blended score
        best_model = None
        best_score = -1.0
        for m in self.models:
            score = _CENTROID_WEIGHT * centroid_norm.get(m, 0.0) + (
                1.0 - _CENTROID_WEIGHT
            ) * heuristic.get(m, 0.0)
            if score > best_score:
                best_score = score
                best_model = m

        return best_model if best_model else self.models[0]
