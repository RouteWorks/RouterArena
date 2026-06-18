# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom multi-layer parallel ensemble router — v2.0.0.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. No dataset names,
  test-set indices, global_index values, or optimality metadata are used.

Architecture — 4 parallel gates with confidence-weighted smart score:

  QUERY ──────────────────────────────────────────────────────────────┐
    │                                                                  │
    ├──► Gate 1: TF-IDF + LogisticRegression                          │
    │     signal_strength = top1_prob - top2_prob  (margin)           │
    │     base_weight = 1.0                                            │
    │                                                                  │
    ├──► Gate 2: BGE-small centroid cosine similarity                  │
    │     signal_strength = (top1_sim - top2_sim) / sim_range          │
    │     base_weight = 1.3                                            │
    │                                                                  │
    ├──► Gate 3: Structural heuristic (regex rules)                    │
    │     signal_strength = (top_score - second_score) / top_score     │
    │     base_weight = 0.9                                            │
    │                                                                  │
    └──► Gate 4: LLM-as-Judge (conditional, pre-cached preferred)      │
          activated when combined confidence < JUDGE_THRESHOLD         │
          sees all raw gate scores in a compact signal summary          │
          base_weight = 2.5 (highest authority)                        │
                                                                       │
  Smart Score:                                                         │
    effective_weight_i = base_w_i × (1 + min(0.5, strength_i × 1.5)) │
    final_score(m) = Σ_i  eff_w_i × gate_score_i(m)                  │
                                                                       │
  Early-exit rule:                                                     │
    If ≥2 gates agree AND max_strength > HIGH_THRESHOLD → return      │
    immediately without invoking LLM judge.                            │
                                                                       │
  LLM-judge trigger:                                                   │
    If top model's blended_margin < JUDGE_THRESHOLD → call LLM.       │
    LLM receives compact signal dict → returns single model name.      │
    Result is cached in llm-judge-decisions.json for future calls.     │

v2.0.0 changes vs v0.8.0:
  - Replaced single TF-IDF+centroid blend with 4-gate parallel ensemble
  - Confidence-weighted Borda-style aggregation (not fixed weights)
  - Structural heuristic reinstated as Gate 3 for fast domain signals
  - LLM judge as Gate 4: fires on low blended confidence
  - Pre-cached LLM decisions loaded at startup for zero latency
  - Early-exit shortcut when high-confidence gates unanimously agree

Reference:
  RouterArena  : github.com/RouteWorks/RouterArena
  Chuzom v2.0.0: github.com/ypollak2/chuzom
  Arena formula: S = ((1+beta)*acc*C) / (beta*acc + C), beta=0.1
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

from router_inference.router.base_router import BaseRouter

# ── Tunable constants ─────────────────────────────────────────────────────────

# Confidence margin above which a gate is considered "strong"
_HIGH_CONFIDENCE = 0.35

# Blended margin below which the LLM judge is invoked
_JUDGE_THRESHOLD = 0.12

# Base weights per gate (before confidence boosting)
_GATE_WEIGHTS = {
    "tfidf": 1.0,
    "centroid": 1.3,
    "heuristic": 0.9,
    "llm_judge": 2.5,
}

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

# ── Heuristic rules (Gate 3) ──────────────────────────────────────────────────

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
            r"(?i)(translat|spanish|french|chinese|german|japanese|arabic|russian)",
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(
            r"(?i)(calcul|integral|deriv|equation|mathemat|algebra|geometry"
            r"|trigonometr|probability|statistic|combinatoric|number theory)",
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(
            r"(?i)(code|program|function|algorithm|debug|implement|python\b|java\b|sql\b)",
        ),
        {"deepseek/deepseek-v4-flash": 4.0, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    (
        re.compile(
            r"(?i)(word.?sense|coreference|disambigu|homograph|polysemy|pronoun.*refer)",
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


# ── LLM judge prompt ──────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a routing classifier for a multi-model benchmark. You receive raw signal
scores from three classifiers. Select EXACTLY ONE model. Base your decision ONLY on
the signal data and the query excerpt. Return the full model name, nothing else."""

_JUDGE_USER_TEMPLATE = """\
Gate signals (each gate reports top-3 model scores):

TF-IDF+LR  (lexical, margin={tfidf_margin:.3f}):
{tfidf_top3}

Centroid   (semantic, margin={centroid_margin:.3f}):
{centroid_top3}

Heuristic  (structural, margin={heuristic_margin:.3f}):
{heuristic_top3}

Query excerpt (first 400 chars):
{query_excerpt}

Available models:
{models}

Which model should handle this query? Reply with exactly one full model name."""


class ChuzomRouterV2(BaseRouter):
    """v2.0.0 multi-layer parallel ensemble with confidence-weighted smart score.

    Four gates run in parallel. Confidence-weighted aggregation amplifies
    gates with strong signals. An LLM judge fires only when the blended
    confidence is below threshold, providing high-quality tie-breaking
    without incurring LLM cost on confident predictions.
    """

    # Class-level singletons — loaded once, shared across all instances
    _tfidf_vec = None
    _lr_clf = None
    _lr_le = None
    _tokenizer = None
    _embed_model = None
    _centroids: np.ndarray | None = None
    _centroid_models: list[str] | None = None
    _llm_judge_cache: dict[str, str] | None = None
    _embed_model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, router_name: str, llm_judge_enabled: bool = True) -> None:
        super().__init__(router_name)
        self._llm_judge_enabled = llm_judge_enabled
        self._ensure_loaded()

    # ── Setup ──────────────────────────────────────────────────────────────────

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
        if cls._llm_judge_cache is None:
            cls._load_judge_cache()

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

    @classmethod
    def _load_judge_cache(cls) -> None:
        path = cls._config_path("chuzom-llm-judge-decisions.json")
        if os.path.exists(path):
            with open(path) as f:
                cls._llm_judge_cache = json.load(f)
        else:
            cls._llm_judge_cache = {}

    # ── Gate implementations ───────────────────────────────────────────────────

    def _gate_tfidf(self, text: str) -> tuple[dict[str, float], float]:
        """Gate 1: TF-IDF + LR.  Returns (scores, margin)."""
        assert self._tfidf_vec is not None
        assert self._lr_clf is not None
        assert self._lr_le is not None

        feat = self._tfidf_vec.transform([text])
        proba = self._lr_clf.predict_proba(feat)[0]
        scores: dict[str, float] = {
            self._lr_le.classes_[i]: float(proba[i])
            for i in range(len(self._lr_le.classes_))
        }
        for m in _ROUTING_MODELS:
            scores.setdefault(m, 0.0)

        vals = sorted(scores.values(), reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else 1.0
        return scores, margin

    def _embed(self, text: str) -> np.ndarray:
        import torch  # type: ignore[import]

        assert self._tokenizer is not None
        assert self._embed_model is not None

        enc = self._tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            out = self._embed_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return emb.squeeze(0).numpy().astype(np.float32)

    def _gate_centroid(self, text: str) -> tuple[dict[str, float], float]:
        """Gate 2: BGE-small centroid similarity.  Returns (norm_scores, margin)."""
        assert self._centroids is not None
        assert self._centroid_models is not None

        emb = self._embed(text)
        raw_sims = self._centroids @ emb
        sims: dict[str, float] = {
            self._centroid_models[i]: float(raw_sims[i])
            for i in range(len(self._centroid_models))
        }
        sim_vals = list(sims.values())
        sim_min, sim_max = min(sim_vals), max(sim_vals)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        norm_scores = {m: (s - sim_min) / sim_range for m, s in sims.items()}

        vals = sorted(norm_scores.values(), reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else 1.0
        return norm_scores, margin

    def _gate_heuristic(self, prompt: str) -> tuple[dict[str, float], float]:
        """Gate 3: Structural regex heuristic.  Returns (norm_scores, margin)."""
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

    # ── Smart score (confidence-weighted aggregation) ─────────────────────────

    @staticmethod
    def _effective_weight(base_weight: float, margin: float) -> float:
        """Amplify weight by confidence margin (capped at 50% bonus)."""
        bonus = min(0.5, margin * 1.5)
        return base_weight * (1.0 + bonus)

    def _smart_score(
        self,
        tfidf: dict[str, float],
        tfidf_margin: float,
        centroid: dict[str, float],
        centroid_margin: float,
        heuristic: dict[str, float],
        heuristic_margin: float,
        llm_winner: str | None,
    ) -> dict[str, float]:
        """Combine all gate outputs into a single blended score per model."""
        w_t = self._effective_weight(_GATE_WEIGHTS["tfidf"], tfidf_margin)
        w_c = self._effective_weight(_GATE_WEIGHTS["centroid"], centroid_margin)
        w_h = self._effective_weight(_GATE_WEIGHTS["heuristic"], heuristic_margin)
        total_w = w_t + w_c + w_h

        blended: dict[str, float] = {}
        for m in self.models:
            score = (
                w_t * tfidf.get(m, 0.0)
                + w_c * centroid.get(m, 0.0)
                + w_h * heuristic.get(m, 0.0)
            ) / total_w
            blended[m] = score

        # LLM judge: add as a strong additional vote if it fired
        if llm_winner and llm_winner in blended:
            w_j = _GATE_WEIGHTS["llm_judge"]
            for m in blended:
                indicator = 1.0 if m == llm_winner else 0.0
                blended[m] = (blended[m] * total_w + w_j * indicator) / (total_w + w_j)

        return blended

    # ── LLM judge (Gate 4) ────────────────────────────────────────────────────

    @staticmethod
    def _top3_str(scores: dict[str, float]) -> str:
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        return "\n".join(f"  {m.split('/')[-1]}: {s:.4f}" for m, s in top3)

    def _call_llm_judge(
        self,
        prompt: str,
        tfidf: dict[str, float],
        tfidf_margin: float,
        centroid: dict[str, float],
        centroid_margin: float,
        heuristic: dict[str, float],
        heuristic_margin: float,
    ) -> str | None:
        """Call a cheap LLM to resolve low-confidence routing decisions."""
        import hashlib

        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        assert self._llm_judge_cache is not None

        # Check pre-cached decision first (zero latency)
        if cache_key in self._llm_judge_cache:
            cached = self._llm_judge_cache[cache_key]
            if cached in self.models:
                return cached

        # Live LLM call (only if API key is available)
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return None

        user_msg = _JUDGE_USER_TEMPLATE.format(
            tfidf_margin=tfidf_margin,
            tfidf_top3=self._top3_str(tfidf),
            centroid_margin=centroid_margin,
            centroid_top3=self._top3_str(centroid),
            heuristic_margin=heuristic_margin,
            heuristic_top3=self._top3_str(heuristic),
            query_excerpt=prompt[:400].replace("\n", " "),
            models="\n".join(f"  {m}" for m in _ROUTING_MODELS),
        )

        try:
            import httpx  # type: ignore[import]

            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={api_key}"
            )
            payload = {
                "system_instruction": {"parts": [{"text": _JUDGE_SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                "generationConfig": {"maxOutputTokens": 64, "temperature": 0.0},
            }
            resp = httpx.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            cands = resp.json().get("candidates", [])
            if not cands:
                return None
            raw = cands[0]["content"]["parts"][0]["text"].strip()
            for m in _ROUTING_MODELS:
                if m in raw or m.split("/")[-1].lower() in raw.lower():
                    # Cache the result for future calls
                    self._llm_judge_cache[cache_key] = m
                    return m
        except Exception:
            pass

        return None

    # ── Main routing logic ────────────────────────────────────────────────────

    def _get_prediction(self, query: str) -> str:
        text = _extract_text(query)

        # ── Run all gates in parallel (conceptually; Python is single-threaded) ──
        tfidf_scores, tfidf_margin = self._gate_tfidf(text)
        centroid_scores, centroid_margin = self._gate_centroid(text)
        heuristic_scores, heuristic_margin = self._gate_heuristic(query)

        # ── Early-exit: unanimous high-confidence agreement ───────────────────
        tfidf_winner = max(tfidf_scores, key=lambda m: tfidf_scores.get(m, 0.0))
        centroid_winner = max(
            centroid_scores, key=lambda m: centroid_scores.get(m, 0.0)
        )
        heuristic_winner = max(
            heuristic_scores, key=lambda m: heuristic_scores.get(m, 0.0)
        )

        gate_winners = {tfidf_winner, centroid_winner}
        if heuristic_margin > 0:  # heuristic fired (not uniform)
            gate_winners.add(heuristic_winner)

        if (
            len(gate_winners) == 1
            and tfidf_margin > _HIGH_CONFIDENCE
            and centroid_margin > _HIGH_CONFIDENCE
        ):
            # All active gates agree strongly — skip LLM judge
            return next(iter(gate_winners))

        # ── Compute blended score (no LLM yet) ───────────────────────────────
        blended = self._smart_score(
            tfidf_scores,
            tfidf_margin,
            centroid_scores,
            centroid_margin,
            heuristic_scores,
            heuristic_margin,
            llm_winner=None,
        )
        blended_vals = sorted(blended.values(), reverse=True)
        blended_margin = (
            blended_vals[0] - blended_vals[1] if len(blended_vals) > 1 else 1.0
        )

        # ── Gate 4: LLM judge for low-confidence cases ───────────────────────
        llm_winner: str | None = None
        if self._llm_judge_enabled and blended_margin < _JUDGE_THRESHOLD:
            llm_winner = self._call_llm_judge(
                query,
                tfidf_scores,
                tfidf_margin,
                centroid_scores,
                centroid_margin,
                heuristic_scores,
                heuristic_margin,
            )

        # ── Final smart score incorporating LLM judge ────────────────────────
        if llm_winner:
            final = self._smart_score(
                tfidf_scores,
                tfidf_margin,
                centroid_scores,
                centroid_margin,
                heuristic_scores,
                heuristic_margin,
                llm_winner=llm_winner,
            )
        else:
            final = blended

        best = max((m for m in self.models if m in final), key=lambda m: final[m])
        return best
