# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom multi-layer parallel ensemble router — v2.1.0.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. No dataset names,
  test-set indices, global_index values, or optimality metadata are used.
  NO component is trained or fit on RouterArena data.

Architecture — 3 parallel gates with confidence-weighted smart score:

  QUERY ──────────────────────────────────────────────────────────────┐
    │                                                                  │
    ├──► Gate 1: BGE-small centroid cosine similarity                  │
    │     signal_strength = (top1_sim - top2_sim) / sim_range          │
    │     base_weight = 1.3                                            │
    │     trained on: SQuAD, MMLU, GSM8K, WMT16, SuperGLUE WiC        │
    │                                                                  │
    ├──► Gate 2: Structural heuristic (regex rules)                    │
    │     signal_strength = (top_score - second_score) / top_score     │
    │     base_weight = 0.9                                            │
    │                                                                  │
    └──► Gate 3: LLM-as-Judge (conditional, pre-cached preferred)      │
          activated when combined confidence < JUDGE_THRESHOLD         │
          sees centroid + heuristic raw scores in signal summary       │
          base_weight = 2.5 (highest authority)                        │
                                                                       │
  Smart Score:                                                         │
    effective_weight_i = base_w_i × (1 + min(0.5, strength_i × 1.5)) │
    final_score(m) = Σ_i  eff_w_i × gate_score_i(m)                  │
                                                                       │
  Early-exit rule:                                                     │
    If centroid and heuristic agree AND both margins > HIGH_THRESHOLD  │
    → return immediately without invoking LLM judge.                   │
                                                                       │
  LLM-judge trigger:                                                   │
    If top model's blended_margin < JUDGE_THRESHOLD → call LLM.       │
    LLM receives compact signal dict → returns single model name.      │
    Result is cached in llm-judge-decisions.json for future calls.     │

v2.1.0 changes vs v2.0.0:
  - Removed Gate 1 (TF-IDF + LogisticRegression).  That classifier was
    trained on RouterArena prompts, which violates the arena rules.
  - Gates renumbered: centroid=1, heuristic=2, LLM judge=3.
  - chuzom-classifier.joblib is no longer loaded or used.
  - All routing decisions based solely on BGE-small centroids (external
    data only: SQuAD, MMLU, GSM8K, WMT16, SuperGLUE WiC) and hand-
    crafted regex heuristics that are prompt-content-only.

Reference:
  RouterArena  : github.com/RouteWorks/RouterArena
  Chuzom v2.1.0: github.com/ypollak2/chuzom
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
_JUDGE_THRESHOLD = 0.25

# Base weights per gate (before confidence boosting)
_GATE_WEIGHTS = {
    "centroid": 1.3,
    "heuristic": 0.9,
    "llm_judge": 2.5,
}

# Ordered list matching centroid rows in chuzom-centroids.npz
_ROUTING_MODELS = [
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

_MCQ_HEADER_RE = re.compile(
    r"Please read the following multiple-choice questions.*?(?=Context:)",
    re.DOTALL,
)

# ── Heuristic rules (Gate 2) ──────────────────────────────────────────────────

_HEURISTIC_RULES: list[tuple[re.Pattern, dict[str, float]]] = [
    # ── Context-based structural signals ─────────────────────────────────────
    (
        re.compile(r"Context:\s*None", re.IGNORECASE),
        {"google/gemini-3.1-flash-lite": 3.0},
    ),
    (
        re.compile(
            r"Context:\s*(?!None|N/A|\bNone\b).{20,}", re.IGNORECASE | re.DOTALL
        ),
        {"google/gemini-3.1-flash-lite": 4.0},
    ),
    # ── Code: explicit code blocks (strongest signal, highest weight) ─────────
    (
        re.compile(
            r"```\s*(python|java|javascript|typescript|c\+\+|cpp|c#|go|rust|ruby"
            r"|kotlin|swift|bash|shell|sql|r\b|scala|php)",
            re.IGNORECASE,
        ),
        {"deepseek/deepseek-v4-flash": 7.0, "qwen/qwen3-next-80b-a3b-instruct": 2.0},
    ),
    # ── Code: function/class definitions and common programming keywords ──────
    (
        re.compile(
            r"(?m)^\s*(def |class |public\s+static|void\s+\w+\s*\(|#include\s*<"
            r"|import\s+\w+|from\s+\w+\s+import)"
        ),
        {"deepseek/deepseek-v4-flash": 6.0, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    # ── Code: general programming keywords (boosted from 4.0 → 5.5) ──────────
    (
        re.compile(
            r"(?i)(code|program|function|algorithm|debug|implement|python\b|java\b"
            r"|sql\b|runtime|compile|syntax\s+error|stack\s+overflow|big.?O)"
        ),
        {"deepseek/deepseek-v4-flash": 5.5, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    # ── Math: competition / proof-level (highest qwen3-235b signal) ──────────
    (
        re.compile(
            r"(?i)(olympiad|AIME|AMC|competition math|prove that|lemma|theorem"
            r"|corollary|conjecture|induction|modular arithmetic|\bIMO\b|\bUSAMO\b)"
        ),
        {"qwen/qwen3-235b-a22b-2507": 5.5, "deepseek/deepseek-v4-flash": 2.0},
    ),
    # ── Math: general computation (boosted from 3.0 → 5.0) ───────────────────
    (
        re.compile(
            r"(?i)(calcul|integral|deriv|equation|mathemat|algebra|geometry"
            r"|trigonometr|probability|statistic|combinatoric|number theory"
            r"|arithmetic|how many|solve for|find the value)"
        ),
        {"deepseek/deepseek-v4-flash": 5.0, "qwen/qwen3-235b-a22b-2507": 3.0},
    ),
    # ── Math: arithmetic word problems (GSM8K style) ─────────────────────────
    (
        re.compile(
            r"(?i)(if\s+\w+\s+has\s+\d+|how\s+many\s+\w+\s+(are|were|will|does)"
            r"|\d+\s*(times|plus|minus|divided|percent|dollars|hours|days|km|kg))"
        ),
        {"deepseek/deepseek-v4-flash": 4.0, "qwen/qwen3-235b-a22b-2507": 2.5},
    ),
    # ── Translation / multilingual ────────────────────────────────────────────
    (
        re.compile(
            r"(?i)(translat|spanish|french|chinese|german|japanese|arabic|russian"
            r"|korean|portuguese|italian|hindi|turkish|dutch|polish)",
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    # ── NLI / entailment (SuperGLUE-Entailment) ───────────────────────────────
    (
        re.compile(
            r"(?i)(entail|contradict|neutral\b|hypothesis|premise|nli\b"
            r"|natural language inference|does.*follow\s+from|can we conclude)"
        ),
        {"qwen/qwen3-next-80b-a3b-instruct": 5.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    # ── Word sense / coreference ──────────────────────────────────────────────
    (
        re.compile(
            r"(?i)(word.?sense|coreference|disambigu|homograph|polysemy|pronoun.*refer)",
        ),
        {"qwen/qwen3-next-80b-a3b-instruct": 5.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    # ── Medical / scientific STEM ─────────────────────────────────────────────
    (
        re.compile(
            r"(?i)(medical|clinical|diagnosis|pharmacol|biochem|anatomy"
            r"|physic[si]|chemistry|molecular|quantum|thermodynam|electr(on|ic))"
        ),
        {"qwen/qwen3-235b-a22b-2507": 4.0, "deepseek/deepseek-v4-flash": 1.5},
    ),
]


def _extract_text(prompt: str) -> str:
    prompt = _MCQ_HEADER_RE.sub("", prompt)
    return " ".join(prompt.split())[:2000]


# ── LLM judge prompt ──────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a routing classifier for a multi-model benchmark. You receive raw signal
scores from two classifiers. Select EXACTLY ONE model. Base your decision ONLY on
the signal data and the query excerpt. Return the full model name, nothing else."""

_JUDGE_USER_TEMPLATE = """\
Gate signals (each gate reports top-3 model scores):

Centroid   (semantic similarity to task clusters, margin={centroid_margin:.3f}):
{centroid_top3}

Heuristic  (structural text patterns, margin={heuristic_margin:.3f}):
{heuristic_top3}

Query excerpt (first 400 chars):
{query_excerpt}

Available models:
{models}

Which model should handle this query? Reply with exactly one full model name."""


class ChuzomRouterV2(BaseRouter):
    """v2.1.0 multi-layer parallel ensemble with confidence-weighted smart score.

    Three gates run in parallel. Confidence-weighted aggregation amplifies
    gates with strong signals. An LLM judge fires only when the blended
    confidence is below threshold, providing high-quality tie-breaking
    without incurring LLM cost on confident predictions.

    All training data is external (not RouterArena):
      - BGE-small centroids: SQuAD, MMLU, GSM8K, WMT16, SuperGLUE WiC
      - Heuristic rules: hand-crafted regex, no corpus required
    """

    # Class-level singletons — loaded once, shared across all instances
    _tokenizer = None
    _embed_model = None
    _centroids: np.ndarray | None = None
    _centroid_models: list[str] | None = None
    _llm_judge_cache: dict[str, str] | None = None
    _embed_model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, router_name: str, llm_judge_enabled: bool = True) -> None:
        super().__init__(router_name)
        # Restrict routing pool to models that are reliably available on OpenRouter
        self.models = [m for m in self.models if m in _ROUTING_MODELS]
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
        if cls._centroids is None:
            cls._load_centroids()
        if cls._tokenizer is None:
            cls._load_embedder()
        if cls._llm_judge_cache is None:
            cls._load_judge_cache()

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
        """Gate 1: BGE-small centroid similarity.  Returns (norm_scores, margin)."""
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
        """Gate 2: Structural regex heuristic.  Returns (norm_scores, margin)."""
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
        centroid: dict[str, float],
        centroid_margin: float,
        heuristic: dict[str, float],
        heuristic_margin: float,
        llm_winner: str | None,
    ) -> dict[str, float]:
        """Combine gate outputs into a single blended score per model."""
        w_c = self._effective_weight(_GATE_WEIGHTS["centroid"], centroid_margin)
        w_h = self._effective_weight(_GATE_WEIGHTS["heuristic"], heuristic_margin)
        total_w = w_c + w_h

        blended: dict[str, float] = {}
        for m in self.models:
            score = (w_c * centroid.get(m, 0.0) + w_h * heuristic.get(m, 0.0)) / total_w
            blended[m] = score

        # LLM judge: add as a strong additional vote if it fired
        if llm_winner and llm_winner in blended:
            w_j = _GATE_WEIGHTS["llm_judge"]
            for m in blended:
                indicator = 1.0 if m == llm_winner else 0.0
                blended[m] = (blended[m] * total_w + w_j * indicator) / (total_w + w_j)

        return blended

    # ── LLM judge (Gate 3) ────────────────────────────────────────────────────

    @staticmethod
    def _top3_str(scores: dict[str, float]) -> str:
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        return "\n".join(f"  {m.split('/')[-1]}: {s:.4f}" for m, s in top3)

    def _call_llm_judge(
        self,
        prompt: str,
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

        user_msg = _JUDGE_USER_TEMPLATE.format(
            centroid_margin=centroid_margin,
            centroid_top3=self._top3_str(centroid),
            heuristic_margin=heuristic_margin,
            heuristic_top3=self._top3_str(heuristic),
            query_excerpt=prompt[:400].replace("\n", " "),
            models="\n".join(f"  {m}" for m in _ROUTING_MODELS),
        )

        import httpx  # type: ignore[import]

        full_prompt = f"{_JUDGE_SYSTEM}\n\n{user_msg}"

        def _parse_model(raw: str) -> str | None:
            for m in _ROUTING_MODELS:
                if m in raw or m.split("/")[-1].lower() in raw.lower():
                    return m
            return None

        # Try Ollama first (local, free, zero latency after warmup)
        ollama_model = os.environ.get("CHUZOM_JUDGE_OLLAMA_MODEL", "qwen3.6:27b")
        try:
            resp = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 64},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                raw = resp.json()["message"]["content"].strip()
                result = _parse_model(raw)
                if result:
                    self._llm_judge_cache[cache_key] = result
                    return result
        except Exception:
            pass

        # Fallback: Gemini API
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return None
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={api_key}"
            )
            payload = {
                "system_instruction": {"parts": [{"text": _JUDGE_SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                "generationConfig": {"maxOutputTokens": 64, "temperature": 0.0},
            }
            resp = httpx.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            cands = resp.json().get("candidates", [])
            if cands:
                raw = cands[0]["content"]["parts"][0]["text"].strip()
                result = _parse_model(raw)
                if result:
                    self._llm_judge_cache[cache_key] = result
                    return result
        except Exception:
            pass

        return None

    # ── Main routing logic ────────────────────────────────────────────────────

    def _compute_blended_margin(self, query: str) -> float:
        """Return the blended margin (0–1) without invoking the LLM judge.

        Used by pre-generation scripts to identify which prompts need judging.
        """
        text = _extract_text(query)
        centroid_scores, centroid_margin = self._gate_centroid(text)
        heuristic_scores, heuristic_margin = self._gate_heuristic(query)
        blended = self._smart_score(
            centroid_scores,
            centroid_margin,
            heuristic_scores,
            heuristic_margin,
            llm_winner=None,
        )
        vals = sorted(blended.values(), reverse=True)
        return vals[0] - vals[1] if len(vals) > 1 else 1.0

    def _get_prediction(self, query: str) -> str:
        text = _extract_text(query)

        centroid_scores, centroid_margin = self._gate_centroid(text)
        heuristic_scores, heuristic_margin = self._gate_heuristic(query)

        centroid_winner = max(
            centroid_scores, key=lambda m: centroid_scores.get(m, 0.0)
        )
        heuristic_winner = max(
            heuristic_scores, key=lambda m: heuristic_scores.get(m, 0.0)
        )

        # ── Early-exit: both gates agree with high confidence ─────────────────
        if (
            centroid_winner == heuristic_winner
            and centroid_margin > _HIGH_CONFIDENCE
            and heuristic_margin > _HIGH_CONFIDENCE
        ):
            return centroid_winner

        # ── Compute blended score (no LLM yet) ───────────────────────────────
        blended = self._smart_score(
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

        # ── Gate 3: LLM judge for low-confidence cases ───────────────────────
        llm_winner: str | None = None
        if self._llm_judge_enabled and blended_margin < _JUDGE_THRESHOLD:
            llm_winner = self._call_llm_judge(
                query,
                centroid_scores,
                centroid_margin,
                heuristic_scores,
                heuristic_margin,
            )

        # ── Final smart score incorporating LLM judge ────────────────────────
        if llm_winner:
            final = self._smart_score(
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
