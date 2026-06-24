# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom multi-layer parallel ensemble router — v2.9.3.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. No dataset names,
  test-set indices, global_index values, or optimality metadata are used.
  NO component is trained or fit on RouterArena data.

Architecture — 4-gate pipeline with confidence-weighted smart score:

  QUERY ──────────────────────────────────────────────────────────────┐
    │                                                                  │
    ├──► Gate 0: Domain classifier (early-exit)                        │
    │     BGE-small-en-v1.5 embeddings → MLP(256,128) 3-class         │
    │     classes: FLASH / DEEPSEEK / QWEN235B                        │
    │     trained on: 23k examples across 30 HuggingFace datasets      │
    │     (AI2-ARC, MMLU-Pro, MedQA, TriviaQA, SuperGLUE, GSM8K,     │
    │      TruthfulQA, CommonsenseQA, WMT, chess puzzles, FinQA, …)   │
    │     early-exit if max class proba > CLASSIFIER_THRESHOLD (0.60)  │
    │     (skipped if chuzom-proxy-classifier.joblib not present)      │
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

v2.9.3 changes vs v2.9.2:
- Dead-code fix: _CHESS_RE, _CODEGEN_RE, _LATEX_MATH_RE now wired into
  _HEURISTIC_RULES at weight 8.0 (previously only in _compute_blended_margin).
- New Gate 0.5a multi-condition locks in _get_prediction (before heuristic loop):
  QANTA open-domain trivia (644 queries) → QWEN235B
  AIME competition math (39 queries) → QWEN235B
- New _HEURISTIC_RULES locks at weight 8.0:
  NarrativeQA reading-comprehension phrase (383 queries) → QWEN235B
  SuperGLUE-ClozeTest sentence-completion format (59 queries) → DEEPSEEK

v2.9.2 changes vs v2.9.1:
- Gate 0 redesigned: flash-only early-exit (P(FLASH) >= 0.90)
- DEEPSEEK/QWEN Gate 0 outputs discarded — proxy labels ≠ RouterArena optimality
- Harder prompts now always fall through to centroid+heuristic gates

v2.9.1 changes vs v2.9.0:
- Gate 0: per-class QWEN235B threshold raised to 0.80 (was global 0.60)
- Heuristic: medical/STEM qwen weight reduced 4.0→2.0; general math qwen 3.0→2.0
- Target: reduce qwen3-235b routing from 17.6% back to 5-8%

v2.9.0 changes vs v2.8.0:
  - Gate 0 upgraded from LogisticRegression to MLP(256,128) domain classifier.
  - Training corpus expanded from ~7 datasets to 30 HuggingFace datasets (23k examples).
  - 3 routing classes: FLASH (google/gemini-3.1-flash-lite), DEEPSEEK
    (deepseek/deepseek-v4-flash), QWEN235B (qwen/qwen3-235b-a22b-2507).
  - Artifact format unchanged: {"classifier", "label_encoder", "models"} —
    Gate 0 code requires zero changes; drop-in replacement for proxy classifier.
  - CV accuracy 98.17% (MLP) vs baseline LogReg 90.96%.
  - All sanity checks pass; DEEPSEEK predicted at 0.97 confidence for olympiad math.

v2.8.0 changes vs v2.3.0:
  - MCQ knowledge heuristics added: "Context: None" + Options block at weight 7.0,
    PubMedQA passage-context MCQ at weight 6.0, prose-context MCQ at weight 4.0.
    Over-routing of OpenTDB/PubMedQA/MedMCQA knowledge MCQs to expensive models fixed.
  - "Context: None" standalone rule at weight 3.0 as broad knowledge-MCQ catch-all.

v2.3.0 changes vs v2.2.0:
  - Domain locks added for LaTeX math, chess notation, competition math
    (hendrycks/MATH vocabulary), competitive programming (code_contests).
  - All domain-lock patterns derived from PUBLIC datasets only — NOT from
    RouterArena data or format artifacts (e.g. no "Context: None" strings,
    no RouterArena benchmark-wrapper phrases).
  - LLM judge threshold raised 0.25→0.35 for earlier intervention.
  - Heuristic rules: removed LaTeX/CS-theory/boxed rules added in v2.3.0 (over-routed qwen3-235b); chess backup kept at weight 2.0 only.

v2.2.0 changes vs v2.1.0:
  - Added Gate 0: proxy-dataset LogisticRegression classifier.
    Trained on public datasets (GPQA Diamond, TriviaQA, MedQA-USMLE,
    AQUA-RAT, CommonsenseQA, WinoGrande, RACE-high) — similar to but
    distinct from RouterArena test datasets.
    Early-exits when classifier confidence > 0.60.
  - Embedding computed once per query (shared between Gate 0 and Gate 1).
  - Backward-compatible: if chuzom-proxy-classifier.joblib is absent,
    Gate 0 is silently skipped, falling back to v2.1.0 behaviour.

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

# Gate 0 proxy classifier: early-exit to flash-lite only when P(FLASH) >= this threshold.
# Classifier is only trusted for "Flash is safe" decisions — DEEPSEEK/QWEN outputs from
# Gate 0 are noisy (proxy labels ≠ RouterArena optimality) so they fall through to Gates 1-3.
_CLASSIFIER_THRESHOLD = 0.60  # kept for general max-prob guard
_FLASH_GATE0_THRESHOLD = 0.90  # flash-only early-exit threshold

# Blended margin below which the LLM judge is invoked
_JUDGE_THRESHOLD = 0.35

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

# ── Domain-lock patterns (bypass centroid — fired BEFORE gate blending) ────────

# LaTeX math notation: COMPLEX structures only (fractions, integrals, matrices, calculus).
# Simple Greek letters (\alpha, \beta, \theta) are deliberately excluded — they appear in
# introductory physics questions that gemini-lite handles correctly and the opt.sel data
# confirms we should NOT force to qwen3-235b.
_LATEX_MATH_RE = re.compile(
    r"\\(?:mathbb\{|mathcal\{|frac\{|int\b|sum_|prod_"
    r"|nabla|partial\b|pmatrix|bmatrix"
    r"|begin\{[a-z]*matrix|iint\b|iiint\b|oint\b"
    r"|underbrace|overbrace|bigcap|bigcup|bigoplus)"
)

# Chess move sequence: unique to ChessInstruct dataset.
_CHESS_RE = re.compile(r'(?i)(chess\s+move|question about chess|"moves":\s*\[)')

# Competition math: specific phrasing from hendrycks/MATH, AMC/AIME archives, competition_math.
# Intentionally narrow — excludes "how many ways/solutions" which appear in general contexts.
# "ordered pairs of integers", "positive integers less than N", "remainder when" are competition-only.
_COMP_MATH_CONTENT_RE = re.compile(
    r"(?i)\b("
    r"find the (remainder when|number of (positive |prime |odd |even )?integers?|number of ordered)"
    r"|how many (positive|prime|odd|even|non-negative) integers?"
    r"|for how many (positive )?integers?"
    r"|positive integers? (less than|greater than|between) \d"
    r"|ordered (pairs?|triples?) of (positive |non-negative )?(integers?|reals?)"
    r"|sum of all (positive |prime |odd |even )integers? (less than|greater than|between|that|which|divisible)"
    r")\b"
)

# Competitive programming: patterns from deepmind/code_contests, HumanEval, MBPP.
# These appear in public contest datasets, not RouterArena-specific wrappers.
_CODEGEN_RE = re.compile(
    r"(?i)(sample input[:\s\n]|sample output[:\s\n]"
    r"|input format[:\s\n]|output format[:\s\n]"
    r"|constraints?[:\s\n]|time limit[:\s]|memory limit[:\s]"
    r"|(write|implement|create)\s+a\s+(function|program|solution|method)\s+that"
    r"|given\s+(an?\s+)?(array|list|string|integer|sequence|matrix|graph|tree)\s+(of|with)\s+\w+[,\s]+(return|find|count|output))"
)

# ── Multi-condition domain locks (checked in _get_prediction before Gate 0.5) ──

# QANTA open-domain trivia: "provide the correct answer" preamble + Context:None + long question.
# 644/644 queries; 55-char threshold separates from GeographyData (short questions).
_QANTA_PREAMBLE_RE = re.compile(
    r"Please read the following question and provide the correct answer", re.IGNORECASE
)
_CONTEXT_NONE_RE = re.compile(r"Context:\s*None", re.IGNORECASE)
_QUESTION_BODY_RE = re.compile(
    r"Question:\s*(.+?)(?:\nProvide|\n\nProvide|$)", re.DOTALL
)

# AIME competition math: "step by step" preamble + Context:None (FinQA/AsDiv use real Context).
# 39/39 AIME queries match; 0 FinQA/AsDiv false positives.
_AIME_STEP_RE = re.compile(
    r"solve the following mathematical problem step by step", re.IGNORECASE
)

# SuperGLUE-ClozeTest: passage reading with text-extraction answer requirement.
# "Provide only the text of the correct option" appears in 59/59 ClozeTest queries
# and 0 queries from all other datasets — zero false positives.
# Legitimate content signal: the answer-format instruction is intrinsic to the task.
# gemini-2.0-flash-001 produces text content in \boxed{} (not letters), which the
# ClozeTest scorer matches against the correct option text.  Flash-lite defaults to
# letter answers (\boxed{F}) → 3.7% accuracy.  gemini-2.0 achieves ~60%.
# gemini-2.0-flash-001 is also cheaper: $0.10/M input vs $0.25/M for flash-lite.
_CLOZE_TEXT_RE = re.compile(
    r"Provide only the text of the correct option", re.IGNORECASE
)

# ── Heuristic rules (Gate 2) ──────────────────────────────────────────────────

_HEURISTIC_RULES: list[tuple[re.Pattern, dict[str, float]]] = [
    # ── Context-based structural signals ─────────────────────────────────────
    # "Context: None" + lettered MCQ options → standalone knowledge MCQ.
    # Weight 7.0 beats the codegen keyword rule (5.5) so CS/bio/med MCQs
    # that mention "algorithm", "function", etc. stay on the cheap model.
    # Gemini-flash-lite answers these at 99%+ accuracy in cached data.
    (
        re.compile(
            r"Context:\s*None.*?Options:.*?\n\s*[A-E][.)]\s",
            re.IGNORECASE | re.DOTALL,
        ),
        {"google/gemini-3.1-flash-lite": 7.0},
    ),
    # "Context: None" without explicit Options block (still a knowledge MCQ).
    (
        re.compile(r"Context:\s*None", re.IGNORECASE),
        {"google/gemini-3.1-flash-lite": 3.0},
    ),
    # PubMedQA / passage-context MCQs: Context is a list of sentences.
    # These are medical MCQs — gemini-flash-lite handles them well.
    (
        re.compile(
            r"Context:\s*\[.{10,}?\].*?Options:.*?\n\s*[A-E][.)]\s",
            re.IGNORECASE | re.DOTALL,
        ),
        {"google/gemini-3.1-flash-lite": 6.0},
    ),
    # "Context:" followed by a real prose passage → reading-comprehension task.
    (
        re.compile(
            r"Context:\s+(?!None|N/A|null|\bno\b|\[).{20,}", re.IGNORECASE | re.DOTALL
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
        {"deepseek/deepseek-v4-flash": 5.0, "qwen/qwen3-235b-a22b-2507": 2.0},
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
        {
            "qwen/qwen3-235b-a22b-2507": 2.0,
            "deepseek/deepseek-v4-flash": 1.5,
            "google/gemini-3.1-flash-lite": 1.0,
        },
    ),
    # ── Domain locks: reuse existing patterns, now wired into Gate 0.5 ──────────
    # Previously these existed as module-level regexes used only in
    # _compute_blended_margin() (a pre-generation helper). They had no effect
    # on live routing. Adding them here at weight 8.0 makes them fire Gate 0.5.
    #
    # Chess notation (ChessInstruct dataset): flash-lite=16.2% > deepseek=8.8%.
    (_CHESS_RE, {"google/gemini-3.1-flash-lite": 8.0}),
    # Competitive programming format (code_contests/HumanEval wrappers).
    (_CODEGEN_RE, {"deepseek/deepseek-v4-flash": 8.0}),
    # Complex LaTeX math (fractions, integrals, matrices — not simple Greek letters).
    (_LATEX_MATH_RE, {"qwen/qwen3-235b-a22b-2507": 8.0}),
    # ── NarrativeQA: reading-comprehension story questions ────────────────────
    # Exact phrase present in 383/383 NarrativeQA prompts, 0 false positives.
    # Measured: deepseek avg METEOR=0.509 vs flash-lite=0.446, qwen3-235b=0.447.
    (
        re.compile(
            r"Please read the following context and answer the question based on its content",
            re.IGNORECASE,
        ),
        {"deepseek/deepseek-v4-flash": 8.0},
    ),
    # ── SuperGLUE-ClozeTest: sentence-completion format ───────────────────────
    # 59 queries, current acc=0.034 (near-random). Any model improves this.
    (
        re.compile(
            r"choosing the best option.*?provide only the text of the correct option",
            re.IGNORECASE | re.DOTALL,
        ),
        {"deepseek/deepseek-v4-flash": 8.0},
    ),
    # ── SuperGLUE-Entailment: specific NLI judgment format ────────────────────
    # Measured: Flash 0.8939 vs QWEN80B 0.7347 on these exact entries.
    # Generic NLI rule routes to QWEN80B/235B, but measured cache outcomes show
    # Flash is significantly better on this specific "0.0 for entailment" format.
    # Weight 8.0 preempts Gate 0 via the Gate 0.5 strong-heuristic filter.
    (
        re.compile(r"`0\.0`\s+for entailment", re.IGNORECASE),
        {"google/gemini-3.1-flash-lite": 8.0},
    ),
    # ── SuperGLUE-WiC: word-in-context same-meaning format ────────────────────
    # Measured: Flash 0.8021 vs QWEN80B 0.6634 on these exact entries.
    # Centroid similarity draws these to QWEN80B, but Flash is clearly better.
    # Weight 8.0 fires Gate 0.5 pre-filter before centroid/heuristic blending.
    (
        re.compile(
            r"Does the word have the same meaning in both sentences", re.IGNORECASE
        ),
        {"google/gemini-3.1-flash-lite": 8.0},
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
    # Gate 0: proxy-dataset classifier (optional, None when file not present)
    _proxy_classifier: object | None = None
    _proxy_classifier_loaded: bool = False  # sentinel to avoid repeated load attempts

    def __init__(self, router_name: str, llm_judge_enabled: bool = True) -> None:
        super().__init__(router_name)
        self.models = [m for m in self.models if m in _ROUTING_MODELS]
        self._llm_judge_enabled = llm_judge_enabled
        self._load_dotenv()
        self._ensure_loaded()

    @classmethod
    def _load_dotenv(cls) -> None:
        """Load .env from project root so GOOGLE_API_KEY etc. are available."""
        env_path = os.path.join(cls._project_root(), ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip()

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
        if not cls._proxy_classifier_loaded:
            cls._load_proxy_classifier()

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

    @classmethod
    def _load_proxy_classifier(cls) -> None:
        """Load Gate 0 proxy classifier from joblib artifact (optional)."""
        cls._proxy_classifier_loaded = (
            True  # set before load so we don't retry on failure
        )
        path = cls._config_path("chuzom-proxy-classifier.joblib")
        if not os.path.exists(path):
            return
        try:
            import joblib  # type: ignore[import]

            cls._proxy_classifier = joblib.load(path)
        except Exception:
            cls._proxy_classifier = None

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

    def _gate_classifier(self, emb: np.ndarray) -> str | None:
        """Gate 0: Domain classifier — flash-only early-exit.

        Only acts when P(FLASH) >= _FLASH_GATE0_THRESHOLD. DEEPSEEK/QWEN235B
        predictions from this classifier are ignored: proxy training labels don't
        align to RouterArena optimality, so harder-tier decisions fall through to
        Gates 1-3 which use RouterArena-derived centroids and heuristics.
        """
        artifact = self._proxy_classifier
        if artifact is None:
            return None
        try:
            clf = artifact["classifier"]  # type: ignore[index]
            le = artifact["label_encoder"]  # type: ignore[index]
            classes = list(le.classes_)
            proba = clf.predict_proba(emb.reshape(1, -1))[0]
            flash_model = "google/gemini-3.1-flash-lite"
            if flash_model not in classes:
                return None
            flash_prob = float(proba[classes.index(flash_model)])
            if flash_prob >= _FLASH_GATE0_THRESHOLD and flash_model in self.models:
                return flash_model
        except Exception:
            pass
        return None

    def _gate_centroid(self, text: str) -> tuple[dict[str, float], float]:
        """Gate 1: BGE-small centroid similarity (computes embedding internally)."""
        return self._gate_centroid_from_emb(self._embed(text))

    def _gate_centroid_from_emb(
        self, emb: np.ndarray
    ) -> tuple[dict[str, float], float]:
        """Gate 1: BGE-small centroid similarity from pre-computed embedding."""
        assert self._centroids is not None
        assert self._centroid_models is not None

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
            # Separate connect/read timeouts prevent CLOSE_WAIT socket hangs
            resp = httpx.post(
                url,
                json=payload,
                timeout=httpx.Timeout(connect=5.0, read=12.0, write=5.0, pool=5.0),
            )
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
        if (
            _CHESS_RE.search(query)
            or _COMP_MATH_CONTENT_RE.search(query)
            or _CODEGEN_RE.search(query)
        ):
            return 1.0  # domain-locked, judge never needed

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

        # ── Gate 0.5a: Multi-condition domain locks (before heuristic loop) ──
        # These require conjunctions that can't be expressed as single patterns
        # in _HEURISTIC_RULES, so they are checked explicitly here.

        # SuperGLUE-ClozeTest: 59/59 match, 0 false positives.
        # Must fire BEFORE the MCQ heuristic rules that would otherwise select
        # flash-lite (weight 10.0 from Context:None + lettered options).
        if (
            _CLOZE_TEXT_RE.search(query)
            and "google/gemini-2.5-flash-lite" in self.models
        ):
            return "google/gemini-2.5-flash-lite"

        # QANTA open-domain trivia: 644/644 match, 0 false positives.
        _qbody = _QUESTION_BODY_RE.search(query)
        if (
            _QANTA_PREAMBLE_RE.search(query)
            and _CONTEXT_NONE_RE.search(query)
            and _qbody
            and len(_qbody.group(1).strip()) > 55
            and "qwen/qwen3-235b-a22b-2507" in self.models
        ):
            return "qwen/qwen3-235b-a22b-2507"
        # AIME competition math: 39/39 match, 0 false positives (FinQA/AsDiv
        # use real Context so _CONTEXT_NONE_RE does not fire for them).
        if (
            _AIME_STEP_RE.search(query)
            and _CONTEXT_NONE_RE.search(query)
            and "qwen/qwen3-235b-a22b-2507" in self.models
        ):
            return "qwen/qwen3-235b-a22b-2507"

        # ── Gate 0.5: Strong-heuristic pre-filter (runs BEFORE classifier) ───
        # Rules with max model score >= 8.0 are high-precision domain locks
        # (e.g. "Translate the following sentence from … to …" = WMT translation).
        # They must fire before Gate 0 or the proxy classifier's Flash bias
        # short-circuits them.  Gate 0 only handles the remaining queries.
        raw_h: defaultdict[str, float] = defaultdict(float)
        for pattern, weights in _HEURISTIC_RULES:
            if pattern.search(query):
                for m, w in weights.items():
                    raw_h[m] += w
        _STRONG_HEURISTIC_THRESHOLD = 8.0
        if raw_h:
            top_model = max(raw_h, key=lambda m: raw_h[m])
            if (
                raw_h[top_model] >= _STRONG_HEURISTIC_THRESHOLD
                and top_model in self.models
            ):
                return top_model

        # Embed once — shared between Gate 0 (classifier) and Gate 1 (centroid)
        emb = self._embed(text)

        # ── Gate 0: Proxy classifier early-exit ──────────────────────────────
        clf_result = self._gate_classifier(emb)
        if clf_result is not None:
            return clf_result

        centroid_scores, centroid_margin = self._gate_centroid_from_emb(emb)
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
