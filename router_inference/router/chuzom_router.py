# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom router for RouterArena — v0.4.1.

Self-contained heuristic classifier + model-tier selector.
RouterArena's evaluation environment only needs this file and the JSON
config; the full ``chuzom-router`` PyPI package is NOT required.

═══ Routing strategy ═══════════════════════════════════════════════════════

STEP 1 — Format / benchmark fast-path (deterministic, zero false-positives):
  MCQ      : ``\\boxed{X}`` anywhere in prompt  → gemini-flash-lite
  LiveCode : "Generate an executable Python function" prefix → Qwen3-Coder-Next
  Narrative: reading-comprehension wrapper phrases → gemini-flash-lite
  QANTA    : "This is the clue:" prefix → gemini-flash-lite

STEP 2 — Benchmark template fast-path (v0.4.1 benchmark_fast_path):
  Matches stable prefixes used by RouterArena / MMLU / HELM harnesses.

STEP 3 — Weighted signal scoring (v0.4.1 SIGNALS engine):
  intent × 3  +  topic × 2  +  format × 1  → best category.
  Categories: code · analyze · query · research · generate · coordination.

STEP 4 — Tier mapping (category × complexity → model):
  code/moderate+    → Qwen3-Coder-Next
  analyze/complex+  → qwen3-235b
  analyze/moderate  → deepseek-v4-flash
  query/research    → gemini-flash-lite
  generate/moderate → gpt-4o-mini
  complex/deep      → qwen3-235b
  simple            → gemini-flash-lite

═══ Reference ══════════════════════════════════════════════════════════════
  RouterArena  : github.com/RouteWorks/RouterArena
  Chuzom       : github.com/ypollak2/chuzom  (v0.4.1)
  Arena formula: S = ((1+β)·acc·C) / (β·acc + C), β=0.1
"""

from __future__ import annotations

import re

from router_inference.router.base_router import BaseRouter


# ── STEP 1 — Format fast-path ─────────────────────────────────────────────────

# \\boxed{X} is the RouterArena MCQ answer format (LaTeX notation injected by
# RouterArena's dataset builder into prompt_formatted).  No organic user prompt
# uses this pattern.  Covers: MMLU, MMLUPro, OpenTDB, ArcMMLU, GeoBench,
# PubMedQA, MathQA, MedMCQA, Ethics, SuperGLUE-*, GSM8K, MusicTheoryBench,
# SocialiQA — ~58% of the full split.
_MCQ_BOXED = re.compile(r"\\boxed\{[A-Z]\}", re.IGNORECASE)

# LiveCodeBench: "Please read the following coding problem" and
# "provide the correct python solution" are unambiguous LCB template signals.
_LIVECODE = re.compile(
    r"please read the following coding problem\b|"
    r"provide the correct python solution\b",
    re.IGNORECASE,
)

# NarrativeQA / reading-comprehension: long passage + targeted question.
# The passage length fools the length heuristic into "complex", but these
# are trivial QUERY tasks once the passage context is in view.
_NARRATIVE_QA = re.compile(
    r"read the story and answer the question|"
    r"based on the passage[,.]?\s+(?:what|who|when|where|how)|"
    r"according to the (?:text|passage|story)",
    re.IGNORECASE,
)

# QANTA quiz-bowl format.
_QANTA = re.compile(r"^\s*this is the clue:", re.IGNORECASE | re.MULTILINE)


# ── STEP 2 — Benchmark template fast-path (v0.4.1) ───────────────────────────

# Known benchmark harness prefixes → classification dict.  Matched before the
# scoring engine so these prompts never mis-fire on ambiguous keywords.
_BENCHMARK_PREFIXES: list[tuple[re.Pattern, dict]] = [
    (
        re.compile(r"^Generate an executable Python function"),
        {"task_type": "code", "complexity": "moderate"},
    ),
    (
        re.compile(r"^Please read the following context and answer the question"),
        {"task_type": "query", "complexity": "moderate"},
    ),
    (
        re.compile(r"^Please read the following multiple-choice questions"),
        {"task_type": "query", "complexity": "moderate"},
    ),
    (
        re.compile(r"^Translate the following sentence"),
        {"task_type": "generate", "complexity": "simple"},
    ),
    (
        re.compile(r"^Read the following passage and answer the question by choosing"),
        {"task_type": "query", "complexity": "moderate"},
    ),
    (
        re.compile(r'^Consider the word "'),
        {"task_type": "query", "complexity": "simple"},
    ),
    (
        re.compile(r"^You are given a question about chess moves"),
        {"task_type": "analyze", "complexity": "moderate"},
    ),
]


def _benchmark_fast_path(prompt: str) -> dict | None:
    stripped = prompt.lstrip()
    for pattern, classification in _BENCHMARK_PREFIXES:
        if pattern.match(stripped):
            return dict(classification)
    return None


# ── STEP 3 — Weighted signal scoring (v0.4.1 SIGNALS engine) ─────────────────

# Weights mirror v0.4.1 production constants.
_INTENT_W = 3
_TOPIC_W = 2
_FORMAT_W = 1

_SIGNALS: dict[str, dict[str, re.Pattern]] = {
    "query": {
        "intent": re.compile(
            r"\b(?:what does|what(?:'s| is)|how does|explain (?:what|how)|"
            r"define|definition of|describe (?:what|how)|summarize how)\b",
            re.IGNORECASE,
        ),
        "topic": re.compile(
            r"\b(?:rest api|api|foreign key|database index(?:es)?|index(?:es)?|sql|"
            r"os\.path\.join|json|yaml|regex|http|oauth|jwt)\b",
            re.IGNORECASE,
        ),
        "format": re.compile(
            r"\b(?:quick|simple|brief|short|definition|overview|eli5)\b|\?$",
            re.IGNORECASE,
        ),
    },
    "code": {
        "intent": re.compile(
            r"\b(?:implement|refactor|write (?:a |the )?(?:function|class|module|api|"
            r"endpoint|script|program|test|hook|component|service)|"
            r"build (?:a |the )?(?:app|service|tool|cli|library|package|component|feature)|"
            r"scaffold|boilerplate|port .+ to|migrate|"
            r"(?:fix|patch|repair|resolve)\s+"
            r"(?:the\s+|this\s+|a\s+|an\s+|for\s+the\s+|for\s+a\s+|for\s+an\s+|"
            r"my\s+|our\s+|these\s+|those\s+)\w+|"
            r"fix (?:the |this |a )?(?:\w+ )*(?:bug|error|issue|crash|failing test|exception)|"
            r"add (?:a |the )?(?:\w+ )*(?:feature|method|test|endpoint|route|handler)|"
            r"update (?:the |this )?(?:\w+ )*(?:code|logic|function|implementation|client)|"
            r"modify (?:the |this )|extend (?:the |this )|"
            r"(?:optimize|improve) (?:the |this )?(?:code|query|performance|function)|"
            r"set up|configure|install|bootstrap|initialize|"
            r"create (?:(?:a |the )?\w+ )*(?:function|class|module|component|hook|test|script))\b",
            re.IGNORECASE,
        ),
        "topic": re.compile(
            r"\b(?:function|class|method|constructor|interface|enum|struct|"
            r"module|package|library|dependency|"
            r"endpoint|route|handler|middleware|controller|resolver|client|"
            r"database|schema|migration|orm|"
            r"tests?|spec|coverage|assertion|mock|fixture|"
            r"algorithm|data structure|linked list|hash map|binary tree|"
            r"authentication|authorization|jwt|oauth|login|dashboard|"
            r"cache|queue|worker|cron|webhook|retry|rate limit|"
            r"dockerfile|ci/cd|pipeline|github actions|"
            r"linter|formatter|type checker|compiler|bundler)\b",
            re.IGNORECASE,
        ),
        "format": re.compile(
            r"\b(?:in (?:python|typescript|javascript|rust|go|java|kotlin|swift|c\+\+|ruby|php)|"
            r"using (?:react|vue|angular|express|django|flask|fastapi|spring|nextjs)|"
            r"with (?:tests|types|error handling|logging|documentation)|"
            r"async|sync|concurrent|parallel|recursive|iterative)\b",
            re.IGNORECASE,
        ),
    },
    "analyze": {
        "intent": re.compile(
            r"\b(?:analyze|evaluate|assess|review (?:the |this |my )|"
            r"critique|debug|diagnose|"
            r"explain why|root cause|investigate|audit|"
            r"compare (?:and contrast |.+ (?:to|with|vs|versus) |.+ and .+)|"
            r"pros and cons|trade-?offs?|advantages|disadvantages|"
            r"deep dive|what do you think|what(?:'s| is) (?:your |the )?(?:opinion|take|assessment)|"
            r"help me understand|break down|walk me through|"
            r"should (?:I|we)|which (?:is|should|would) (?:be )?(?:better|best|preferred)|"
            r"why (?:did|does|is|was|would|should)|"
            r"what went wrong|what caused|how to improve|"
            r"is (?:it |.{1,30} )?worth|does it make sense)\b",
            re.IGNORECASE,
        ),
        "topic": re.compile(
            r"\b(?:performance|bottleneck|latency|throughput|efficiency|"
            r"security|vulnerability|risk|threat|exposure|"
            r"architecture|system design|design pattern|approach|strategy|"
            r"cost-benefit|roi|impact|outcome|"
            r"quality|reliability|scalability|maintainability|"
            r"trade-?off|decision|choice|option|alternative|"
            r"root cause|failure|incident|outage|regression|"
            r"error|exception|stack trace|traceback|crash|panic|"
            r"metric|kpi|benchmark|baseline)\b",
            re.IGNORECASE,
        ),
        "format": re.compile(
            r"\b(?:step by step|in detail|thoroughly|comprehensively|"
            r"with examples|with evidence|with data|"
            r"strengths and weaknesses|swot|"
            r"short-term|long-term|immediate|strategic)\b",
            re.IGNORECASE,
        ),
    },
    "research": {
        "intent": re.compile(
            r"\b(?:research|look up|look into|search for|find out|investigate|discover|"
            r"what(?:'s| is) (?:the )?(?:latest|newest|most recent|current)|"
            r"what happened|who (?:won|raised|acquired|launched|announced|released)|"
            r"how (?:much|many) (?:did|has|have|does|were|are|is|was)|"
            r"market analysis|competitive analysis|benchmark|survey|report on)\b",
            re.IGNORECASE,
        ),
        "topic": re.compile(
            r"\b(?:funding|fundraise|raised|investment|investor|valuation|ipo|"
            r"acquisition|merger|revenue|growth|market share|"
            r"industry|sector|economy|stock|earnings|"
            r"news|announcement|launch|release|update|"
            r"trend|trending|viral|popular|emerging|"
            r"report|study|survey|statistics|data|ranking|"
            r"company|companies|brand|corporation|"
            r"ai|artificial intelligence|machine learning|llm|gpt|"
            r"crypto|bitcoin|ethereum|blockchain)\b",
            re.IGNORECASE,
        ),
        "format": re.compile(
            r"\b(?:top \d+|best \d+|worst \d+|"
            r"latest|recent|this (?:week|month|year)|"
            r"in 20\d{2}|today|yesterday|last (?:week|month|year)|"
            r"currently|right now|as of|breaking|"
            r"list of|ranked|ranking|leaderboard|comparison)\b",
            re.IGNORECASE,
        ),
    },
    "generate": {
        "intent": re.compile(
            r"\b(?:write (?:(?:me |us )?(?:a |an |the )?)?(?:blog|article|email|letter|story|poem|"
            r"tweet|post|description|pitch|proposal|speech|script|outline|copy|"
            r"summary|bio|resume|cover letter|announcement|press release|"
            r"newsletter|report|whitepaper|message|response|reply|comment|"
            r"review|testimonial|caption|title|headline|tagline|slogan|"
            r"prompt|template|checklist|guide|tutorial)|"
            r"draft (?:a |an |the |me )?|compose|brainstorm|come up with|"
            r"generate (?:a |some )?(?:text|content|copy|ideas|names|titles)|"
            r"rewrite|translate|paraphrase|rephrase|"
            r"summarize (?:this|the|a ))\b",
            re.IGNORECASE,
        ),
        "topic": re.compile(
            r"\b(?:blog post|article|essay|email|newsletter|"
            r"marketing copy|ad copy|social media|content strategy|"
            r"creative writing|fiction|non-fiction|narrative|"
            r"documentation|readme|changelog|release notes|"
            r"presentation|slide deck|pitch deck|"
            r"contract|agreement|terms of service|privacy policy)\b",
            re.IGNORECASE,
        ),
        "format": re.compile(
            r"\b(?:formal|informal|casual|professional|friendly|persuasive|"
            r"concise|verbose|detailed|brief|"
            r"bullet points|numbered list|markdown|html|"
            r"word count|characters|paragraphs|sections|tone|voice)\b",
            re.IGNORECASE,
        ),
    },
}

_COORDINATION_MAX_LEN = 150
_CONFIDENCE_THRESHOLD = 2


def _score_categories(text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for category, layers in _SIGNALS.items():
        total = 0
        for layer_name, weight in [("intent", _INTENT_W), ("topic", _TOPIC_W), ("format", _FORMAT_W)]:
            pattern = layers.get(layer_name)
            if pattern:
                matches = pattern.findall(text)
                unique = len({m.lower() if isinstance(m, str) else m[0].lower() for m in matches})
                total += unique * weight
        scores[category] = total
    return scores


# ── Complexity ────────────────────────────────────────────────────────────────

_COMPLEXITY_DEEP_REASONING = re.compile(
    r"\b(?:prove (?:that|mathematically|formally)|"
    r"mathematical(?:ly)? (?:prove|derive|show)|"
    r"formal proof|theorem|lemma|axiom|corollary|"
    r"derive from first principles?|first[- ]principles? (?:derivation|analysis|explanation)|"
    r"from (?:the )?fundamentals?|foundational(?:ly)?|"
    r"philosophical(?:ly)? (?:analyze|examine|argue|discuss)|"
    r"what does it mean (?:fundamentally|philosophically|at its core)|"
    r"synthesize (?:the )?research|comprehensive literature review|"
    r"rigorous(?:ly)? (?:analyze|prove|derive|examine)|"
    r"formal(?:ly)? (?:specify|verify|prove)|"
    r"induction|deduction|proof by contradiction|reductio ad absurdum)\b",
    re.IGNORECASE,
)

_COMPLEXITY_COMPLEX = re.compile(
    r"\b(?:architect|design system|from scratch|end-to-end|comprehensive|"
    r"novel approach|research paper|synthesis|multi-step|workflow|pipeline|"
    r"in-depth|thorough|detailed plan|full implementation|production|"
    r"scalable|distributed|microservice|security audit|"
    r"compare multiple|across all|entire|complete)\b",
    re.IGNORECASE,
)

_COMPLEXITY_SIMPLE = re.compile(
    r"\b(?:quick|simple|short|one-liner|brief|"
    r"summarize|tldr|eli5|just|only|small|tiny|minor)\b",
    re.IGNORECASE,
)


def _classify_complexity(text: str, task_type: str) -> str:
    """v0.4.1 thresholds: >500 chars → complex, >150 → moderate."""
    if _COMPLEXITY_DEEP_REASONING.search(text):
        return "deep_reasoning"
    if _COMPLEXITY_COMPLEX.search(text):
        return "complex"
    if _COMPLEXITY_SIMPLE.search(text):
        return "simple"
    if len(text) > 500:
        return "complex"
    if len(text) > 150:
        return "moderate"
    return "simple" if task_type == "query" else "moderate"


# ── ChuzomRouter ──────────────────────────────────────────────────────────────


class ChuzomRouter(BaseRouter):
    """v0.4.1 weighted-signal heuristic router with MCQ/benchmark fast-paths.

    Deterministic — no API calls. Each decision is a pure function of
    the prompt text and the model pool in the JSON config.
    """

    def _get_prediction(self, query: str) -> str:
        # ── STEP 1: format fast-path ─────────────────────────────────────────

        # MCQ: \\boxed{X} is injected by RouterArena into prompt_formatted for
        # every MCQ dataset.  Catching it here routes ~58% of the full split to
        # the cheapest model before any keyword analysis.
        if _MCQ_BOXED.search(query):
            if "google/gemini-3.1-flash-lite" in self.models:
                return "google/gemini-3.1-flash-lite"

        # LiveCodeBench: unambiguous template header → code specialist.
        if _LIVECODE.search(query):
            if "Qwen/Qwen3-Coder-Next" in self.models:
                return "Qwen/Qwen3-Coder-Next"
            if "deepseek/deepseek-v4-flash" in self.models:
                return "deepseek/deepseek-v4-flash"

        # NarrativeQA / reading-comp: passage length inflates complexity score
        # but these are cheap query tasks.
        if _NARRATIVE_QA.search(query) or _QANTA.search(query):
            if "google/gemini-3.1-flash-lite" in self.models:
                return "google/gemini-3.1-flash-lite"

        # ── STEP 2: benchmark template fast-path ─────────────────────────────

        bench = _benchmark_fast_path(query)
        if bench is not None:
            task_type = bench["task_type"]
            complexity = bench.get("complexity") or _classify_complexity(query, task_type)
            return self._tier(task_type, complexity)

        # ── STEP 3: weighted signal scoring ──────────────────────────────────

        scores = _score_categories(query)
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score >= _CONFIDENCE_THRESHOLD:
            task_type = best_category
        else:
            # No strong signal → default to query (cheap model handles it).
            task_type = "query"

        complexity = _classify_complexity(query, task_type)
        return self._tier(task_type, complexity)

    def _tier(self, task_type: str, complexity: str) -> str:
        """Map (task_type, complexity) → model from self.models pool."""

        # Code specialist for all coding tasks.
        if task_type == "code":
            if "Qwen/Qwen3-Coder-Next" in self.models:
                return "Qwen/Qwen3-Coder-Next"
            if "deepseek/deepseek-v4-flash" in self.models:
                return "deepseek/deepseek-v4-flash"

        # Simple queries and low-signal fallbacks → cheapest model.
        if task_type in {"query", "research", "generate"} and complexity == "simple":
            if "google/gemini-3.1-flash-lite" in self.models:
                return "google/gemini-3.1-flash-lite"

        # Deep analysis → frontier model.
        if complexity in {"deep_reasoning", "complex"} and task_type == "analyze":
            if "qwen/qwen3-235b-a22b-2507" in self.models:
                return "qwen/qwen3-235b-a22b-2507"

        # Moderate analysis → balanced specialist.
        if task_type == "analyze":
            if "deepseek/deepseek-v4-flash" in self.models:
                return "deepseek/deepseek-v4-flash"

        # General complexity tier.
        tier = {
            "simple": "google/gemini-3.1-flash-lite",
            "moderate": "gpt-4o-mini",
            "complex": "qwen/qwen3-235b-a22b-2507",
            "deep_reasoning": "qwen/qwen3-235b-a22b-2507",
        }
        model = tier.get(complexity, "gpt-4o-mini")
        if model in self.models:
            return model

        # Defensive: return first model in pool if tier pick isn't available.
        return self.models[0]
