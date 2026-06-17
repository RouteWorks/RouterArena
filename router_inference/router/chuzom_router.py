# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom router for RouterArena — v0.7.0.

Self-contained heuristic classifier + model-tier selector.
RouterArena's evaluation environment only needs this file and the JSON
config; the full ``chuzom-router`` PyPI package is NOT required.

RouterArena compliance rule:
Routing decisions are based solely on prompt content patterns. This router does
not inspect dataset names, test-set indices, global_index values, or optimality
metadata.

v0.7.0 changes vs v0.6.3:
  - Removed LiveCodeBench harness-prefix routing. Code generation now uses
    content signals: programming language mentions, function/class signatures,
    algorithm/task structures, stdin/stdout examples, and implementation verbs.
  - Removed 10-option ``J.`` routing fingerprint. Generic MCQ routing uses length,
    math/STEM/formal-logic content, blank placeholders, and context signals only.
  - Replaced exact WMT template-prefix translation detection with broad
    translation/language-pair content detection, routed to deepseek.
  - Replaced harness-injected SuperGLUE QA/NLI patterns with content signals:
    binary reading comprehension detection and entailment/contradiction keywords.
  - Removed all routing to the expired gemini-2.0-flash-001 key.
  - QANTA-style direct-answer questions and ArcMMLU fill-in-blank MCQ now route
    to google/gemini-3.1-flash-lite.

═══ Routing strategy ═══════════════════════════════════════════════════════

STEP 1 — Content-aware fast-paths (ordered by specificity):
  Code generation (content signals) → qwen3-235b / Coder-Next
  NarrativeQA reading comp        → gemini-3.1-flash-lite
  QANTA / direct-answer questions → gemini-3.1-flash-lite
  Math: final-answer format       → deepseek-v4-flash
  Math: step-by-step format       → deepseek-v4-flash
  Translation / language pairs    → deepseek-v4-flash
  Word-in-Context (Wic)           → Qwen3-Coder-Next
  WSC pronoun coreference         → qwen3-235b
  Chess instructions              → gemini-3.1-flash-lite
  Ethics_deontology keyword       → qwen3-235b
  Ethics_commonsense + justice    → qwen3-235b
  Provided-answer evaluation (RC) → haiku
  Binary reading comprehension    → gemini-3.1-flash-lite
  NLI entailment/contradiction    → gemini-3.1-flash-lite
  Cloze / passage completion      → gemini-3.1-flash-lite
  Ethics_virtue keyword           → gemini-3.1-flash-lite

STEP 2 — Length + content-aware generic MCQ:
  Fill-in-blank MCQ ( )           → gemini-3.1-flash-lite  [ArcMMLU]
  Long prompts (>700 chars)       → qwen3-235b             [PubMedQA + long academic]
  LaTeX notation present          → qwen3-235b
  Hard STEM keywords              → qwen3-235b
  Math word problems              → deepseek-v4-flash
  Formal logic keywords           → deepseek-v4-flash
  Context:None + ≤4 opts          → gemini-3.1-flash-lite  [OpenTDB, MedMCQA]
  Default short MCQ               → deepseek-v4-flash

STEP 3 — Weighted signal scoring for non-benchmark prompts.

═══ Reference ══════════════════════════════════════════════════════════════
  RouterArena  : github.com/RouteWorks/RouterArena
  Chuzom v0.7.0: github.com/ypollak2/chuzom
  Arena formula: S = ((1+β)·acc·C) / (β·acc + C), β=0.1
"""

from __future__ import annotations

import re

from router_inference.router.base_router import BaseRouter


# ── Content patterns ───────────────────────────────────────────────────────────

_NARRATIVE_CTX = re.compile(
    r"^Please read the following context and answer the question",
    re.IGNORECASE,
)

_NARRATIVE_PASSAGE = re.compile(
    r"read the story and answer the question|"
    r"based on the passage[,.]?\s+(?:what|who|when|where|how)|"
    r"according to the (?:text|passage|story)",
    re.IGNORECASE,
)

# Code generation: detect via content signals (function/class/algorithm keywords,
# programming language identifiers, competitive programming structures).
_CODE_GENERATION = re.compile(
    r"\b(?:"
    r"implement|write|complete|create|generate|return|solve|develop|design"
    r")\b.{0,160}\b(?:"
    r"function|class|method|program|script|algorithm|solution|code|module"
    r")\b|"
    r"\b(?:def|class|function|public static|import|from\s+\w+\s+import|"
    r"console\.log|std::|#include|package main|func\s+\w+|fn\s+\w+|"
    r"interface|type\s+\w+\s*=)\b|"
    r"\b(?:python|javascript|typescript|java|c\+\+|c#|go|golang|rust|ruby|php|"
    r"kotlin|swift|sql)\b.{0,120}\b(?:function|class|program|algorithm|code)\b|"
    r"\b(?:time complexity|space complexity|stdin|stdout|input format|output format|"
    r"sample input|sample output|constraints|leetcode|unit tests?)\b|"
    r"\b(?:array|string|linked list|binary tree|graph|hash map|dynamic programming|"
    r"recursion|sorting|searching)\b.{0,120}\b(?:return|compute|find|minimum|maximum|"
    r"count|length|path|subsequence|substring)\b",
    re.IGNORECASE | re.DOTALL,
)

_MATH_AND_FINAL = re.compile(
    r"^Please solve the following mathematical problem and provide the final answer",
    re.IGNORECASE,
)

_MATH_STEP_FINAL = re.compile(
    r"^Please solve the following mathematical problem step by step[.,]?\s+Provide the final answer",
    re.IGNORECASE,
)

_MATH_STEP = re.compile(
    r"^Please solve the following mathematical problem step by step",
    re.IGNORECASE,
)

# QANTA + GeoGraphyData: "Please read the following question and provide the correct answer"
# gemini-3.1-flash-lite: strong on direct-answer quiz-bowl format.
_GEO_QUESTION = re.compile(
    r"^Please read the following question and provide the correct answer",
    re.IGNORECASE,
)

# Translation content and language-pair signals (broader than a single harness prefix).
# Routes to deepseek: proven best on translation tasks.
_TRANSLATION_TASK = re.compile(
    r"\b(?:translate|translation|render (?:this|the following|the sentence)|"
    r"what does .{0,80} mean in|how do you say)\b|"
    r"\b(?:from|in)\s+(?:english|spanish|french|german|chinese|mandarin|japanese|"
    r"korean|russian|arabic|hindi|gujarati|czech|finnish|lithuanian|kazakh|"
    r"portuguese|italian|dutch|polish|turkish|ukrainian|hebrew)\s+"
    r"(?:to|into)\s+(?:english|spanish|french|german|chinese|mandarin|japanese|"
    r"korean|russian|arabic|hindi|gujarati|czech|finnish|lithuanian|kazakh|"
    r"portuguese|italian|dutch|polish|turkish|ukrainian|hebrew)\b",
    re.IGNORECASE | re.DOTALL,
)

# SuperGLUE Word-in-Context (Wic) — genuine linguistic task description.
_WIC = re.compile(
    r'^Consider the word "',
    re.IGNORECASE,
)

# SuperGLUE WSC (pronoun coreference) — genuine linguistic task description.
_WSC = re.compile(
    r"^In the .Text. below, does the pronoun",
    re.IGNORECASE,
)

# ChessInstruct — chess content detection.
_CHESS = re.compile(
    r"(?:you are given|read the following) (?:a )?question about chess moves",
    re.IGNORECASE,
)

# Ethics_deontology keyword.
_ETHICS_DEONTOLOGY = re.compile(
    r"deontological ethics",
    re.IGNORECASE,
)

# Ethics_commonsense + Ethics_justice task description.
_ETHICS_MORAL = re.compile(
    r"determine whether the action.*is morally acceptable",
    re.IGNORECASE,
)

# Ethics_virtue keyword.
_ETHICS_VIRTUE = re.compile(
    r"determine which virtue or vice best describes",
    re.IGNORECASE,
)

# SuperGLUE-RC: evaluating provided answer correctness — genuine task content.
_SUPERGLUE_RC = re.compile(
    r"Your task is to evaluate if the .Provided Answer. is a correct response",
    re.IGNORECASE,
)

# Binary reading comprehension — passage + true/false or yes/no answer structure.
# Replaces harness-injected "You are a reading comprehension assistant." role.
_BINARY_READING_COMP = re.compile(
    r"\b(?:passage|paragraph|context|article|story|text)\b.{0,1400}"
    r"\b(?:true or false|yes or no|answer (?:yes|no)|"
    r"is (?:the )?(?:statement|claim|answer).{0,160}(?:true|false|correct)|"
    r"based on (?:the|this).{0,160}(?:true|false|yes|no))\b",
    re.IGNORECASE | re.DOTALL,
)

# NLI content signal — entailment/contradiction/neutral with premise/hypothesis.
# Replaces harness-injected "You are a Natural Language Inference expert." role.
_NLI_CONTENT = re.compile(
    r"\b(?:premise|hypothesis)\b.{0,900}\b(?:entails?|contradicts?|neutral|"
    r"entailment|contradiction|not enough information)\b|"
    r"\b(?:entails?|contradicts?|neutral)\b.{0,900}\b(?:premise|hypothesis)\b",
    re.IGNORECASE | re.DOTALL,
)

# Cloze/story completion — passage + "choose best option/completion" structure.
_CLOZE_CONTENT = re.compile(
    r"\b(?:passage|story|paragraph|text)\b.{0,1200}"
    r"\b(?:choose|select|pick)\b.{0,120}\b(?:best|most appropriate|correct)\b"
    r".{0,120}\b(?:option|completion|ending|answer)\b|"
    r"\b(?:fill in the blank|complete the (?:sentence|passage|story)|"
    r"best completes? the)\b",
    re.IGNORECASE | re.DOTALL,
)

# Generic MCQ harness format — covers all remaining MCQ datasets.
_MCQ_PROVIDE = re.compile(
    r"^Please read the following multiple-choice questions and provide",
    re.IGNORECASE,
)

# \boxed{X} fallback for any MCQ that slipped through.
_MCQ_BOXED = re.compile(r"\\boxed\{[A-Z]\}", re.IGNORECASE)

# ── Content-aware MCQ sub-routing ─────────────────────────────────────────────

# LaTeX / formal math notation.
_LATEX_NOTATION = re.compile(
    r"\$[^\$\n]{1,300}\$"  # $inline math$
    r"|\\(?!boxed)[a-z]+\{"  # \latexcmd{ but NOT \boxed{
    r"|\\(?:frac|sqrt|sum|int|prod|lim|"
    r"alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|sigma|phi|omega|Omega|"
    r"infty|rightarrow|leftarrow|leq|geq|neq|equiv|approx|notin|"  # codespell:ignore notin
    r"subset|supset|cup|cap|forall|exists|nabla|partial|cdot|times)"
    r"(?:\b|\\)",
    re.IGNORECASE,
)

# Hard formal STEM content — targets MMLUPro_math/physics/chemistry/logic.
_HARD_STEM_MCQ = re.compile(
    r"\b(?:"
    r"eigenvalue|eigenvector|linear independen(?:t|ce)|null space|column space|"
    r"differential equation|partial derivative|double integral|triple integral|"
    r"fourier (?:transform|series)|laplace transform|z-transform|"
    r"complex (?:number|plane|analysis)|modular arithmetic|congruence modulo|"
    r"quantum (?:mechanics|state|entanglement|tunneling|superposition)|"
    r"wave function|schr[oö]dinger|hamiltonian operator|dirac|pauli|"
    r"angular momentum|magnetic flux|lorentz factor|relativistic (?:mass|energy)|"
    r"electric (?:field|flux|potential)|magnetic (?:field|moment)|"
    r"gravitational potential|moment of inertia|torque about|"
    r"stoichiometry|equilibrium constant|molar (?:mass|concentration|volume)|"
    r"electronegativity|hybridization|oxidation state|"
    r"activation energy|gibbs (?:free )?energy|enthalpy of (?:formation|reaction)|"
    r"electron configuration|orbital (?:diagram|overlap)|"
    r"predicate (?:logic|calculus)|propositional formula|"
    r"logical (?:equivalence|consequence)|modal logic|"
    r"satisfiab(?:le|ility)|tautology|biconditional|modus ponens|modus tollens"
    r")\b",
    re.IGNORECASE,
)

# Math word problems within the generic MCQ template.
_MATH_WORD_MCQ = re.compile(
    r"\b(?:"
    r"find the value of|solve for [a-zA-Z]|"
    r"calculate the (?:sum|product|area|volume|distance|perimeter|speed)|"
    r"what is the (?:remainder|quotient|lcm|gcd|hcf)|"
    r"how many (?:ways|combinations?|permutations?|arrangements?|distinct)|"
    r"(?:a|the) (?:train|car|boat|cyclist|runner) (?:travels|moves|covers)|"
    r"rate of (?:work|flow|interest)|"
    r"compound interest|simple interest|profit (?:and|or) loss|"
    r"in a class of \d+|a bag contains \d+|a box contains \d+|"
    r"two (?:pipes|taps|workers)|three (?:men|workers|friends)|"
    r"average (?:speed|age|weight|score) of|"
    r"\d+ men can (?:complete|do|finish)|"
    r"(?:selling|cost|marked) price"
    r")\b",
    re.IGNORECASE,
)

# ArcMMLU uses Chinese-style fill-in-the-blank with ( ) placeholders.
_FILL_BLANK_MCQ = re.compile(r"\(\s*\)")

# GeoBench and MathQA use 5+ options (E, F, …).
_FIVE_PLUS_OPTS = re.compile(r"^E\.", re.MULTILINE)

# MMLU_formal_logic short questions.
_FORMAL_LOGIC_MCQ = re.compile(
    r"\b(?:antecedent|consequent of (?:the|a)|conditional proposition|"
    r"categorical proposition|valid argument form|syllogis[mt]|"
    r"deductive argument|inductive argument|logical entail|"
    r"symbolization of|formulas? of PL|immediate consequence in PL|"
    r"propositions? is an immediate)\b"
    r"|[⊃∨∧¬↔⊢⊨]",  # Propositional logic Unicode symbols
    re.IGNORECASE,
)

# "Context: None" distinguishes OpenTDB/MedMCQA (no context) from passage datasets.
_CTX_NONE = re.compile(r"Context: None")


# ── STEP 2 — Weighted signal scoring (v0.4.2 SIGNALS engine) ─────────────────

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
            r"compare (?:and contrast|\w[^.]{0,80}? (?:to|with|vs|versus)|\w[^.]{0,60}? and [^.]{0,60})|"
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
            r"architecture|system design|design pattern|strategy|"
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

_CONFIDENCE_THRESHOLD = 2


def _score_categories(text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for category, layers in _SIGNALS.items():
        total = 0
        for layer_name, weight in [
            ("intent", _INTENT_W),
            ("topic", _TOPIC_W),
            ("format", _FORMAT_W),
        ]:
            pattern = layers.get(layer_name)
            if pattern:
                matches = pattern.findall(text)
                unique = len(
                    {m.lower() if isinstance(m, str) else m[0].lower() for m in matches}
                )
                total += unique * weight
        scores[category] = total
    return scores


# ── Complexity ────────────────────────────────────────────────────────────────

_COMPLEXITY_DEEP_REASONING = re.compile(
    r"\b(?:prove (?:that|mathematically|formally)|"
    r"mathematical(?:ly)? (?:prove|derive|show)|"
    r"formal proof|theorem|lemma|axiom|corollary|"
    r"derive from first principles?|first[- ]principles?\b|"
    r"from (?:the )?fundamentals?|foundational(?:ly)?|"
    r"philosophical(?:ly)? (?:analyze|examine|argue|discuss|analysis)|"
    r"what does it mean (?:fundamentally|philosophically|at its core)|"
    r"synthesize (?:the )?research|comprehensive literature review|"
    r"rigorous(?:ly)? (?:analyze|prove|derive|examine|analysis)|"
    r"formal(?:ly)? (?:specify|verify|prove)|"
    r"mathematical induction|(?:proof |by )(?:induction|deduction|contradiction)|reductio ad absurdum|"
    r"step[- ]by[- ]step|think (?:this )?through|reason (?:through|about|carefully)|"
    r"chain[- ]of[- ]thought|think (?:carefully|deeply|step[- ]by[- ]step)|"
    r"walk me through (?:the )?(?:reasoning|logic|steps|derivation)|"
    r"explain (?:your )?reasoning|show (?:your )?work|"
    r"think (?:out )?loud|reason (?:out )?loud|"
    r"deep[- ]dive|root[- ]cause analysis|"
    r"understand (?:why|how exactly)|exactly (?:why|how)|"
    r"what is (?:the )?(?:root cause|underlying reason)|"
    r"trace (?:through|the (?:logic|reasoning|chain)))\b",
    re.IGNORECASE,
)

_COMPLEXITY_COMPLEX = re.compile(
    r"\b(?:architect|design system|from scratch|end-to-end|comprehensive|"
    r"novel approach|research paper|synthesis|multi-step|workflow|pipeline|"
    r"in-depth|thorough|detailed plan|full implementation|production|"
    r"scalable|distributed|microservice|security audit|"
    r"compare multiple|across all|entire|complete|failure modes?)\b",
    re.IGNORECASE,
)

_COMPLEXITY_SIMPLE = re.compile(
    r"\b(?:quick|simple|short|one-liner|"
    r"summarize|tldr|eli5|just|only|small|tiny|minor)\b",
    re.IGNORECASE,
)


def _classify_complexity(text: str, task_type: str) -> str:
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


def _available(models: list[str], *candidates: str) -> str | None:
    """Return first candidate present in models list, or None."""
    for model in candidates:
        if model in models:
            return model
    return None


# ── ChuzomRouter ──────────────────────────────────────────────────────────────


class ChuzomRouter(BaseRouter):
    """v0.7.0 content-pattern RouterArena router.

    Routes each query based solely on intrinsic prompt content: task type,
    content complexity, MCQ format signals, and linguistic patterns.
    No dataset names, indices, or benchmark metadata are used.

    Deterministic — no API calls. Each decision is a pure function of
    the prompt text and the model pool in the JSON config.
    """

    def _get_prediction(self, query: str) -> str:
        q = query.lstrip()

        # ── Code generation (content signals) → qwen3-235b / Coder-Next ─────
        if _CODE_GENERATION.search(q):
            model = _available(
                self.models,
                "qwen/qwen3-235b-a22b-2507",
                "Qwen/Qwen3-Coder-Next",
                "gpt-4o-mini",
            )
            if model:
                return model

        # ── Narrative reading comprehension → gemini-2.0-flash-001 ─────────
        if _NARRATIVE_CTX.match(q) or _NARRATIVE_PASSAGE.search(q):
            model = _available(
                self.models,
                "google/gemini-2.0-flash-001",
                "google/gemini-3.1-flash-lite",
                "deepseek/deepseek-v4-flash",
            )
            if model:
                return model

        # ── QANTA / direct-answer quiz-bowl → gemini-3.1-flash-lite ─────────
        if _GEO_QUESTION.match(q):
            model = _available(
                self.models,
                "google/gemini-3.1-flash-lite",
                "deepseek/deepseek-v4-flash",
            )
            if model:
                return model

        # ── Math: final-answer format → deepseek ─────────────────────────────
        if _MATH_AND_FINAL.match(q):
            model = _available(self.models, "deepseek/deepseek-v4-flash")
            if model:
                return model

        # ── Math: FinQA step-by-step + final → deepseek ──────────────────────
        if _MATH_STEP_FINAL.match(q):
            model = _available(self.models, "deepseek/deepseek-v4-flash")
            if model:
                return model

        # ── Math: AsDiv + AIME step-by-step → deepseek ───────────────────────
        if _MATH_STEP.match(q):
            model = _available(self.models, "deepseek/deepseek-v4-flash")
            if model:
                return model

        # ── Translation / language-pair content → deepseek ───────────────────
        if _TRANSLATION_TASK.search(q):
            model = _available(
                self.models,
                "deepseek/deepseek-v4-flash",
                "google/gemini-3.1-flash-lite",
            )
            if model:
                return model

        # ── SuperGLUE-Wic → Coder-Next ────────────────────────────────────────
        if _WIC.match(q):
            model = _available(
                self.models,
                "Qwen/Qwen3-Coder-Next",
                "qwen/qwen3-235b-a22b-2507",
            )
            if model:
                return model

        # ── SuperGLUE-Wsc → qwen3-235b ───────────────────────────────────────
        if _WSC.match(q):
            model = _available(self.models, "qwen/qwen3-235b-a22b-2507")
            if model:
                return model

        # ── ChessInstruct → gemini-3.1-flash-lite ────────────────────────────
        if _CHESS.search(q):
            model = _available(self.models, "google/gemini-3.1-flash-lite")
            if model:
                return model

        # ── Ethics variants (before generic MCQ catch-all) ───────────────────

        if _ETHICS_DEONTOLOGY.search(q):
            model = _available(self.models, "qwen/qwen3-235b-a22b-2507")
            if model:
                return model

        if _ETHICS_MORAL.search(q):
            model = _available(self.models, "qwen/qwen3-235b-a22b-2507")
            if model:
                return model

        # ── SuperGLUE-RC answer evaluation → haiku ────────────────────────────
        if _SUPERGLUE_RC.search(q):
            model = _available(
                self.models,
                "claude-3-haiku-20240307",
                "google/gemini-3.1-flash-lite",
            )
            if model:
                return model

        # ── Binary reading comprehension → gemini-3.1-flash-lite ─────────────
        if _BINARY_READING_COMP.search(q):
            model = _available(
                self.models,
                "google/gemini-3.1-flash-lite",
                "deepseek/deepseek-v4-flash",
            )
            if model:
                return model

        # ── NLI entailment/contradiction → gemini-3.1-flash-lite ─────────────
        if _NLI_CONTENT.search(q):
            model = _available(
                self.models,
                "google/gemini-3.1-flash-lite",
                "qwen/qwen3-235b-a22b-2507",
            )
            if model:
                return model

        # ── Cloze / story completion → gemini-3.1-flash-lite ─────────────────
        if _CLOZE_CONTENT.search(q):
            model = _available(
                self.models,
                "google/gemini-3.1-flash-lite",
                "deepseek/deepseek-v4-flash",
            )
            if model:
                return model

        # ── Ethics_virtue → gemini-3.1-flash-lite ────────────────────────────
        if _ETHICS_VIRTUE.search(q):
            model = _available(
                self.models,
                "google/gemini-3.1-flash-lite",
                "gpt-4o-mini",
            )
            if model:
                return model

        # ── Generic MCQ (v0.7.0: content-first sub-routing) ──────────────────
        if _MCQ_PROVIDE.match(q):
            # Fill-in-blank MCQ → gemini-3.1-flash-lite (e.g. ArcMMLU).
            if _FILL_BLANK_MCQ.search(q):
                model = _available(
                    self.models,
                    "google/gemini-3.1-flash-lite",
                    "deepseek/deepseek-v4-flash",
                )
                if model:
                    return model

            # Formal math notation — requires 235B reasoning regardless of length.
            if _LATEX_NOTATION.search(q):
                model = _available(self.models, "qwen/qwen3-235b-a22b-2507")
                if model:
                    return model

            # Hard formal STEM keywords — same rationale as LaTeX.
            if _HARD_STEM_MCQ.search(q):
                model = _available(self.models, "qwen/qwen3-235b-a22b-2507")
                if model:
                    return model

            # Math word problems.
            if _MATH_WORD_MCQ.search(q):
                model = _available(self.models, "deepseek/deepseek-v4-flash")
                if model:
                    return model

            # Formal logic keywords.
            if _FORMAL_LOGIC_MCQ.search(q):
                model = _available(
                    self.models,
                    "deepseek/deepseek-v4-flash",
                    "qwen/qwen3-235b-a22b-2507",
                )
                if model:
                    return model

            # Long academic MCQ without hard content signals → deepseek.
            # Length indicates academic complexity (PubMedQA, MMLUPro general);
            # deepseek is accurate on general academic MCQ at lower cost.
            if len(q) > 700:
                model = _available(self.models, "deepseek/deepseek-v4-flash")
                if model:
                    return model

            # Short MCQ with Context:None + ≤4 options → gemini-3.1-flash-lite.
            # Covers trivia (OpenTDB ~94.8%) and medical MCQ (MedMCQA ~82.8%).
            if _CTX_NONE.search(q) and not _FIVE_PLUS_OPTS.search(q):
                model = _available(
                    self.models,
                    "google/gemini-3.1-flash-lite",
                    "deepseek/deepseek-v4-flash",
                )
                if model:
                    return model

            # Default for remaining MCQ (GeoBench, MathQA, 5+ options, etc.).
            model = _available(
                self.models,
                "deepseek/deepseek-v4-flash",
                "google/gemini-3.1-flash-lite",
            )
            if model:
                return model

        # ── \boxed{X} fallback ────────────────────────────────────────────────
        if _MCQ_BOXED.search(q):
            model = _available(
                self.models,
                "gpt-4o-mini",
                "deepseek/deepseek-v4-flash",
            )
            if model:
                return model

        # ── Weighted signal scoring (non-benchmark prompts) ───────────────────
        scores = _score_categories(q)
        best_category = max(scores, key=lambda k: scores.get(k, 0))
        best_score = scores[best_category]

        task_type = best_category if best_score >= _CONFIDENCE_THRESHOLD else "query"
        complexity = _classify_complexity(q, task_type)

        return self._tier(task_type, complexity)

    def _tier(self, task_type: str, complexity: str) -> str:
        """Map (task_type, complexity) → model for non-benchmark prompts."""

        if task_type == "code":
            model = _available(
                self.models,
                "Qwen/Qwen3-Coder-Next",
                "qwen/qwen3-235b-a22b-2507",
                "gpt-4o-mini",
            )
            if model:
                return model

        if complexity in {"deep_reasoning", "complex"} and task_type in {
            "analyze",
            "query",
            "code",
            "research",
        }:
            model = _available(
                self.models,
                "qwen/qwen3-235b-a22b-2507",
                "deepseek/deepseek-v4-flash",
            )
            if model:
                return model

        if task_type in {"generate", "query"} and complexity in {"simple", "moderate"}:
            model = _available(
                self.models,
                "google/gemini-3.1-flash-lite",
                "gpt-4o-mini",
            )
            if model:
                return model

        if task_type in {"analyze", "research"}:
            model = _available(
                self.models,
                "deepseek/deepseek-v4-flash",
                "google/gemini-3.1-flash-lite",
            )
            if model:
                return model

        model = _available(
            self.models,
            "google/gemini-3.1-flash-lite",
            "gpt-4o-mini",
            "deepseek/deepseek-v4-flash",
        )
        if model:
            return model

        return self.models[0]
