# SPDX-FileCopyrightText: Copyright (c) 2026 Yali Pollak
# SPDX-License-Identifier: Apache-2.0

"""Tier 1A — self-consistency utilities for multiple-choice queries.

Pure functions for the inference runner. Each function uses only prompt
text or model output text — never accuracy, cost, or any signal derived
from prior evaluations. See ``docs/ROUTERARENA_IMPROVEMENT_PLAN.md`` §"Allowed
signals" for the policy.

Public API
----------
``is_multiple_choice(prompt)``
    Return True iff ``prompt`` opens with the canonical RouterArena MC
    template ("Please read the following multiple-choice questions...").
    Detection is by prompt opening only; no per-query labels are read.

``extract_boxed_answer(text)``
    Extract the first ``\\boxed{X}`` answer where ``X`` is a single
    A-J letter. Returns the letter or ``None``.

``extract_mc_letter(text)``
    Fallback extractor for outputs that omit ``\\boxed{}``. Looks for
    common patterns like "The answer is C" or "Answer: B".

``majority_vote(letters, min_agreement=2)``
    Take a list of extracted letters (may include ``None``). Return the
    most-common non-None letter iff it appears at least ``min_agreement``
    times; otherwise ``None``. Ties at the top resolve to ``None`` to
    signal "no consensus" so the runner can fall back to the first sample.

``system_prompt_for(prompt)``
    Tier 1B — pick a task-family-tailored system prompt from the user
    prompt text alone. Returns the system message string or ``None``
    (caller sends no system message). Uses only prompt content; no
    accuracy/cost/label signals.

``SYSTEM_PROMPT_VERSION``
    Version tag for the prompt set. Bump this when you change any
    prompt body so the inference runner can invalidate cached samples
    that were drawn from the old prompts.

Reference
---------
The ``\\boxed{X}`` regex matches RouterArena's own
``llm_evaluation.enhanced_extractor.EnhancedExtractor._extract_standard_boxed``
so the vote tally agrees with what the leaderboard will eventually score.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

__all__ = [
    "is_multiple_choice",
    "extract_boxed_answer",
    "extract_mc_letter",
    "majority_vote",
    "system_prompt_for",
    "SYSTEM_PROMPT_VERSION",
]


_MC_OPENING_SIGNATURES = (
    "multiple-choice question",
    "multiple-choice questions",
    "most likely correct answer based on the options",
)


def is_multiple_choice(prompt: str) -> bool:
    """Return True for canonical RouterArena MC prompts.

    Detection is by prompt opening only. Empirically (065cca5 baseline,
    8400 prompts) every dataset prefix is either 100% MC or 0% MC, so a
    text-only check is sufficient and stable across the full split.
    """
    if not prompt:
        return False
    head = prompt[:300].lower()
    return any(sig in head for sig in _MC_OPENING_SIGNATURES)


_BOXED_RE = re.compile(r"\\boxed\{\s*([A-J])\s*\}")
_LETTER_AFTER_PHRASE_RE = re.compile(
    r"\b(?:the\s+)?(?:correct\s+)?(?:answer|choice|option|letter)"
    r"\s*(?:is|:|=)\s*\(?([A-J])\)?\b",
    re.IGNORECASE,
)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the first ``\\boxed{X}`` letter, or ``None``."""
    if not text:
        return None
    match = _BOXED_RE.search(text)
    return match.group(1).upper() if match else None


def extract_mc_letter(text: str) -> Optional[str]:
    """Fallback letter extractor for outputs without ``\\boxed{}``."""
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    phrase_match = _LETTER_AFTER_PHRASE_RE.search(text)
    if phrase_match:
        return phrase_match.group(1).upper()
    return None


def majority_vote(
    letters: list[Optional[str]],
    min_agreement: int = 2,
) -> Optional[str]:
    """Return the majority letter, or ``None`` if no consensus.

    Args:
        letters: Per-sample extracted letters. ``None`` entries are
            treated as "no answer" and excluded from the tally.
        min_agreement: Minimum agreement count required to declare a
            winner. Default 2 — appropriate for 3-sample voting.

    Returns:
        The single most-common letter if it appears at least
        ``min_agreement`` times AND beats every other letter outright.
        Returns ``None`` for ties at the top so the caller can fall
        back to the first sample. Returns ``None`` if no valid letters
        were extracted.
    """
    tally = Counter(letter for letter in letters if letter is not None)
    if not tally:
        return None
    most_common = tally.most_common(2)
    top_letter, top_count = most_common[0]
    if top_count < min_agreement:
        return None
    if len(most_common) > 1 and most_common[1][1] == top_count:
        return None
    return top_letter


# ── Tier 1B: task-family system prompts ───────────────────────────────────────
#
# Each prompt is short and focused. They all instruct the model to:
#   1. produce a brief reasoning trace (helps small models, doesn't hurt big),
#   2. emit a final answer in the canonical RouterArena format
#      (``\\boxed{LETTER}`` for MC, ``\\boxed{value}`` for math),
#   3. avoid refusals or "I don't know" — the eval scores those as wrong.
#
# Legitimacy: the *selector* below only reads ``prompt`` text. The strings
# themselves are static — they don't encode per-query labels.

SYSTEM_PROMPT_VERSION = "v1"

_SP_MC_GENERAL = (
    "You answer multiple-choice questions. "
    "Read the question and options carefully. "
    "Eliminate clearly wrong options, then pick the single best one. "
    "End your reply with the final letter on its own in \\boxed{X} exactly once. "
    "Do not refuse; if uncertain, choose the most plausible option."
)

_SP_MC_REASONING = (
    "You answer multiple-choice questions in math, physics, chemistry, "
    "engineering, and related quantitative sciences. "
    "Reason step by step. Check units and signs. "
    "Then pick the single best option and end your reply with the final letter "
    "in \\boxed{X} exactly once. Do not refuse."
)

_SP_MATH_OPEN = (
    "You solve math problems. "
    "Show concise step-by-step reasoning, then give the final numerical answer "
    "in \\boxed{value} exactly once. Do not refuse."
)

_SP_CODE = (
    "You write code that compiles and runs correctly the first time. "
    "Return only the requested function or program in the requested language. "
    "No prose before or after the code block. Do not refuse."
)


# Detectors. All operate on prompt text only.

_RE_SP_CODE_PROMPT = re.compile(
    r"generate an executable [a-z]+ function|"
    r"write a (?:python|java|c\+\+|javascript|typescript|rust|go) function|"
    r"return the function body",
    re.IGNORECASE,
)

_RE_SP_MATH_OPEN_PROMPT = re.compile(
    r"solve the following math(?:ematical)? problem|"
    r"step[- ]by[- ]step solution|"
    r"final answer (?:as|in) (?:a |an )?\\boxed",
    re.IGNORECASE,
)

# Quantitative MC subjects — keyword detector for MC prompts that benefit
# from explicit step-by-step reasoning instructions. Mirrors the science
# keywords used in the router; deliberately conservative.
_RE_SP_MC_REASONING_KW = re.compile(
    r"\b(?:velocity|acceleration|momentum|kinetic energy|"
    r"electric field|magnetic field|wavelength|frequency|"
    r"thermodynamic|entropy|enthalpy|quantum|photon|electron|"
    r"derivative|integral|matrix|polynomial|equation|"
    r"stoichiometr|mole|molar|reaction rate|equilibrium constant|"
    r"force|torque|circuit|voltage|current|resistance|"
    r"calculate|compute|determine the value|"
    r"hertz|joule|newton|pascal|tesla|coulomb)\b",
    re.IGNORECASE,
)


def system_prompt_for(prompt: str) -> Optional[str]:
    """Return a task-family-tailored system prompt, or ``None``.

    Selection priority (first match wins):
      1. Code generation prompts → ``_SP_CODE``
      2. Open-ended math problems → ``_SP_MATH_OPEN``
      3. Multiple-choice with quantitative-science signals → ``_SP_MC_REASONING``
      4. Other multiple-choice → ``_SP_MC_GENERAL``
      5. Everything else → ``None`` (caller sends no system message)

    The selector is pure: it inspects only ``prompt`` text and returns one
    of a small set of static strings. There is no per-query state.
    """
    if not prompt:
        return None

    # Restrict pattern scans to the first ~600 chars so very long context-bearing
    # prompts (NarrativeQA, PubMedQA) don't trigger reasoning detectors via
    # keywords inside the passage.
    head = prompt[:600]

    if _RE_SP_CODE_PROMPT.search(head):
        return _SP_CODE

    if _RE_SP_MATH_OPEN_PROMPT.search(head):
        return _SP_MATH_OPEN

    if is_multiple_choice(prompt):
        # Look at the question body (after the boilerplate) for quantitative cues.
        if _RE_SP_MC_REASONING_KW.search(prompt):
            return _SP_MC_REASONING
        return _SP_MC_GENERAL

    return None
