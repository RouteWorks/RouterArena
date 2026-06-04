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
