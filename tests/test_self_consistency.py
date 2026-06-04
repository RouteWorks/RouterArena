# SPDX-FileCopyrightText: Copyright (c) 2026 Yali Pollak
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/self_consistency.py``.

Verify the Tier 1A primitives behave correctly on canonical RouterArena
prompt shapes and on adversarial model outputs (partial agreement, ties,
empty samples, non-letter \\boxed{} contents).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from self_consistency import (  # noqa: E402
    SYSTEM_PROMPT_VERSION,
    extract_boxed_answer,
    extract_mc_letter,
    is_multiple_choice,
    majority_vote,
    system_prompt_for,
)


# ── is_multiple_choice ────────────────────────────────────────────────────────


class TestIsMultipleChoice:
    def test_canonical_mc_prompt(self):
        prompt = (
            "Please read the following multiple-choice questions and "
            "provide the most likely correct answer based on the options "
            "given.\n\nContext: None\n\nQuestion: What is 2+2?\n\nA. 3\nB. 4"
        )
        assert is_multiple_choice(prompt) is True

    def test_math_prompt_is_not_mc(self):
        prompt = "Please solve the following mathematical problem step by step."
        assert is_multiple_choice(prompt) is False

    def test_code_prompt_is_not_mc(self):
        prompt = "Generate an executable Python function from the given prompt."
        assert is_multiple_choice(prompt) is False

    def test_empty_prompt(self):
        assert is_multiple_choice("") is False

    def test_case_insensitive(self):
        # Some datasets capitalize "Multiple-Choice"
        prompt = "Please read the following Multiple-Choice Questions and..."
        assert is_multiple_choice(prompt) is True

    def test_mc_signature_only_in_head(self):
        # We only look at the first 300 chars to keep detection cheap;
        # an MC signature buried deep in a non-MC prompt does NOT count.
        prompt = (
            "Please solve this math problem step by step."
            + "x" * 500
            + " multiple-choice question"
        )
        assert is_multiple_choice(prompt) is False


# ── extract_boxed_answer ──────────────────────────────────────────────────────


class TestExtractBoxedAnswer:
    def test_single_letter(self):
        assert extract_boxed_answer("The correct answer is \\boxed{F}.") == "F"

    def test_letter_with_whitespace(self):
        assert extract_boxed_answer("\\boxed{ C }") == "C"

    def test_no_boxed(self):
        assert extract_boxed_answer("The answer is F") is None

    def test_empty(self):
        assert extract_boxed_answer("") is None

    def test_non_letter_content(self):
        # Math answers like \boxed{42} are not MC letters
        assert extract_boxed_answer("\\boxed{42}") is None

    def test_first_match_wins(self):
        # If a model emits multiple boxed answers, take the first
        text = "First I thought \\boxed{B} but then \\boxed{C}."
        assert extract_boxed_answer(text) == "B"

    def test_lowercase_letter_rejected(self):
        # MC options are always uppercase
        assert extract_boxed_answer("\\boxed{a}") is None


# ── extract_mc_letter ─────────────────────────────────────────────────────────


class TestExtractMcLetter:
    def test_boxed_takes_priority(self):
        text = "The answer is C, so \\boxed{D}"
        assert extract_mc_letter(text) == "D"

    def test_answer_is_letter(self):
        assert extract_mc_letter("The answer is B.") == "B"

    def test_answer_colon_letter(self):
        assert extract_mc_letter("Answer: A") == "A"

    def test_choice_is_letter(self):
        assert extract_mc_letter("The correct choice is (E).") == "E"

    def test_no_pattern_match(self):
        assert extract_mc_letter("I don't know.") is None

    def test_empty_text(self):
        assert extract_mc_letter("") is None


# ── majority_vote ─────────────────────────────────────────────────────────────


class TestMajorityVote:
    def test_clear_majority_three(self):
        assert majority_vote(["A", "A", "B"]) == "A"

    def test_unanimous(self):
        assert majority_vote(["C", "C", "C"]) == "C"

    def test_tie_returns_none(self):
        # 3 samples, all different — no consensus
        assert majority_vote(["A", "B", "C"]) is None

    def test_tie_at_top_returns_none(self):
        # Even with 4 samples, a 2-2 split has no winner
        assert majority_vote(["A", "A", "B", "B"]) is None

    def test_none_entries_excluded(self):
        # Two samples extracted "A", one failed — A still wins
        assert majority_vote([None, "A", "A"]) == "A"

    def test_all_none(self):
        assert majority_vote([None, None, None]) is None

    def test_min_agreement_enforced(self):
        # Single A with two Nones isn't enough at default min_agreement=2
        assert majority_vote(["A", None, None]) is None

    def test_min_agreement_can_be_relaxed(self):
        assert majority_vote(["A", None, None], min_agreement=1) == "A"

    def test_empty_list(self):
        assert majority_vote([]) is None

    def test_winner_must_beat_runner_up(self):
        # 3 votes for A, 3 votes for B → no consensus
        assert majority_vote(["A", "A", "A", "B", "B", "B"]) is None

    def test_clear_majority_beats_runner_up(self):
        # 4 A, 2 B → A wins
        assert majority_vote(["A", "A", "A", "A", "B", "B"]) == "A"


# ── integration: full pipeline behaves on realistic outputs ───────────────────


class TestSelfConsistencyPipeline:
    def test_three_samples_with_format_drift(self):
        # Sample 1: clean boxed
        # Sample 2: answered conversationally
        # Sample 3: boxed
        # All three agree the answer is F
        samples = [
            "The correct answer is \\boxed{F}.",
            "After analysis, the answer is F.",
            "Therefore \\boxed{F}.",
        ]
        letters = [extract_mc_letter(s) for s in samples]
        assert letters == ["F", "F", "F"]
        assert majority_vote(letters) == "F"

    def test_one_dissenter_does_not_block_majority(self):
        samples = [
            "\\boxed{D}",
            "\\boxed{D}",
            "I think the answer is E.",
        ]
        letters = [extract_mc_letter(s) for s in samples]
        assert letters == ["D", "D", "E"]
        assert majority_vote(letters) == "D"

    def test_all_refusals_return_none(self):
        samples = [
            "I'm not sure.",
            "This is ambiguous.",
            "Could be several options.",
        ]
        letters = [extract_mc_letter(s) for s in samples]
        assert letters == [None, None, None]
        assert majority_vote(letters) is None


# ── system_prompt_for ─────────────────────────────────────────────────────────


class TestSystemPromptFor:
    """Tier 1B task-family system prompt selector."""

    def test_version_is_set(self):
        assert SYSTEM_PROMPT_VERSION  # truthy, non-empty string

    def test_empty_prompt_returns_none(self):
        assert system_prompt_for("") is None
        assert system_prompt_for(None) is None  # type: ignore[arg-type]

    def test_code_generation_prompt(self):
        prompt = (
            "Generate an executable Python function from the given prompt. "
            "Return the function body."
        )
        sp = system_prompt_for(prompt)
        assert sp is not None
        assert "code" in sp.lower()
        # Should not be the MC general prompt
        assert "multiple-choice" not in sp.lower()

    def test_open_math_prompt(self):
        prompt = (
            "Solve the following mathematical problem step by step. "
            "Provide the final answer in \\boxed{value}."
        )
        sp = system_prompt_for(prompt)
        assert sp is not None
        assert "math" in sp.lower()

    def test_general_mc_prompt(self):
        # Non-quantitative MC — a trivia-style question.
        prompt = (
            "Please read the following multiple-choice questions and provide "
            "the most likely correct answer based on the options given.\n\n"
            "Question: Who wrote Hamlet?\n\nA. Dickens\nB. Shakespeare\n"
            "C. Twain\nD. Austen\n\nProvide the correct letter choice in "
            "\\boxed{X}."
        )
        sp = system_prompt_for(prompt)
        assert sp is not None
        assert "multiple-choice" in sp.lower()
        # General MC should not push "step by step" specifically
        assert "\\boxed{x}" in sp.lower() or "\\boxed" in sp.lower()

    def test_quantitative_mc_prompt(self):
        # MC question with physics keywords → reasoning variant
        prompt = (
            "Please read the following multiple-choice questions and provide "
            "the most likely correct answer.\n\nQuestion: A particle moves "
            "with constant acceleration of 2 m/s^2. What is its velocity "
            "after 5 seconds, starting from rest?\n\nA. 5\nB. 10\nC. 15\n"
            "D. 20\n\nProvide the correct letter choice in \\boxed{X}."
        )
        sp = system_prompt_for(prompt)
        assert sp is not None
        # Reasoning prompt instructs step-by-step thinking
        assert "step by step" in sp.lower()

    def test_quantitative_mc_does_not_override_open_math(self):
        # If the prompt looks like open-ended math, return the math prompt
        # even if quantitative keywords are present.
        prompt = (
            "Solve the following math problem step by step. "
            "A particle's velocity is 10 m/s. Find its kinetic energy. "
            "Final answer in \\boxed{value}."
        )
        sp = system_prompt_for(prompt)
        assert sp is not None
        # Math prompt mentions "numerical answer" not "multiple-choice"
        assert "math" in sp.lower()
        assert "multiple-choice" not in sp.lower()

    def test_non_mc_non_math_prompt_returns_none(self):
        # Free-form translation prompt that isn't math/code/MC.
        prompt = "Translate the following sentence to French: 'Good morning.'"
        sp = system_prompt_for(prompt)
        assert sp is None

    def test_long_context_does_not_misfire_reasoning(self):
        # A general MC prompt with quantitative keywords buried deep in the
        # context body should still get the general MC prompt (we scan the
        # whole prompt for MC reasoning, so this verifies the priority rules
        # behave: it WILL match reasoning if keywords are present anywhere).
        prompt = (
            "Please read the following multiple-choice questions and provide "
            "the most likely correct answer.\n\nContext: "
            + "x" * 800
            + "\n\nQuestion: Who wrote Hamlet?\n\nA. Dickens\nB. Shakespeare"
        )
        sp = system_prompt_for(prompt)
        # No quantitative keywords anywhere → general MC
        assert sp is not None
        assert "multiple-choice" in sp.lower()

    def test_returns_same_string_for_repeat_calls(self):
        # Same prompt → same system prompt (deterministic, pure function)
        prompt = (
            "Please read the following multiple-choice questions and provide "
            "the most likely correct answer.\n\nQuestion: What is 2+2?"
        )
        assert system_prompt_for(prompt) == system_prompt_for(prompt)
