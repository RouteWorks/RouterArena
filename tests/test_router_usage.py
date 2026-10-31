# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Tests for router-call costs (global_utils/router_usage.py and the evaluator)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "llm_evaluation"))

from global_utils.router_usage import (  # noqa: E402
    router_usage_cost,
    router_usage_models,
    validate_router_usage,
)
from evaluate_models import ModelEvaluator  # noqa: E402
from run import compute_row_cost  # noqa: E402

CALL = {"model": "gpt-5-nano", "input_tokens": 1000, "output_tokens": 500}


def fake_price(model, usage):
    return usage["input_tokens"] * 1e-6 + usage["output_tokens"] * 2e-6


@pytest.fixture(scope="module")
def evaluator():
    return ModelEvaluator(cached_results_dir=str(REPO_ROOT / "cached_results"))


def test_valid_record_has_no_errors():
    assert validate_router_usage([CALL, {**CALL, "purpose": "routing"}]) == []
    assert validate_router_usage([]) == []


@pytest.mark.parametrize(
    "value, problem",
    [
        ({"model": "x"}, "must be a list"),
        (["x"], "must be an object"),
        ([{**CALL, "model": ""}], "model must be a non-empty string"),
        ([{**CALL, "input_tokens": -1}], "input_tokens must be an integer"),
        ([{**CALL, "output_tokens": 1.5}], "output_tokens must be an integer"),
        ([{**CALL, "output_tokens": True}], "output_tokens must be an integer"),
        ([{**CALL, "purpose": 3}], "purpose must be a string"),
    ],
)
def test_invalid_records_are_reported(value, problem):
    assert any(problem in e for e in validate_router_usage(value))


def test_cost_sums_every_call_and_missing_field_is_free():
    assert router_usage_cost(None, fake_price) == 0.0
    assert router_usage_cost([CALL, CALL], fake_price) == pytest.approx(2 * 0.002)
    assert router_usage_models([CALL, "x"]) == ["gpt-5-nano"]


def test_malformed_record_is_never_priced_as_free():
    with pytest.raises(ValueError):
        router_usage_cost([{"model": "gpt-5-nano"}], fake_price)


def test_row_cost_adds_router_calls_at_their_own_price(evaluator):
    row = {
        "prediction": "gpt-5-mini",
        "generated_result": {
            "token_usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        },
        "router_usage": [
            {"model": "gpt-5-nano", "input_tokens": 1_000_000, "output_tokens": 0}
        ],
    }
    answer, router = compute_row_cost(row, evaluator, "gpt-5-mini")
    assert answer == pytest.approx(0.25 + 2.0)  # gpt-5-mini: $0.25 / $2.00 per 1M
    assert router == pytest.approx(0.05)  # gpt-5-nano input: $0.05 per 1M


def test_unpriced_router_model_pays_the_maximum_price(evaluator):
    row = {
        "prediction": "gpt-5-mini",
        "generated_result": {"token_usage": {"input_tokens": 0, "output_tokens": 0}},
        "router_usage": [
            {"model": "no-such-model", "input_tokens": 1_000_000, "output_tokens": 0}
        ],
    }
    _, router = compute_row_cost(row, evaluator, "gpt-5-mini")
    assert router == pytest.approx(
        evaluator.max_price_info()["input_token_price_per_million"]
    )
