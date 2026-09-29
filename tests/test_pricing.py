# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Tests for model price profiles and the evaluator's price lookup."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "llm_evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pricing"))

import build_model_cost  # noqa: E402
from evaluate_models import ModelEvaluator  # noqa: E402

USAGE = {
    "input_tokens": 1_000_000,
    "output_tokens": 1_000_000,
    "total_tokens": 2_000_000,
}


@pytest.fixture(scope="module")
def evaluator():
    return ModelEvaluator(cached_results_dir=str(REPO_ROOT / "cached_results"))


def test_model_cost_json_is_generated_from_profiles():
    profiles = build_model_cost.load_profiles()
    expected = build_model_cost.render(build_model_cost.build_cost_table(profiles))
    actual = (REPO_ROOT / "model_cost" / "model_cost.json").read_text(encoding="utf-8")
    assert actual == expected, "run: python scripts/pricing/build_model_cost.py"


def test_every_name_is_unique_across_ids_and_aliases(tmp_path):
    bad = tmp_path / "profiles.yaml"
    bad.write_text(
        "models:\n"
        "  a: {price: {input: 1, output: 2}, aliases: [b]}\n"
        "  b: {price: {input: 1, output: 2}}\n"
    )
    with pytest.raises(ValueError, match="used by both"):
        build_model_cost.load_profiles(bad)


def test_negative_or_missing_price_is_rejected(tmp_path):
    bad = tmp_path / "profiles.yaml"
    bad.write_text("models:\n  a: {price: {input: -1}}\n")
    with pytest.raises(ValueError, match="price.input"):
        build_model_cost.load_profiles(bad)


def test_alias_is_charged_the_profile_price(evaluator):
    # Lynkr names this model "openai/gpt-oss-120b"; its profile id is "gpt-oss-120b".
    assert evaluator.has_price("openai/gpt-oss-120b")
    assert evaluator.calculate_inference_cost(
        "openai/gpt-oss-120b", USAGE
    ) == pytest.approx(evaluator.calculate_inference_cost("gpt-oss-120b", USAGE))


def test_no_substring_matching(evaluator):
    # Used to match "gpt-oss-120b" because that key is a substring of the name.
    assert not evaluator.has_price("gpt-oss-120b-some-finetune")
    assert not evaluator.has_price("gpt")


def test_unmatched_model_is_charged_the_maximum_price(evaluator):
    max_price = evaluator.max_price_info()
    table = json.loads((REPO_ROOT / "model_cost" / "model_cost.json").read_text())
    assert max_price["input_token_price_per_million"] == max(
        v["input_token_price_per_million"] for v in table.values()
    )
    assert max_price["output_token_price_per_million"] == max(
        v["output_token_price_per_million"] for v in table.values()
    )
    cost = evaluator.calculate_inference_cost("not-a-registered-model", USAGE)
    assert cost == pytest.approx(
        max_price["input_token_price_per_million"]
        + max_price["output_token_price_per_million"]
    )
    assert evaluator.unpriced_models.get("not-a-registered-model", 0) >= 1


def test_unmatched_model_is_never_cheaper_than_a_registered_one(evaluator):
    table = json.loads((REPO_ROOT / "model_cost" / "model_cost.json").read_text())
    unmatched = evaluator.calculate_inference_cost("not-a-registered-model", USAGE)
    assert all(
        evaluator.calculate_inference_cost(name, USAGE) <= unmatched for name in table
    )
