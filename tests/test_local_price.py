# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local-model price rule (scripts/pricing/local_price.py)."""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pricing"))

import build_model_cost  # noqa: E402
import local_price  # noqa: E402

RULE = {
    "size_curve": {
        "input": {"coef": 0.1, "exponent": 0.5},
        "output": {"coef": 0.4, "exponent": 0.5},
    },
    "quantization": [{"min_bits": 8, "factor": 1.0}, {"min_bits": 0, "factor": 0.8}],
}

PROFILES = """\
models:
  "vendor/hosted":
    price: {input: 0.5, output: 2}
    aliases: ["hosted"]
  "me/hosted-q4":
    price: {input: 1, output: 1}
    local: {base: "hosted", bits: 4}
    sync: false
  "me/tiny":
    price: {input: 1, output: 1}
    local: {active_params_b: 4, bits: 16}
    sync: false
"""


def test_quantization_factor_by_bits():
    assert local_price.quantization_factor(16, RULE) == 1.0
    assert local_price.quantization_factor(8, RULE) == 1.0
    assert local_price.quantization_factor(4, RULE) == 0.8
    assert local_price.quantization_factor(2, RULE) == 0.8


def test_listed_weights_take_the_listed_price_times_the_factor():
    profiles = yaml.safe_load(PROFILES)["models"]
    price, basis = local_price.local_price(
        {"base": "hosted", "bits": 4}, profiles, RULE
    )
    assert price == (0.4, 1.6)
    assert "vendor/hosted" in basis and "x0.8" in basis


def test_unlisted_weights_take_the_size_curve():
    price, basis = local_price.local_price({"active_params_b": 16, "bits": 8}, {}, RULE)
    assert price == (0.4, 1.6)  # 0.1 * 16 ** 0.5, 0.4 * 16 ** 0.5
    assert "16B" in basis


@pytest.mark.parametrize(
    "spec",
    [
        {"base": "missing", "bits": 4},
        {"base": "me/tiny", "bits": 4},  # a local model can't be another's base
        {"active_params_b": 4},  # no bits
        {"bits": 4},  # neither base nor size
    ],
)
def test_bad_specs_are_rejected(spec):
    profiles = yaml.safe_load(PROFILES)["models"]
    with pytest.raises(ValueError):
        local_price.local_price(spec, profiles, RULE)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    profiles = tmp_path / "model_profiles.yaml"
    profiles.write_text(PROFILES, encoding="utf-8")
    rule = tmp_path / "local_pricing.yaml"
    rule.write_text(yaml.safe_dump(RULE), encoding="utf-8")
    monkeypatch.setattr(build_model_cost, "PROFILES_PATH", profiles)
    monkeypatch.setattr(build_model_cost, "COST_PATH", tmp_path / "model_cost.json")
    monkeypatch.setattr(local_price, "RULE_PATH", rule)
    return tmp_path, profiles


def test_check_fails_until_write_updates_the_local_profiles(repo):
    tmp_path, profiles = repo
    assert local_price.main(["--check"]) == 1
    assert local_price.main(["--write"]) == 0
    data = yaml.safe_load(profiles.read_text(encoding="utf-8"))["models"]
    assert data["me/hosted-q4"]["price"] == {"input": 0.4, "output": 1.6}
    assert data["me/tiny"]["price"] == {"input": 0.2, "output": 0.8}
    assert data["vendor/hosted"]["price"] == {"input": 0.5, "output": 2}
    table = json.loads((tmp_path / "model_cost.json").read_text())
    assert table["me/hosted-q4"]["input_token_price_per_million"] == 0.4
    assert local_price.main(["--check"]) == 0


def test_the_repository_rule_and_profiles_are_consistent():
    rule = local_price.load_rule()
    assert local_price.quantization_factor(16, rule) == 1.0
    profiles = build_model_cost.load_profiles()
    assert local_price.plan(profiles, rule) == []
