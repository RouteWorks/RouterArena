# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenRouter price sync (scripts/pricing/sync_openrouter.py)."""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pricing"))

import build_model_cost  # noqa: E402
import sync_openrouter  # noqa: E402

PROFILES = """\
# A comment that must survive.
models:
  "vendor/model-a":
    price: {input: 1.0, output: 2.0}
    aliases: ["model-a"]
    as_of: "2026-09-27"
    source: null
  "model-b":
    price: {input: 0.5, output: 3}
    openrouter_id: "google/model-b"
    as_of: "2026-09-27"
    source: null
    note: "keep me"
  "pinned":
    price: {input: 9.0, output: 9.0}
    sync: false
    as_of: "2026-09-27"
  "local-only":
    price: {input: 0.1, output: 0.1}
    as_of: "2026-09-27"
"""

MODELS = {
    "data": [
        {
            "id": "vendor/model-a",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        {
            "id": "google/model-b",
            "pricing": {"prompt": "0.000000825", "completion": "0.000004125"},
        },
        {"id": "pinned", "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


def endpoint(
    provider, prompt, completion, discount=0, tier=None, quant="unknown", status=0
):
    return {
        "provider_name": provider,
        "service_tier": tier,
        "quantization": quant,
        "status": status,
        "pricing": {
            "prompt": str(prompt / 1e6),
            "completion": str(completion / 1e6),
            "discount": discount,
        },
    }


# google/model-b is sold by its lab at a 50% introductory discount (standard 1.5 / 7.5);
# the model-level list price above is one regional endpoint's discounted price.
ENDPOINTS = {
    "vendor/model-a": [endpoint("SomeHost", 1.0, 2.0)],
    "google/model-b": [
        endpoint("Google AI Studio", 0.375, 1.875, 0.5, tier="flex"),
        endpoint("Google AI Studio", 0.75, 3.75, 0.5),
        endpoint("Google", 0.825, 4.125, 0.5),
        endpoint("Google", 1.35, 6.75, 0.5, tier="priority"),
    ],
}


def write_endpoints(directory, endpoints):
    directory.mkdir(exist_ok=True)
    for or_id, eps in endpoints.items():
        path = directory / f"{or_id.replace('/', '__')}.json"
        path.write_text(json.dumps({"data": {"endpoints": eps}}), encoding="utf-8")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    profiles = tmp_path / "model_profiles.yaml"
    profiles.write_text(PROFILES, encoding="utf-8")
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS), encoding="utf-8")
    monkeypatch.setattr(build_model_cost, "PROFILES_PATH", profiles)
    monkeypatch.setattr(build_model_cost, "COST_PATH", tmp_path / "model_cost.json")
    monkeypatch.setattr(sync_openrouter, "SNAPSHOT_DIR", tmp_path / "snapshots")
    write_endpoints(tmp_path / "endpoints", ENDPOINTS)
    return tmp_path, profiles, models


def run(tmp_path, models, *extra):
    return sync_openrouter.main(
        ["--models-json", str(models), "--endpoints-dir", str(tmp_path / "endpoints")]
        + list(extra)
    )


def test_prices_convert_to_per_million_and_skip_variable_prices(setup):
    _, _, models = setup
    prices = sync_openrouter.load_openrouter_prices(str(models))
    assert prices["google/model-b"] == (0.825, 4.125)
    assert "openrouter/auto" not in prices


def test_plan_resolves_by_explicit_id_and_respects_sync_false(setup):
    tmp_path, profiles, models = setup
    loaded = build_model_cost.load_profiles(profiles)
    changes, resolved, unresolved, no_data = sync_openrouter.plan_updates(
        loaded,
        sync_openrouter.load_openrouter_prices(str(models)),
        sync_openrouter.endpoints_loader(str(tmp_path / "endpoints")),
    )
    assert {k: v["openrouter_id"] for k, v in resolved.items()} == {
        "vendor/model-a": "vendor/model-a",
        "model-b": "google/model-b",
    }
    assert [c["model"] for c in changes] == ["model-b"]  # model-a is unchanged
    assert unresolved == ["local-only"]  # "pinned" is skipped, not unresolved
    assert no_data == []


def test_standard_price_is_the_labs_own_without_the_promotion():
    found = sync_openrouter.standard_price(
        "google/model-b", ENDPOINTS["google/model-b"]
    )
    assert found["price"] == (1.5, 7.5)
    assert found["discount_removed"] == 0.5
    assert "Google AI Studio" in found["basis"]


def test_standard_price_without_the_lab_is_the_median_of_full_precision_hosts():
    eps = [
        endpoint("HostA", 0.14, 0.28, quant="fp8"),
        endpoint("HostB", 0.0399, 0.0798, 0.715, quant="fp8"),  # promo of 0.14 / 0.28
        endpoint("HostC", 0.2, 0.5),
        endpoint("Cheap4bit", 0.01, 0.02, quant="fp4"),
        endpoint("Down", 0.01, 0.01, status=-5),
    ]
    found = sync_openrouter.standard_price("deepseek/model-x", eps)
    assert found["price"] == (0.14, 0.28)
    assert found["basis"] == "median of 3 host(s)"


def test_only_low_bit_hosts_still_give_a_price():
    eps = [endpoint("A", 0.1, 0.2, quant="fp4"), endpoint("B", 0.3, 0.4, quant="int4")]
    assert sync_openrouter.standard_price("x/y", eps)["price"] == (0.2, 0.3)


def test_no_standard_tier_endpoint_means_no_price():
    eps = [endpoint("OpenAI", 2, 10, tier="flex")]
    assert sync_openrouter.standard_price("openai/m", eps) is None
    assert sync_openrouter.standard_price("openai/m", None) is None


def test_unreadable_endpoints_keep_the_price(setup):
    tmp_path, profiles, models = setup
    (tmp_path / "endpoints" / "google__model-b.json").unlink()
    run(tmp_path, models, "--date", "2026-10-30")
    data = yaml.safe_load(profiles.read_text(encoding="utf-8"))["models"]
    assert data["model-b"]["price"] == {"input": 0.5, "output": 3}


def test_run_updates_only_changed_lines_and_writes_snapshot(setup):
    tmp_path, profiles, models = setup
    assert run(tmp_path, models, "--date", "2026-10-30") == 0

    text = profiles.read_text(encoding="utf-8")
    assert text.startswith("# A comment that must survive.")
    assert 'note: "keep me"' in text
    data = yaml.safe_load(text)["models"]
    assert data["model-b"]["price"] == {"input": 1.5, "output": 7.5}
    assert data["model-b"]["as_of"] == "2026-10-30"
    assert data["model-b"]["source"] == "https://openrouter.ai/google/model-b"
    assert (
        data["vendor/model-a"]["as_of"] == "2026-09-27"
    )  # unchanged profile untouched
    assert data["pinned"]["price"] == {"input": 9.0, "output": 9.0}

    snapshot = json.loads((tmp_path / "snapshots" / "2026-10-30.json").read_text())
    assert set(snapshot["prices"]) == {"vendor/model-a", "google/model-b"}
    assert snapshot["prices"]["google/model-b"]["discount_removed"] == 0.5

    table = json.loads((tmp_path / "model_cost.json").read_text())
    assert table["model-b"]["output_token_price_per_million"] == 7.5


def test_dry_run_writes_nothing(setup):
    tmp_path, profiles, models = setup
    before = profiles.read_text(encoding="utf-8")
    run(tmp_path, models, "--dry-run")
    assert profiles.read_text(encoding="utf-8") == before
    assert not (tmp_path / "snapshots").exists()


def test_missing_source_and_as_of_lines_are_added(setup):
    _, profiles, _ = setup
    change = {
        "model": "local-only",
        "openrouter_id": "x/local-only",
        "old": (0.1, 0.1),
        "new": (0.2, 0.3),
    }
    text = sync_openrouter.apply_updates(
        profiles.read_text(encoding="utf-8"), [change], "2026-10-30"
    )
    block = yaml.safe_load(text)["models"]["local-only"]
    assert block["price"] == {"input": 0.2, "output": 0.3}
    assert block["source"] == "https://openrouter.ai/x/local-only"


def test_no_change_writes_no_snapshot_unless_asked(setup):
    tmp_path, profiles, models = setup
    write_endpoints(
        tmp_path / "endpoints", {"google/model-b": [endpoint("Google", 0.5, 3)]}
    )
    run(tmp_path, models, "--date", "2026-10-30")
    assert not (tmp_path / "snapshots").exists()
    run(tmp_path, models, "--date", "2026-10-30", "--always-snapshot")
    assert (tmp_path / "snapshots" / "2026-10-30.json").exists()
