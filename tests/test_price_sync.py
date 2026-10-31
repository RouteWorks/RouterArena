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


@pytest.fixture
def setup(tmp_path, monkeypatch):
    profiles = tmp_path / "model_profiles.yaml"
    profiles.write_text(PROFILES, encoding="utf-8")
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS), encoding="utf-8")
    monkeypatch.setattr(build_model_cost, "PROFILES_PATH", profiles)
    monkeypatch.setattr(build_model_cost, "COST_PATH", tmp_path / "model_cost.json")
    monkeypatch.setattr(sync_openrouter, "SNAPSHOT_DIR", tmp_path / "snapshots")
    return tmp_path, profiles, models


def test_prices_convert_to_per_million_and_skip_variable_prices(setup):
    _, _, models = setup
    prices = sync_openrouter.load_openrouter_prices(str(models))
    assert prices["google/model-b"] == (0.825, 4.125)
    assert "openrouter/auto" not in prices


def test_plan_resolves_by_explicit_id_and_respects_sync_false(setup):
    _, profiles, models = setup
    loaded = build_model_cost.load_profiles(profiles)
    changes, resolved, unresolved = sync_openrouter.plan_updates(
        loaded, sync_openrouter.load_openrouter_prices(str(models))
    )
    assert resolved == {"vendor/model-a": "vendor/model-a", "model-b": "google/model-b"}
    assert [c["model"] for c in changes] == ["model-b"]  # model-a is unchanged
    assert unresolved == ["local-only"]  # "pinned" is skipped, not unresolved


def test_run_updates_only_changed_lines_and_writes_snapshot(setup):
    tmp_path, profiles, models = setup
    assert (
        sync_openrouter.main(["--models-json", str(models), "--date", "2026-10-30"])
        == 0
    )

    text = profiles.read_text(encoding="utf-8")
    assert text.startswith("# A comment that must survive.")
    assert 'note: "keep me"' in text
    data = yaml.safe_load(text)["models"]
    assert data["model-b"]["price"] == {"input": 0.825, "output": 4.125}
    assert data["model-b"]["as_of"] == "2026-10-30"
    assert data["model-b"]["source"] == "https://openrouter.ai/google/model-b"
    assert (
        data["vendor/model-a"]["as_of"] == "2026-09-27"
    )  # unchanged profile untouched
    assert data["pinned"]["price"] == {"input": 9.0, "output": 9.0}

    snapshot = json.loads((tmp_path / "snapshots" / "2026-10-30.json").read_text())
    assert set(snapshot["prices"]) == {"vendor/model-a", "google/model-b"}

    table = json.loads((tmp_path / "model_cost.json").read_text())
    assert table["model-b"]["output_token_price_per_million"] == 4.125


def test_dry_run_writes_nothing(setup):
    tmp_path, profiles, models = setup
    before = profiles.read_text(encoding="utf-8")
    sync_openrouter.main(["--models-json", str(models), "--dry-run"])
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
    data = json.loads(models.read_text())
    data["data"][1]["pricing"] = {"prompt": "0.0000005", "completion": "0.000003"}
    models.write_text(json.dumps(data))
    sync_openrouter.main(["--models-json", str(models), "--date", "2026-10-30"])
    assert not (tmp_path / "snapshots").exists()
    sync_openrouter.main(
        ["--models-json", str(models), "--date", "2026-10-30", "--always-snapshot"]
    )
    assert (tmp_path / "snapshots" / "2026-10-30.json").exists()
