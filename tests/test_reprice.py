# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Tests for the re-pricing tool (scripts/pricing/reprice.py)."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "llm_evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pricing"))

import reprice  # noqa: E402
from evaluate_models import ModelEvaluator  # noqa: E402
from run import compute_router_metrics, router_metrics  # noqa: E402

PRICES = {
    "cheap-model": {
        "input_token_price_per_million": 1.0,
        "output_token_price_per_million": 2.0,
    },
    "pricey-model": {
        "input_token_price_per_million": 10.0,
        "output_token_price_per_million": 20.0,
    },
}


@pytest.fixture
def evaluator():
    ev = ModelEvaluator(cached_results_dir=str(REPO_ROOT / "cached_results"))
    ev.cost_config = dict(PRICES)
    return ev


def row(model, accuracy, cost, input_tokens=1_000_000, output_tokens=1_000_000):
    return {
        "global index": f"q-{model}-{accuracy}",
        "prediction": model,
        "accuracy": accuracy,
        "cost": cost,
        "generated_result": {
            "success": True,
            "generated_answer": "x",
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    }


def test_rows_are_recosted_at_the_current_prices(evaluator):
    rows = [row("cheap-model", 1.0, 99.0), row("pricey-model", 0.0, 99.0)]
    assert reprice.reprice_rows(rows, evaluator) == 2
    assert rows[0]["cost"] == pytest.approx(3.0)
    assert rows[1]["cost"] == pytest.approx(30.0)
    assert rows[0]["accuracy"] == 1.0 and rows[1]["accuracy"] == 0.0


def test_rows_the_evaluator_did_not_cost_are_left_alone(evaluator):
    rows = [row("cheap-model", 1.0, None)]
    del rows[0]["cost"]
    assert reprice.reprice_rows(rows, evaluator) == 0
    assert "cost" not in rows[0]


def test_served_model_is_billed_when_it_has_a_price(evaluator):
    rows = [row("cheap-model", 1.0, 0.5)]
    rows[0]["generated_result"]["model_used"] = "pricey-model"
    reprice.reprice_rows(rows, evaluator)
    assert rows[0]["cost"] == pytest.approx(30.0)


def test_router_calls_are_charged(evaluator):
    rows = [row("cheap-model", 1.0, 3.0)]
    rows[0]["router_usage"] = [
        {"model": "pricey-model", "input_tokens": 100_000, "output_tokens": 0}
    ]
    reprice.reprice_rows(rows, evaluator)
    assert rows[0]["router_cost"] == pytest.approx(1.0)
    assert rows[0]["cost"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "style",
    [
        dict(indent=2, ensure_ascii=False),
        dict(indent=4, ensure_ascii=True),
        dict(indent=None, separators=(", ", ": ")),
        dict(indent=None, separators=(",", ":")),
    ],
)
@pytest.mark.parametrize("newline", ["", "\n"])
def test_dump_keeps_the_file_style(style, newline):
    data = [{"a": "é", "cost": 1.5}, {"a": "b", "cost": 2.0}]
    original = json.dumps(data, **style) + newline
    data[0]["cost"] = 0.25
    assert reprice.dump_like(original, data) == json.dumps(data, **style) + newline


README = """intro
| Rank | Router | Affiliation | Arena | Accuracy | Cost/1K | Opt Sel | Opt Cost | Opt Acc | Latency | Robustness |
|------|---|---|---|---|---|---|---|---|---|---|
| 🥇 | [Alpha]() | a | 77.00 | 80.00 | $0.50 | 10.00 | 20.00 | 90.00 | — | 70.00 |
| 🥈 | [Beta Router](https://x) | b | 76.00 | 79.00 | $0.40 | — | — | — | — | 60.00 |
| 🥉 | [Gamma]() | c | 75.00 | 78.00 | $0.30 | — | — | — | — | 50.00 |

after"""


def test_readme_rows_update_and_rerank():
    metrics = {
        "Alpha": {
            "arena_score": 0.7412,
            "accuracy": 0.80,
            "avg_cost_per_1000": 2.5,
            "optimality": {"opt_sel": 0.5, "opt_cost": 0.25, "opt_acc": 0.9},
        },
        "Beta Router": {
            "arena_score": 0.7700,
            "accuracy": 0.79,
            "avg_cost_per_1000": 0.3,
        },
    }
    text, updated = reprice.update_readme(README, metrics)
    assert sorted(updated) == ["Alpha", "Beta Router"]
    rows = [
        line
        for line in text.split("\n")
        if line.startswith("| ") and "Rank" not in line
    ]
    assert (
        rows[0].startswith("| 🥇 | [Beta Router]")
        and "| 77.00 | 79.00 | $0.30 | — |" in rows[0]
    )
    assert (
        rows[1].startswith("| 🥈 | [Gamma]()")
        and "| 75.00 | 78.00 | $0.30 |" in rows[1]
    )
    assert rows[2].startswith("| 🥉 | [Alpha]()")
    assert "| 74.12 | 80.00 | $2.50 | 50.00 | 25.00 | 90.00 | — | 70.00 |" in rows[2]
    assert text.startswith("intro\n") and text.endswith("\nafter")


def test_router_metrics_match_metrics_json(tmp_path, monkeypatch):
    rows = [row("cheap-model", 1.0, 3.0), row("pricey-model", 0.0, 30.0)]
    rows.append(row("cheap-model", 1.0, 3.0))
    rows[-1]["generated_result"]["success"] = False  # abnormal: scored 0, no cost
    monkeypatch.chdir(tmp_path)
    compute_router_metrics(rows, "test-router")
    saved = json.loads((tmp_path / "metrics.json").read_text())
    metrics, details = router_metrics(rows, "test-router")
    assert saved == metrics
    assert metrics["abnormal_count"] == 1 and details["valid_cost_count"] == 2
    assert metrics["avg_cost_per_1000"] == pytest.approx(33.0 / 3 * 1000)
