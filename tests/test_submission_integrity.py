# SPDX-FileCopyrightText: Copyright (c) 2026 Yali Pollak
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/check_submission_integrity.py.

Each test verifies that a specific known-bad pattern is detected by the
integrity check, AND that legitimate patterns pass cleanly. The Lever #3
pattern (commit 401ad54) is reproduced in test_lever_3_pattern_is_detected
as a regression fixture.

Run:
    uv run pytest tests/test_submission_integrity.py -v
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest


# Make sibling scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_submission_integrity as cs  # noqa: E402


# ── Check 1: diff analysis ────────────────────────────────────────────────────


def _write_predictions(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries, indent=2))


def test_lever_3_pattern_is_detected(tmp_path: Path) -> None:
    """Reproduce commit 401ad54: 100% of reassignments target accuracy=0 queries.

    Baseline has 5 queries, 3 of which are wrong (accuracy=0).
    Current reassigns exactly those 3 to a new model.
    Expected: integrity check flags it as leakage.
    """
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    baseline = [
        {"global index": f"Q{i}", "prediction": "model_a", "accuracy": acc}
        for i, acc in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])
    ]
    current = [
        # Only the previously-wrong (acc=0) queries get reassigned
        {
            "global index": f"Q{i}",
            "prediction": ("model_b" if acc == 0.0 else "model_a"),
        }
        for i, acc in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])
    ]
    _write_predictions(baseline_path, baseline)
    _write_predictions(current_path, current)

    errors = cs.check_reassignments_vs_prior_accuracy(
        current_path, baseline_path, correlation_threshold=0.95
    )

    assert errors, (
        "Lever #3 pattern (100% reassignments target accuracy=0) should be flagged"
    )
    joined = "\n".join(errors)
    assert "LEAKAGE" in joined
    assert "3/3" in joined or "100%" in joined


def test_legitimate_reassignment_passes(tmp_path: Path) -> None:
    """A reassignment that targets queries regardless of prior accuracy is clean.

    Here we reassign the first 3 queries (mixed accuracy) — could be a
    legitimate per-dataset routing change.
    """
    baseline = [
        {"global index": f"Q{i}", "prediction": "model_a", "accuracy": acc}
        for i, acc in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])
    ]
    current = [
        {"global index": f"Q{i}", "prediction": ("model_b" if i < 3 else "model_a")}
        for i, acc in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])
    ]
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_predictions(baseline_path, baseline)
    _write_predictions(current_path, current)

    errors = cs.check_reassignments_vs_prior_accuracy(
        current_path, baseline_path, correlation_threshold=0.95
    )
    # 1 of 3 reassigned queries had accuracy=0 → 33% < 95% threshold → clean
    assert not errors, f"Mixed-accuracy reassignment should pass; got: {errors}"


def test_no_baseline_skips_check(tmp_path: Path) -> None:
    """First submission has no baseline — integrity check skips this branch."""
    current_path = tmp_path / "current.json"
    _write_predictions(current_path, [{"global index": "Q0", "prediction": "model_a"}])
    errors = cs.check_reassignments_vs_prior_accuracy(
        current_path, baseline_path=None, correlation_threshold=0.95
    )
    assert not errors


def test_no_reassignments_passes(tmp_path: Path) -> None:
    """If predictions are unchanged, no leakage possible."""
    entries = [
        {"global index": f"Q{i}", "prediction": "model_a", "accuracy": 0.0}
        for i in range(5)
    ]
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_predictions(baseline_path, entries)
    _write_predictions(current_path, entries)
    errors = cs.check_reassignments_vs_prior_accuracy(
        current_path, baseline_path, correlation_threshold=0.95
    )
    assert not errors


# ── Check 2: source-code scan ─────────────────────────────────────────────────


def test_detects_accuracy_to_prediction_pattern(tmp_path: Path) -> None:
    """The Lever #3 code pattern in scripts/ should be flagged.

    Reproduces the exact accuracy→prediction code that triggered the violation.
    """
    bad_script = tmp_path / "scripts" / "bad_lever_3.py"
    bad_script.parent.mkdir(parents=True, exist_ok=True)
    bad_script.write_text(
        textwrap.dedent("""
        import json
        preds = json.load(open('predictions.json'))
        for p in preds:
            if float(p.get('accuracy', 0) or 0) == 0.0:
                p['prediction'] = 'deepseek/deepseek-v3.2'
    """).strip()
    )

    errors = cs.check_source_code_for_leakage([tmp_path / "scripts"])
    assert errors, (
        "Lever #3 code pattern (accuracy check → prediction write) should be flagged"
    )
    assert any("LEAKAGE" in e for e in errors)
    assert any(str(bad_script) in e for e in errors)


def test_legitimate_inference_signal_passes(tmp_path: Path) -> None:
    """Code that routes based on inference failure (success=False) is OK.

    This is the 83a7425 pattern — legitimate. Should NOT be flagged.
    """
    good_script = tmp_path / "scripts" / "regen_failed.py"
    good_script.parent.mkdir(parents=True, exist_ok=True)
    good_script.write_text(
        textwrap.dedent("""
        import json
        preds = json.load(open('predictions.json'))
        for p in preds:
            gr = p.get('generated_result', {})
            if not gr.get('success', True):
                p['prediction'] = 'qwen/qwen3-235b-a22b-2507'
    """).strip()
    )

    errors = cs.check_source_code_for_leakage([tmp_path / "scripts"])
    assert not errors, (
        f"Inference-failure routing is legitimate but was flagged: {errors}"
    )


def test_accuracy_read_without_prediction_write_passes(tmp_path: Path) -> None:
    """Just reading accuracy for reporting (not routing) shouldn't trigger."""
    report_script = tmp_path / "scripts" / "report.py"
    report_script.parent.mkdir(parents=True, exist_ok=True)
    report_script.write_text(
        textwrap.dedent("""
        import json
        preds = json.load(open('predictions.json'))
        for p in preds:
            if float(p.get('accuracy', 0) or 0) == 0.0:
                print(p['global index'])
    """).strip()
    )

    errors = cs.check_source_code_for_leakage([tmp_path / "scripts"])
    assert not errors, (
        f"Reading accuracy without writing prediction should pass: {errors}"
    )


def test_self_skip(tmp_path: Path) -> None:
    """The integrity check script itself contains the strings but is skipped."""
    # Create a file named like the integrity check
    fake_self = tmp_path / "scripts" / "check_submission_integrity.py"
    fake_self.parent.mkdir(parents=True, exist_ok=True)
    fake_self.write_text(
        textwrap.dedent("""
        for p in preds:
            if float(p.get('accuracy', 0) or 0) == 0.0:
                p['prediction'] = 'should not flag'
    """).strip()
    )

    errors = cs.check_source_code_for_leakage([tmp_path / "scripts"])
    assert not errors, "Integrity check itself must be skipped"


# ── Check 3: reassignment plan scan ───────────────────────────────────────────


def test_plan_with_accuracy_key_is_flagged(tmp_path: Path) -> None:
    """Reassignment plans containing accuracy/cost keys indicate label use."""
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            [
                {"gi": "Q0", "from": "a", "to": "b", "accuracy": 0.0},
                {"gi": "Q1", "from": "a", "to": "b", "accuracy": 0.0},
            ]
        )
    )
    errors = cs.check_reassignment_plans([plan])
    assert errors
    assert any("LEAKAGE" in e for e in errors)


def test_plan_without_label_keys_passes(tmp_path: Path) -> None:
    """Plans built from prompt-only features are legitimate."""
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            [
                {"gi": "Q0", "from": "a", "to": "b", "category": "math"},
                {"gi": "Q1", "from": "a", "to": "b", "category": "code"},
            ]
        )
    )
    errors = cs.check_reassignment_plans([plan])
    assert not errors


def test_missing_plan_file_silent(tmp_path: Path) -> None:
    """Non-existent plan files are OK (the plan might have been deleted)."""
    errors = cs.check_reassignment_plans([tmp_path / "does-not-exist.json"])
    assert not errors


# ── Integration test: full check on a leaky scenario ──────────────────────────


def test_full_check_catches_lever_3_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive main() on a tmp_path replicating Lever #3 — should exit non-zero."""
    pred_dir = tmp_path / "router_inference" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    baseline = [
        {"global index": f"Q{i}", "prediction": "model_a", "accuracy": acc}
        for i, acc in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])
    ]
    current = [
        {
            "global index": f"Q{i}",
            "prediction": ("model_b" if acc == 0.0 else "model_a"),
        }
        for i, acc in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])
    ]
    (pred_dir / "llm-router.json").write_text(json.dumps(current))
    (pred_dir / "llm-router.json.bak.honest").write_text(json.dumps(baseline))

    (scripts_dir / "bad.py").write_text(
        textwrap.dedent("""
        import json
        for p in json.load(open('x.json')):
            if float(p.get('accuracy', 0) or 0) == 0.0:
                p['prediction'] = 'evil'
    """).strip()
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_submission_integrity.py",
            "--predictions",
            str(pred_dir / "llm-router.json"),
            "--baseline",
            str(pred_dir / "llm-router.json.bak.honest"),
            "--scripts-dir",
            str(scripts_dir),
        ],
    )

    rc = cs.main()
    assert rc == 1, "Integrity check should exit 1 on Lever #3 pattern"


def test_full_check_passes_on_clean_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive main() with no leakage — should exit 0."""
    pred_dir = tmp_path / "router_inference" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    entries = [{"global index": f"Q{i}", "prediction": "model_a"} for i in range(5)]
    (pred_dir / "llm-router.json").write_text(json.dumps(entries))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_submission_integrity.py",
            "--predictions",
            str(pred_dir / "llm-router.json"),
            "--scripts-dir",
            str(tmp_path / "scripts"),  # doesn't exist - OK
        ],
    )

    rc = cs.main()
    assert rc == 0, "Clean state should pass integrity check"
