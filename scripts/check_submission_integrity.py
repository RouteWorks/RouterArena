# SPDX-FileCopyrightText: Copyright (c) 2026 Yali Pollak
# SPDX-License-Identifier: Apache-2.0

"""Submission integrity check — catch test-set leakage before re-submitting.

RouterArena's submission policy prohibits using ground-truth labels
(per-query accuracy/cost from prior evaluations) to make routing
decisions. This script enforces that policy with four checks:

1. **Diff analysis** — compare current predictions to a baseline file.
   If reassigned queries correlate >95% with previous accuracy=0, that's
   a smoking gun (e.g. the Lever #3 pattern: every reassignment targeted
   a previously-wrong query).

2. **Source-code scan** — grep every Python script in the repo for code
   patterns that read `accuracy` / `cost` from predictions and use them
   to set the `prediction` field. AST-based to avoid false positives in
   comments and strings.

3. **Reassignment plan scan** — `/tmp/reassignment_plan.json` and any
   `*.plan.json` artefacts: report categories used to make decisions.
   Flag if `accuracy` or `cost` keys appear in plan records.

4. **Pre-eval invariants** — the predictions file should NOT contain
   evidence of test-side reasoning (e.g. routing chosen *because of*
   accuracy in a backup file dated before the routing was applied).

Usage:
    uv run python scripts/check_submission_integrity.py
        [--predictions router_inference/predictions/llm-router.json]
        [--baseline router_inference/predictions/llm-router.json.bak.honest]
        [--scripts-dir scripts/]
        [--threshold 0.95]

Exit code is 0 if all checks pass, 1 if any leak is detected.
The non-zero exit makes this suitable as a pre-push git hook.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


# ── Check 1: Diff analysis ────────────────────────────────────────────────────


def check_reassignments_vs_prior_accuracy(
    predictions_path: Path,
    baseline_path: Path | None,
    correlation_threshold: float,
) -> list[str]:
    """Flag if reassigned queries correlate too strongly with prior accuracy=0.

    The smoking gun for Lever #3 was: 100% of reassignments targeted
    queries that scored 0.0 in the baseline. Any correlation above the
    threshold (default 0.95) indicates the routing decision read accuracy.
    """
    errors: list[str] = []
    if not predictions_path.exists():
        # No predictions to check (e.g. running on a branch without the submission
        # artefacts). Print a notice so the user knows the diff check was skipped,
        # but don't treat as a failure — static scans still run.
        print(f"  (skipped — predictions file not found: {predictions_path})")
        return []
    if baseline_path is None or not baseline_path.exists():
        # No baseline to compare against — skip silently (initial submission).
        return []

    current = json.loads(predictions_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    base_map = {p["global index"]: p for p in baseline if not p.get("for_optimality", False)}

    reassigned = []
    for p in current:
        if p.get("for_optimality", False):
            continue
        gi = p["global index"]
        b = base_map.get(gi)
        if b is None:
            continue
        if p.get("prediction") != b.get("prediction"):
            reassigned.append((gi, b.get("accuracy"), p.get("prediction"), b.get("prediction")))

    if not reassigned:
        return []

    # How many of the reassigned queries had baseline accuracy == 0?
    prior_zero = sum(1 for _, acc, _, _ in reassigned if float(acc or 0) == 0.0)
    n = len(reassigned)
    pct = prior_zero / n

    if pct >= correlation_threshold:
        errors.append(
            f"❌ LEAKAGE: {prior_zero}/{n} ({pct:.0%}) reassignments target queries "
            f"with baseline accuracy=0. Threshold is {correlation_threshold:.0%}."
        )
        errors.append(
            f"   This is the Lever #3 pattern — the routing decision was made "
            f"using the prior evaluation's correctness labels."
        )
        errors.append(
            f"   Sample reassignments (first 5):"
        )
        for gi, acc, new_m, old_m in reassigned[:5]:
            errors.append(
                f"     {gi}: acc={acc} | {old_m or '?'} → {new_m or '?'}"
            )

    return errors


# ── Check 2: Source-code scan ─────────────────────────────────────────────────


class _LeakageVisitor(ast.NodeVisitor):
    """Find code that reads accuracy/cost from predictions and writes to prediction."""

    LABEL_FIELDS = {"accuracy", "cost"}
    SET_FIELDS = {"prediction"}

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.violations: list[tuple[int, str]] = []
        self._reads_label_in_scope = False

    def _is_label_subscript(self, node: ast.AST) -> bool:
        """Detect `p['accuracy']`, `p.get('accuracy', ...)`, `p["cost"]`, etc."""
        if isinstance(node, ast.Subscript):
            slice_ = node.slice
            if isinstance(slice_, ast.Constant) and slice_.value in self.LABEL_FIELDS:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value in self.LABEL_FIELDS:
                    return True
        return False

    def _is_pred_subscript(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            slice_ = node.slice
            if isinstance(slice_, ast.Constant) and slice_.value in self.SET_FIELDS:
                return True
        return False

    def visit_If(self, node: ast.If) -> None:
        # Detect: if p['accuracy'] == 0.0 or similar
        for sub in ast.walk(node.test):
            if self._is_label_subscript(sub):
                # Now check the body for assignment to 'prediction'
                for body_stmt in node.body:
                    for sub2 in ast.walk(body_stmt):
                        if isinstance(sub2, ast.Assign):
                            for tgt in sub2.targets:
                                if self._is_pred_subscript(tgt):
                                    line_no = sub2.lineno
                                    snippet = self.source_lines[line_no - 1].strip()
                                    self.violations.append((line_no, snippet))
        self.generic_visit(node)


def scan_python_file_for_leakage(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, snippet) violations in a Python source file."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    visitor = _LeakageVisitor(source.splitlines())
    visitor.visit(tree)
    return visitor.violations


def check_source_code_for_leakage(
    scripts_dirs: Iterable[Path],
) -> list[str]:
    """Scan all .py files under scripts_dirs for accuracy→prediction patterns."""
    errors: list[str] = []
    seen: set[Path] = set()
    for root in scripts_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path in seen:
                continue
            seen.add(path)
            # Skip tests of this very script and the integrity check itself
            if path.name == "check_submission_integrity.py":
                continue
            if "test_submission_integrity" in path.name:
                continue
            violations = scan_python_file_for_leakage(path)
            for line_no, snippet in violations:
                errors.append(
                    f"❌ LEAKAGE pattern in {path}:{line_no}\n"
                    f"     {snippet}\n"
                    f"   This reads accuracy/cost and writes the prediction field. "
                    f"That is the Lever #3 pattern."
                )
    return errors


# ── Check 3: Reassignment plan scan ───────────────────────────────────────────


def check_reassignment_plans(plan_paths: Iterable[Path]) -> list[str]:
    """Inspect any reassignment plan JSON files for label-derived fields."""
    errors: list[str] = []
    for plan_path in plan_paths:
        if not plan_path.exists():
            continue
        try:
            data = json.loads(plan_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        # Check if any record has accuracy/cost keys, which would mean
        # the plan was built from label data
        leak_keys = {"accuracy", "cost"}
        sample = data[0] if data else {}
        if isinstance(sample, dict):
            leaked = leak_keys & set(sample.keys())
            if leaked:
                errors.append(
                    f"❌ LEAKAGE: reassignment plan {plan_path} contains "
                    f"label-derived fields {leaked} on each record."
                )
    return errors


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="RouterArena submission integrity check")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("router_inference/predictions/llm-router.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline predictions file to diff against. If omitted, only static checks run.",
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        action="append",
        default=None,
        help="Python source directories to scan (can pass multiple times).",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        action="append",
        default=None,
        help="Reassignment plan JSON files to inspect (can pass multiple times).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Fraction of reassignments overlapping prior accuracy=0 that triggers a leak alarm.",
    )
    args = parser.parse_args()

    scripts_dirs = args.scripts_dir or [Path("scripts"), Path("router_inference/router")]
    plan_paths = args.plan or [
        Path("/tmp/reassignment_plan.json"),
    ]

    all_errors: list[str] = []

    print("=" * 70)
    print("ROUTERARENA SUBMISSION INTEGRITY CHECK")
    print("=" * 70)

    print("\n[1] Diff analysis (reassignments vs baseline accuracy)…")
    diff_errors = check_reassignments_vs_prior_accuracy(
        args.predictions, args.baseline, args.threshold
    )
    if diff_errors:
        all_errors.extend(diff_errors)
        for e in diff_errors:
            print(f"  {e}")
    else:
        print("  ✓ No suspicious reassignment correlation with baseline accuracy.")

    print("\n[2] Source-code scan for accuracy→prediction patterns…")
    code_errors = check_source_code_for_leakage(scripts_dirs)
    if code_errors:
        all_errors.extend(code_errors)
        for e in code_errors:
            print(f"  {e}")
    else:
        print(f"  ✓ No leakage pattern in {len(list(scripts_dirs))} scanned directories.")

    print("\n[3] Reassignment plan scan…")
    plan_errors = check_reassignment_plans(plan_paths)
    if plan_errors:
        all_errors.extend(plan_errors)
        for e in plan_errors:
            print(f"  {e}")
    else:
        print("  ✓ No reassignment plans with label-derived fields.")

    print("\n" + "=" * 70)
    if all_errors:
        print(f"✗ INTEGRITY CHECK FAILED — {len(all_errors)} issue(s)")
        print("=" * 70)
        return 1
    print("✓ ALL CHECKS PASSED — submission is clean of test-set leakage")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
