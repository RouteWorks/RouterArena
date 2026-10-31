#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0


"""Utility helpers for evaluating PR router submissions on a host server.

This script is designed for maintainers who want to process leaderboard
submissions on a trusted machine (e.g., the server backing RouterArena). It
automates the repetitive steps involved in pulling a contributor's pull request,
validating the prediction file, running the evaluation pipeline, and capturing
artifacts for later review.

Typical usage:

    uv run python automation/process_pr_submission.py --pr 123 --router my-router \
        --split sub_10

Key features:

* Grades with the base repository's code, never the pull request's: it creates an
  isolated git worktree of ``<remote>/main`` (so the current working copy stays
  untouched) and copies in only the submission's data files from the pull request
  (prediction, robustness and config files, plus the model names and prices it adds).
* Runs the existing validation and evaluation scripts via ``uv run``.
* Saves evaluation summaries and the evaluated prediction file under
  ``pr_evaluations/pr-<PR_NUMBER>/`` for auditing.
* Cleans up the temporary worktree and branch after completion unless instructed
  otherwise.

The script requires ``git`` and ``uv`` to be available on the host machine. A
GitHub token with read access to the repository is only needed if the default
``git fetch`` command cannot access pull request refs without authentication.
"""

from __future__ import annotations

import ast
import math
import argparse
import json
import shutil
import subprocess
import sys
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from global_utils.robustness import compute_robustness_score


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKTREES_DIR = REPO_ROOT / ".pr_worktrees"
ARTIFACTS_DIR = REPO_ROOT / "pr_evaluations"

# The only files taken from a pull request; everything else comes from the base branch.
SUBMISSION_FILES = (
    "router_inference/predictions/{router}.json",
    "router_inference/predictions/{router}-robustness.json",
    "router_inference/config/{router}.json",
)
MODEL_COST_PATH = "model_cost/model_cost.json"
MODEL_NAMES_PATH = "universal_model_names.py"


class CommandError(RuntimeError):
    """Raised when a subprocess fails and we want a cleaner error message."""

    def __init__(
        self,
        message: str,
        *,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def run_command(
    args: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, raising ``CommandError`` on non-zero exit codes."""

    display_cmd = " ".join(args)
    print(f"→ {display_cmd}")
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            check=True,
            text=True,
            capture_output=capture,
        )
        if capture and completed.stdout:
            print(completed.stdout)
        if capture and completed.stderr:
            # ``uv`` sometimes emits warnings on stderr even on success.
            print(completed.stderr, file=sys.stderr)
        return completed
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else None
        stderr = exc.stderr if isinstance(exc.stderr, str) else None
        raise CommandError(
            f"Command failed (exit code {exc.returncode}): {display_cmd}",
            stdout=stdout,
            stderr=stderr,
        ) from exc


def ensure_worktree(
    pr_number: int, remote: str, eval_ref: str, *, keep_existing: bool = False
) -> tuple[Path, str]:
    """Fetch a PR head and create a worktree of the evaluation code to grade it in.

    The worktree is checked out at ``eval_ref``, so the validator, evaluator and
    dependencies come from the base repository, never from the pull request. The PR
    head is only fetched to a local branch, from which ``stage_submission`` copies
    the submission's data files.
    """

    branch_name = f"pr-{pr_number}"
    # "+" lets the fetch replace a branch left from an earlier run after a force-push.
    fetch_ref = f"+pull/{pr_number}/head:{branch_name}"
    # Ensure the base branch is up-to-date for later diffs.
    run_command(["git", "fetch", remote, "main"], cwd=REPO_ROOT)
    run_command(["git", "fetch", remote, fetch_ref], cwd=REPO_ROOT)

    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    worktree_path = WORKTREES_DIR / branch_name

    if worktree_path.exists() and not keep_existing:
        run_command(
            ["git", "worktree", "remove", "--force", str(worktree_path)], cwd=REPO_ROOT
        )

    run_command(
        [
            "git",
            "worktree",
            "add",
            "--force",
            "--detach",
            str(worktree_path),
            eval_ref,
        ],
        cwd=REPO_ROOT,
    )

    return worktree_path, branch_name


def cleanup_worktree(worktree_path: Path, branch_name: str, *, keep: bool) -> None:
    """Remove the temporary worktree and local branch (unless ``keep``)."""

    if keep:
        print(f"Skipping cleanup for worktree {worktree_path}")
        return

    if worktree_path.exists():
        run_command(
            ["git", "worktree", "remove", "--force", str(worktree_path)], cwd=REPO_ROOT
        )

    # Delete the local branch reference so repeated runs stay clean.
    run_command(["git", "branch", "-D", branch_name], cwd=REPO_ROOT)


def ensure_prediction_file_added(
    pr_ref: str, base_ref: str, router_name: str, *, robustness: bool = False
) -> None:
    """Verify the PR adds or modifies a prediction file for the specified router."""

    suffix = "-robustness" if robustness else ""
    target_path = (
        Path("router_inference") / "predictions" / f"{router_name}{suffix}.json"
    )

    diff_cmd = [
        "git",
        "diff",
        "--name-status",
        f"{base_ref}...{pr_ref}",
        "--",
        str(target_path),
    ]

    completed = subprocess.run(
        diff_cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in lines:
        # Allow both added (A) and modified (M) files
        if line[0] in ("A", "M"):
            return

    raise RuntimeError(
        textwrap.dedent(
            f"""
            Expected pull request to add or modify a prediction file {target_path}.
            Diff against {base_ref} did not show a newly added or modified file.
            """
        ).strip()
    )


def read_file_at(ref: str, path: str) -> Optional[bytes]:
    """Return a file's contents at ``ref``, or None if it doesn't exist there.

    Reads the git blob directly, so nothing from the pull request is checked out,
    and refuses anything other than a regular file (a symlink, for example).
    """

    listing = subprocess.run(
        ["git", "ls-tree", ref, "--", path],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.split()
    if not listing:
        return None
    mode, kind = listing[0], listing[1]
    if kind != "blob" or mode not in ("100644", "100755"):
        raise RuntimeError(f"{path} at {ref} is not a regular file.")
    return subprocess.run(
        ["git", "cat-file", "blob", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def literal_assignments(source: str, names: tuple[str, ...]) -> dict[str, object]:
    """Evaluate top-level ``name = <literal>`` assignments without running the code."""

    values: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names:
            values[target.id] = ast.literal_eval(value)
    return values


def stage_model_prices(worktree_path: Path, pr_ref: str, base_ref: str) -> None:
    """Add prices for the models the PR introduces; priced models keep their price.

    Only entries the PR itself adds or edits count, so a PR branched from an older
    base doesn't undo price changes made on the base branch since.
    """

    content = read_file_at(pr_ref, MODEL_COST_PATH)
    if content is None:
        return
    pr_prices = json.loads(content)
    if not isinstance(pr_prices, dict):
        raise ValueError(f"{MODEL_COST_PATH} in the pull request is not a JSON object.")
    merge_base = subprocess.run(
        ["git", "merge-base", base_ref, pr_ref],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    before = json.loads(read_file_at(merge_base, MODEL_COST_PATH) or b"{}")
    pr_edits = {
        model: price for model, price in pr_prices.items() if before.get(model) != price
    }

    cost_path = worktree_path / MODEL_COST_PATH
    prices = json.loads(cost_path.read_text(encoding="utf-8"))
    added = {model: price for model, price in pr_edits.items() if model not in prices}
    changed = sorted(
        model
        for model, price in pr_edits.items()
        if model in prices and price != prices[model]
    )
    if changed:
        print(
            f"⚠ The PR changes the price of {len(changed)} existing model(s); "
            f"evaluating with the base branch's price: {', '.join(changed)}"
        )
    if added:
        prices.update(added)
        cost_path.write_text(json.dumps(prices, indent=2) + "\n", encoding="utf-8")
        print(
            f"  Added prices for {len(added)} new model(s): {', '.join(sorted(added))}"
        )


def stage_model_names(worktree_path: Path, pr_ref: str) -> None:
    """Register the model names and name mappings the PR adds.

    The PR's ``universal_model_names.py`` is parsed, not imported, so none of its
    code runs; ``universal_names`` and ``mapping`` must be plain literals.
    """

    content = read_file_at(pr_ref, MODEL_NAMES_PATH)
    if content is None:
        return
    keys = ("universal_names", "mapping")
    try:
        pr_values = literal_assignments(content.decode("utf-8"), keys)
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Could not read {MODEL_NAMES_PATH} from the pull request; universal_names "
            f"and mapping must be plain literals ({error})."
        ) from error
    pr_names = pr_values.get("universal_names", [])
    pr_mapping = pr_values.get("mapping", {})
    if not (
        isinstance(pr_names, list)
        and all(isinstance(name, str) for name in pr_names)
        and isinstance(pr_mapping, dict)
        and all(
            isinstance(k, str) and isinstance(v, str) for k, v in pr_mapping.items()
        )
    ):
        raise RuntimeError(
            f"{MODEL_NAMES_PATH} in the pull request must define universal_names as a "
            "list of strings and mapping as a dict of strings."
        )

    names_path = worktree_path / MODEL_NAMES_PATH
    base_values = literal_assignments(names_path.read_text(encoding="utf-8"), keys)
    base_names = base_values.get("universal_names", [])
    base_mapping = base_values.get("mapping", {})
    assert isinstance(base_names, list) and isinstance(base_mapping, dict)
    new_names = [name for name in pr_names if name not in base_names]
    new_mapping = {k: v for k, v in pr_mapping.items() if k not in base_mapping}
    if not (new_names or new_mapping):
        return
    with names_path.open("a", encoding="utf-8") as handle:
        handle.write("\n# Added from the pull request under evaluation.\n")
        handle.write(f"universal_names.extend({new_names!r})\n")
        handle.write(f"mapping.update({new_mapping!r})\n")
    print(
        f"  Added {len(new_names)} model name(s) and {len(new_mapping)} name mapping(s)"
    )


def stage_submission(
    worktree_path: Path, pr_ref: str, base_ref: str, router_name: str
) -> None:
    """Copy a submission's data files from the pull request into the worktree.

    Only the router's prediction, robustness and config files are copied as they
    are, plus the model names and prices the PR adds. Every other file the PR
    changes is evaluated with the base branch's version and listed in the log.
    """

    staged = {template.format(router=router_name) for template in SUBMISSION_FILES}
    for path in sorted(staged):
        content = read_file_at(pr_ref, path)
        if content is None:
            continue
        destination = worktree_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        print(f"  Copied {path}")

    stage_model_prices(worktree_path, pr_ref, base_ref)
    stage_model_names(worktree_path, pr_ref)

    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{pr_ref}"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    ignored = [
        path
        for path in changed
        if path not in staged | {MODEL_COST_PATH, MODEL_NAMES_PATH}
    ]
    if ignored:
        print("⚠ Changed in the PR but not used for evaluation:")
        for path in ignored:
            print(f"    {path}")


def find_dataset_source() -> Optional[Path]:
    """Locate a dataset directory accessible from the current environment."""

    env_path = os.getenv("ROUTERARENA_DATASET_DIR")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    # Start with the repo root (either canonical checkout or temporary worktree).
    candidates = [REPO_ROOT / "dataset"]

    # Also consider parent directories, which covers the main checkout when running
    # inside a git worktree located at <repo>/.pr_worktrees/<branch>.
    for parent in REPO_ROOT.parents:
        candidates.append(parent / "dataset")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return None


def sync_dataset_into_worktree(worktree_path: Path) -> None:
    """Ensure the evaluation dataset is present inside the temporary worktree."""

    source = find_dataset_source()
    if not source:
        print("⚠ Maintainer dataset directory not found; skipping dataset sync.")
        return

    destination = worktree_path / "dataset"
    shutil.copytree(source, destination, dirs_exist_ok=True)


def compute_scores(prediction_file: Path) -> dict[str, float]:
    """
    Compute aggregate metrics from an evaluated prediction file.

    Only uses regular entries (for_optimality=False) for RouterArena score.
    Optimality entries are excluded from arena_score calculation.
    """

    with prediction_file.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    if not isinstance(predictions, list):
        raise ValueError(f"Unexpected prediction payload in {prediction_file}")

    # Filter to regular predictions only (exclude optimality entries)
    regular_predictions = [p for p in predictions if not p.get("for_optimality", False)]

    # Every regular query contributes to the accuracy denominator. Entries with no
    # valid generation (success=False / missing / empty) are scored as 0, not dropped,
    # so accuracy cannot be inflated over a self-selected answered subset.
    accuracies = []
    abnormal_count = 0
    for entry in regular_predictions:
        generated_result = entry.get("generated_result")
        # Untrusted contributor input: require success to be exactly True (a
        # truthy string like "False" must not pass).
        has_valid_generation = (
            isinstance(generated_result, dict)
            and generated_result.get("success") is True
        )
        accuracy = entry.get("accuracy")
        # Accept accuracy only if it is a real number (reject bool, since
        # isinstance(True, int) is True, and strings that would break sum()).
        if (
            has_valid_generation
            and isinstance(accuracy, (int, float))
            and not isinstance(accuracy, bool)
        ):
            accuracies.append(float(accuracy))
        else:
            accuracy = 0.0
            abnormal_count += 1
            accuracies.append(0.0)

    costs = [
        entry["cost"]
        for entry in regular_predictions
        if entry.get("cost") not in (None, 0)
    ]

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    total_cost = float(sum(costs)) if costs else 0.0
    num_queries = len(regular_predictions)
    avg_cost_per_query = total_cost / num_queries if num_queries else 0.0
    avg_cost_per_1000 = avg_cost_per_query * 1000

    arena_score = compute_arena_score(avg_cost_per_1000, avg_accuracy)

    return {
        "num_queries": num_queries,
        "accuracy": avg_accuracy,
        "total_cost": total_cost,
        "avg_cost_per_query": avg_cost_per_query,
        "avg_cost_per_1000": avg_cost_per_1000,
        "arena_score": arena_score,
        "abnormal_count": abnormal_count,
    }


def compute_robustness_score_from_predictions(
    full_prediction_file: Path, robustness_prediction_file: Path
) -> Optional[float]:
    """Compute robustness flip ratio between full/sub_10 and robustness splits."""

    with full_prediction_file.open("r", encoding="utf-8") as full_handle:
        full_predictions = json.load(full_handle)
    with robustness_prediction_file.open("r", encoding="utf-8") as robustness_handle:
        robustness_predictions = json.load(robustness_handle)

    if not isinstance(full_predictions, list) or not isinstance(
        robustness_predictions, list
    ):
        raise ValueError("Prediction payload must be a list of entries.")

    return compute_robustness_score(full_predictions, robustness_predictions)


def append_robustness_score_to_metrics(
    metrics: dict[str, object],
    prediction_file: Path,
    robustness_prediction_file: Path,
    metrics_path: Path,
) -> dict[str, object]:
    """
    Ensure robustness_score is present in metrics, computing it if necessary.
    """

    if "robustness_score" in metrics:
        return metrics

    if not robustness_prediction_file.exists():
        print(
            "⚠ Robustness prediction file not found; skipping robustness score computation."
        )
        return metrics

    score = compute_robustness_score_from_predictions(
        prediction_file, robustness_prediction_file
    )
    if score is None:
        print(
            "⚠ Could not compute robustness score because no overlapping entries were found."
        )
        return metrics

    metrics["robustness_score"] = score
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"✔ Appended robustness_score={score:.4f} to metrics.json")
    return metrics


def compute_arena_score(
    cost: float,
    accuracy: float,
    *,
    beta: float = 0.1,
    c_max: float = 200.0,
    c_min: float = 0.0044,
) -> float:
    """Mirror the project-wide arena score calculation."""

    if cost <= 0:
        # Avoid math domain errors – treat missing cost as neutral.
        return 0.0

    numerator = math.log2(c_max) - math.log2(cost)
    denominator = math.log2(c_max) - math.log2(c_min)
    C_i = numerator / denominator

    return (
        ((1 + beta) * accuracy * C_i) / (beta * accuracy + C_i)
        if (beta * accuracy + C_i)
        else 0.0
    )


def write_summary(summary: dict[str, object], destination: Path) -> None:
    """Persist a human-readable and JSON summary next to the evaluated file."""

    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = destination / f"summary-{timestamp}.json"
    txt_path = destination / f"summary-{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = ["Evaluation summary"]
    lines.append("=" * 80)
    for key, value in summary.items():
        if key == "logs":
            continue
        lines.append(f"{key}: {value}")

    logs = summary.get("logs")
    if isinstance(logs, str) and logs.strip():
        lines.append("\nCommand log excerpt")
        lines.append("-" * 80)
        lines.append(logs.strip())

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate router submissions from a pull request on this host server."
    )
    parser.add_argument(
        "--pr", type=int, required=True, help="Pull request number to evaluate."
    )
    parser.add_argument(
        "--router", required=True, help="Router name (prediction file <router>.json)."
    )
    parser.add_argument(
        "--split",
        choices=["sub_10", "full"],
        default="sub_10",
        help="Dataset split to evaluate (default: sub_10).",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remote name to fetch from (default: origin).",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Do not delete the temporary git worktree/branch after completion.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip running 'uv sync --locked' inside the worktree.",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Git ref to diff against when validating submissions (default: <remote>/main).",
    )
    parser.add_argument(
        "--eval-ref",
        default=None,
        help="Commit whose validator and evaluator grade the submission "
        "(default: <remote>/main).",
    )
    parser.add_argument(
        "--allow-existing-prediction",
        action="store_true",
        help="Skip the check that enforces adding a new prediction file in this PR.",
    )

    args = parser.parse_args(argv)

    worktree_path: Optional[Path] = None
    branch_name = ""

    base_ref = args.base_ref or f"{args.remote}/main"
    eval_ref = args.eval_ref or f"{args.remote}/main"

    try:
        worktree_path, branch_name = ensure_worktree(
            args.pr, args.remote, eval_ref, keep_existing=args.keep_worktree
        )
        evaluator_commit = run_command(
            ["git", "rev-parse", "HEAD"], cwd=worktree_path, capture=True
        ).stdout.strip()
        print(
            f"✔ Created worktree at {worktree_path} "
            f"(evaluator: {eval_ref} @ {evaluator_commit})"
        )

        sync_dataset_into_worktree(worktree_path)

        if not args.allow_existing_prediction:
            ensure_prediction_file_added(branch_name, base_ref, args.router)
            ensure_prediction_file_added(
                branch_name, base_ref, args.router, robustness=True
            )

        print("▶ Copying the submission's files from the pull request...", flush=True)
        stage_submission(worktree_path, branch_name, base_ref, args.router)

        if not args.skip_sync:
            print("▶ Syncing dependencies with uv...", flush=True)
            run_command(["uv", "sync", "--locked"], cwd=worktree_path, capture=False)
            print("✔ Synced dependencies")

        validation_cmd = [
            "uv",
            "run",
            "--active",
            "router_inference/check_config_prediction_files.py",
            args.router,
            args.split,
            "--check-generated-result",
        ]
        print("▶ Validating prediction/config files...", flush=True)
        validation_result = run_command(
            validation_cmd, cwd=worktree_path, capture=False
        )
        print("✔ Validated prediction files")

        evaluation_cmd = [
            "uv",
            "run",
            "--active",
            "llm_evaluation/run.py",
            args.router,
            args.split,
            "--force",
        ]

        evaluation_logs = ""
        try:
            print("▶ Running evaluation...", flush=True)
            evaluation_result = run_command(
                evaluation_cmd, cwd=worktree_path, capture=False
            )
            print("✔ Evaluated predictions")
            evaluation_logs = (evaluation_result.stdout or "") + (
                evaluation_result.stderr or ""
            )
        except CommandError as error:
            evaluation_logs = (error.stdout or "") + (error.stderr or "")
            raise

        prediction_file = (
            worktree_path / "router_inference" / "predictions" / f"{args.router}.json"
        )
        if not prediction_file.exists():
            raise FileNotFoundError(
                textwrap.dedent(
                    f"""
                    Prediction file not found after evaluation: {prediction_file}
                    Ensure the pull request contains router_inference/predictions/{args.router}.json
                    """
                ).strip()
            )

        robustness_prediction_file = prediction_file.with_name(
            f"{args.router}-robustness.json"
        )
        if not robustness_prediction_file.exists():
            raise FileNotFoundError(
                textwrap.dedent(
                    f"""
                    Robustness prediction file not found: {robustness_prediction_file}
                    Ensure the pull request includes router_inference/predictions/{args.router}-robustness.json
                    """
                ).strip()
            )

        # Read metrics from metrics.json (required - no fallback)
        # llm_evaluation/run.py writes metrics.json to the current working directory (worktree_path)
        metrics_path = worktree_path / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                textwrap.dedent(
                    f"""
                    metrics.json not found at {metrics_path}.
                    Evaluation must produce metrics.json file.
                    Ensure llm_evaluation/run.py completed successfully.
                    """
                ).strip()
            )

        print("\n✔ Reading metrics from metrics.json...")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics = append_robustness_score_to_metrics(
            metrics, prediction_file, robustness_prediction_file, metrics_path
        )

        # Copy metrics.json to base directory (REPO_ROOT) for workflow to read
        base_metrics_path = REPO_ROOT / "metrics.json"
        shutil.copy2(metrics_path, base_metrics_path)
        print(f"✔ Copied metrics.json to {base_metrics_path}")

        # Display optimality scores if available
        if "optimality" in metrics:
            opt = metrics["optimality"]
            print("\n✔ Optimality scores:")
            print(
                f"  Opt.Sel: {opt.get('opt_sel', 0):.4f} ({opt.get('opt_sel', 0) * 100:.2f}%)"
            )
            print(f"  Opt.Cost: {opt.get('opt_cost', 0):.4f}")
            print(f"  Opt.Acc: {opt.get('opt_acc', 0):.4f}")

        # Persist artifacts under pr_evaluations/pr-<num>/run-<timestamp>/
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ARTIFACTS_DIR / f"pr-{args.pr}" / f"run-{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        archived_prediction = run_dir / f"{args.router}.json"
        shutil.copy2(prediction_file, archived_prediction)
        archived_robust_prediction = run_dir / f"{args.router}-robustness.json"
        shutil.copy2(robustness_prediction_file, archived_robust_prediction)

        summary_payload: dict[str, object] = {
            "pr": args.pr,
            "router": args.router,
            "split": args.split,
            "worktree": str(worktree_path),
            "evaluator_commit": evaluator_commit,
            "validation_stdout": validation_result.stdout
            if validation_result.stdout
            else "",
            "metrics": metrics,
            "logs": evaluation_logs,
        }

        write_summary(summary_payload, run_dir)

        print("\n✔ Evaluation completed successfully")
        print(f"  Metrics: {json.dumps(metrics, indent=2)}")
        print(f"  Artifacts saved to: {run_dir}")

        return 0

    except CommandError as error:
        if error.stdout:
            print(error.stdout, file=sys.stdout)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        print(f"✗ {error}", file=sys.stderr)
        return 1
    except Exception as error:  # pylint: disable=broad-except
        print(f"✗ {error}", file=sys.stderr)
        return 1
    finally:
        if worktree_path and branch_name:
            cleanup_worktree(worktree_path, branch_name, keep=args.keep_worktree)


if __name__ == "__main__":
    sys.exit(main())
