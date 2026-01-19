# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Batch Model Evaluation Script

This script runs evaluation for multiple models sequentially.
It processes models one by one, but each model evaluation uses
query-level parallelism for efficiency.

The script evaluates all models that:
1. Have a cached results file (.jsonl) in the cached_results directory
2. Are listed in model_cost.json (for cost calculation)

Usage:
    python batch_evaluate.py [--cached-results-dir CACHED_RESULTS_DIR] [--num-workers NUM_WORKERS] [--model-cost-path PATH]
"""

import os
import sys
import argparse
import subprocess
import json
from typing import List, Optional
import time

# Add parent directory to path to import universal_model_names
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from universal_model_names import universal_names

    print(f"Loaded {len(universal_names)} universal model names")
except ImportError:
    print(
        "Warning: Could not import universal_model_names. Model name validation may be limited."
    )
    universal_names = []


def load_models_from_cost_config(
    cost_config_path: Optional[str], project_root: str
) -> List[str]:
    """
    Load list of model names from model_cost.json.

    Args:
        cost_config_path: Path to model_cost.json (can be relative or absolute).
                         If None or empty, constructs path as project_root/model_cost/model_cost.json
        project_root: Path to project root directory

    Returns:
        List of model names (keys from model_cost.json)
    """
    # If no path provided, use default location in project root
    if not cost_config_path:
        cost_config_path = os.path.join(project_root, "model_cost", "model_cost.json")
    elif not os.path.isabs(cost_config_path):
        # If relative path, make it relative to project root
        cost_config_path = os.path.join(project_root, cost_config_path)

    if not os.path.exists(cost_config_path):
        print(f"Error: model_cost.json not found at: {cost_config_path}")
        return []

    try:
        with open(cost_config_path, "r", encoding="utf-8") as f:
            cost_config = json.load(f)

        models = list(cost_config.keys())
        print(f"Loaded {len(models)} models from {cost_config_path}")
        return models
    except (json.JSONDecodeError, IOError) as e:
        print(
            f"Error: Could not load or parse cost configuration from {cost_config_path}: {e}"
        )
        return []


def check_cached_results_exist(cached_results_dir: str, model_name: str) -> bool:
    """Check if cached results file exists for a model."""
    cached_file = os.path.join(cached_results_dir, f"{model_name}.jsonl")
    return os.path.exists(cached_file)


def get_available_models(
    cached_results_dir: str, model_list: Optional[List[str]] = None
) -> List[str]:
    """Get list of models that have cached results available."""
    if model_list is None:
        model_list = universal_names

    available_models = []
    missing_models = []

    for model_name in model_list:
        if check_cached_results_exist(cached_results_dir, model_name):
            available_models.append(model_name)
        else:
            missing_models.append(model_name)

    if missing_models:
        print(f"Warning: {len(missing_models)} models don't have cached results:")
        for model in missing_models[:10]:  # Show first 10
            print(f"  - {model}")
        if len(missing_models) > 10:
            print(f"  ... and {len(missing_models) - 10} more")

    print(
        f"Found {len(available_models)} models with cached results available for evaluation"
    )
    return available_models


def run_evaluation(
    model_name: str, cached_results_dir: str, num_workers: int = 16, rerun: bool = False
) -> dict:
    """Run evaluation for a single model."""
    start_time = time.time()

    try:
        # Get the path to evaluate_models.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        evaluate_script = os.path.join(script_dir, "evaluate_models.py")

        # Run the evaluation
        cmd = [
            sys.executable,
            evaluate_script,
            model_name,
            "--cached-results-dir",
            cached_results_dir,
            "--num-workers",
            str(num_workers),
        ]

        # Add rerun flag if specified
        if rerun:
            cmd.append("--rerun")

        print(f"🔄 Starting evaluation for {model_name}")

        # Run with real-time output streaming
        # Use Popen to stream output while still being able to capture it for error reporting
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            universal_newlines=True,
            bufsize=1,  # Line buffered
        )

        # Stream output in real time and collect it
        stdout_lines: list[str] = []
        try:
            import sys as sys_module

            # Read lines and print in real time
            stdout_stream = process.stdout
            if stdout_stream is None:
                raise RuntimeError("stdout is None despite PIPE")
            while True:
                line = stdout_stream.readline()
                if not line:
                    if process.poll() is not None:
                        break  # Process finished
                    continue
                sys_module.stdout.write(line)
                sys_module.stdout.flush()
                stdout_lines.append(line)

            # Process will finish when readline returns None
            # Wait to ensure it's fully terminated (12-hour timeout)
            process.wait(timeout=43200)

        except (KeyboardInterrupt, Exception) as e:
            if process.poll() is None:
                process.kill()
                process.wait()
            if isinstance(e, KeyboardInterrupt):
                raise
            # Continue to handle as error below

        stdout_text = "".join(stdout_lines)
        duration = time.time() - start_time

        if process.returncode == 0:
            print(f"✅ Completed {model_name} in {duration:.1f}s")
            return {
                "model_name": model_name,
                "status": "success",
                "duration": duration,
                "stdout": stdout_text,
                "stderr": "",  # Merged into stdout
            }
        else:
            print(f"❌ Failed {model_name} after {duration:.1f}s")
            # Extract error from last few lines if available
            error_msg = (
                "\n".join(stdout_lines[-10:]) if stdout_lines else "Unknown error"
            )
            print(f"   Error: {error_msg.strip()}")
            return {
                "model_name": model_name,
                "status": "failed",
                "duration": duration,
                "stdout": stdout_text,
                "stderr": "",  # Merged into stdout
                "return_code": process.returncode,
            }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Timeout {model_name} after {duration:.1f}s")
        return {
            "model_name": model_name,
            "status": "timeout",
            "duration": duration,
            "error": "Evaluation timed out after 12 hours",
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 Exception {model_name} after {duration:.1f}s: {str(e)}")
        return {
            "model_name": model_name,
            "status": "exception",
            "duration": duration,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate multiple models in parallel"
    )
    parser.add_argument(
        "--cached-results-dir",
        type=str,
        default="../cached_results/",
        help="Directory containing cached results (default: ../cached_results/)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers for query-level evaluation (default: 16)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force re-evaluation of all entries, even if already evaluated",
    )
    parser.add_argument(
        "--model-cost-path",
        type=str,
        default=None,
        help="Path to model_cost.json file (default: {project_root}/model_cost/model_cost.json). Can be absolute or relative to project root.",
    )

    args = parser.parse_args()

    # Handle deprecated --max-workers if it was provided and not default
    # If the user used --max-workers but not --num-workers, use that value for num-workers
    # We still process models sequentially as requested.
    num_workers = args.num_workers

    # Get project root directory (parent of llm_evaluation/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Validate cached results directory
    if not os.path.exists(args.cached_results_dir):
        print(
            f"Error: Cached results directory does not exist: {args.cached_results_dir}"
        )
        return 1

    # Load models from model_cost.json instead of universal_names
    # Default path is project_root/model_cost/model_cost.json
    model_list = load_models_from_cost_config(args.model_cost_path, project_root)

    if not model_list:
        print(
            "Error: No models found in model_cost.json. Cannot proceed with evaluation."
        )
        return 1

    # Get available models (those with cached results)
    available_models = get_available_models(args.cached_results_dir, model_list)

    if not available_models:
        print("No models with cached results found for evaluation.")
        return 1

    print(f"\n🚀 Starting batch evaluation of {len(available_models)} models")
    print(f"📊 Using {num_workers} parallel workers per model (sequential models)")
    print(f"📁 Cached results directory: {args.cached_results_dir}")
    print("=" * 80)

    # Run evaluations sequentially (one model at a time)
    # Each model evaluation will internally use query-level parallelism
    start_time = time.time()
    results = []

    for model in available_models:
        result = run_evaluation(model, args.cached_results_dir, num_workers, args.rerun)
        results.append(result)

    # Print final summary
    total_duration = time.time() - start_time
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    timeouts = [r for r in results if r["status"] == "timeout"]
    exceptions = [r for r in results if r["status"] == "exception"]

    print("=" * 80)
    print(f"🏁 Batch evaluation completed in {total_duration:.1f}s")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏰ Timeouts: {len(timeouts)}")
    print(f"💥 Exceptions: {len(exceptions)}")

    if failed:
        print("\nFailed models:")
        for result in failed:
            print(
                f"  - {result['model_name']}: {result.get('stderr', 'Unknown error')}"
            )

    if timeouts:
        print("\nTimeout models:")
        for result in timeouts:
            print(f"  - {result['model_name']}")

    if exceptions:
        print("\nException models:")
        for result in exceptions:
            print(
                f"  - {result['model_name']}: {result.get('error', 'Unknown exception')}"
            )

    # Calculate average duration for successful evaluations
    if successful:
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        print(f"\nAverage evaluation time: {avg_duration:.1f}s")

    return 0 if len(failed) + len(timeouts) + len(exceptions) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
