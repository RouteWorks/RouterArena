#!/usr/bin/env python3
"""
Batch Model Evaluation Script

This script runs evaluation for multiple models in parallel using the universal model names
from universal_model_names.py. It can process up to 16 models concurrently for efficiency.

Usage:
    python batch_evaluate.py [--cached-results-dir CACHED_RESULTS_DIR] [--max-workers MAX_WORKERS] [--models MODEL1 MODEL2 ...]
"""

import os
import sys
import argparse
import subprocess
import concurrent.futures
from typing import List, Optional
import time

# Add parent directory to path to import universal_model_names
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from universal_model_names import universal_names

    print(f"Loaded {len(universal_names)} universal model names")
except ImportError:
    print(
        "Error: Could not import universal_model_names. Make sure the file exists in the parent directory."
    )
    sys.exit(1)


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
    model_name: str, cached_results_dir: str, rerun: bool = False
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
        ]

        # Add rerun flag if specified
        if rerun:
            cmd.append("--rerun")

        print(f"🔄 Starting evaluation for {model_name}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=43200,  # 12 hours timeout per model
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Completed {model_name} in {duration:.1f}s")
            return {
                "model_name": model_name,
                "status": "success",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        else:
            print(f"❌ Failed {model_name} after {duration:.1f}s")
            print(f"   Error: {result.stderr.strip()}")
            return {
                "model_name": model_name,
                "status": "failed",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
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
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of parallel evaluations (default: 16)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force re-evaluation of all entries, even if already evaluated",
    )

    args = parser.parse_args()

    # Validate cached results directory
    if not os.path.exists(args.cached_results_dir):
        print(
            f"Error: Cached results directory does not exist: {args.cached_results_dir}"
        )
        return 1

    model_list = universal_names

    # Get available models (those with cached results)
    available_models = get_available_models(args.cached_results_dir, model_list)

    if not available_models:
        print("No models with cached results found for evaluation.")
        return 1

    print(f"\n🚀 Starting batch evaluation of {len(available_models)} models")
    print(f"📊 Using {min(args.max_workers, len(available_models))} parallel workers")
    print(f"📁 Cached results directory: {args.cached_results_dir}")
    print("=" * 80)

    # Run evaluations in parallel
    start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(
                run_evaluation, model, args.cached_results_dir, args.rerun
            ): model
            for model in available_models
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            result = future.result()
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
