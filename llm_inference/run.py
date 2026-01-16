# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Run LLM inference for router predictions.

This script processes router prediction files and makes LLM API calls
for each prediction, using cached results when available.

Uses parallel inference system for efficient processing with multiple workers.
"""

import argparse
import json
import os
import sys
import logging
import datetime
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../"))
os.chdir(base_dir)

# Load environment variables from .env file
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from universal_model_names import ModelNameManager  # noqa: E402
from pipeline import load_jsonl_file  # noqa: E402
from parallel_inference import ParallelInferenceManager  # noqa: E402

logger = logging.getLogger(__name__)


def load_predictions_file(router_name: str) -> List[Dict[str, Any]]:
    """
    Load router predictions from JSON file.

    Args:
        router_name: Name of the router

    Returns:
        List of prediction dictionaries
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}\n"
            f"Please create the prediction file first. See README.md for format."
        )

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions from {prediction_path}")
    return predictions


def load_cached_results_for_predictions(
    predictions: List[Dict[str, Any]], cached_results_dir: str = "./cached_results"
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load cached results for all model-query pairs in predictions.

    When multiple runs exist for the same query (due to num_runs > 1), this function
    selects one result per (model, global_index) pair using the following priority:
    1. Prefer run_number=1 (first run)
    2. If run_number=1 doesn't exist, prefer successful runs
    3. Otherwise keep the first one encountered

    Note: Each prediction entry will have one generated_result, even if multiple runs exist.

    Args:
        predictions: List of prediction dictionaries
        cached_results_dir: Directory containing cached results

    Returns:
        Dictionary mapping (universal_model_name, global_index) to cached result
    """
    cached_results = {}

    # Track statistics
    queries_with_multiple_runs = defaultdict(list)  # {(model, gidx): [run_numbers]}

    # Group by model to load cache files efficiently
    model_to_predictions = defaultdict(list)
    for pred in predictions:
        model_name = pred.get("prediction", "")
        if model_name:
            try:
                universal_model_name = ModelNameManager.get_universal_name(model_name)
                model_to_predictions[universal_model_name].append(pred)
            except Exception as e:
                logger.warning(
                    f"Could not get universal name for model '{model_name}': {e}"
                )
                continue

    # Load cache for each model
    for universal_model_name, model_predictions in model_to_predictions.items():
        cached_file = os.path.join(cached_results_dir, f"{universal_model_name}.jsonl")

        if not os.path.exists(cached_file):
            continue

        # Load all cached results for this model
        model_cached_results = load_jsonl_file(cached_file)

        # Create lookup by global_index
        # When multiple runs exist for the same query, prefer run_number=1
        # If run_number=1 doesn't exist, prefer successful runs, then the first one encountered
        for result in model_cached_results:
            global_index = result.get("global_index")
            if not global_index:
                continue

            key = (universal_model_name, global_index)
            current_run_number = result.get("run_number", 1)

            if key not in cached_results:
                cached_results[key] = result
                queries_with_multiple_runs[key] = [current_run_number]
            else:
                existing = cached_results[key]
                existing_run_number = existing.get("run_number", 1)
                queries_with_multiple_runs[key].append(current_run_number)

                # Prefer run_number=1
                if current_run_number == 1 and existing_run_number != 1:
                    cached_results[key] = result
                elif existing_run_number == 1 and current_run_number != 1:
                    # Keep existing (run_number=1)
                    pass
                # If neither is run_number=1, prefer successful runs
                elif result.get("success", False) and not existing.get(
                    "success", False
                ):
                    cached_results[key] = result
                # Otherwise keep existing (first one encountered)

    # Log statistics about multiple runs
    queries_with_multiples = {
        k: sorted(runs)
        for k, runs in queries_with_multiple_runs.items()
        if len(runs) > 1
    }
    if queries_with_multiples:
        total_multiples = len(queries_with_multiples)
        logger.debug(
            f"Found {total_multiples} queries with multiple runs. "
            f"Selected run_number=1 (or best available) for each."
        )

    return cached_results


def save_predictions_file(
    predictions: List[Dict[str, Any]], router_name: str, create_backup: bool = False
) -> None:
    """
    Save predictions back to file.

    Args:
        predictions: List of prediction dictionaries
        router_name: Name of the router
        create_backup: Whether to create a backup before saving (only needed once)
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.debug(f"Saved predictions to {prediction_path}")


def process_router_predictions(
    router_name: str,
    num_workers: int = 16,
    num_runs: int = 1,
    cached_results_dir: str = "./cached_results",
) -> None:
    """
    Process router predictions using parallel inference system.

    Groups predictions by model and processes each model's queries in parallel.

    Args:
        router_name: Name of the router
        num_workers: Number of parallel workers per model (default: 16)
        num_runs: Target number of successful runs per query (default: 1)
        cached_results_dir: Directory containing cached results
    """
    logger.info(f"Starting LLM inference for router: {router_name}")
    logger.info(f"Using parallel inference with {num_workers} workers per model")
    logger.info(f"Target runs per query: {num_runs}")

    # Load predictions
    predictions = load_predictions_file(router_name)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Create backup of original predictions file
    save_predictions_file(predictions, router_name, create_backup=True)

    # Filter out entries without required fields and convert to universal model names
    valid_predictions = []
    model_to_queries = defaultdict(list)  # {universal_model_name: [query_entries]}

    for pred in predictions:
        global_index = pred.get("global index") or pred.get("global_index")
        prompt = pred.get("prompt", "") or pred.get("prompt_formatted", "")
        model_name = pred.get("prediction", "")

        if not global_index or not prompt or not model_name:
            logger.warning("Skipping prediction: missing required fields")
            continue

        # Convert to universal model name
        try:
            universal_model_name = ModelNameManager.get_universal_name(model_name)
        except Exception as e:
            logger.error(
                f"Error converting model name '{model_name}' to universal name: {e}\n"
                f"Make sure the model is in universal_model_names.py"
            )
            continue

        # Create data entry format for parallel inference
        query_entry = {
            "global_index": global_index,
            "global index": global_index,  # Support both formats
            "prompt": prompt,
            "prompt_formatted": prompt,  # Support both formats
        }

        model_to_queries[universal_model_name].append(query_entry)
        valid_predictions.append((pred, universal_model_name, global_index))

    logger.info(f"Grouped predictions into {len(model_to_queries)} models")
    for model, queries in model_to_queries.items():
        logger.info(f"  {model}: {len(queries)} queries")

    # Initialize parallel inference manager
    manager = ParallelInferenceManager(
        cache_dir=cached_results_dir, workers=num_workers
    )

    start_time = datetime.datetime.now()
    all_stats = {}

    # Process each model with its assigned queries
    for model_idx, (universal_model_name, queries) in enumerate(
        model_to_queries.items(), 1
    ):
        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"Processing model {model_idx}/{len(model_to_queries)}: {universal_model_name}"
        )
        logger.info(f"{'=' * 80}")

        # Process this model's queries
        stats = manager.process_single_model(
            model=universal_model_name,
            data=queries,
            num_workers=num_workers,
            num_runs=num_runs,
        )
        all_stats[universal_model_name] = stats

    # Load all cached results to update predictions
    logger.info("\nLoading cached results to update predictions...")
    cached_results = load_cached_results_for_predictions(
        predictions, cached_results_dir
    )

    # Update predictions with results from cache
    updated_count = 0
    for pred, universal_model_name, global_index in valid_predictions:
        key = (universal_model_name, global_index)
        if key in cached_results:
            result_entry = cached_results[key]

            # Update prediction entry with generated_result
            pred["generated_result"] = {
                "generated_answer": result_entry.get("generated_answer", ""),
                "success": result_entry.get("success", False),
                "token_usage": result_entry.get("token_usage", {}),
                "provider": result_entry.get("provider", "unknown"),
                "error": result_entry.get("error", None),
            }
            updated_count += 1

    # Save updated predictions
    save_predictions_file(predictions, router_name, create_backup=False)

    # Final summary
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60

    logger.info("\n" + "=" * 80)
    logger.info("Processing completed!")
    logger.info("=" * 80)
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Valid predictions: {len(valid_predictions)}")
    logger.info(f"Predictions updated: {updated_count}")
    logger.info(f"Total duration: {total_duration:.1f} minutes")

    # Model statistics
    logger.info("\nModel Statistics:")
    for model, stats in all_stats.items():
        logger.info(f"  {model}:")
        logger.info(f"    Processed: {stats['processed']}")
        logger.info(f"    Successful: {stats['successful']}")
        logger.info(f"    Failed: {stats['failed']}")

    logger.info(
        f"\nPredictions saved to: ./router_inference/predictions/{router_name}.json"
    )
    logger.info("=" * 80)


def main():
    """Main function to handle command line arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Run LLM inference for router predictions using parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process router predictions with default 16 workers
  uv run python llm_inference/run.py my-router

  # Process with 8 workers and 2 runs per query
  uv run python llm_inference/run.py my-router --num-workers 8 --num-runs 2

  # Process with custom cache directory
  uv run python llm_inference/run.py my-router --cached-results-dir ./my_cache
        """,
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to ./router_inference/predictions/<router_name>.json)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers per model (default: 16)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Target number of successful inference runs per query (default: 1)",
    )
    parser.add_argument(
        "--cached-results-dir",
        type=str,
        default="./cached_results",
        help="Directory containing cached results (default: ./cached_results)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up environment (change to project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../"))
    os.chdir(base_dir)

    # Run inference
    try:
        process_router_predictions(
            router_name=args.router_name,
            num_workers=args.num_workers,
            num_runs=args.num_runs,
            cached_results_dir=args.cached_results_dir,
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Partial results have been saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
