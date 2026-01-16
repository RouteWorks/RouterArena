# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Batch LLM Inference Script

This script processes multiple models from model_cost.json sequentially,
using parallel workers for query processing within each model.

Architecture:
- Processes models sequentially (one at a time)
- Within each model, uses k workers for parallel query processing
- Example: 8400 queries with 16 workers → each worker handles ~525 queries

Usage:
    # Process all models from model_cost.json with 16 workers per model
    uv run python llm_inference/batch_inference.py --num-workers 16

    # Process specific models only
    uv run python llm_inference/batch_inference.py \
        --models gemini-2.0-flash-001 gpt-5-mini \
        --num-workers 16
"""

import argparse
import json
import os
import sys
import logging
import datetime
from typing import List, Optional
from parallel_inference import ParallelInferenceManager

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(base_dir)

# Load environment variables from .env file (use absolute path)
try:
    from dotenv import load_dotenv

    env_path = os.path.join(base_dir, ".env")
    load_dotenv(env_path)
except ImportError:
    # dotenv is optional
    pass


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_list_from_cost_file(
    model_cost_path: str = "./model_cost/model_cost.json",
    specified_models: Optional[List[str]] = None,
) -> List[str]:
    """
    Load list of models to process from model_cost.json

    Args:
        model_cost_path: Path to model_cost.json
        specified_models: Optional list of specific models to process

    Returns:
        List of model names to process
    """
    if not os.path.exists(model_cost_path):
        raise FileNotFoundError(f"model_cost.json not found at {model_cost_path}")

    with open(model_cost_path, "r", encoding="utf-8") as f:
        model_cost = json.load(f)

    all_models = list(model_cost.keys())

    if specified_models:
        # Validate specified models exist in model_cost.json
        invalid_models = [m for m in specified_models if m not in all_models]
        if invalid_models:
            logger.warning(
                f"These models not found in model_cost.json: {invalid_models}"
            )

        models = [m for m in specified_models if m in all_models]
        logger.info(
            f"Processing {len(models)} specified models (out of {len(all_models)} available)"
        )
    else:
        models = all_models
        logger.info(f"Processing all {len(models)} models from model_cost.json")

    return models


def main():
    """Main function to handle batch inference."""
    parser = argparse.ArgumentParser(
        description="Batch LLM Inference - Process multiple models from model_cost.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models from model_cost.json with 16 workers per model
  uv run python llm_inference/batch_inference.py --num-workers 16

  # Process all models with 2 runs per query
  uv run python llm_inference/batch_inference.py \\
      --num-workers 16 \\
      --num-runs 2

  # Process specific models only with 3 runs per query
  uv run python llm_inference/batch_inference.py \\
      --models gemini-2.0-flash-001 gpt-5-mini \\
      --num-workers 16 \\
      --num-runs 3

  # Process with custom cache directory and 8 workers
  uv run python llm_inference/batch_inference.py \\
      --cache-dir ./my_cache \\
      --num-workers 8
        """,
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
        "--models",
        nargs="+",
        help="Specific models to process (default: all models from model_cost.json)",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cached_results",
        help="Directory where cached results are stored (default: ./cached_results)",
    )
    parser.add_argument(
        "--model-cost-path",
        default="./model_cost/model_cost.json",
        help="Path to model_cost.json (default: ./model_cost/model_cost.json)",
    )
    parser.add_argument(
        "--input-file",
        default="./llm_inference/datasets/router_data.json",
        help="Path to input data file (default: ./llm_inference/datasets/router_data.json)",
    )

    args = parser.parse_args()

    try:
        # Convert relative paths to absolute paths based on project root
        if not os.path.isabs(args.model_cost_path):
            args.model_cost_path = os.path.join(base_dir, args.model_cost_path)
        if not os.path.isabs(args.input_file):
            args.input_file = os.path.join(base_dir, args.input_file)
        if not os.path.isabs(args.cache_dir):
            args.cache_dir = os.path.join(base_dir, args.cache_dir)

        start_time = datetime.datetime.now()

        logger.info("\n" + "=" * 80)
        logger.info("BATCH INFERENCE STARTING")
        logger.info("=" * 80)
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Workers per model: {args.num_workers}")
        logger.info(f"Target runs per query: {args.num_runs}")
        logger.info(f"Cache directory: {args.cache_dir}")
        logger.info(f"Input file: {args.input_file}")
        logger.info("=" * 80 + "\n")

        # Validate input file exists
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(
                f"Input file not found: {args.input_file}\n"
                f"Please run: uv run python scripts/process_datasets/prep_datasets.py"
            )

        # Load models to process
        models = load_model_list_from_cost_file(
            model_cost_path=args.model_cost_path, specified_models=args.models
        )

        if not models:
            logger.error("No models to process!")
            return 1

        logger.info(f"Models to process: {models}\n")

        # Initialize parallel inference manager
        manager = ParallelInferenceManager(
            cache_dir=args.cache_dir, workers=args.num_workers
        )

        # Load input data once (will be reused for all models)
        data = manager.load_input_data(args.input_file)
        logger.info(f"Loaded {len(data)} queries from input file\n")

        # Process all models sequentially
        all_stats = manager.process_all_models(
            models=models,
            data=data,
            num_workers=args.num_workers,
            num_runs=args.num_runs,
        )

        # Final summary
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        logger.info("\n" + "=" * 80)
        logger.info("BATCH INFERENCE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration:.1f} minutes")
        logger.info("=" * 80)

        # Summary statistics
        total_processed = sum(s["processed"] for s in all_stats.values())
        total_successful = sum(s["successful"] for s in all_stats.values())
        total_failed = sum(s["failed"] for s in all_stats.values())

        logger.info("\nSummary Statistics:")
        logger.info(f"  Models processed: {len(models)}")
        logger.info(f"  Total queries processed: {total_processed}")
        logger.info(f"  Total successful: {total_successful}")
        logger.info(f"  Total failed: {total_failed}")

        if total_processed > 0:
            success_rate = (total_successful / total_processed) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")

        logger.info("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error in batch inference: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
