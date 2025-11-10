# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Run LLM Evaluation for Router Predictions.

This script evaluates router predictions that have been processed by llm_inference/run.py.
It loads the router prediction file, extracts generated results, and evaluates them using
the evaluation framework from this folder.

The script:
1. Loads router predictions from router_inference/predictions/<router_name>.json
2. Evaluates each generated_result based on the query's global_index (determines dataset) and generated_answer
3. Saves evaluation results (accuracy, cost, etc.) back to the prediction file
4. Saves incrementally every N steps to preserve progress if halted mid-way

Usage:
    python llm_evaluation/run.py <router_name> <split> [--save-interval N]
"""

import argparse
import json
import os
import sys
import logging
import datetime
import math
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional
    pass

from universal_model_names import ModelNameManager

# Import model evaluator from current directory
from evaluate_models import ModelEvaluator

# Import evaluation components
from eval_reasoning import get_scorers_for_dataset
from evaluate_models import load_eval_config_for_dataset

logger = logging.getLogger(__name__)


def compute_arena_score(cost, accuracy, beta=0.1, c_max=200, c_min=0.0044):
    """
    Compute the RouterArena score S_i,β for a given cost and accuracy.

    Parameters:
    -----------
    cost : float
        The cost c_i of the model or router (per 1000 queries).
    accuracy : float
        The accuracy A_i of the model or router.
    beta : float, optional
        Weighting factor between accuracy and cost (default = 0.1).
    c_max : float, optional
        Maximum cost (default = 200).
    c_min : float, optional
        Minimum cost (default = 0.0044).

    Returns:
    --------
    float
        The computed RouterArena score S_i,β.
    """
    # Validate inputs
    if cost is None or cost <= 0:
        raise ValueError(
            f"Invalid cost value: {cost}. Cost must be positive. "
            "This usually means no entries were evaluated (all entries skipped)."
        )

    if accuracy is None or accuracy < 0 or accuracy > 1:
        raise ValueError(
            f"Invalid accuracy value: {accuracy}. Accuracy must be between 0 and 1."
        )

    # Clamp cost to valid range for log2 calculation
    cost = max(c_min, min(cost, c_max))

    # Compute normalized cost C_i
    C_i = (math.log2(c_max) - math.log2(cost)) / (math.log2(c_max) - math.log2(c_min))

    # Compute score S_i,β
    S = ((1 + beta) * accuracy * C_i) / (beta * accuracy + C_i)

    return S


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
            f"Please create the prediction file first."
        )

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions from {prediction_path}")
    return predictions


def save_predictions_file(predictions: List[Dict[str, Any]], router_name: str) -> None:
    """
    Save predictions back to file.

    Args:
        predictions: List of prediction dictionaries
        router_name: Name of the router
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.debug(f"Saved predictions to {prediction_path}")


def load_ground_truth_dataset(split: str) -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth dataset based on split from local disk or private repo.

    For "full" split: If HF_TOKEN is available, loads from RouteWorks/RouterEvalBenchmark
    (private repo with answers). Otherwise, loads from local disk (public dataset without answers).

    Args:
        split: Dataset split ("sub_10" for testing or "full" for submission)

    Returns:
        Dictionary mapping global_index to ground truth data
    """
    from datasets import load_dataset, load_from_disk
    import pandas as pd

    if split not in ["sub_10", "full"]:
        raise ValueError(f"Invalid split: {split}. Must be 'sub_10' or 'full'")

    router_eval_bench_df = None
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    # For "full" split, try private repo first if token is available
    if split == "full" and hf_token:
        logger.info("Loading full dataset with answers from private repo...")
        try:
            router_arena_dataset = load_dataset(
                "RouteWorks/RouterEvalBenchmark",
                split="full",
                token=hf_token,
            )
            router_eval_bench_df = pd.DataFrame(router_arena_dataset)
            logger.info("Successfully loaded from private repo.")
        except Exception as e:
            logger.warning(
                f"Could not load from private repo: {e}. Falling back to local dataset."
            )

    # Load from local disk if not already loaded
    if router_eval_bench_df is None:
        dataset_path = (
            "./dataset/routerarena_10" if split == "sub_10" else "./dataset/routerarena"
        )
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Please run: python scripts/process_datasets/prep_datasets.py"
            )
        logger.info(f"Loading dataset from {dataset_path}...")
        router_arena_dataset = load_from_disk(dataset_path)
        router_eval_bench_df = pd.DataFrame(router_arena_dataset)

    # Verify answers exist for "full" split
    if split == "full":
        sample_size = min(100, len(router_eval_bench_df))
        sample_answers = router_eval_bench_df.head(sample_size)["Answer"]
        has_answers = any(
            answer and str(answer).strip() != "" for answer in sample_answers
        )

        if not has_answers:
            logger.error("=" * 80)
            logger.error(
                "WARNING: The 'full' split does not contain ground truth answers."
            )
            logger.error("")
            logger.error("To submit predictions for the full dataset evaluation:")
            logger.error("1. Generate predictions for the full dataset")
            logger.error("2. Create a pull request to the RouterArena repository")
            logger.error("3. Include your predictions file in the PR")
            logger.error(
                "4. The official evaluation would be automatically conducted for you"
            )
            logger.error("=" * 80)
            raise ValueError(
                "The 'full' split does not have ground truth answers. "
                "Use 'sub_10' for local testing, or submit your predictions via issue for full evaluation."
            )

    # Convert to dictionary keyed by global_index
    ground_truth_map = {}
    for _, row in router_eval_bench_df.iterrows():
        global_index = row["Global Index"]
        ground_truth_map[global_index] = {
            "question": row["Question"],
            "global_index": global_index,
            "context": row["Context"],
            "answer": row["Answer"],
            "options": row["Options"],
            "metadata": row["Metadata"],
        }

    logger.info(f"Loaded {len(ground_truth_map)} ground truth samples")
    return ground_truth_map


# Module-level cache for LiveCodeBench dataset
_livecodebench_cache: Optional[List[Dict[str, Any]]] = None


def get_livecodebench_ground_truth(global_index: str) -> Optional[Dict[str, Any]]:
    """
    Get LiveCodeBench ground truth for a specific global_index.

    Args:
        global_index: Global index of the entry

    Returns:
        LiveCodeBench entry if found, None otherwise
    """
    global _livecodebench_cache
    try:
        from datasets import load_from_disk

        # Load LiveCodeBench dataset (cache it if needed)
        if _livecodebench_cache is None:
            dataset_path = "./dataset/livecodebench"
            if not os.path.exists(dataset_path):
                logger.warning(f"LiveCodeBench dataset not found at {dataset_path}")
                return None
            _livecodebench_cache = load_from_disk(dataset_path).to_list()

        # Find the entry with matching global_idx
        if _livecodebench_cache is None:
            return None
        for entry in _livecodebench_cache:
            if entry.get("global_idx") == global_index:
                return entry

        return None
    except Exception as e:
        logger.error(f"Error loading LiveCodeBench dataset: {e}")
        return None


def evaluate_single_prediction(
    prediction: Dict[str, Any],
    ground_truth_map: Dict[str, Dict[str, Any]],
    evaluator: ModelEvaluator,
) -> bool:
    """
    Evaluate a single prediction entry.

    Args:
        prediction: Prediction dictionary with global_index, prediction, and generated_result
        ground_truth_map: Map from global_index to ground truth data
        evaluator: ModelEvaluator instance

    Returns:
        True if evaluation succeeded, False otherwise
    """
    # Extract global_index (handle both formats)
    global_index = prediction.get("global index") or prediction.get("global_index")
    if not global_index:
        logger.warning("Skipping entry: missing global_index")
        return False

    # Extract model name and generated result
    model_name = prediction.get("prediction", "")
    generated_result = prediction.get("generated_result")

    if not model_name:
        logger.warning(
            f"Skipping entry (global_index: {global_index}): missing prediction/model name"
        )
        return False

    # Check if we have generated results
    if not generated_result or not isinstance(generated_result, dict):
        logger.warning(
            f"Skipping entry (global_index: {global_index}): no generated_result found. "
            f"Run llm_inference/run.py first."
        )
        return False

    # Check if generation was successful
    if not generated_result.get("success", False):
        logger.debug(
            f"Skipping entry (global_index: {global_index}): inference was unsuccessful"
        )
        return False

    # Convert to universal model name
    try:
        universal_model_name = ModelNameManager.get_universal_name(model_name)
    except Exception as e:
        logger.error(
            f"Error converting model name '{model_name}' to universal name: {e}"
        )
        return False

    # Determine dataset name from global_index
    dataset_name = evaluator.determine_dataset_from_global_index(global_index)

    # Get evaluation metric and scorer for this dataset
    eval_metrics = load_eval_config_for_dataset(dataset_name)
    scorers = get_scorers_for_dataset(dataset_name, eval_metrics)

    if not scorers:
        logger.warning(f"No scorers found for dataset {dataset_name}, skipping")
        return False

    try:
        # Get ground truth
        if dataset_name == "LiveCodeBench":
            ground_truth = get_livecodebench_ground_truth(global_index)
            if ground_truth is None:
                logger.warning(
                    f"No LiveCodeBench ground truth found for {global_index}"
                )
                return False
        else:
            if global_index not in ground_truth_map:
                logger.warning(
                    f"No ground truth found for global_index: {global_index}"
                )
                return False
            ground_truth = ground_truth_map[global_index]["answer"]

        # Get generated answer
        generated_answer = generated_result.get("generated_answer", "")

        # Evaluate using the appropriate scorer
        # For LiveCodeBench, ground_truth is a dict, but _evaluate_single_entry expects str
        # The scorer functions handle this internally (code_accuracy accepts dict)
        scorer_func, metric_name = scorers[0]
        if dataset_name == "LiveCodeBench":
            # LiveCodeBench ground_truth is a dict, but we pass it as-is to the scorer
            # _evaluate_single_entry will call scorer(generated_answer, ground_truth)
            # which works because code_accuracy accepts dict as ground_truth
            score, metric_name = evaluator._evaluate_single_entry(
                generated_answer,
                ground_truth,
                scorer_func,
                dataset_name,
            )
        else:
            # For other datasets, ground_truth is a string
            assert isinstance(ground_truth, str), (
                f"Expected str for {dataset_name}, got {type(ground_truth)}"
            )
            score, metric_name = evaluator._evaluate_single_entry(
                generated_answer, ground_truth, scorer_func, dataset_name
            )

        # Calculate inference cost
        token_usage = generated_result.get("token_usage", {})
        inference_cost = evaluator.calculate_inference_cost(
            universal_model_name, token_usage
        )

        # Update the prediction with evaluation results
        prediction["accuracy"] = score
        prediction["cost"] = inference_cost

        return True

    except Exception as e:
        logger.error(f"Error evaluating entry (global_index: {global_index}): {e}")
        return False


def process_router_predictions(
    router_name: str, split: str, save_interval: int = 50
) -> None:
    """
    Process router predictions by evaluating generated results with incremental saving.

    Args:
        router_name: Name of the router
        split: Dataset split ("sub_10" or "full")
        save_interval: Number of entries to process before saving (default: 50)
    """
    logger.info(f"Starting LLM evaluation for router: {router_name} (split: {split})")

    # Load predictions
    predictions = load_predictions_file(router_name)

    # Load ground truth dataset
    ground_truth_map = load_ground_truth_dataset(split)

    # Initialize model evaluator
    evaluator = ModelEvaluator(cached_results_dir="./cached_results")

    # Statistics
    total = len(predictions)
    evaluated_count = 0
    skipped_count = 0
    failed_count = 0
    already_evaluated_count = 0

    start_time = datetime.datetime.now()

    logger.info(
        "The dataset contains entries from LiveCodeBench, and it is common to wait for ~10 minutes to evaluate the sub_10 split of the dataset."
    )

    # Process each prediction with incremental saving
    for i, prediction in enumerate(predictions):
        # Check if already evaluated (has accuracy and cost)
        if (
            prediction.get("accuracy") is not None
            and prediction.get("cost") is not None
        ):
            already_evaluated_count += 1
            evaluated_count += 1
            continue

        # Evaluate the prediction
        success = evaluate_single_prediction(prediction, ground_truth_map, evaluator)

        if success:
            evaluated_count += 1
        else:
            skipped_count += 1

        # Incremental save every save_interval entries (if save_interval is reasonable)
        if save_interval <= total and (i + 1) % save_interval == 0:
            save_predictions_file(predictions, router_name)
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds() / 60
            logger.info(
                f"Progress: {i + 1}/{total} processed | "
                f"Evaluated: {evaluated_count} | Skipped: {skipped_count} | "
                f"Elapsed: {elapsed_time:.1f}min | Saved checkpoint"
            )

    # Final save
    save_predictions_file(predictions, router_name)

    # Final summary
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60

    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info(
        f"Total: {total} | Evaluated: {evaluated_count} (already done: {already_evaluated_count}) | "
        f"Skipped: {skipped_count} | Failed: {failed_count}"
    )
    logger.info(f"Total duration: {total_duration:.1f} minutes")
    logger.info(
        f"Predictions saved to: ./router_inference/predictions/{router_name}.json"
    )
    logger.info("=" * 60)

    # Compute and display router-level metrics
    compute_router_metrics(predictions, router_name)


def compute_router_metrics(predictions: List[Dict[str, Any]], router_name: str) -> None:
    """
    Compute router-level metrics (accuracy, cost, RouterArena score, etc.) and display them.

    Args:
        predictions: List of prediction dictionaries with evaluation results
        router_name: Name of the router
    """
    accuracies = []
    costs = []
    valid_cost_count = 0

    for prediction in predictions:
        accuracy = prediction.get("accuracy")
        if accuracy is not None:
            accuracies.append(accuracy)

        cost = prediction.get("cost")
        if cost is not None and cost > 0:
            costs.append(cost)
            valid_cost_count += 1

    # Check if any entries were evaluated
    if not accuracies and not costs:
        raise ValueError(
            "No entries were evaluated. All prediction entries are missing 'generated_result' fields. "
            "Please run llm_inference/run.py first to generate model outputs before evaluation."
        )

    if not accuracies:
        raise ValueError(
            "No entries have accuracy values. Cannot compute RouterArena score without accuracy data."
        )

    if not costs:
        raise ValueError(
            "No entries have valid cost values. Cannot compute RouterArena score without cost data."
        )

    # Compute average accuracy
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    # Compute total cost (sum of all costs)
    total_cost = sum(costs) if costs else 0.0

    # Compute average cost per 1000 queries for RouterArena score calculation
    num_queries = len(predictions)
    avg_cost_per_1000 = (total_cost / num_queries * 1000) if num_queries > 0 else 0.0

    # Compute RouterArena score using average cost per 1000 queries and average accuracy
    arena_score = compute_arena_score(avg_cost_per_1000, avg_accuracy)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info(f"Router: {router_name}")
    logger.info("=" * 80)
    logger.info(f"Total Queries: {num_queries}")
    logger.info(f"Queries with Accuracy: {len(accuracies)}")
    logger.info(f"Queries with Valid Cost: {valid_cost_count}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Total Cost: ${total_cost:.6f}")
    if num_queries > 0:
        logger.info(f"Average Cost per Query: ${total_cost / num_queries:.6f}")
    else:
        logger.info("Average Cost per Query: $0.00")
    logger.info(f"Average Cost per 1K Queries: ${avg_cost_per_1000:.4f}")
    logger.info(f"RouterArena Score: {arena_score:.4f}")
    logger.info(
        "PLEASE NOTE: The sub_10 dataset is a subset of the full dataset and is used for testing purposes. It is generally easier than the full dataset."
    )
    logger.info("=" * 80 + "\n")


def main():
    """Main function to handle command line arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation for router predictions"
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to ./router_inference/predictions/<router_name>.json)",
    )
    parser.add_argument(
        "split",
        type=str,
        choices=["sub_10", "full"],
        help="Dataset split to use for evaluation ('sub_10' for testing with answers, 'full' for submission)",
    )
    parser.add_argument(
        "--cached-results-dir",
        type=str,
        default="./cached_results",
        help="Directory containing cached results (default: ./cached_results)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Number of entries to process before saving (default: 10). Set to 0 to save only at the end.",
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

    # Run evaluation
    try:
        # If save_interval is 0, only save at the end
        predictions = load_predictions_file(args.router_name)
        save_interval = (
            args.save_interval if args.save_interval > 0 else len(predictions) + 1
        )
        process_router_predictions(args.router_name, args.split, save_interval)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving partial results...")
        try:
            # Try to save current state if possible
            predictions = load_predictions_file(args.router_name)
            save_predictions_file(predictions, args.router_name)
            logger.info("Partial results saved successfully.")
        except Exception as e:
            logger.warning(f"Could not save partial results: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
