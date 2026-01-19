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
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from universal_model_names import ModelNameManager
from global_utils.robustness import compute_robustness_score
from parallel_evaluation import ParallelEvaluationManager

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional
    pass


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


def load_predictions_from_path(path: str) -> List[Dict[str, Any]]:
    """
    Load predictions directly from an absolute or relative file path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    Load ground truth dataset based on split from local disk.

    The dataset is loaded from the public RouteWorks/RouterArena repository
    and saved locally by prep_datasets.py.

    Args:
        split: Dataset split ("sub_10" for testing or "full" for submission)

    Returns:
        Dictionary mapping global_index to ground truth data
    """
    from datasets import load_from_disk
    import pandas as pd

    if split not in ["sub_10", "full"]:
        raise ValueError(f"Invalid split: {split}. Must be 'sub_10' or 'full'")

    # Load from local disk if not already loaded
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
    router_name: str,
    split: str,
    save_interval: int = 50,
    num_workers: int = 4,
    force: bool = False,
) -> None:
    """
    Process router predictions by evaluating generated results with incremental saving.
    Uses multi-threading for parallel evaluation.

    Handles both regular entries and optimality entries:
    - Regular entries (for_optimality=False): Used for RouterArena score calculation
    - Optimality entries (for_optimality=True): Used only for optimality metrics, excluded from RouterArena score

    Args:
        router_name: Name of the router
        split: Dataset split ("sub_10" or "full")
        save_interval: Number of entries to process before saving (default: 50)
        num_workers: Number of worker threads for parallel processing (default: 4)
        force: If True, re-evaluate all entries even if already evaluated (default: False)
    """
    logger.info(f"Starting LLM evaluation for router: {router_name} (split: {split})")
    logger.info(f"Using {num_workers} worker threads for parallel processing")

    # Load predictions
    predictions = load_predictions_file(router_name)

    # Separate regular and optimality entries
    regular_predictions = [p for p in predictions if not p.get("for_optimality", False)]
    optimality_predictions = [p for p in predictions if p.get("for_optimality", False)]

    if optimality_predictions:
        logger.info(
            f"Found {len(optimality_predictions)} optimality entries for automatic optimality score calculation"
        )
        logger.info(
            f"Regular entries: {len(regular_predictions)}, Optimality entries: {len(optimality_predictions)}"
        )

    # Load ground truth dataset
    # For evaluation, use the appropriate split based on entry type
    ground_truth_map = load_ground_truth_dataset(split)

    # Initialize model evaluator (shared across threads - should be thread-safe for read operations)
    try:
        evaluator = ModelEvaluator(cached_results_dir="./cached_results")
    except Exception as e:
        logger.error(f"Failed to initialize ModelEvaluator: {e}")
        raise RuntimeError(f"ModelEvaluator initialization failed: {e}")

    # Initialize parallel evaluation manager
    manager = ParallelEvaluationManager(workers=num_workers)

    # Statistics
    total = len(predictions)
    already_evaluated_count = 0

    logger.info(
        "The dataset contains entries from LiveCodeBench, and it is common to wait for ~10 minutes to evaluate the sub_10 split of the dataset."
    )

    # Prepare tasks: filter out already evaluated entries (unless force is True)
    tasks = []
    for i, prediction in enumerate(predictions):
        # Check if already evaluated (has accuracy and cost)
        # Skip if already evaluated AND force is False
        if not force and (
            prediction.get("accuracy") is not None
            and prediction.get("cost") is not None
        ):
            already_evaluated_count += 1
            continue

        # Store (index, prediction) - index is the sequence number in the original list
        tasks.append((i, prediction))

    # Set already evaluated count in manager
    with manager.stats_lock:
        manager.already_evaluated_count = already_evaluated_count

    logger.info(
        f"Found {len(tasks)} entries to evaluate ({already_evaluated_count} already evaluated)"
    )

    def evaluate_task_wrapper(
        seq_idx: int, prediction: Dict[str, Any], **kwargs: Any
    ) -> bool:
        """
        Wrapper for evaluate_single_prediction to be used with ParallelEvaluationManager.
        """
        gt_map: Dict[str, Dict[str, Any]] = kwargs.get("ground_truth_map", {})
        eval_instance: ModelEvaluator = kwargs["evaluator"]

        return evaluate_single_prediction(prediction, gt_map, eval_instance)

    def save_callback():
        """Callback to save predictions file."""
        save_predictions_file(predictions, router_name)

    # Run parallel evaluation
    manager.evaluate_entries_parallel(
        tasks=tasks,
        evaluation_func=evaluate_task_wrapper,
        save_func=save_callback,
        save_interval=save_interval,
        total_count=total,
        ground_truth_map=ground_truth_map,
        evaluator=evaluator,
    )

    # Final summary
    stats = manager.get_stats()
    evaluated_count = stats["evaluated"]
    skipped_count = stats["skipped"]
    failed_count = stats["failed"]
    total_duration = stats["total_duration_min"]

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


def _prepare_optimality_data(
    predictions: List[Dict[str, Any]],
) -> Optional[Tuple[Set[str], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Load sub_10 dataset and separate predictions into router selections and optimality entries.

    Args:
        predictions: List of all prediction dictionaries (regular + optimality)

    Returns:
        Tuple of (sub10_indices, router_selections, optimality_entries), or None if dataset not found
    """
    # Load sub_10 dataset to identify which entries are relevant
    dataset_path = "./dataset/router_data_10.json"
    if not os.path.exists(dataset_path):
        logger.warning(f"Sub_10 dataset not found at {dataset_path}")
        return None

    with open(dataset_path, "r", encoding="utf-8") as f:
        sub10_dataset = json.load(f)

    sub10_indices: Set[str] = set()
    for entry in sub10_dataset:
        idx = entry.get("global index") or entry.get("global_index")
        if idx is not None:
            sub10_indices.add(str(idx))

    # Separate predictions into router selections and optimality entries
    router_selections: Dict[str, Dict[str, Any]] = {}  # {global_index: prediction_dict}
    optimality_entries: List[Dict[str, Any]] = []  # List of optimality prediction dicts

    for prediction in predictions:
        global_index_raw = prediction.get("global index") or prediction.get(
            "global_index"
        )
        if global_index_raw is None:
            continue
        global_index = str(global_index_raw)

        # Only process sub_10 entries
        if global_index not in sub10_indices:
            continue

        if prediction.get("for_optimality", False):
            optimality_entries.append(prediction)
        else:
            router_selections[global_index] = prediction

    if not router_selections:
        logger.warning("No router selections found for sub_10 queries")
        return None

    return sub10_indices, router_selections, optimality_entries


def _build_evaluation_dict(
    router_selections: Dict[str, Dict[str, Any]],
    optimality_entries: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Build evaluation dictionary mapping global_index to model results.

    Args:
        router_selections: Dictionary mapping global_index to router prediction dict
        optimality_entries: List of optimality prediction dicts

    Returns:
        Dictionary mapping global_index to model results: {global_index: {model: (accuracy, cost)}}
    """
    evaluation_dict: Dict[str, Dict[str, Tuple[float, float]]] = {}

    # Add router selections
    for global_index, pred in router_selections.items():
        if not global_index:  # Skip if None
            continue
        model = pred.get("prediction")
        accuracy = pred.get("accuracy")
        cost = pred.get("cost")

        if model and accuracy is not None and cost is not None:
            if global_index not in evaluation_dict:
                evaluation_dict[global_index] = {}
            evaluation_dict[global_index][model] = (accuracy, cost)

    # Add optimality entries (other models evaluated for same queries)
    for pred in optimality_entries:
        global_index_raw = pred.get("global index") or pred.get("global_index")
        if global_index_raw is None:
            continue
        global_index = str(global_index_raw)
        model = pred.get("prediction")
        accuracy = pred.get("accuracy")
        cost = pred.get("cost")

        if model and accuracy is not None and cost is not None:
            if global_index not in evaluation_dict:
                evaluation_dict[global_index] = {}
            evaluation_dict[global_index][model] = (accuracy, cost)

    return evaluation_dict


def _find_optimal_model(
    model_results: Dict[str, Tuple[float, float]],
) -> Optional[Tuple[str, float, float]]:
    """
    Find optimal model for a single query.

    Optimal model is defined as the model with perfect accuracy (acc >= 1.0) and lowest cost.

    Args:
        model_results: Dictionary mapping model names to (accuracy, cost) tuples

    Returns:
        Tuple of (optimal_model_name, optimal_accuracy, optimal_cost), or None if no optimal model found
    """
    optimal_model_name: Optional[str] = None
    optimal_accuracy: Optional[float] = None
    optimal_cost = float("inf")

    for model, (acc, cost) in model_results.items():
        # Only consider models with perfect accuracy
        if acc >= 1.0 and cost is not None and cost < optimal_cost:
            optimal_accuracy = acc
            optimal_cost = cost
            optimal_model_name = model

    if optimal_model_name is None or optimal_accuracy is None:
        return None

    return optimal_model_name, optimal_accuracy, optimal_cost


def _calculate_optimality_metrics(
    optimal_selections: int,
    queries_with_optimal_data: int,
    total_optimal_cost: float,
    total_router_cost: float,
    total_optimal_accuracy: float,
    total_router_accuracy: float,
) -> Dict[str, float]:
    """
    Calculate final optimality metrics from accumulated data.

    Args:
        optimal_selections: Number of queries where router selected optimal model
        queries_with_optimal_data: Total number of queries with optimal data
        total_optimal_cost: Sum of optimal costs
        total_router_cost: Sum of router costs
        total_optimal_accuracy: Sum of optimal accuracies
        total_router_accuracy: Sum of router accuracies

    Returns:
        Dictionary with opt_sel, opt_cost, and opt_acc metrics
    """
    # Compute final metrics as decimals (0-1, not percentages)
    opt_sel = (
        (optimal_selections / queries_with_optimal_data)
        if queries_with_optimal_data > 0
        else 0.0
    )
    # Cost efficiency: total optimal cost / total router cost
    opt_cost = total_optimal_cost / total_router_cost if total_router_cost > 0 else 0.0
    # Accuracy efficiency: total router accuracy / total optimal accuracy
    opt_acc = (
        total_router_accuracy / total_optimal_accuracy
        if total_optimal_accuracy > 0
        else 0.0
    )

    return {
        "opt_sel": opt_sel,
        "opt_cost": opt_cost,
        "opt_acc": opt_acc,
    }


def compute_optimality_from_predictions(
    predictions: List[Dict[str, Any]], router_name: str
) -> Optional[Dict[str, Any]]:
    """
    Compute optimality scores from prediction file.

    Uses:
    - Sub_10 entries with router-selected model (for_optimality=False)
    - Sub_10 entries with other models in pool (for_optimality=True)

    Computes:
    - Opt.Sel: Percentage of queries where router selected the optimal model
    - Opt.Cost: Average ratio of optimal cost to router cost
    - Opt.Acc: Average ratio of router accuracy to optimal accuracy

    Args:
        predictions: List of all prediction dictionaries (regular + optimality)
        router_name: Name of the router

    Returns:
        Dictionary with optimality metrics, or None if computation fails
    """
    try:
        # 1. Prepare data: load dataset and separate predictions
        prepared_data = _prepare_optimality_data(predictions)
        if prepared_data is None:
            return None
        sub10_indices, router_selections, optimality_entries = prepared_data

        # 2. Build evaluation dictionary
        evaluation_dict = _build_evaluation_dict(router_selections, optimality_entries)
        if not evaluation_dict:
            logger.warning("No evaluated entries found for optimality computation")
            return None

        # 3. For each query, find optimal model and compare with router selection
        optimal_selections = 0
        total_optimal_cost = 0.0
        total_router_cost = 0.0
        total_optimal_accuracy = 0.0
        total_router_accuracy = 0.0
        queries_with_optimal_data = 0

        model_name_manager = ModelNameManager()

        for global_index, router_pred in router_selections.items():
            # Get router's selection
            router_model = router_pred.get("prediction")
            router_accuracy = router_pred.get("accuracy")
            router_cost = router_pred.get("cost")

            if not router_model:
                continue

            # Skip if accuracy or cost is None or both are 0
            if (
                router_accuracy is None
                or router_cost is None
                or (router_accuracy == 0 and router_cost == 0)
            ):
                continue

            # Get all model results for this query
            if global_index not in evaluation_dict:
                continue

            model_results = evaluation_dict[global_index]

            # Find optimal model
            optimal_result = _find_optimal_model(model_results)
            if optimal_result is None:
                continue

            optimal_model_name, optimal_accuracy, optimal_cost = optimal_result

            # Skip if router didn't achieve perfect accuracy
            if router_accuracy < 1.0:
                continue

            queries_with_optimal_data += 1

            # Convert router model name to universal name for comparison (same as leaderboard)
            try:
                router_model_universal = model_name_manager.get_universal_name(
                    router_model
                )
            except Exception:
                # If conversion fails, try using the model name as-is
                router_model_universal = router_model

            # Convert optimal model name to universal for comparison
            try:
                optimal_model_universal = model_name_manager.get_universal_name(
                    optimal_model_name
                )
            except Exception:
                optimal_model_universal = optimal_model_name

            # Check if router selected the optimal model
            if router_model_universal == optimal_model_universal:
                optimal_selections += 1

            # Accumulate accuracies (both should be >= 1.0 at this point)
            if optimal_accuracy is not None:
                total_optimal_accuracy += optimal_accuracy
                total_router_accuracy += router_accuracy

            # Accumulate costs (only for queries where both achieved perfect accuracy)
            if optimal_cost != float("inf") and router_cost > 0:
                total_optimal_cost += optimal_cost
                total_router_cost += router_cost

        # 4. Calculate final metrics
        metrics = _calculate_optimality_metrics(
            optimal_selections,
            queries_with_optimal_data,
            total_optimal_cost,
            total_router_cost,
            total_optimal_accuracy,
            total_router_accuracy,
        )

        logger.info(
            f"Optimality computation: {queries_with_optimal_data} queries, "
            f"{optimal_selections} optimal selections"
        )

        return {
            "router_name": router_name,
            "num_sub10_queries": len(sub10_indices),
            "queries_with_optimal_data": queries_with_optimal_data,
            "optimal_selections": optimal_selections,
            **metrics,
        }

    except Exception as e:
        logger.error(f"Error computing optimality scores: {e}", exc_info=True)
        return None


def run_robustness_only(router_name: str, robustness_path: Optional[str]) -> None:
    """
    Compute robustness score without running full evaluation.

    Args:
        router_name: Name of the router whose full split predictions will be used.
        robustness_path: Path to robustness predictions; if None, resolve to default.
    """

    default_path = os.path.join(
        "./router_inference/predictions", f"{router_name}-robustness.json"
    )
    target_path = robustness_path or default_path

    logger.info(
        "Computing robustness score using %s (full) and %s (robustness)",
        f"./router_inference/predictions/{router_name}.json",
        target_path,
    )

    predictions = load_predictions_file(router_name)

    try:
        robustness_predictions = load_predictions_from_path(target_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Robustness predictions not found at "
            f"{target_path}. Generate them with "
            "router_inference/generate_prediction_file.py <router> robustness."
        )
    except json.JSONDecodeError:
        raise RuntimeError(f"Unable to load robustness predictions from {target_path}")

    score = compute_robustness_score(predictions, robustness_predictions)
    if score is None:
        raise ValueError(
            "Could not compute robustness score because no overlapping global indices were found."
        )

    logger.info("Robustness score: %.4f", score)

    metrics_path = "./metrics.json"
    metrics_payload = {"robustness_score": score}
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)
    logger.info("Robustness metrics saved to %s", metrics_path)


def compute_router_metrics(predictions: List[Dict[str, Any]], router_name: str) -> None:
    """
    Compute router-level metrics (accuracy, cost, RouterArena score, etc.) and display them.

    Separates regular and optimality entries:
    - Regular entries (for_optimality=False): Used for RouterArena score
    - Optimality entries (for_optimality=True): Used for optimality metrics only

    Args:
        predictions: List of prediction dictionaries with evaluation results
        router_name: Name of the router
    """
    # Separate regular and optimality predictions
    regular_predictions = [p for p in predictions if not p.get("for_optimality", False)]
    optimality_predictions = [p for p in predictions if p.get("for_optimality", False)]

    # Extract accuracy and cost ONLY from regular predictions for RouterArena score
    accuracies = []
    costs = []
    valid_cost_count = 0

    for prediction in regular_predictions:
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
    # Use only regular predictions count
    num_queries = len(regular_predictions)
    avg_cost_per_1000 = (total_cost / num_queries * 1000) if num_queries > 0 else 0.0

    # Compute RouterArena score using average cost per 1000 queries and average accuracy
    arena_score = compute_arena_score(avg_cost_per_1000, avg_accuracy)

    # Print RouterArena score results
    logger.info("\n" + "=" * 80)
    logger.info(f"Router: {router_name}")
    logger.info("=" * 80)
    logger.info(f"Total Queries (Regular): {num_queries}")
    if optimality_predictions:
        logger.info(
            f"Optimality Entries: {len(optimality_predictions)} (excluded from RouterArena score)"
        )
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

    # Compute optimality scores if we have optimality entries (compute once, reuse for logging and metrics)
    optimality_scores = None
    if optimality_predictions:
        logger.info("\n" + "-" * 80)
        logger.info("Computing Optimality Scores...")
        logger.info("-" * 80)
        try:
            optimality_scores = compute_optimality_from_predictions(
                predictions, router_name
            )
            if optimality_scores:
                logger.info(
                    f"Opt.Sel (Optimal Selection): {optimality_scores['opt_sel']:.4f}\n"
                    f"Opt.Cost (Optimal Cost Ratio): {optimality_scores['opt_cost']:.4f}\n"
                    f"Opt.Acc (Optimal Accuracy Ratio): {optimality_scores['opt_acc']:.4f}\n"
                    f"Queries Evaluated for Optimality: {optimality_scores['queries_with_optimal_data']}/{optimality_scores['num_sub10_queries']}"
                )

                # Print optimality scores in JSON format for easy parsing by automation
                logger.info(
                    "\nOptimality Scores: "
                    + json.dumps(
                        {
                            "opt_sel": optimality_scores["opt_sel"],
                            "opt_cost": optimality_scores["opt_cost"],
                            "opt_acc": optimality_scores["opt_acc"],
                            "queries_with_optimal_data": optimality_scores[
                                "queries_with_optimal_data"
                            ],
                            "num_sub10_queries": optimality_scores["num_sub10_queries"],
                        }
                    )
                )
        except Exception as e:
            logger.warning(f"Could not compute optimality scores: {e}")
            logger.warning(
                "This is expected if optimality entries were not generated during prediction file creation."
            )

    logger.info(
        "\nPLEASE NOTE: The sub_10 dataset is a subset of the full dataset and is used for testing purposes. It is generally easier than the full dataset."
    )
    logger.info("=" * 80 + "\n")

    # Save metrics to metrics.json for automation/workflows
    metrics_dict = {
        "arena_score": arena_score,
        "accuracy": avg_accuracy,
        "total_cost": total_cost,
        "avg_cost_per_query": total_cost / num_queries if num_queries > 0 else 0.0,
        "avg_cost_per_1000": avg_cost_per_1000,
        "num_queries": num_queries,
    }

    # Add optimality scores if available (reuse previously computed result)
    if optimality_scores:
        metrics_dict["optimality"] = {
            "opt_sel": optimality_scores["opt_sel"],
            "opt_cost": optimality_scores["opt_cost"],
            "opt_acc": optimality_scores["opt_acc"],
            "queries_with_optimal_data": optimality_scores["queries_with_optimal_data"],
            "num_sub10_queries": optimality_scores["num_sub10_queries"],
        }

    # Save to metrics.json
    metrics_path = "./metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"\n✓ Metrics saved to {metrics_path}")


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
        nargs="?",
        type=str,
        choices=["sub_10", "full", "robustness"],
        help=(
            "Dataset split to use for evaluation ('sub_10' for testing with answers, "
            "'full' for submission, 'robustness' to compute robustness score only)."
        ),
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker threads for parallel processing (default: 16). Set to 1 for sequential processing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-evaluation of all entries, even if already evaluated (default: False)",
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

    default_robustness_path = os.path.join(
        "./router_inference/predictions", f"{args.router_name}-robustness.json"
    )

    if args.split is None:
        parser.error("split is required (sub_10, full, or robustness).")

    if args.split == "robustness":
        run_robustness_only(args.router_name, default_robustness_path)
        return

    # Run evaluation
    try:
        # If save_interval is 0, only save at the end
        predictions = load_predictions_file(args.router_name)
        save_interval = (
            args.save_interval if args.save_interval > 0 else len(predictions) + 1
        )

        process_router_predictions(
            args.router_name,
            args.split,
            save_interval,
            args.num_workers,
            args.force,
        )
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
