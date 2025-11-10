# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import os
import json
from typing import Any, Dict
from metrics import (
    mcq_exact_match,
    mcq_accuracy,
    math_metric,
    code_accuracy,
    exact_match,
    meteor_score,
    chess_accuracy,
    superglue_exact_match,
    superglue_clozetest,
)
from datasets import load_from_disk

# Dataset to metric mapping
dataset2metric = {
    "lsat-ar": [mcq_exact_match],
    "AIME2024": [math_metric],
    "MATH-500": [math_metric],
    "MMLUPro": [mcq_accuracy],
    "livecodebench": [code_accuracy],
    "OpenTDB": [mcq_accuracy],
    "MATH": [math_metric],
    "NarrativeQA": [meteor_score],
}

# Evaluation metric name mapping
eval_metric2scorer = {
    "mcq_accuracy": mcq_accuracy,
    "mcq_exact_match": mcq_exact_match,
    "math_metric": math_metric,
    "exact_match": exact_match,
    "code_accuracy": code_accuracy,
    "meteor_score": meteor_score,
    "chess_accuracy": chess_accuracy,
    "superglue_exact_match": superglue_exact_match,
    "superglue_clozetest": superglue_clozetest,
}


def get_scorers_for_dataset(dataset_name, eval_metrics):
    """
    Select the appropriate scorers based on dataset name and evaluation metrics.

    Args:
        dataset_name: Name of the dataset (e.g., "MMLUPro")
        eval_metrics: List of evaluation metrics (e.g., ["mcq_accuracy"])

    Returns:
        list of (scorer function, metric name) tuples
    """
    scorers = []

    # If eval_metrics is provided, use those
    if eval_metrics:
        for metric_name in eval_metrics:
            if metric_name in eval_metric2scorer:
                scorers.append((eval_metric2scorer[metric_name], metric_name))
            else:
                print(f"Warning: Unknown metric {metric_name}, skipping")

    # If no valid metrics from eval_metrics, fall back to dataset-based mapping
    if not scorers and dataset_name in dataset2metric:
        for scorer_func in dataset2metric[dataset_name]:
            scorers.append((scorer_func, scorer_func.__name__))

    # Default fallback
    if not scorers:
        print(
            f"Warning: No specific scorer found for dataset {dataset_name}, using mcq_accuracy"
        )
        scorers.append((mcq_accuracy, "mcq_accuracy"))

    return scorers


def match_predictions_with_ground_truth(pred_file_path, all_data, dataset_name=None):
    """
    Match predictions with ground truth data using global indices.

    Args:
        pred_file_path: Path to the prediction JSONL file
        all_data: List of ground truth data with global indices
        dataset_name: Name of the dataset to determine ground truth format

    Returns:
        predictions: List of prediction strings
        ground_truths: List of corresponding ground truth data (answers or full problems)
        matched_data: List of matched prediction-ground truth pairs with metadata
    """
    # Create a mapping from global index to ground truth data
    gt_index_map = {item["global index"]: item for item in all_data}

    # For LiveCodeBench, also load the complete dataset for matching
    livecodebench_dataset = load_from_disk("./dataset/livecodebench").to_list()

    predictions = []
    ground_truths = []
    matched_data = []
    unmatched_count = 0

    with open(pred_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Extract global index from the ID (e.g., "MMLUPro_business_0" -> 0)
            # This might need adjustment based on your ID format
            pred_id = data.get("id", "")

            # Find corresponding ground truth
            if pred_id in gt_index_map:
                gt_data = gt_index_map[pred_id]
                predictions.append(data["prediction"])

                # For other datasets, just use the answer
                if dataset_name == "LiveCodeBench":
                    ground_truths.append(
                        livecodebench_dataset[int(pred_id.split("_")[-1])]
                    )
                elif dataset_name == "SuperGLUE-ClozeTest":
                    # Find the index of the answer in options, with fallback handling
                    try:
                        answer_index = gt_data["options"].index(gt_data["answer"])
                        ground_truths.append(answer_index)
                    except ValueError:
                        # If exact match fails, try to find the best match
                        answer_text = gt_data["answer"]
                        best_match_index = None
                        best_match_score = 0

                        for i, option in enumerate(gt_data["options"]):
                            # Simple similarity check - if answer is contained in option or vice versa
                            if (
                                answer_text.lower() in option.lower()
                                or option.lower() in answer_text.lower()
                            ):
                                # Calculate a simple similarity score
                                score = len(
                                    set(answer_text.lower().split())
                                    & set(option.lower().split())
                                )
                                if score > best_match_score:
                                    best_match_score = score
                                    best_match_index = i

                        if best_match_index is not None:
                            ground_truths.append(best_match_index)
                        else:
                            # If no match found, use the answer text directly
                            print(
                                f"Warning: Could not find answer '{answer_text}' in options for SuperGLUE-ClozeTest, using answer text directly"
                            )
                            ground_truths.append(answer_text)
                else:
                    ground_truths.append(gt_data["answer"])

                matched_data.append(
                    {
                        "prediction_id": pred_id,
                        "global_index": pred_id,
                        "prediction": data["prediction"],
                        "ground_truth": gt_data["answer"],
                        "question": gt_data["question"],
                        "context": gt_data.get("context", ""),
                        "options": gt_data.get("options", []),
                        "metadata": gt_data.get("metadata", {}),
                    }
                )

                if "provider" in data:
                    matched_data[-1]["provider"] = data["provider"]
                if "router" in data:
                    matched_data[-1]["router"] = data["router"]
            else:
                print(f"Warning: No ground truth found for global index {pred_id}")
                unmatched_count += 1

    if unmatched_count > 0:
        raise ValueError(
            f"Warning: {unmatched_count} predictions could not be matched with ground truth"
        )

    return predictions, ground_truths, matched_data


def eval(pred_dir, eval_params, pipeline_config, all_data):
    """
    Evaluate predictions in pred_dir against ground truth data.

    Args:
        pred_dir: Path to the prediction JSONL file
        eval_params: Evaluation parameters including dataset name and metrics
        pipeline_config: Pipeline configuration
        all_data: Ground truth data from utils.py

    Returns:
        scores: Dictionary of evaluation scores
        raw_results: Detailed results for each prediction
    """
    scores: Dict[str, float] = {}
    all_raw_results: Dict[str, Dict[str, Any]] = {}

    # Get the appropriate scorers for this dataset and metrics
    dataset_name = eval_params["dataset"]
    eval_metrics = eval_params.get("eval_metrics", [])
    scorers = get_scorers_for_dataset(dataset_name, eval_metrics)

    print(
        f"Using scorers: {[metric_name for _, metric_name in scorers]} for dataset: {dataset_name}"
    )
    print(f"Evaluating predictions from: {pred_dir}")

    # Match predictions with ground truth
    predictions, ground_truths, matched_data = match_predictions_with_ground_truth(
        pred_dir, all_data, dataset_name
    )

    if len(predictions) == 0:
        print(f"Warning: No valid predictions found in {pred_dir}")
        return scores, all_raw_results

    print(f"Found {len(predictions)} matched prediction-ground truth pairs")

    filename = os.path.basename(pred_dir)
    all_raw_results[filename] = {
        "total_predictions": len(predictions),
        "matched_data": matched_data,
        "metrics": {},
    }

    # Calculate scores using all selected scorers
    for scorer, metric_name in scorers:
        try:
            score, raw_results = scorer(
                predictions, ground_truths, matched_data=matched_data
            )
            scores[f"{filename}_{metric_name}"] = score
            all_raw_results[filename]["metrics"][metric_name] = {
                "score": score,
                "raw_results": raw_results,
            }
            print(f"Score for {filename} ({metric_name}): {score:.4f}")

        except Exception as e:
            print(f"Error evaluating {pred_dir} with {metric_name}: {str(e)}")
            scores[f"{filename}_{metric_name}"] = 0.0
            all_raw_results[filename]["metrics"][metric_name] = {
                "score": 0.0,
                "error": str(e),
            }

    # Save results in the same directory as the prediction file
    pred_dir_path = os.path.dirname(pred_dir)
    out_path = os.path.join(pred_dir_path, "processed_result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Save detailed results
    detailed_out_path = os.path.join(pred_dir_path, "detailed_results.json")
    with open(detailed_out_path, "w") as f:
        json.dump(all_raw_results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {out_path}")
    print(f"Detailed results saved to {detailed_out_path}")

    return scores, all_raw_results
