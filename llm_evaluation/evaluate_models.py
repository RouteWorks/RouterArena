# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Model Evaluation Pipeline

This script evaluates model outputs from cached results using the evaluation
framework from eval_reasoning.py. It processes cached results and updates them
in-place with evaluation scores.

Usage:
    python evaluate_models.py MODEL_NAME [--cached-results-dir CACHED_RESULTS_DIR]
"""

import os
import json
import argparse
import glob
from typing import Dict, List, Any, Optional
import sys
from tqdm import tqdm

# Add the current directory to Python path to import eval modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to import model name manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_reasoning import get_scorers_for_dataset

# Import ModelNameManager for model name validation and conversion
try:
    from universal_model_names import ModelNameManager

    model_name_manager: Optional[ModelNameManager]
    model_name_manager = ModelNameManager()
except ImportError:
    print("Warning: Could not import ModelNameManager. Model name validation disabled.")
    model_name_manager = None


def load_eval_config_for_dataset(dataset_name: str) -> List[str]:
    """
    Load evaluation configuration for a dataset and return the eval_metrics.

    Args:
        dataset_name: Name of the dataset (e.g., "AIME", "MMLU")

    Returns:
        List of evaluation metrics for the dataset
    """
    if dataset_name == "GeoGraphyData_100k":
        dataset_name = "GeoGraphyData"
    # Try to find the config file in the eval_config directory
    config_paths = [
        f"../config/eval_config/zero-shot/{dataset_name}.json",
        f"config/eval_config/zero-shot/{dataset_name}.json",
        f"/home/jy101/yifan/RouterArena/config/eval_config/zero-shot/{dataset_name}.json",
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    eval_metrics = config.get("eval_params", {}).get("eval_metrics", [])
                    return eval_metrics
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load config for {dataset_name}: {e}")
                continue

    # If no config found, return empty list (will fall back to dataset-based mapping)
    print(
        f"Warning: No eval config found for dataset {dataset_name}, using default metrics"
    )
    return []


class ModelEvaluator:
    """Handles evaluation of model outputs using the existing evaluation framework."""

    def __init__(self, cached_results_dir: str = "../cached_results/"):
        self.cached_results_dir = cached_results_dir
        self.all_data: Optional[List[Dict[str, Any]]] = None
        self.dataset_configs: Dict[str, Dict[str, Any]] = {}
        self.existing_results: Dict[
            str, Any
        ] = {}  # Store existing results for incremental evaluation
        self.cost_config: Dict[str, Any] = {}  # Store cost configuration

        # Load dataset configurations
        self.load_dataset_configs()

        # Load cost configuration
        self.load_cost_config()

    def load_all_data(self):
        """Load the complete ground truth data from utils.py"""
        print("Loading ground truth data...")
        try:
            # Load data directly without LiveCodeBench dependency
            from datasets import load_from_disk
            import pandas as pd

            # Load the router eval benchmark dataset
            router_eval_bench = load_from_disk("./dataset/routerarena")
            router_eval_bench_df = pd.DataFrame(router_eval_bench)

            # Convert to the expected format
            self.all_data = []
            for _, row in router_eval_bench_df.iterrows():
                self.all_data.append(
                    {
                        "question": row["Question"],
                        "global index": row["Global Index"],
                        "context": row["Context"],
                        "answer": row["Answer"],
                        "options": row["Options"],
                        "metadata": row["Metadata"],
                    }
                )

            print(f"Loaded {len(self.all_data)} ground truth samples")
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            raise

    def load_dataset_configs(self):
        """Load all dataset configuration files."""
        config_dir = "./config/eval_config/zero-shot"
        self.dataset_configs = {}

        # Get all dataset names from the available config files
        config_files = glob.glob(f"{config_dir}/*.json")

        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    dataset_name = config["eval_params"]["dataset"]
                    self.dataset_configs[dataset_name] = config["eval_params"]
            except Exception as e:
                print(f"Warning: Could not load config {config_file}: {e}")

    def load_cost_config(self):
        """Load cost configuration from model_cost/cost.json"""
        # Try multiple possible paths for cost file
        possible_paths = [
            "./model_cost/cost.json",
            "../model_cost/cost.json",
            "model_cost/cost.json",
        ]

        cost_file = None
        for path in possible_paths:
            if os.path.exists(path):
                cost_file = path
                break

        if not cost_file:
            print(
                f"Warning: Could not find cost configuration file. Tried: {possible_paths}"
            )
            self.cost_config = {}
            return

        try:
            with open(cost_file, "r") as f:
                self.cost_config = json.load(f)
            print(
                f"Loaded cost configuration for {len(self.cost_config)} models from {cost_file}"
            )
        except Exception as e:
            print(f"Warning: Could not load cost configuration from {cost_file}: {e}")
            self.cost_config = {}

    def calculate_inference_cost(
        self, model_name: str, token_usage: Dict[str, int]
    ) -> float:
        """Calculate inference cost based on token usage and model pricing."""
        if not token_usage or not self.cost_config:
            return 0.0

        # Remove _batch suffix if present for cost lookup
        cost_lookup_name = model_name
        if model_name.endswith("_batch"):
            cost_lookup_name = model_name[:-6]  # Remove '_batch' suffix

        # Normalize model name to match cost config
        if model_name_manager:
            normalized_name = model_name_manager.get_universal_name(cost_lookup_name)
        else:
            normalized_name = cost_lookup_name

        # Try to find exact match first
        if normalized_name in self.cost_config:
            cost_info = self.cost_config[normalized_name]
        else:
            # Try to find partial matches
            cost_info = None
            for config_name in self.cost_config.keys():
                if config_name in normalized_name or normalized_name in config_name:
                    cost_info = self.cost_config[config_name]
                    break

        if not cost_info:
            print(
                f"Warning: No cost configuration found for model {model_name} (lookup: {cost_lookup_name}, normalized: {normalized_name})"
            )
            return 0.0

        # Calculate cost
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)

        input_cost_per_million = cost_info.get("input_token_price_per_million", 0.0)
        output_cost_per_million = cost_info.get("output_token_price_per_million", 0.0)

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million

        total_cost = input_cost + output_cost
        return total_cost

    def determine_dataset_from_global_index(self, global_index: str) -> str:
        """Determine dataset name from global index."""
        # Extract dataset prefix from global index (e.g., "AIME_112" -> "AIME")
        if "_" in global_index:
            dataset_prefix = global_index.split("_")[0]
        else:
            dataset_prefix = global_index

        # Map common prefixes to full dataset names based on available configs
        dataset_mapping = {
            "AIME": "AIME",
            "ArcMMLU": "MMLUPro",
            "AsDiv": "AsDiv",
            "ChessInstruct": "ChessInstruct",
            "Ethics": "Ethics_commonsense",  # Default to commonsense
            "FinQA": "FinQA",
            "GeoBench": "GeoBench",
            "GeoGraphyData": "GeoGraphyData_100k",  # Fix the dataset name
            "GSM8K": "GSM8K",
            "LiveCodeBench": "LiveCodeBench",
            "MATH": "MATH",
            "MathQA": "MathQA",
            "MedMCQA": "MedMCQA",
            "MMLUPro": "MMLUPro",
            "MMLU": "MMLU",
            "MusicTheoryBench": "MusicTheoryBench",
            "NarrativeQA": "NarrativeQA",
            "OpenTDB": "OpenTDB",
            "PubMedQA": "PubMedQA",
            "QANTA": "QANTA",
            "SocialiQA": "SocialiQA",
            "SuperGLUE": "SuperGLUE-RC",  # Default to RC
            "WMT19": "WMT19-cs-en",  # Default to cs-en
        }

        return dataset_mapping.get(dataset_prefix, dataset_prefix)

    def group_cached_results_by_dataset(
        self, cached_results: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Group cached results by dataset based on global_index."""
        dataset_groups: Dict[str, List[Dict[str, Any]]] = {}

        for entry in cached_results:
            global_index = entry.get("global_index", "")
            dataset_name = self.determine_dataset_from_global_index(global_index)

            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(entry)

        return dataset_groups

    def evaluate_model(self, model_name: str, rerun=False) -> Dict[str, Any]:
        """Evaluate a single model's outputs from cached_results with incremental evaluation support."""

        # Convert input model name to universal name using ModelNameManager
        if model_name_manager:
            universal_model_name = model_name_manager.get_universal_name(model_name)
            print(f"\nEvaluating model: {model_name} -> {universal_model_name}")
        else:
            universal_model_name = model_name
        print(f"\nEvaluating model: {model_name}")

        self.load_cost_config()

        # Load cached results for this model
        cached_file = os.path.join(
            self.cached_results_dir, f"{universal_model_name}.jsonl"
        )
        if not os.path.exists(cached_file):
            print(f"Error: Cached results file not found: {cached_file}")
            return {
                "error": f"Cached results file not found for {universal_model_name}"
            }

        # Load all cached results
        cached_results = []
        with open(cached_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    cached_results.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line: {e}")
                    continue

        print(f"Loaded {len(cached_results)} cached results")

        # Check which entries need evaluation (don't have evaluation_result or have None)
        entries_to_evaluate = []
        already_evaluated = 0

        for entry in cached_results:
            eval_result = entry.get("evaluation_result")
            # Consider an entry already evaluated if it has evaluation_result with a score
            if eval_result and eval_result.get("score") is not None and not rerun:
                already_evaluated += 1
            else:
                # Only evaluate entries that have successful inference
                if entry.get("success", False):
                    entries_to_evaluate.append(entry)

        print(f"Found {already_evaluated} already evaluated entries")
        print(
            f"Found {len(entries_to_evaluate)} entries to evaluate (successful inference only)"
        )

        if not entries_to_evaluate:
            print("No new entries to evaluate.")
            # Still return existing results structure
            return self._compile_final_results(universal_model_name, cached_results)

        # Group entries by dataset for evaluation
        dataset_groups = self.group_cached_results_by_dataset(entries_to_evaluate)
        print(
            f"Found {len(dataset_groups)} datasets with new samples: {list(dataset_groups.keys())}"
        )

        # Evaluate each dataset group
        evaluated_count = 0
        dataset_scores: Dict[str, int] = {}

        # Create progress bar for datasets
        dataset_progress = tqdm(
            dataset_groups.items(), desc="Evaluating new samples", leave=False
        )

        for dataset_name, dataset_entries in dataset_progress:
            # if dataset_name == "LiveCodeBench":
            #     continue
            dataset_progress.set_description(
                f"Evaluating {dataset_name} ({len(dataset_entries)} samples)"
            )
            print(f"\nEvaluating {len(dataset_entries)} samples from {dataset_name}")

            # Get eval metrics from config for this dataset
            eval_metrics = load_eval_config_for_dataset(dataset_name)

            # Get scorers for this dataset
            scorers = get_scorers_for_dataset(dataset_name, eval_metrics)
            if not scorers:
                print(f"Warning: No scorers found for dataset {dataset_name}, skipping")
                continue

            # Evaluate each entry in this dataset
            for entry in dataset_entries:
                global_index_val = entry.get("global_index")
                generated_answer = entry.get("generated_answer", "")

                try:
                    # Get ground truth for this entry
                    if not isinstance(global_index_val, str):
                        print(
                            f"Warning: Invalid global_index {global_index_val} for dataset {dataset_name}"
                        )
                        continue
                    ground_truth = self._get_ground_truth(
                        global_index_val, dataset_name
                    )
                    if ground_truth is None:
                        print(f"Warning: No ground truth found for {global_index_val}")
                        continue

                    # Evaluate using the appropriate scorer
                    scorer_func, metric_name = scorers[
                        0
                    ]  # Use first scorer for the dataset
                    score, metric_name = self._evaluate_single_entry(
                        generated_answer, ground_truth, scorer_func, dataset_name
                    )

                    # Calculate inference cost
                    token_usage = entry.get("token_usage", {})
                    inference_cost = self.calculate_inference_cost(
                        universal_model_name, token_usage
                    )

                    # Create evaluation result matching the expected format
                    evaluation_result = {
                        "extracted_answer": generated_answer,  # For now, use full answer
                        "ground_truth": ground_truth
                        if dataset_name != "LiveCodeBench"
                        else "See dataset for testcases",
                        "score": score,
                        "metric": metric_name,
                        "inference_cost": inference_cost,
                    }

                    # Update the entry with evaluation result
                    entry["evaluation_result"] = evaluation_result
                    evaluated_count += 1

                except Exception as e:
                    entry["evaluation_result"] = {
                        "extracted_answer": generated_answer,
                        "ground_truth": ground_truth,
                        "score": 0.0,
                        "metric": "error",
                        "inference_cost": 0.0,
                    }
                    print(f"Error evaluating {global_index_val}: {e}")
                    continue

            dataset_scores[dataset_name] = len(dataset_entries)

        # Save updated cached results back to file
        print(
            f"\nSaving {evaluated_count} newly evaluated entries to cached results..."
        )
        with open(cached_file, "w", encoding="utf-8") as f:
            for entry in cached_results:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

        print(f"Evaluation completed. Evaluated {evaluated_count} new entries.")
        return self._compile_final_results(universal_model_name, cached_results)

    def _get_ground_truth(self, global_index: str, dataset_name: str) -> Optional[Any]:
        """Get ground truth for a specific global_index from the dataset."""
        # Load dataset if not already loaded
        if self.all_data is None:
            self.load_all_data()

        if dataset_name == "LiveCodeBench":
            # For LiveCodeBench, find the entry with matching global_idx and return the entire instance
            try:
                # Load LiveCodeBench dataset
                from datasets import load_from_disk

                livecodebench_dataset = load_from_disk(
                    "./RouterArena/dataset/livecodebench"
                ).to_list()

                # Find the entry with matching global_idx
                for entry in livecodebench_dataset:
                    if entry.get("global_idx") == global_index:
                        # Return the entire LiveCodeBench instance as ground truth
                        return entry

                print(
                    f"Warning: No LiveCodeBench entry found with global_idx {global_index}"
                )
                return None
            except Exception as e:
                print(f"Error loading LiveCodeBench dataset: {e}")
                return None

        # For other datasets, find the entry with matching global_index
        if self.all_data is None:
            return None
        for item in self.all_data:
            if (
                item.get("global index") == global_index
                or item.get("global_index") == global_index
            ):
                return item.get("answer", item.get("ground_truth"))

        return None

    def _evaluate_single_entry(
        self, generated_answer: str, ground_truth: Any, scorer, dataset_name: str
    ) -> tuple:
        """Evaluate a single entry using the appropriate scorer."""
        try:
            result = scorer(generated_answer, ground_truth)
            metric_name = getattr(scorer, "__name__", "unknown_metric")

            # Handle different return formats from scorers
            if isinstance(result, tuple) and len(result) == 2:
                # Most metrics return (score, details) tuple
                score, details = result
                return score, metric_name
            elif isinstance(result, (int, float)):
                # Some metrics might return just a score
                return result, metric_name
            else:
                print(f"Unexpected result format from scorer {metric_name}: {result}")
                return 0.0, "error"
        except Exception as e:
            print(f"Error in scorer for {dataset_name}: {e}")
            return 0.0, "error"

    def _compile_final_results(
        self, model_name: str, cached_results: List[Dict]
    ) -> Dict[str, Any]:
        """Compile final results in the expected format."""
        # Calculate overall statistics
        total_entries = len(cached_results)
        evaluated_entries = sum(
            1
            for entry in cached_results
            if entry.get("evaluation_result")
            and entry["evaluation_result"].get("score") is not None
        )

        # Calculate average cost per sample
        total_cost = 0.0
        cost_count = 0
        for entry in cached_results:
            eval_result = entry.get("evaluation_result")
            if eval_result and eval_result.get("inference_cost"):
                total_cost += eval_result["inference_cost"]
                cost_count += 1

        avg_cost = total_cost / cost_count if cost_count > 0 else 0.0

        # Group results by dataset for detailed reporting
        dataset_results: Dict[str, List[Dict[str, Any]]] = {}
        for entry in cached_results:
            if not entry.get("evaluation_result"):
                continue

            global_index = entry.get("global_index", "")
            dataset_name = self.determine_dataset_from_global_index(global_index)

            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = []

            eval_result = entry["evaluation_result"]
            dataset_results[dataset_name].append(
                {
                    "global_index": global_index,
                    "extracted_answer": eval_result.get("extracted_answer"),
                    "ground_truth": eval_result.get("ground_truth"),
                    "score": eval_result.get("score"),
                    "metric": eval_result.get("metric"),
                    "inference_cost": eval_result.get("inference_cost"),
                }
            )

        # Calculate metrics per dataset
        detailed_results = []
        for dataset_name, results in dataset_results.items():
            if results:
                scores = [r["score"] for r in results if r["score"] is not None]
                avg_score = sum(scores) / len(scores) if scores else 0.0

                detailed_results.append(
                    {
                        "dataset": dataset_name,
                        "num_samples": len(results),
                        "metrics": {results[0]["metric"]: avg_score} if results else {},
                        "detailed_results": results,
                    }
                )

        return {
            "model_name": model_name,
            "average_cost_per_sample": avg_cost,
            "total_entries": total_entries,
            "evaluated_entries": evaluated_entries,
            "detailed_results": detailed_results,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs from cached results"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to evaluate (will be converted to universal name)",
    )
    parser.add_argument(
        "--cached-results-dir",
        type=str,
        default="../cached_results/",
        help="Directory containing cached results (default: ../cached_results/)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force re-evaluation of all entries, even if already evaluated",
    )

    args = parser.parse_args()

    if model_name_manager is not None:
        universal_name = model_name_manager.get_universal_name(args.model_name)
    else:
        universal_name = args.model_name
    print(f"Input model name: {args.model_name}")
    print(f"Universal model name: {universal_name}")

    # Check if cached results exist for this model
    cached_file = os.path.join(args.cached_results_dir, f"{universal_name}.jsonl")
    if not os.path.exists(cached_file):
        print(f"Error: No cached results found for {universal_name}")
        print(f"Expected file: {cached_file}")
        return

    # Initialize evaluator
    evaluator = ModelEvaluator(args.cached_results_dir)

    # Evaluate the model
    try:
        results = evaluator.evaluate_model(args.model_name, rerun=args.rerun)

        # Print summary
        if "detailed_results" in results:
            print("\nEvaluation Summary:")
            print(f"Total entries: {results.get('total_entries', 0)}")
            print(f"Evaluated entries: {results.get('evaluated_entries', 0)}")
            print(
                f"Average cost per sample: ${results.get('average_cost_per_sample', 0):.6f}"
            )
            print(f"Datasets evaluated: {len(results['detailed_results'])}")

            for dataset_result in results["detailed_results"]:
                dataset_name = dataset_result["dataset"]
                num_samples = dataset_result["num_samples"]
                metrics = dataset_result.get("metrics", {})
                print(f"  - {dataset_name}: {num_samples} samples, metrics: {metrics}")

        print(
            f"\nEvaluation results have been updated in the cached results file: {cached_file}"
        )

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
