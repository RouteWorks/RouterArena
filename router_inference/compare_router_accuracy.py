# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple, DefaultDict, Any
import json
import os
from collections import defaultdict
from universal_model_names import ModelNameManager


def load_cost_data(cost_file: str = "./model_cost/cost.json") -> Dict:
    """Load model cost information from JSON file."""
    with open(cost_file, "r") as f:
        return json.load(f)


def load_router_config(router_name: str) -> Dict:
    """Load router configuration from JSON file."""
    config_path = f"./router_inference/config/{router_name}.json"
    with open(config_path, "r") as f:
        return json.load(f)


def build_complete_evaluation_dictionary() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Build a complete dictionary mapping model -> {global_index -> (accuracy, cost)}
    for ALL evaluation results available.

    Returns:
        Dictionary where key is model_name and value is {global_index -> (accuracy, cost)}
    """
    print("Building complete evaluation dictionary...")

    results_dir = "./cached_results2"
    cost_data = load_cost_data()

    evaluation_dict: DefaultDict[str, Dict[Any, Tuple[float, float]]] = defaultdict(
        dict
    )

    # Get all evaluation result files
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory {results_dir} not found")

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".jsonl")]
    print(f"Found {len(result_files)} evaluation result files")

    for file_name in result_files:
        file_path = os.path.join(results_dir, file_name)
        print(f"Processing file: {file_path}")

        # Load JSONL file (JSON Lines format)
        results = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))

        # Extract model name from JSON content (more reliable than filename parsing)
        model_name = file_name.replace(".jsonl", "")

        # Extract detailed results using existing logic
        extracted_results = {}

        for result in results:
            global_index = result["global_index"]
            extracted_results[global_index] = result

        unvalidated_count = 0

        # Add to main dictionary
        for global_index, result in extracted_results.items():
            if (
                "evaluation_result" in result
                and result["evaluation_result"] is not None
            ):
                # Get cost for this inference
                inference_cost = result["evaluation_result"]["inference_cost"]

                if inference_cost == 0.0:
                    # Handle different token naming conventions
                    # Token info is in result["token_usage"], not in evaluation_result
                    token_usage = result["token_usage"]
                    if token_usage is None:
                        continue
                    input_tokens = token_usage.get(
                        "input_tokens", token_usage.get("prompt_tokens", 0)
                    )
                    output_tokens = token_usage.get(
                        "output_tokens", token_usage.get("completion_tokens", 0)
                    )

                    if input_tokens and output_tokens:
                        input_cost_per_million = cost_data[model_name][
                            "input_token_price_per_million"
                        ]
                        output_cost_per_million = cost_data[model_name][
                            "output_token_price_per_million"
                        ]

                        inference_cost = (
                            input_tokens / 1_000_000
                        ) * input_cost_per_million + (
                            output_tokens / 1_000_000
                        ) * output_cost_per_million

                # Get accuracy (score)
                accuracy = result["evaluation_result"]["score"]

            else:
                unvalidated_count += 1
                inference_cost = 0.0
                accuracy = 0.0

            # Store in dictionary
            evaluation_dict[model_name][global_index] = (accuracy, inference_cost)

        # print(f"  Added {len(extracted_results)} results for {model_name}")
        # print(
        #     f"  Added {len(extracted_results)} results for {model_name} (unvalidated: {unvalidated_count})"
        # )

    total_pairs = sum(len(queries) for queries in evaluation_dict.values())
    print(
        f"\nCompleted building evaluation dictionary with {total_pairs} total (model, global_index) pairs across {len(evaluation_dict)} models"
    )

    with open("./router_inference/llm_evaluation_dict.json", "w") as f:
        json.dump(dict(evaluation_dict), f)

    return dict(evaluation_dict)


def load_predictions(router_name: str, config: Dict):
    print(f"Loading predictions for {router_name}.......")
    model_name_manager = ModelNameManager()

    predictions_path = f"./router_inference/predictions2/{router_name}.json"
    if not os.path.exists(predictions_path):
        print(f"Warning: Predictions file {predictions_path} not found")
        return {}
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    if router_name == "vllm":
        predictions = predictions["results"]

        for pred in predictions:
            pred["all_confidence"] = {
                model_name_manager.get_universal_name(pred["prediction"]): 1.0
            }

        # For VLLM, we've already set all_confidence correctly, so return early
        return predictions

    # Get the list of models from config to match with all_confidence array
    models_list = list(config["pipeline_params"]["models"].keys())
    costs = list(config["pipeline_params"]["models"].values())

    # Transform all_confidence array to dict with universal model names
    for prediction in predictions:
        # Handle both 'all_confidence' and 'all confidence' field names
        confidence_field = None
        if "all_confidence" in prediction:
            confidence_field = "all_confidence"
        elif "all confidence" in prediction:
            confidence_field = "all confidence"

        if confidence_field and prediction[confidence_field]:
            confidence_dict = {}
            for i, model_name in enumerate(models_list):
                if i < len(prediction[confidence_field]) and costs[i] is not None:
                    universal_name = model_name_manager.get_universal_name_non_static(
                        model_name
                    )
                    if isinstance(prediction[confidence_field], dict):
                        confidence_dict[universal_name] = list(
                            prediction[confidence_field].values()
                        )[i]
                    else:
                        confidence_dict[universal_name] = prediction[confidence_field][
                            i
                        ]

            # Replace the array with the dict
            prediction[confidence_field] = confidence_dict

    for model_name in model_name_manager.missing_models:
        print(f"Missing model: {model_name}")

    return predictions


def main():
    print("Loading model results...")

    # Build complete evaluation dictionary from llm_evaluation results
    evaluation_dict = build_complete_evaluation_dictionary()

    # Load RouterEvalBench dataset and create global_index to bloom_level mapping
    from datasets import load_dataset  # type: ignore[import-untyped]

    # Load the routerevalbench dataset from local path
    dataset_path = "./dataset/routerevalbench"
    routerevalbench = load_dataset(
        "arrow", data_files=f"{dataset_path}/data-00000-of-00001.arrow"
    )

    # Create global_index to bloom_level mapping
    global_index_to_bloom_level = {}
    for example in routerevalbench["train"]:
        global_index = example["Global Index"]
        bloom_level = example["bloom_level"]
        global_index_to_bloom_level[global_index] = bloom_level

    print(f"Created mapping for {len(global_index_to_bloom_level)} entries")
    print(f"Sample mapping: {dict(list(global_index_to_bloom_level.items())[:5])}")

    router_names = [
        "carrot",
        "graphrouter",
        "notdiamond",
        "gpt5",
        "azure",
        "vllm",
        "mirt_bert",
        "nirt_bert",
        "routellm",
        "routerbench_knn",
        "routerbench_mlp",
        "RouterDC",
    ]

    all_router_data = {}

    missing_data = defaultdict(list)
    model_name_manager = ModelNameManager()

    for router_name in router_names:
        config = load_router_config(router_name.split("_")[0])

        router_data = {}

        predictions = load_predictions(router_name, config)

        for prediction in predictions:
            global_index = prediction.get(
                "global_index", prediction.get("global index", "")
            )
            all_confidence = prediction.get(
                "all_confidence", prediction.get("all confidence", {})
            )
            if router_name in ["gpt5", "azure", "notdiamond"]:
                model_selected = prediction.get("prediction")
            else:
                model_selected = max(all_confidence, key=all_confidence.get)

            if router_name == "vllm":
                model_selected = model_name_manager.get_universal_name(model_selected)

            assert model_selected in evaluation_dict, (
                f"Model {model_selected} not found in evaluation dictionary"
            )
            bloom_level = global_index_to_bloom_level[global_index]

            try:
                accuracy, cost = evaluation_dict[model_selected][global_index]

            except Exception:
                print(
                    f"Router {router_name}: Model {model_selected} not found in evaluation dictionary for global index {global_index}"
                )
                missing_data[router_name].append((global_index, model_selected))
                accuracy = 0.0
                cost = 0.0

            router_data[global_index] = {
                "model": model_selected,
                "accuracy": accuracy,
                "cost": cost,
                "bloom_level": bloom_level,
            }

        all_router_data[router_name] = router_data

    with open("./router_inference/all_router_data.json", "w") as f:
        json.dump(all_router_data, f)

    print(f"\n✓ Saved all_router_data.json with {len(all_router_data)} routers")
    for router_name in all_router_data:
        print(f"  - {router_name}: {len(all_router_data[router_name])} queries")

    for router_name, data in missing_data.items():
        for global_index, model_selected in data:
            print(f"{global_index} --- {model_selected}")


if __name__ == "__main__":
    main()
