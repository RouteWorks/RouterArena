# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Check Config and Prediction Files Format.

This script validates:
1. Config file contains valid model names that can be found in ModelNameManager
2. Prediction file has the correct number of entries (809 for 10% split, 8400 for full)
3. Each prediction has the correct fields:
   - global_index exactly matches the dataset
   - prompt exactly matches the dataset (either prompt_formatted or prompt field)
   - prediction comes from models defined in the config

Usage:
    python router_inference/check_config_prediction_files.py <router_name> <split>

    split: either "sub_10" for 10% split or "full" for full dataset
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Set, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from universal_model_names import ModelNameManager

# Expected dataset sizes
EXPECTED_SIZES = {
    "sub_10": 809,
    "full": 8400,
}

# Dataset file paths
DATASET_PATHS = {
    "sub_10": "./dataset/router_data_10.json",
    "full": "./dataset/router_data.json",
}


def load_config(router_name: str) -> Dict[str, Any]:
    """
    Load router config file.

    Args:
        router_name: Name of the router

    Returns:
        Configuration dictionary
    """
    config_path = f"./router_inference/config/{router_name}.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def load_predictions(router_name: str) -> List[Dict[str, Any]]:
    """
    Load router predictions file.

    Args:
        router_name: Name of the router

    Returns:
        List of prediction dictionaries
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    return predictions


def load_dataset(split: str) -> List[Dict[str, Any]]:
    """
    Load dataset file.

    Args:
        split: Either "sub_10" or "full"

    Returns:
        List of dataset entries
    """
    dataset_path = DATASET_PATHS.get(split)

    if not dataset_path:
        raise ValueError(f"Invalid split: {split}. Must be 'sub_10' or 'full'")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_cost_config() -> Dict[str, Any]:
    """
    Load cost configuration from model_cost/cost.json.
    Uses a canonical path relative to the project root.

    Returns:
        Dictionary mapping model names to cost configurations
    """
    # Get the project root directory (parent of router_inference/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    cost_file_path = os.path.join(project_root, "model_cost", "cost.json")

    if not os.path.exists(cost_file_path):
        return {}

    try:
        with open(cost_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load cost configuration from {cost_file_path}: {e}")
        return {}


def check_model_costs(
    predictions: List[Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check that all models used in predictions have cost configurations.

    Args:
        predictions: List of prediction dictionaries
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    cost_config = load_cost_config()

    if not cost_config:
        errors.append(
            "Cost configuration file (model_cost/cost.json) not found. "
            "Cannot validate model costs."
        )
        return False, errors

    # Get all unique models from predictions
    used_models = set()
    model_manager = ModelNameManager()

    for prediction in predictions:
        model_prediction = prediction.get("prediction")
        if model_prediction:
            used_models.add(model_prediction)

    # Also include models from config
    config_models = config.get("pipeline_params", {}).get("models", [])
    for model in config_models:
        used_models.add(model)

    # Check each model has cost configuration
    missing_costs = []
    for model_name in used_models:
        try:
            # Convert to universal name for cost lookup
            universal_name = model_manager.get_universal_name(model_name)

            # Remove _batch suffix if present
            cost_lookup_name = universal_name
            if universal_name.endswith("_batch"):
                cost_lookup_name = universal_name[:-6]

            # Check if cost exists (exact match or partial match)
            has_cost = False
            if cost_lookup_name in cost_config:
                has_cost = True
            else:
                # Try partial matches
                for config_name in cost_config.keys():
                    if (
                        config_name in cost_lookup_name
                        or cost_lookup_name in config_name
                    ):
                        has_cost = True
                        break

            if not has_cost:
                missing_costs.append(f"{model_name} (universal: {cost_lookup_name})")
        except Exception as e:
            # If we can't convert, try original name
            if model_name not in cost_config:
                missing_costs.append(f"{model_name} (conversion failed: {str(e)})")

    if missing_costs:
        errors.append(f"Missing cost configuration for {len(missing_costs)} model(s):")
        for model in missing_costs:
            errors.append(f"  - {model}")
        errors.append(
            "\nPlease add cost configuration to model_cost/cost.json or update "
            "your router config to use models with existing cost configurations."
        )

    return len(missing_costs) == 0, errors


def check_config_models(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check that all model names in config can be found in ModelNameManager.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    models = config.get("pipeline_params", {}).get("models", [])

    model_manager = ModelNameManager()

    for model_name in models:
        try:
            # Try to get universal name for this model
            model_manager.get_universal_name(model_name)
        except Exception as e:
            errors.append(
                f"Model '{model_name}' not found in ModelNameManager: {str(e)}"
            )

    return len(errors) == 0, errors


def check_prediction_size(
    predictions: List[Dict[str, Any]], split: str
) -> Tuple[bool, str]:
    """
    Check that predictions have the correct number of entries.

    Args:
        predictions: List of prediction dictionaries
        split: Either "sub_10" or "full"

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_size = EXPECTED_SIZES.get(split)

    if expected_size is None:
        return False, f"Invalid split: {split}. Must be 'sub_10' or 'full'"

    actual_size = len(predictions)

    if actual_size != expected_size:
        return False, (
            f"Prediction size mismatch: expected {expected_size} entries "
            f"for split '{split}', got {actual_size}"
        )

    return True, ""


def check_prediction_fields(
    predictions: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    valid_models: Set[str],
    check_generated_result: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Check that each prediction has correct fields matching the dataset.

    Args:
        predictions: List of prediction dictionaries
        dataset: List of dataset entries
        valid_models: Set of valid model names from config

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Create a mapping from global_index to dataset entry
    dataset_map = {}
    for entry in dataset:
        global_index = entry.get("global index")
        if global_index:
            dataset_map[global_index] = entry

    for i, prediction in enumerate(predictions):
        # Check global_index
        pred_global_index = prediction.get("global index") or prediction.get(
            "global_index"
        )

        if not pred_global_index:
            errors.append(f"Entry {i}: missing global_index")
            continue

        if pred_global_index not in dataset_map:
            errors.append(
                f"Entry {i}: global_index '{pred_global_index}' not found in dataset"
            )
            continue

        # Check prompt - try both "prompt" and "prompt_formatted" fields
        pred_prompt = prediction.get("prompt") or prediction.get("prompt_formatted")
        dataset_entry = dataset_map[pred_global_index]
        assert dataset_entry is not None, (
            f"dataset_entry should exist for {pred_global_index}"
        )
        dataset_prompt = dataset_entry.get("prompt_formatted") or dataset_entry.get(
            "prompt"
        )

        if not pred_prompt:
            errors.append(
                f"Entry {i} (global_index: {pred_global_index}): missing prompt"
            )
            continue

        if pred_prompt != dataset_prompt:
            errors.append(
                f"Entry {i} (global_index: {pred_global_index}): prompt mismatch with dataset"
            )
            # Show first 100 chars of each for debugging
            dataset_prompt_str = str(dataset_prompt) if dataset_prompt else ""
            pred_prompt_str = str(pred_prompt) if pred_prompt else ""
            errors.append(f"  Expected: {dataset_prompt_str[:100]}...")
            errors.append(f"  Got: {pred_prompt_str[:100]}...")

        # Check prediction (model selection)
        model_prediction = prediction.get("prediction")

        if not model_prediction:
            errors.append(
                f"Entry {i} (global_index: {pred_global_index}): missing prediction"
            )
            continue

        # Check if the predicted model is in the valid models set
        # First try to convert to universal name
        model_manager = ModelNameManager()
        try:
            universal_model_name = model_manager.get_universal_name(model_prediction)
            valid_models_universal = set()
            for model in valid_models:
                try:
                    universal_model = model_manager.get_universal_name(model)
                    valid_models_universal.add(universal_model)
                except Exception:
                    valid_models_universal.add(model)  # Fallback to original

            if (
                universal_model_name not in valid_models_universal
                and model_prediction not in valid_models
            ):
                errors.append(
                    f"Entry {i} (global_index: {pred_global_index}): "
                    f"prediction '{model_prediction}' not in config models"
                )
        except Exception as e:
            # If we can't convert, check if it's in the original set
            if model_prediction not in valid_models:
                errors.append(
                    f"Entry {i} (global_index: {pred_global_index}): "
                    f"prediction '{model_prediction}' not in config models "
                    f"(also failed to convert: {str(e)})"
                )

        # Check generated_result - only if flag is enabled (for post-inference validation)
        # This check is skipped by default since validation runs before inference
        if check_generated_result:
            generated_result = prediction.get("generated_result")
            if generated_result is None:
                errors.append(
                    f"Entry {i} (global_index: {pred_global_index}): "
                    f"missing generated_result (must be a dictionary). "
                    f"Please run llm_inference/run.py first to generate model outputs."
                )
            elif not isinstance(generated_result, dict):
                errors.append(
                    f"Entry {i} (global_index: {pred_global_index}): "
                    f"generated_result must be a dictionary, got {type(generated_result).__name__}"
                )
            else:
                # Validate dictionary structure
                required_fields = ["generated_answer", "success", "token_usage"]
                missing_fields = [
                    field for field in required_fields if field not in generated_result
                ]
                if missing_fields:
                    errors.append(
                        f"Entry {i} (global_index: {pred_global_index}): "
                        f"generated_result dictionary missing required fields: {', '.join(missing_fields)}"
                    )
                elif not isinstance(generated_result.get("generated_answer"), str):
                    errors.append(
                        f"Entry {i} (global_index: {pred_global_index}): "
                        f"generated_result.generated_answer must be a string"
                    )
                elif (
                    generated_result.get("success", False)
                    and not generated_result.get("generated_answer", "").strip()
                ):
                    # Only require non-empty generated_answer if success is True
                    # Failed entries (success=False) may have empty generated_answer
                    errors.append(
                        f"Entry {i} (global_index: {pred_global_index}): "
                        f"generated_result.generated_answer is empty but success is True"
                    )
                elif not isinstance(generated_result.get("success"), bool):
                    errors.append(
                        f"Entry {i} (global_index: {pred_global_index}): "
                        f"generated_result.success must be a boolean"
                    )

    return len(errors) == 0, errors


def main():
    """Main function to handle command line arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Check config and prediction files format"
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to config and predictions files)",
    )
    parser.add_argument(
        "split",
        type=str,
        choices=["sub_10", "full"],
        help="Dataset split: 'sub_10' for 10%% split (809 entries) or 'full' (8400 entries)",
    )
    parser.add_argument(
        "--check-generated-result",
        action="store_true",
        default=False,
        help="Check that generated_result field is present and valid (for post-inference validation)",
    )

    args = parser.parse_args()

    # Change to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../"))
    os.chdir(base_dir)

    print(f"Checking router: {args.router_name}")
    print(f"Dataset split: {args.split}")
    print("=" * 80)

    all_valid = True
    errors_summary = []
    config = None  # Initialize config variable
    predictions = None  # Initialize predictions variable

    # Check 1: Load and validate config
    print("\n[1] Checking config file...")
    try:
        config = load_config(args.router_name)
        print(f"✓ Config loaded from ./router_inference/config/{args.router_name}.json")

        # Get valid models from config
        valid_models = set(config.get("pipeline_params", {}).get("models", []))
        print(f"✓ Found {len(valid_models)} models in config")

        # Check if all models are valid
        config_valid, config_errors = check_config_models(config)
        if config_valid:
            print("✓ All models in config are valid (found in ModelNameManager)")
        else:
            print("✗ Invalid models found in config:")
            for error in config_errors:
                print(f"  - {error}")
            all_valid = False
            errors_summary.extend(config_errors)

    except Exception as e:
        print(f"✗ Error loading config: {e}")
        all_valid = False
        errors_summary.append(f"Config error: {str(e)}")
        valid_models = set()
        config = None

    # Check 2: Load and validate predictions
    print("\n[2] Checking prediction file...")
    try:
        predictions = load_predictions(args.router_name)
        print(
            f"✓ Predictions loaded from ./router_inference/predictions/{args.router_name}.json"
        )

        # Check size
        size_valid, size_error = check_prediction_size(predictions, args.split)
        if size_valid:
            print(f"✓ Prediction file has correct size: {len(predictions)} entries")
        else:
            print(f"✗ {size_error}")
            all_valid = False
            errors_summary.append(size_error)

    except Exception as e:
        print(f"✗ Error loading predictions: {e}")
        all_valid = False
        errors_summary.append(f"Predictions error: {str(e)}")
        predictions = None

    # Check 3: Load dataset and validate fields
    print("\n[3] Checking prediction fields against dataset...")
    try:
        dataset = load_dataset(args.split)
        print(f"✓ Dataset loaded: {len(dataset)} entries")

        if predictions is not None and valid_models:
            fields_valid, field_errors = check_prediction_fields(
                predictions, dataset, valid_models, args.check_generated_result
            )
            if fields_valid:
                print("✓ All prediction fields match dataset correctly")
            else:
                print(f"✗ Found {len(field_errors)} field validation errors:")
                # Show first 10 errors
                for error in field_errors[:10]:
                    print(f"  - {error}")
                if len(field_errors) > 10:
                    print(f"  ... and {len(field_errors) - 10} more errors")
                all_valid = False
                errors_summary.extend(field_errors)

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        all_valid = False
        errors_summary.append(f"Dataset error: {str(e)}")

    # Check 4: Validate model costs
    print("\n[4] Checking model cost configurations...")
    try:
        if predictions is not None and config is not None:
            cost_valid, cost_errors = check_model_costs(predictions, config)
            if cost_valid:
                cost_config = load_cost_config()
                print(
                    f"✓ All models have cost configurations ({len(cost_config)} models in cost file)"
                )
            else:
                print("✗ Missing cost configurations found:")
                for error in cost_errors:
                    print(f"  {error}")
                all_valid = False
                errors_summary.extend(cost_errors)
        else:
            print("⚠ Skipping cost check (predictions or config not loaded)")

    except Exception as e:
        print(f"✗ Error checking model costs: {e}")
        all_valid = False
        errors_summary.append(f"Cost check error: {str(e)}")

    # Final summary
    print("\n" + "=" * 80)
    if all_valid:
        print("✓ ALL CHECKS PASSED!")
        print(f"Router '{args.router_name}' is configured correctly.")
    else:
        print("✗ VALIDATION FAILED")
        print(f"Found {len(errors_summary)} error(s). Please fix the issues above.")
    print("=" * 80)

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
