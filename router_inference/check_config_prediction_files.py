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
        predictions = []

    # Check 3: Load dataset and validate fields
    print("\n[3] Checking prediction fields against dataset...")
    try:
        dataset = load_dataset(args.split)
        print(f"✓ Dataset loaded: {len(dataset)} entries")

        if predictions and valid_models:
            fields_valid, field_errors = check_prediction_fields(
                predictions, dataset, valid_models
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
