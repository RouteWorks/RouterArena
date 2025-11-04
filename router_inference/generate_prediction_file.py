# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Generate Prediction File for Toy Router.

This script generates a prediction file for a toy router that cycles through
models in the config file using a simple modulo operation. This is useful for
testing the RouterArena pipeline.

Usage:
    python router_inference/generate_prediction_file.py <router_name> <split>

    split: either "sub_10" for 10% split (809 entries) or "full" (8400 entries)
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

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


def generate_predictions(
    dataset: List[Dict[str, Any]], models: List[str]
) -> List[Dict[str, Any]]:
    """
    Generate predictions by cycling through models using modulo operation.

    Args:
        dataset: List of dataset entries
        models: List of model names to cycle through

    Returns:
        List of prediction dictionaries
    """
    predictions = []

    # Get the list of model names (only those with non-null cost)
    model_names = [model for model in models]

    if not model_names:
        raise ValueError("No models with non-null cost found in config")

    for i, entry in enumerate(dataset):
        global_index = entry.get("global index")
        prompt = entry.get("prompt_formatted") or entry.get("prompt")

        if not global_index or not prompt:
            continue

        # Use modulo operation to cycle through models
        # NOTE: This is a toy router, you could implement your own logic here, or create a prediction file in your own router loop.
        model_index = i % len(model_names)
        selected_model = model_names[model_index]

        # Create prediction entry
        prediction_entry = {
            "global index": global_index,
            "prompt": prompt,
            "prediction": selected_model,
            "generated_result": None,
            "cost": None,
            "accuracy": None,
        }

        predictions.append(prediction_entry)

    return predictions


def save_predictions(predictions: List[Dict[str, Any]], router_name: str) -> None:
    """
    Save predictions to file.

    Args:
        predictions: List of prediction dictionaries
        router_name: Name of the router
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(predictions)} predictions to {prediction_path}")


def main():
    """Main function to handle command line arguments and generate predictions."""
    parser = argparse.ArgumentParser(
        description="Generate prediction file for toy router"
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to config file)",
    )
    parser.add_argument(
        "split",
        type=str,
        choices=["sub_10", "full"],
        help="Dataset split: 'sub_10' for 10%% split or 'full'",
    )

    args = parser.parse_args()

    # Change to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../"))
    os.chdir(base_dir)

    print(f"Generating predictions for router: {args.router_name}")
    print(f"Dataset split: {args.split}")
    print("=" * 80)

    # Load config
    print("\n[1] Loading config...")
    config = load_config(args.router_name)
    models = config.get("pipeline_params", {}).get("models", [])
    print(f"✓ Config loaded: {len(models)} models in config")
    print(f"  Models: {', '.join(models)}")

    # Load dataset
    print("\n[2] Loading dataset...")
    dataset = load_dataset(args.split)
    print(f"✓ Dataset loaded: {len(dataset)} entries")

    # Generate predictions
    print("\n[3] Generating predictions...")
    predictions = generate_predictions(dataset, models)
    print(f"✓ Generated {len(predictions)} predictions")
    print("  Using toy router logic: cycling through models")

    # Save predictions
    print("\n[4] Saving predictions...")
    save_predictions(predictions, args.router_name)

    print("\n" + "=" * 80)
    print("✓ Prediction file generation completed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
