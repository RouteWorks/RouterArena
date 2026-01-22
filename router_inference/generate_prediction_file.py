# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Generate Prediction File using configured router.

This script generates a prediction file using the router class specified
in the config file's pipeline_params.router_cls_name field.

Usage:
    python router_inference/generate_prediction_file.py <router_name> <split>

    split: one of "sub_10", "full", or "robustness"
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from router_inference.router import BaseRouter

# Dataset file paths
DATASET_PATHS = {
    "sub_10": "./dataset/router_data_10.json",
    "full": "./dataset/router_data.json",
    "robustness": "./dataset/router_robustness.json",
}


def load_dataset(split: str) -> List[Dict[str, Any]]:
    """
    Load dataset file.

    Args:
        split: One of the supported dataset splits (sub_10, full, robustness)

    Returns:
        List of dataset entries
    """
    dataset_path = DATASET_PATHS.get(split)

    if not dataset_path:
        raise ValueError(
            f"Invalid split: {split}. Must be one of {list(DATASET_PATHS.keys())}"
        )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def generate_predictions(
    dataset: List[Dict[str, Any]],
    router: BaseRouter,
    model_pool: List[str],
    split: str,
    include_optimality: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate predictions using the router, optionally including optimality entries.

    For the `full` split:
    - Generates 8400 regular entries
    - For sub_10 queries within the full split, generates optimality entries for other models

    For `robustness`:
    - Generates regular entries only (no optimality augmentation)

    Args:
        dataset: List of dataset entries
        router: Router instance to use for predictions
        model_pool: List of all models in the router's pool
        split: Dataset split ("sub_10", "full", or "robustness")
        include_optimality: Whether to include optimality entries (default: True for supported splits)

    Returns:
        List of prediction dictionaries including optimality entries when applicable
    """
    predictions = []

    # Only full/sub_10 support optimality augmentation
    if split not in {"sub_10", "full"}:
        include_optimality = False

    # Load sub_10 indices to identify which entries need optimality calculations
    sub10_indices = set()
    if include_optimality:
        try:
            sub10_dataset = load_dataset("sub_10")
            sub10_indices = {entry.get("global index") for entry in sub10_dataset}
            print(
                f"  Loaded {len(sub10_indices)} sub_10 indices for optimality calculation"
            )
        except Exception as e:
            print(f"  Warning: Could not load sub_10 dataset: {e}")
            print("  Optimality entries will not be generated")
            include_optimality = False

    # Track selected models for sub_10 entries
    sub10_selected_models = {}  # {global_index: (selected_model, prompt)}

    # Generate regular entries for all queries
    for entry in dataset:
        global_index = entry.get("global index")
        prompt = entry.get("prompt_formatted") or entry.get("prompt")

        if not global_index or not prompt:
            continue

        # Use the router to get prediction (validation is handled by BaseRouter)
        selected_model = router.get_prediction(prompt)

        # Track selected model for sub_10 entries (for optimality generation)
        if global_index in sub10_indices:
            sub10_selected_models[global_index] = (selected_model, prompt)

        # Create prediction entry
        prediction_entry = {
            "global index": global_index,
            "prompt": prompt,
            "prediction": selected_model,
            "generated_result": None,
            "cost": None,
            "accuracy": None,
            "for_optimality": False,  # Regular entry
        }

        predictions.append(prediction_entry)

    # Generate optimality entries for sub_10 queries
    if include_optimality and sub10_selected_models:
        print(
            f"\n  Generating optimality entries for {len(sub10_selected_models)} sub_10 queries..."
        )
        optimality_count = 0

        for global_index, (selected_model, prompt) in sub10_selected_models.items():
            # Generate entries for all OTHER models in pool
            other_models = [m for m in model_pool if m != selected_model]

            for model in other_models:
                optimality_entry = {
                    "global index": global_index,
                    "prompt": prompt,
                    "prediction": model,  # Other model, not the one router selected
                    "generated_result": None,
                    "cost": None,
                    "accuracy": None,
                    "for_optimality": True,  # Flag for optimality calculation
                }
                predictions.append(optimality_entry)
                optimality_count += 1

        print(f"  Generated {optimality_count} optimality entries")
        print(
            f"  Total entries: {len(predictions)} ({len(dataset)} regular + {optimality_count} optimality)"
        )

    return predictions


def save_predictions(
    predictions: List[Dict[str, Any]], router_name: str, split: str
) -> None:
    """
    Save predictions to file.

    Args:
        predictions: List of prediction dictionaries
        router_name: Name of the router
    """
    filename = router_name
    if split == "robustness":
        filename = f"{router_name}-robustness"
    prediction_path = f"./router_inference/predictions/{filename}.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(predictions)} predictions to {prediction_path}")


def main():
    """Main function to handle command line arguments and generate predictions."""
    parser = argparse.ArgumentParser(
        description="Generate prediction file using router specified in config"
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to config file)",
    )
    parser.add_argument(
        "split",
        type=str,
        choices=list(DATASET_PATHS.keys()),
        help="Dataset split: 'sub_10', 'full', or 'robustness'",
    )
    parser.add_argument(
        "--no-optimality",
        action="store_true",
        help="Skip generating optimality entries (default: include optimality entries)",
    )

    args = parser.parse_args()

    # Change to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../"))
    os.chdir(base_dir)

    print(f"Generating predictions for router: {args.router_name}")
    print(f"Dataset split: {args.split}")
    print("=" * 80)

    # Load router config first to get router_cls_name
    print("\n[1] Loading router config...")
    config_path = f"./router_inference/config/{args.router_name}.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    pipeline_params = config.get("pipeline_params", {})
    model_pool = pipeline_params.get("models", [])
    router_cls_name = pipeline_params.get("router_cls_name", "ExampleRouter")

    print(f"✓ Config loaded: {config_path}")
    print(f"  Router class: {router_cls_name}")
    print(f"  Model pool: {len(model_pool)} models")
    print(f"  Models: {', '.join(model_pool)}")

    # Initialize router dynamically based on router_cls_name
    print("\n[2] Initializing router...")

    # Import the router module to access router classes
    import router_inference.router as router_module

    # Get the router class by name
    if not hasattr(router_module, router_cls_name):
        raise ValueError(
            f"Router class '{router_cls_name}' not found in router_inference.router module. "
            f"Available routers: {', '.join([name for name in dir(router_module) if not name.startswith('_')])}"
        )

    router_cls = getattr(router_module, router_cls_name)
    router = router_cls(args.router_name)

    print(f"✓ Router initialized: {router.router_name}")
    print(f"  Available models: {', '.join(router.models)}")

    # Load dataset
    print("\n[3] Loading dataset...")
    dataset = load_dataset(args.split)
    print(f"✓ Dataset loaded: {len(dataset)} entries")

    # Generate predictions
    print("\n[4] Generating predictions...")
    include_optimality = not args.no_optimality
    optimality_reason = None
    if args.no_optimality:
        optimality_reason = "--no-optimality flag set"
    elif args.split not in {"sub_10", "full"}:
        optimality_reason = "not supported for robustness split"

    if optimality_reason:
        include_optimality = False
        print(f"  Skipping optimality entries ({optimality_reason})")
    else:
        print(
            "  Including optimality entries for automatic optimality score calculation"
        )

    predictions = generate_predictions(
        dataset, router, model_pool, args.split, include_optimality
    )
    print(f"✓ Generated {len(predictions)} total entries")

    # Save predictions
    print("\n[5] Saving predictions...")
    save_predictions(predictions, args.router_name, args.split)

    print("\n" + "=" * 80)
    print("✓ Prediction file generation completed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
