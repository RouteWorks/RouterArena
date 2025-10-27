# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
from shared_utils import (
    setup_environment,
    setup_logging_and_seed,
    get_timezone_info,
    log_pipeline_start,
    log_pipeline_completion,
)

# Set up environment
current_dir, base_dir = setup_environment()


def main():
    """Main function to handle command line arguments and run inference pipeline."""
    parser = argparse.ArgumentParser(description="LLM Inference Pipeline")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for inference (e.g., WizardLM/WizardLM-13B-V1.2)",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Run in full mode (processes all available data)",
    )

    args = parser.parse_args()

    # Set up logging and seed
    logger, seed = setup_logging_and_seed(current_dir, args.model_name)
    ct_timezone, start_time = get_timezone_info()

    # Log pipeline start
    log_pipeline_start(logger, "LLM INFERENCE", args.model_name, start_time, seed)

    # Log run_full mode if enabled
    if args.run_full:
        logger.info("Running in FULL mode - processing all available data")
    else:
        logger.info("Running in standard mode")

    # Import and run the standard pipeline
    from pipeline import inference_pipeline

    logger.info("Using standard processing pipeline")

    # Create config for the model
    config = {"model_name": args.model_name, "run_full": args.run_full}

    # Run the inference pipeline
    results = inference_pipeline(config, args.run_full)

    # Log pipeline completion
    end_time = datetime.datetime.now(ct_timezone)
    log_pipeline_completion(
        logger, "LLM INFERENCE", args.model_name, start_time, end_time, results
    )


if __name__ == "__main__":
    main()
