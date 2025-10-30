# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import json
import os
import logging
import datetime
from typing import Dict, Any, List
from model_inference import ModelInference
from universal_model_names import ModelNameManager
import time

logger = logging.getLogger(__name__)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries parsed from the JSONL file
    """
    results: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        return results

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Could not parse JSON line {line_num} in {file_path}: {line[:100]}... Error: {e}"
                        )
                        continue
    except Exception as e:
        logger.error(f"Error reading JSONL file {file_path}: {e}")

    return results


def convert_json_to_jsonl_if_needed(json_file_path: str, jsonl_file_path: str) -> bool:
    """
    Convert a JSON file to JSONL format if the JSONL file doesn't exist.

    Args:
        json_file_path: Path to the input JSON file
        jsonl_file_path: Path to the output JSONL file

    Returns:
        True if conversion was successful or JSONL file already exists, False otherwise
    """
    if os.path.exists(jsonl_file_path):
        return True

    if not os.path.exists(json_file_path):
        return False

    try:
        # Read the JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Write to JSONL format (one JSON object per line)
        with open(jsonl_file_path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        logger.info(
            f"Converted {len(data)} entries from {json_file_path} to {jsonl_file_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Error converting {json_file_path} to {jsonl_file_path}: {e}")
        return False


def inference_pipeline(config: Dict[str, Any], on_full=False):
    """
    Run inference pipeline for a specific model on the dataset.
    Uses cached results and only processes missing entries (where success != true).

    Args:
        config: Configuration dictionary containing model_name
        on_full: Whether to use full dataset or subset
    """
    model_name = config["model_name"]

    # Convert to universal model name using model name manager
    universal_model_name = ModelNameManager.get_universal_name(model_name)

    logger.info(f"Starting inference pipeline for model: {model_name}")
    logger.info(f"Universal model name: {universal_model_name}")

    # Initialize model inference
    model_inferencer = ModelInference()

    data_path = "./llm_inference/datasets/router_data.json"

    # Load data (must be prepared by prep_datasets.py)
    logger.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/process_datasets/prep_datasets.py first to prepare the dataset."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} data points")

    # Create output directory and file path using universal model name
    output_dir = "./cached_results/"
    os.makedirs(output_dir, exist_ok=True)

    # Use universal model name for output file
    output_file = os.path.join(output_dir, f"{universal_model_name}.jsonl")

    logger.info(f"Output will be saved to: {output_file}")

    # Load existing cached results if the file exists
    cached_results_by_global_index = {}
    successful_indices = set()

    if os.path.exists(output_file):
        try:
            # Load all cached results
            all_cached_results = load_jsonl_file(output_file)
            logger.info(
                f"Loaded {len(all_cached_results)} cached results from {output_file}"
            )

            # Create lookup by global_index
            for result in all_cached_results:
                global_index = result.get("global_index")
                if global_index is not None:
                    cached_results_by_global_index[global_index] = result

                    # Track successful entries to skip
                    if result.get("success", False):
                        successful_indices.add(global_index)

            logger.info(f"Found {len(successful_indices)} successful entries to skip")
            logger.info(
                f"Will process {len(all_cached_results) - len(successful_indices)} missing/failed entries"
            )

        except Exception as e:
            logger.warning(f"Could not load cached results: {e}. Starting fresh.")
            cached_results_by_global_index = {}
            successful_indices = set()

    # Process each data point
    results = []

    logger.info(f"Starting to process {len(data)} data points")
    if successful_indices:
        remaining_to_process = len(data) - len(successful_indices)
        logger.info(f"Will skip {len(successful_indices)} successful entries")
        logger.info(f"Will process {remaining_to_process} entries (missing/failed)")
    else:
        logger.info("Starting fresh - no existing successful results found")

    start_processing_time = datetime.datetime.now()
    successful_count = 0
    failed_count = 0

    for i, item in enumerate(data):
        time.sleep(0.3)

        global_index = item.get(
            "global index", item.get("global_index", i)
        )  # Use 'global index' from data or fallback to enumerate index

        # Skip if global_index exists in cached results AND success == True
        if (
            global_index in cached_results_by_global_index
            and cached_results_by_global_index[global_index].get("success", False)
        ):
            logger.debug(f"✓ Skipping successful entry {global_index}")
            continue

        # Extract question from the data
        question = item.get("prompt_formatted", item.get("prompt", ""))

        if not question:
            logger.warning(f"No question found for item {global_index}, skipping")
            continue

        # Log progress
        logger.info(f"🔄 Processing {i + 1}/{len(data)} | Global ID: {global_index}")

        # Run inference
        item_start_time = datetime.datetime.now()
        try:
            inference_result = model_inferencer.infer(model_name, question)

            # Calculate processing time
            item_duration = (datetime.datetime.now() - item_start_time).total_seconds()

            # Log consolidated inference result
            if inference_result.get("success", False):
                token_usage = inference_result.get("token_usage", {})
                logger.info(
                    f"✅ {global_index} | {inference_result.get('provider', 'unknown')} | "
                    f"Duration: {item_duration:.2f}s | "
                    f"Tokens: {token_usage.get('input_tokens', 0)}/{token_usage.get('output_tokens', 0)}/{token_usage.get('total_tokens', 0)}"
                )
                successful_count += 1
            else:
                error_msg = inference_result.get("error", "Unknown error")
                logger.error(
                    f"❌ {global_index} | {inference_result.get('provider', 'unknown')} | {model_name} | "
                    f"Duration: {item_duration:.2f}s | Error: {error_msg}"
                )
                failed_count += 1

            # Create result entry matching cached_results golden standard structure
            result_entry = {
                "global_index": global_index,
                "question": question,
                "llm_selected": model_name,
                "generated_answer": inference_result.get("response", ""),
                "token_usage": inference_result.get(
                    "token_usage",
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                ),
                "success": inference_result.get("success", False),
                "provider": inference_result.get("provider", "unknown"),
                "error": inference_result.get("error", None)
                if not inference_result.get("success", False)
                else None,
                "batch_size": inference_result.get("batch_size", None),
                "batch_number": inference_result.get("batch_number", None),
                "batch_index": inference_result.get("batch_index", None),
                "processing_mode": inference_result.get(
                    "processing_mode", "individual"
                ),
                "evaluation_result": None,  # Preserved from cached_results or null for new entries
            }

            # Add to results
            results.append(result_entry)

            # Update cached results lookup, preserving evaluation_result if it exists
            if (
                global_index in cached_results_by_global_index
                and "evaluation_result" in cached_results_by_global_index[global_index]
            ):
                result_entry["evaluation_result"] = cached_results_by_global_index[
                    global_index
                ]["evaluation_result"]
            cached_results_by_global_index[global_index] = result_entry

        except Exception as e:
            item_duration = (datetime.datetime.now() - item_start_time).total_seconds()
            logger.error(
                f"💥 EXCEPTION | Global ID: {global_index} | Duration: {item_duration:.2f}s | Error: {str(e)}"
            )
            failed_count += 1

            # Still save the failed attempt - matching cached_results golden standard structure
            result_entry = {
                "global_index": global_index,
                "question": question,
                "llm_selected": model_name,
                "generated_answer": "",
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "success": False,
                "provider": "unknown",
                "error": str(e),
                "batch_size": None,
                "batch_number": None,
                "batch_index": None,
                "processing_mode": "individual",
                "evaluation_result": None,  # Preserved from cached_results or null for new entries
            }

            results.append(result_entry)

            # Update cached results lookup for failed entries too, preserving evaluation_result if it exists
            if (
                global_index in cached_results_by_global_index
                and "evaluation_result" in cached_results_by_global_index[global_index]
            ):
                result_entry["evaluation_result"] = cached_results_by_global_index[
                    global_index
                ]["evaluation_result"]
            cached_results_by_global_index[global_index] = result_entry

        # Log progress every 10 items
        if (i + 1) % 10 == 0:
            elapsed_time = (
                datetime.datetime.now() - start_processing_time
            ).total_seconds() / 60
            logger.info(
                f"Progress: {i + 1}/{len(data)} completed | "
                f"Success: {successful_count} | Failed: {failed_count} | "
                f"Elapsed: {elapsed_time:.1f}min"
            )

    # Save complete merged results (cached + new) to maintain structure
    all_final_results = list(cached_results_by_global_index.values())

    # Save complete results to file
    with open(output_file, "w") as f:
        for result in all_final_results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    total_entries = len(all_final_results)
    successful_entries = sum(1 for r in all_final_results if r.get("success", False))
    failed_entries = total_entries - successful_entries

    logger.info("Inference pipeline completed.")
    logger.info(f"Total entries in cache: {total_entries}")
    logger.info(f"Successful: {successful_entries} | Failed: {failed_entries}")
    logger.info(f"New entries processed this run: {len(results)}")
    logger.info(f"Results saved to: {output_file}")

    return all_final_results
