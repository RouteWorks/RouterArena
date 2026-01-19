# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Parallel LLM Inference Manager

This module provides parallel inference capabilities for RouterArena.
It processes models sequentially, but uses multiple workers to parallelize
query processing within each model.

Adapted from How2TrainARouter's sequential_inference.py.
"""

import json
import os
import logging
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock

from model_inference import ModelInference

# Thread-local storage for ModelInference instances
_thread_local = threading.local()

logger = logging.getLogger(__name__)


class ParallelInferenceManager:
    """
    Manages parallel inference for multiple models.

    Architecture:
    - Process models sequentially (one at a time)
    - Within each model, use k workers for parallel query processing
    - Example: 320 queries with 16 workers → each worker handles ~20 queries
    """

    def __init__(self, cache_dir: str = "./cached_results", workers: int = 16):
        """
        Initialize the Parallel Inference Manager

        Args:
            cache_dir: Directory where cached results are stored
            workers: Number of parallel workers per model
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.file_lock = threading.Lock()  # Lock for file operations
        self.results_lock = threading.Lock()  # Lock for results dictionary

    @staticmethod
    def load_model_list(
        model_cost_path: str = "./model_cost/model_cost.json",
    ) -> List[str]:
        """
        Load list of model names from model_cost.json

        Args:
            model_cost_path: Path to model_cost.json

        Returns:
            List of model names (keys from model_cost.json)
        """
        if not os.path.exists(model_cost_path):
            raise FileNotFoundError(f"model_cost.json not found at {model_cost_path}")

        with open(model_cost_path, "r", encoding="utf-8") as f:
            model_cost = json.load(f)

        models = list(model_cost.keys())
        logger.info(f"Loaded {len(models)} models from {model_cost_path}")
        return models

    def load_cached_indices(self, model: str) -> Set[str]:
        """
        Load all global_indices that have already been successfully inferred for a model
        (Legacy method for backward compatibility)

        Args:
            model: Model name (e.g., 'gemini-2.0-flash-001')

        Returns:
            Set of global_indices that have been cached successfully
        """
        run_counts = self.load_cached_run_counts(model)
        return set(run_counts.keys())

    def load_cached_run_counts(self, model: str) -> Dict[str, Set[int]]:
        """
        Load run counts for each global_index that has been successfully inferred.
        Returns a dictionary mapping global_index to a set of successful run numbers.

        This method reads the entire cache file to ensure all entries are captured,
        even if the file is being written to concurrently. Uses file locking to ensure
        a consistent snapshot.

        Args:
            model: Model name (e.g., 'gemini-2.0-flash-001')

        Returns:
            Dictionary mapping global_index to set of successful run numbers
            Example: {"MMLUPro_1": {1, 2}, "MMLUPro_2": {1}}
        """
        # Sanitize model name for filename (replace / with _)
        model_filename = model.replace("/", "_")
        cache_file = self.cache_dir / f"{model_filename}.jsonl"
        run_counts: Dict[str, Set[int]] = {}

        if cache_file.exists():
            logger.info(f"Loading cached run counts from {cache_file}")

            # Use cross-platform file lock to ensure consistent read (prevents reading while writing)
            lock_file = FileLock(str(cache_file) + ".lock")
            with self.file_lock:  # Thread-level lock for this process
                try:
                    with lock_file:  # Process-level lock across all processes
                        with open(cache_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    entry = json.loads(line)
                                    global_index = entry.get("global_index")
                                    if not global_index:
                                        continue

                                    # Get run_number (default to 1 for backward compatibility)
                                    run_number = entry.get("run_number", 1)

                                    # Only count successful entries
                                    if entry.get("success", False):
                                        if global_index not in run_counts:
                                            run_counts[global_index] = set()
                                        run_counts[global_index].add(run_number)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse line in {cache_file}"
                                    )
                                    continue
                except Exception as e:
                    logger.error(f"Error loading cached run counts: {e}")

            total_queries = len(run_counts)
            total_runs = sum(len(runs) for runs in run_counts.values())
            logger.info(
                f"Found {total_queries} queries with {total_runs} total successful runs for {model}"
            )
        else:
            logger.info(f"No cache file found for {model}, starting fresh")

        return run_counts

    def load_input_data(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load input data from JSON file

        Args:
            input_file: Path to input JSON file

        Returns:
            List of data entries
        """
        logger.info(f"Loading input data from {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} entries from {input_file}")
        return data

    def filter_uninferred_data(
        self, data: List[Dict[str, Any]], model: str, num_runs: int = 1
    ) -> List[tuple]:
        """
        Filter data to determine which queries need inference and which run number to use.
        Returns list of (entry, run_number) tuples.

        Logic:
        - If query has < num_runs successful runs, it needs more runs
        - If query has run_number N but not run_number 1, new result should be run_number 1
        - If query has all runs 1..num_runs, skip it

        Args:
            data: List of data entries
            model: Model name
            num_runs: Target number of successful runs per query

        Returns:
            List of (entry, run_number) tuples for queries that need inference
        """
        run_counts = self.load_cached_run_counts(model)

        filtered_data = []
        skipped_count = 0

        for entry in data:
            global_index = entry.get("global index") or entry.get("global_index")
            if not global_index:
                continue

            existing_runs = run_counts.get(global_index, set())

            # Determine which run numbers are needed
            # Add missing runs up to num_runs (this handles gaps automatically)
            needed_runs = []
            for run_num in range(1, num_runs + 1):
                if run_num not in existing_runs:
                    needed_runs.append(run_num)

            # Add entry for each needed run
            if needed_runs:
                for run_number in needed_runs:
                    filtered_data.append((entry, run_number))
            else:
                skipped_count += 1

        # Validate: ensure no run_number exceeds num_runs
        invalid_runs = [
            (e.get("global index") or e.get("global_index"), rn)
            for e, rn in filtered_data
            if rn > num_runs
        ]
        if invalid_runs:
            logger.error(
                f"ERROR: Found invalid run numbers > {num_runs}: {invalid_runs[:5]}"
            )

        logger.info(
            f"Filtered to {len(filtered_data)} entries needing inference for {model} "
            f"(target: {num_runs} runs per query, skipped {skipped_count} queries with all runs complete)"
        )

        return filtered_data

    def clear_failed_entries(self, model: str) -> int:
        """
        Remove all failed entries from the cache file, keeping only successful ones.

        This allows failed entries to be retried in subsequent runs.
        Called automatically before processing each model to ensure failed entries
        are retried.

        Args:
            model: Model name

        Returns:
            Number of failed entries removed
        """
        model_filename = model.replace("/", "_")
        cache_file = self.cache_dir / f"{model_filename}.jsonl"

        if not cache_file.exists():
            return 0

        # Read all entries, filter out failed ones
        successful_entries = []
        failed_count = 0

        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    # Keep only successful entries
                    if entry.get("success", False):
                        successful_entries.append(entry)
                    else:
                        failed_count += 1
                except json.JSONDecodeError:
                    logger.warning("Failed to parse line, skipping")
                    failed_count += 1
                    continue

        # Write back only successful entries if any were removed
        if failed_count > 0:
            # Use cross-platform file lock for process-safe file writing
            lock_file = FileLock(str(cache_file) + ".lock")
            with self.file_lock:  # Thread-level lock for this process
                try:
                    with lock_file:  # Process-level lock across all processes
                        with open(cache_file, "w", encoding="utf-8") as f:
                            for entry in successful_entries:
                                json.dump(entry, f, ensure_ascii=False)
                                f.write("\n")
                    logger.info(
                        f"Cleared {failed_count} failed entries from {cache_file}"
                    )
                    logger.info(f"Kept {len(successful_entries)} successful entries")
                except Exception as e:
                    logger.error(f"Error clearing failed entries: {e}")

        return failed_count

    def load_existing_cache(self, model: str) -> Dict[str, Dict[str, Any]]:
        """
        Load all existing cached results for a model (all entries, including failed)
        Uses composite key: global_index_run_N for entries with run_number,
        or just global_index for legacy entries without run_number

        Args:
            model: Model name

        Returns:
            Dictionary mapping key to cached result
        """
        model_filename = model.replace("/", "_")
        cache_file = self.cache_dir / f"{model_filename}.jsonl"
        cached_results = {}

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        global_index = entry.get("global_index")
                        if not global_index:
                            continue

                        # Create key: use composite key if run_number exists, else just global_index
                        # Legacy entries (without run_number field) use global_index as key for backward compatibility
                        # New entries (with run_number field) always use composite key
                        if "run_number" not in entry:
                            # Legacy entry without run_number field - use global_index as key
                            key = global_index
                        else:
                            # Entry with run_number - always use composite key
                            run_number = entry.get("run_number", 1)
                            key = f"{global_index}_run_{run_number}"

                        cached_results[key] = entry
                    except json.JSONDecodeError:
                        continue

        return cached_results

    def save_single_result(self, result: Dict[str, Any], model: str):
        """
        Save a single result to the cache file with lock (thread-safe append)
        Ensures run_number is always present (defaults to 1 if missing)

        Note: This method assumes the filtering logic correctly identifies missing entries.
        Only entries that need inference (as determined by filter_uninferred_data) should
        be passed to this method, so we can safely append without duplicate checking.

        Args:
            result: Result dictionary
            model: Model name
        """
        # Ensure run_number is present (default to 1 for backward compatibility)
        if "run_number" not in result:
            result["run_number"] = 1

        model_filename = model.replace("/", "_")
        cache_file = self.cache_dir / f"{model_filename}.jsonl"

        # Use cross-platform file lock to ensure process-safe file writing
        lock_file = FileLock(str(cache_file) + ".lock")
        with self.file_lock:  # Thread-level lock for this process
            try:
                with lock_file:  # Process-level lock across all processes
                    with open(cache_file, "a", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")
            except Exception as e:
                logger.error(f"Error saving single result: {e}")

    def save_all_results(self, results: Dict[str, Dict[str, Any]], model: str):
        """
        Save all results to the cache file (overwrites existing file) with lock
        First reloads existing file to merge with new results
        Ensures run_number is always present in all entries

        Args:
            results: Dictionary mapping key to result (key may be global_index or global_index_run_N)
            model: Model name
        """
        model_filename = model.replace("/", "_")
        cache_file = self.cache_dir / f"{model_filename}.jsonl"

        # Use lock to ensure thread-safe file writing
        with self.file_lock:
            try:
                # Reload existing cache to merge
                existing_results = {}
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                global_index = entry.get("global_index")
                                if not global_index:
                                    continue

                                # Ensure run_number exists (default to 1 for legacy entries)
                                if "run_number" not in entry:
                                    entry["run_number"] = 1

                                # Create key: use composite key if run_number exists, else just global_index
                                # Legacy entries (without run_number field) use global_index as key for backward compatibility
                                # New entries (with run_number field) always use composite key
                                if "run_number" not in entry:
                                    # Legacy entry - use global_index as key
                                    key = global_index
                                else:
                                    # Entry with run_number - always use composite key
                                    run_number = entry.get("run_number", 1)
                                    key = f"{global_index}_run_{run_number}"

                                existing_results[key] = entry
                            except json.JSONDecodeError:
                                continue

                # Ensure all new results have run_number
                for key, result in results.items():
                    if "run_number" not in result:
                        result["run_number"] = 1

                # Merge: new results override existing ones
                merged_results = {**existing_results, **results}

                # Write all results with cross-platform file lock
                lock_file = FileLock(str(cache_file) + ".lock")
                with lock_file:  # Process-level lock across all processes
                    with open(cache_file, "w", encoding="utf-8") as f:
                        for result in merged_results.values():
                            # Ensure run_number is present before saving
                            if "run_number" not in result:
                                result["run_number"] = 1
                            json.dump(result, f, ensure_ascii=False)
                            f.write("\n")

                logger.info(f"Saved {len(merged_results)} results to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")

    def _process_single_entry(
        self,
        entry: Dict[str, Any],
        run_number: int,
        model: str,
        cached_results: Dict[str, Dict[str, Any]],
        index: int,
        total: int,
    ) -> tuple:
        """
        Process a single entry (worker function)

        Args:
            entry: Data entry to process
            run_number: Run number for this inference (1, 2, ..., N)
            model: Model name
            cached_results: Shared dictionary of cached results (thread-safe access needed)
            index: Index of this entry
            total: Total number of entries

        Returns:
            Tuple of (global_index, result_entry, success_flag)
        """
        global_index = entry.get("global index") or entry.get("global_index")
        prompt = entry.get("prompt_formatted") or entry.get("prompt")

        if not prompt:
            logger.warning(f"Entry {index} missing prompt, skipping")
            return global_index, None, False

        logger.info(
            f"Processing {index + 1}/{total} | Global ID: {global_index} | Run: {run_number}"
        )

        # Get or create model inferencer for this thread (reuse per thread to avoid overhead)
        if not hasattr(_thread_local, "model_inferencer"):
            _thread_local.model_inferencer = ModelInference()
        model_inferencer = _thread_local.model_inferencer

        try:
            inference_result = model_inferencer.infer(model.replace("_", "/"), prompt)

            # Create result entry matching cache_results format
            result_entry = {
                "global_index": global_index,
                "question": prompt,
                "llm_selected": model,
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
                "run_number": run_number,  # Add run_number field
                "evaluation_result": None,
            }

            # Preserve evaluation_result if it exists in cache (thread-safe access)
            # Note: We don't preserve evaluation_result across runs as each run is independent
            # But we keep the field for consistency

            if inference_result.get("success", False):
                token_usage = inference_result.get("token_usage", {})
                logger.info(
                    f"✅ {global_index} | Run {run_number} | {inference_result.get('provider', 'unknown')} | "
                    f"Tokens: {token_usage.get('input_tokens', 0)}/{token_usage.get('output_tokens', 0)}/{token_usage.get('total_tokens', 0)}"
                )
                return global_index, result_entry, True
            else:
                error_msg = inference_result.get("error", "Unknown error")
                logger.error(
                    f"❌ {global_index} | Run {run_number} | Error: {error_msg}"
                )
                return global_index, result_entry, False

        except Exception as e:
            logger.error(
                f"💥 EXCEPTION | Global ID: {global_index} | Run {run_number} | Error: {str(e)}"
            )

            # Save failed attempt
            result_entry = {
                "global_index": global_index,
                "question": prompt,
                "llm_selected": model,
                "generated_answer": "",
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "success": False,
                "provider": "unknown",
                "error": str(e),
                "run_number": run_number,  # Add run_number field
                "evaluation_result": None,
            }

            return global_index, result_entry, False

    def process_single_model(
        self,
        model: str,
        data: List[Dict[str, Any]],
        num_workers: Optional[int] = None,
        num_runs: int = 1,
    ) -> Dict[str, Any]:
        """
        Process a single model with parallel query processing

        Args:
            model: Model name to process
            data: List of all data entries (8400 queries)
            num_workers: Number of parallel workers (defaults to self.workers)
            num_runs: Target number of successful runs per query (default: 1)

        Returns:
            Dictionary with processing statistics
        """
        if num_workers is None:
            num_workers = self.workers

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing model: {model}")
        logger.info(f"Workers: {num_workers}")
        logger.info(f"Target runs per query: {num_runs}")
        logger.info(f"{'=' * 80}")

        # Clear failed entries before processing to allow retries
        # This ensures that failed entries can be retried in this run
        failed_cleared = self.clear_failed_entries(model)
        if failed_cleared > 0:
            logger.info(f"Cleared {failed_cleared} failed entries to allow retries")

        # Filter data to determine which queries need inference and which run number
        filtered_data = self.filter_uninferred_data(data, model, num_runs)

        if not filtered_data:
            logger.info(
                f"No new data to infer for model {model} (all queries have {num_runs} runs)"
            )
            return {
                "model": model,
                "total": len(data),
                "cached": len(data),
                "processed": 0,
                "successful": 0,
                "failed": 0,
            }

        # Load existing cache to preserve all entries
        cached_results = self.load_existing_cache(model)
        logger.info(f"Loaded {len(cached_results)} existing cache entries")

        logger.info(
            f"Starting parallel inference for {len(filtered_data)} entries with {num_workers} workers"
        )

        successful_count = 0
        failed_count = 0
        completed_count = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks (entry, run_number pairs)
            future_to_entry = {
                executor.submit(
                    self._process_single_entry,
                    entry,
                    run_number,
                    model,
                    cached_results,
                    i,
                    len(filtered_data),
                ): (i, entry, run_number)
                for i, (entry, run_number) in enumerate(filtered_data)
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_entry):
                completed_count += 1
                try:
                    global_index, result_entry, success = future.result()

                    if result_entry is None:
                        continue

                    # Create unique key for this result (global_index + run_number)
                    result_key = (
                        f"{global_index}_run_{result_entry.get('run_number', 1)}"
                    )

                    # Thread-safe update of cached_results
                    with self.results_lock:
                        cached_results[result_key] = result_entry

                    # Save immediately with lock (append mode)
                    self.save_single_result(result_entry, model)

                    if success:
                        successful_count += 1
                    else:
                        failed_count += 1

                    # Log progress every 10 entries
                    if completed_count % 10 == 0:
                        logger.info(
                            f"Progress: {completed_count}/{len(filtered_data)} | "
                            f"Success: {successful_count} | Failed: {failed_count}"
                        )

                except Exception as e:
                    logger.error(f"Error processing future: {e}")
                    failed_count += 1

        # Final save of all results - reload everything first to ensure we have all entries
        with self.results_lock:
            # Reload ALL existing cache entries from file
            final_cache = self.load_existing_cache(model)
            # Merge with our in-memory results (our results take precedence)
            final_cache.update(cached_results)
            # Save everything
            self.save_all_results(final_cache, model)
            # Update cached_results to final state for statistics
            cached_results = final_cache

        # Count unique queries and runs
        unique_queries = set()
        for key in cached_results.keys():
            if "_run_" in key:
                global_index = key.split("_run_")[0]
            else:
                global_index = key
            unique_queries.add(global_index)

        total_cached = len(unique_queries)
        run_counts = self.load_cached_run_counts(model)
        total_successful_runs = sum(len(runs) for runs in run_counts.values())

        logger.info("=" * 80)
        logger.info(f"Inference completed for {model}")
        logger.info(f"Total unique queries in cache: {total_cached}")
        logger.info(f"Total successful runs: {total_successful_runs}")
        logger.info(f"New entries processed this run: {len(filtered_data)}")
        logger.info(f"  - Successful: {successful_count}")
        logger.info(f"  - Failed: {failed_count}")
        logger.info("=" * 80)

        return {
            "model": model,
            "total": len(data),
            "cached": len(data)
            - len(
                set(
                    e.get("global index") or e.get("global_index")
                    for e, _ in filtered_data
                )
            ),
            "processed": len(filtered_data),
            "successful": successful_count,
            "failed": failed_count,
        }

    def process_all_models(
        self,
        models: List[str],
        data: List[Dict[str, Any]],
        num_workers: Optional[int] = None,
        num_runs: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all models sequentially, each with parallel query processing

        Args:
            models: List of model names to process
            data: List of all data entries (8400 queries)
            num_workers: Number of parallel workers per model
            num_runs: Target number of successful runs per query

        Returns:
            Dictionary mapping model names to their statistics
        """
        if num_workers is None:
            num_workers = self.workers

        logger.info(f"\n{'#' * 80}")
        logger.info(f"BATCH INFERENCE - Processing {len(models)} models")
        logger.info(f"Dataset size: {len(data)} queries")
        logger.info(f"Workers per model: {num_workers}")
        logger.info(f"Target runs per query: {num_runs}")
        logger.info(f"{'#' * 80}\n")

        all_stats = {}

        for i, model in enumerate(models, 1):
            logger.info(f"\n[Model {i}/{len(models)}]")
            stats = self.process_single_model(model, data, num_workers, num_runs)
            all_stats[model] = stats

        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("BATCH INFERENCE COMPLETED")
        logger.info("=" * 80)

        for model, stats in all_stats.items():
            logger.info(f"\nModel: {model}")
            logger.info(f"  Total entries: {stats['total']}")
            logger.info(f"  Already cached: {stats['cached']}")
            logger.info(f"  Processed this run: {stats['processed']}")
            logger.info(f"    - Successful: {stats['successful']}")
            logger.info(f"    - Failed: {stats['failed']}")

        logger.info("\n" + "=" * 80)

        return all_stats
