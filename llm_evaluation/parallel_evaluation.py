# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Parallel Evaluation Manager

This module provides parallel evaluation capabilities for RouterArena.
It centralizes the query-level parallelism logic used by both run.py and evaluate_models.py.
"""

import logging
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class ParallelEvaluationManager:
    """
    Manages parallel evaluation of queries using multiple workers.
    """

    def __init__(self, workers: int = 16):
        """
        Initialize the Parallel Evaluation Manager.

        Args:
            workers: Number of parallel workers for query processing
        """
        self.workers = workers
        self.stats_lock = threading.Lock()
        self.save_lock = threading.Lock()

        # Statistics
        self.reset_stats()

    def reset_stats(self):
        """Reset evaluation statistics."""
        with self.stats_lock:
            self.processed_count = 0
            self.evaluated_count = 0
            self.skipped_count = 0
            self.failed_count = 0
            self.already_evaluated_count = 0
            self.start_time = datetime.datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Return current evaluation statistics."""
        with self.stats_lock:
            return {
                "processed": self.processed_count,
                "evaluated": self.evaluated_count,
                "skipped": self.skipped_count,
                "failed": self.failed_count,
                "already_evaluated": self.already_evaluated_count,
                "total_duration_min": (
                    datetime.datetime.now() - self.start_time
                ).total_seconds()
                / 60,
            }

    def evaluate_entries_parallel(
        self,
        tasks: List[tuple],
        evaluation_func: Callable,
        save_func: Optional[Callable] = None,
        save_interval: int = 50,
        total_count: int = 0,
        **extra_kwargs,
    ) -> Dict[str, Any]:
        """
        Process entries in parallel using ThreadPoolExecutor.

        Args:
            tasks: List of (index, data) tuples to process
            evaluation_func: Function to call for each task (index, data, **extra_kwargs)
            save_func: Optional function to call for periodic saving
            save_interval: Interval for periodic saving
            total_count: Total count for progress reporting
            **extra_kwargs: Additional keyword arguments for evaluation_func

        Returns:
            Dictionary of final statistics
        """
        if not tasks:
            logger.info("No entries to evaluate.")
            return self.get_stats()

        if total_count == 0:
            total_count = len(tasks)

        logger.info(f"Starting parallel evaluation with {self.workers} workers")

        def worker_task(idx: int, data: Any) -> bool:
            try:
                # Execute evaluation
                success = evaluation_func(idx, data, **extra_kwargs)

                # Update statistics
                with self.stats_lock:
                    if success:
                        self.evaluated_count += 1
                    else:
                        self.skipped_count += 1
                    self.processed_count += 1
                    current_processed = self.processed_count

                # Handle periodic saving
                if (
                    save_func
                    and save_interval > 0
                    and current_processed % save_interval == 0
                ):
                    with self.save_lock:
                        save_func()

                        stats = self.get_stats()
                        logger.info(
                            f"Progress: {current_processed}/{total_count} | "
                            f"Evaluated: {stats['evaluated']} | "
                            f"Skipped: {stats['skipped']} | "
                            f"Elapsed: {stats['total_duration_min']:.1f}min | Saved checkpoint"
                        )

                return success
            except Exception as e:
                logger.error(f"Error in worker task (index {idx}): {e}", exc_info=True)
                with self.stats_lock:
                    self.failed_count += 1
                    self.processed_count += 1
                return False

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_task = {
                executor.submit(worker_task, idx, data): (idx, data)
                for idx, data in tasks
            }

            for future in as_completed(future_to_task):
                try:
                    future.result()
                except Exception as e:
                    idx, _ = future_to_task[future]
                    logger.error(f"Task at index {idx} failed with exception: {e}")

        # Final save if function provided
        if save_func:
            with self.save_lock:
                save_func()

        return self.get_stats()
