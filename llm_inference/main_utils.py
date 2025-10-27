# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import datetime
from zoneinfo import ZoneInfo
import sys
import random

# import torch
import numpy as np


def lock_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)


logger = logging.getLogger(__name__)


def set_logger(current_dir, output_folder_dir=None, args=None, model_name=None):
    """Set up logger to store all log output in llm_inference_logs file"""
    ct_timezone = ZoneInfo("America/Chicago")
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
    log_formatter.converter = lambda *args: datetime.datetime.now(
        ct_timezone
    ).timetuple()

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Set up file handler to write to llm_inference_logs with timestamp and model name
    timestamp = datetime.datetime.now(ct_timezone).strftime("%Y%m%d_%H%M%S")

    if model_name:
        # Sanitize model name for filename (replace / with _ and remove other problematic chars)
        safe_model_name = (
            model_name.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        log_file_path = os.path.join(logs_dir, f"{safe_model_name}_{timestamp}.log")
    else:
        log_file_path = os.path.join(logs_dir, f"llm_inference_logs_{timestamp}")

    # Configure the root logger so all child loggers inherit the configuration
    root_logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Set up console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Set logging level to INFO so we see detailed logs
    root_logger.setLevel(logging.INFO)

    # Suppress verbose HTTP request logs from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Log the log file location
    root_logger.info(f"Logging initialized. Log file: {log_file_path}")

    return logger
