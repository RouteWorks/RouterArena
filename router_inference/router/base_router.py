# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Abstract base class for router implementations.

All router implementations must inherit from this class and implement
the _get_prediction() method. The public get_prediction() method handles
validation automatically.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseRouter(ABC):
    """
    Abstract base class for router implementations.

    This class provides the foundation for all router implementations.
    It handles config loading and validation, while requiring subclasses
    to implement the core routing logic.

    Args:
        router_name: Name of the router (used to load config file)

    Attributes:
        router_name: Name of the router
        config: Router configuration dictionary
        models: List of available models from config
    """

    def __init__(self, router_name: str):
        """
        Initialize the router with a router name.

        Args:
            router_name: Name of the router (used to load config file)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        self.router_name = router_name
        self.config = self._load_config()
        self.models = self._extract_models()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load router configuration from JSON file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config structure is invalid
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_path = os.path.join(
            project_root, "router_inference", "config", f"{self.router_name}.json"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validate config structure
        if "pipeline_params" not in config:
            raise ValueError(
                f"Invalid config structure: missing 'pipeline_params' in {config_path}"
            )

        if "models" not in config["pipeline_params"]:
            raise ValueError(
                "Invalid config structure: missing 'models' in pipeline_params"
            )

        return config

    def _extract_models(self) -> List[str]:
        """
        Extract list of models from config.

        Returns:
            List of model names
        """
        return self.config["pipeline_params"]["models"]

    def _validate_model(self, model_name: str) -> None:
        """
        Validate that the selected model is in the config.

        Args:
            model_name: Name of the model to validate

        Raises:
            ValueError: If model is not in the config
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' is not in the router config. "
                f"Available models: {self.models}"
            )

    @abstractmethod
    def _get_prediction(self, query: str) -> str:
        """
        Get the model prediction for a given query (internal implementation).

        This is the core method that must be implemented by all router subclasses.
        It should analyze the query and return the name of the model to use.
        Subclasses should not validate the model - that is handled by get_prediction().

        Args:
            query: The input query string

        Returns:
            Name of the selected model
        """
        pass

    def get_prediction(self, query: str) -> str:
        """
        Get the model prediction for a given query (public method with validation).

        This method calls the subclass's _get_prediction() implementation and
        validates that the returned model is in the config.

        Args:
            query: The input query string

        Returns:
            Name of the selected model (validated to be in config)

        Raises:
            ValueError: If the returned model is not in the config
        """
        model_name = self._get_prediction(query)
        self._validate_model(model_name)
        return model_name
