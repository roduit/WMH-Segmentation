# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-30 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Base Model Class for Image Segmentation -*-

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self) -> None:
        """Abstract base class for image segmentation models."""
        super().__init__()
        self.run_name = ""

    def set_run_name(self, run_name: str) -> None:
        """Set the MLflow run name.

        Args:
            run_name (str): The name of the MLflow run.

        """
        self.run_name = run_name

    @abstractmethod
    def train(self) -> None:
        """Train the model."""

    @abstractmethod
    def predict_test(self) -> None:
        """Predict the test set."""

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict) -> "BaseModel":
        """Generate the model from a config file.

        Args:
            cfg (dict): The config file

        Returns:
            Model: The model

        """
