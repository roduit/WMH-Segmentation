# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-16 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Implement Random Forest Classifier for Image Segmentation -*-

import logging

import numpy as np
import optuna
from mlflow.utils.autologging_utils import disable_autologging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from models.base_model_classical import BaseModelClassical

logger = logging.getLogger(__name__)


class RfClassifier(BaseModelClassical):
    """Random Forest Classifier for Image Segmentation."""

    def __init__(
        self,
        rf_base_params: dict,
        num_folds: int,
        num_thresholds: int,
        grid_search_params: dict,
    ) -> None:
        """Initialize Random Forest Classifier.

        Args:
            rf_base_params (dict): Base parameters for the Random Forest.
            num_folds (int): Number of folds for cross-validation.
            num_thresholds (int): Number of thresholds for tuning.
            grid_search_params (dict): Parameters for grid search.

        """
        super().__init__(rf_base_params, num_folds, num_thresholds, grid_search_params)
        self.rf_base_params = rf_base_params

    def _create_model(self, params: dict) -> RandomForestClassifier:
        """Create LightGBM Classifier with given parameters.

        Args:
            params (dict): Parameters for the Random Forest model.

        Returns:
            RandomForestClassifier: An instance of Random Forest Classifier.

        """
        return RandomForestClassifier(
            **params, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )

    def _get_x_format(self, x: np.ndarray) -> np.ndarray:
        """Return the input format for Random Forest.

        The only purpose of this function is to be consistent with LGBM and to be able
          to use the same architecture of BaseModel.

        Args:
            x (np.ndarray): Feature array.

        Returns:
            np.ndarray: Formatted feature array.

        """
        return x

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Create the objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.

        Returns:
            float: Mean auc score across folds.

        """
        cfg_grid = self.grid_search_params.get("param_grid", {})

        # ---- Convert config ranges to distributions ----
        params = {}

        # ---- build the Grid search ----
        numeric_params = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        ]
        categorical_params = [
            "max_features",
        ]
        for param in cfg_grid:
            if param in numeric_params:
                low, high = cfg_grid[param]
                params[param] = trial.suggest_int(param, int(low), int(high))
            elif param in categorical_params:
                params[param] = trial.suggest_categorical(param, cfg_grid[param])

        # ---- grouped CV ----
        gkf = StratifiedGroupKFold(
            n_splits=self.num_folds, shuffle=True, random_state=42
        )
        pr_aucs = []
        for train_idx, test_idx in gkf.split(
            self.x_train, self.y_train, groups=self.groups_train
        ):
            x_train, x_val = self.x_train[train_idx], self.x_train[test_idx]
            y_train, y_val = self.y_train[train_idx], self.y_train[test_idx]
        with disable_autologging():
            bst = self._create_model(params)
            bst.fit(x_train, y_train)
            preds = bst.predict_proba(x_val)[:, 1]
            pr = average_precision_score(y_val, preds)
            pr_aucs.append(pr)

        return np.mean(pr_aucs)

    def _get_model_name(self) -> str:
        """Get the name of the model for logging purposes.

        Returns:
            str: The model name.

        """
        return "random_forest"

    @classmethod
    def from_config(cls, cfg_model: dict) -> "RfClassifier":
        """Train the model based on the dataset configuration.

        Args:
            cfg_model (dict): Model configuration dictionary.

        """
        return RfClassifier(
            rf_base_params=cfg_model.get("rf_base", {}),
            num_folds=cfg_model.get("num_folds", 2),
            num_thresholds=cfg_model.get("num_thresholds", 100),
            grid_search_params=cfg_model.get("grid_search", {}),
        )
