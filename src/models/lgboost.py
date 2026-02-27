# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-04 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Light Gradient Boost Class -*-

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from models.base_model_classical import BaseModelClassical

logger = logging.getLogger(__name__)


class LgbClassifier(BaseModelClassical):
    """Light Gradient Boosting Model Class."""

    def __init__(
        self,
        base_params: dict,
        num_folds: int,
        num_thresholds: int,
        grid_search_params: dict,
    ) -> None:
        """Initialize Light Gradient Boosting Model.

        Args:
            base_params (dict): Base parameters for the LightGBM model.
            num_folds (int): Number of folds for cross-validation.
            num_thresholds (int): Number of thresholds for tuning.
            grid_search_params (dict): Parameters for grid search.

        """
        super().__init__(base_params, num_folds, num_thresholds, grid_search_params)

    def _create_model(self, params: dict) -> LGBMClassifier:
        """Create LightGBM Classifier with given parameters.

        Args:
            params (dict): Parameters for the LightGBM model.

        Returns:
            LGBMClassifier: An instance of LightGBM Classifier.

        """
        return LGBMClassifier(**params, device="cuda", n_jobs=-1)

    def _get_x_format(self, x: np.ndarray) -> pd.DataFrame:
        """Create LightGBM Dataset for training.

        Args:
            x (np.ndarray): Feature array.

        Returns:
            pd.DataFrame: DataFrame containing training features.

        """
        return pd.DataFrame(x, columns=self.feature_names_)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Define the Optuna objective.

        By default, the objective metric is the average precision.

        Args:
            trial (optuna.trial.Trial): The optuna trial.

        Returns:
            float: The objective value.

        """
        int_params = [
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample_freq",
            "n_estimators",
        ]
        float_params = ["subsample", "colsample_bytree", "min_split_gain"]
        log_float_params = [
            "learning_rate",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        ]
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "num_threads": -1,
            "device": "cuda",
        }
        # ---- Add config param to the gridsearch ----
        grid_search_items = self.grid_search_params.get("param_grid", {})
        for param in grid_search_items:
            low = grid_search_items[param][0]
            high = grid_search_items[param][1]
            if param in int_params:
                params[param] = trial.suggest_int(param, int(low), int(high))
            elif param in float_params:
                params[param] = trial.suggest_float(param, float(low), float(high))
            elif param in log_float_params:
                params[param] = trial.suggest_float(
                    param, float(low), float(high), log=True
                )
            else:
                logger.warning("Parameter %s does not exist. Please add it.", param)

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
            df_train = pd.DataFrame(x_train, columns=self.feature_names_)
            df_val = pd.DataFrame(x_val, columns=self.feature_names_)
            dtrain = lgb.Dataset(df_train, label=y_train)
            dval = lgb.Dataset(df_val, label=y_val, reference=dtrain)
            bst = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
            )
            preds = bst.predict(x_val, num_iteration=bst.best_iteration)
            pr = average_precision_score(y_val, preds)
            pr_aucs.append(pr)

        return np.mean(pr_aucs)

    def _get_model_name(self) -> str:
        """Get the name of the model for logging purposes.

        Returns:
            str: The model name.

        """
        return "lgboost"

    @classmethod
    def from_config(cls, cfg_model: dict) -> object:
        """Create an instance of LgbClassifier from configuration.

        Args:
            cfg_model (dict): Model configuration dictionary.

        Returns:
            LgbClassifier: An instance of LgbClassifier.

        """
        base_params = cfg_model.get("lgb_base", {})
        num_folds = cfg_model.get("num_folds", 5)
        num_thresholds = cfg_model.get("num_thresholds", 100)
        grid_search_params = cfg_model.get("grid_search", {})
        return cls(
            base_params=base_params,
            num_folds=num_folds,
            num_thresholds=num_thresholds,
            grid_search_params=grid_search_params,
        )
