# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-30 -*-
# -*- Last revision: 2026-02-18 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Base Model for Image Segmentation (Classical Framework) -*-

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import mlflow
import nibabel as nib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, TunedThresholdClassifierCV
from tqdm import tqdm

from analysis.metrics import get_all_metrics
from data_func.data_load import (
    get_array,
    load_csv_description,
    load_subject,
    load_train,
)
from data_func.utils import filter_csv_description, pad_to_original_shape
from misc.constants import (
    DATA_DIR,
    DATASET,
    DEFAULT_MAP_ORDER,
    FAZEKAS,
    GROUP,
    H5_PATH,
    ID,
    OPTUNA_STORAGE,
    REGION,
    REGION_NAME,
    SPATIAL_REGIONS_BY_INT,
    TEST,
)
from mlflow_func.mlflow_logger import (
    generate_test_plots,
    log_3d_mask,
    log_dataframe_as_csv,
    log_fold_metrics,
    log_metrics_summary,
    log_results_optuna,
)
from models.base_model import BaseModel
from models.model_utils import split_test_set

logger = logging.getLogger(__name__)


class BaseModelClassical(BaseModel):
    """Abstract base class for image segmentation models."""

    def __init__(
        self,
        base_params: dict[str, Any],
        num_folds: int,
        num_thresholds: int,
        grid_search_params: dict[str, Any],
    ) -> None:
        """Initialize the base model.

        Args:
            base_params (Dict[str, Any]): Base parameters for the model.
            num_folds (int): Number of folds for cross-validation.
            num_thresholds (int): Number of thresholds for tuning.
            grid_search_params (Dict[str, Any]): Parameters for grid search.

        Attributes:
            grid_search_enabled (bool): Boolean to indicate wether gridsearch is enable.
            y_true (np.ndarray): Array containing batch gt labels. Shape [n_voxels,]
            y_pred (np.ndarray): Array containing batch prediction labels.
              Shape [n_voxels,]
            x_train (np.ndarray): Array containing voxel features.
              Shape [n_voxels, n_features]
            y_train (np.ndarray): Array containing train gt labels. Shape [n_voxels,]
            groups_train (np.ndarray): Array containing group labels. Shape [n_voxels,]
            classifier (LGBMClassifier | RandomForestClassifier): The classifier.
            best_threshold_ (float): The best threshold founded after tuning.
            feature_names_ (list): The name of the features.
            df_metrics (pd.DataFrame): DataFrame containing the results for each
              participants.
            df_spatial_metrics (pd.DataFrame): DataFrame containing the results for each
              participants, split by WM regions.
            subject_data (SubjectData): the subject data.
            metrics_ (dict): Dict containing metrics for hyperparameter tuning.
            pred_3d (np.ndarray): Array containing the binary prediction. Shape [H,W,D]
            pred_3d_proba (np.ndarray): Array containing the probability prediction.
              Shape [H,W,D]
            x_test (np.ndarray): Array containing voxel features.
              Shape [n_voxels, n_features]
            y_test (np.ndarray): Array containing train gt labels. Shape [n_voxels,]

        """
        self.base_params = base_params
        self.num_folds = num_folds
        self.num_thresholds = num_thresholds
        self.grid_search_params = grid_search_params

        self.grid_search_enabled = grid_search_params != {}
        self.y_true = []
        self.y_pred = []
        self.x_train = None
        self.y_train = None
        self.groups_train = None

        # ---- Initialize model-specific attributes ----
        self.classifier = LGBMClassifier | RandomForestClassifier
        self.best_threshold_ = None
        self.feature_names_ = None
        self.df_metrics = None
        self.df_spatial_metrics = None
        self.subject_data = None
        self.metrics_ = None
        self.pred_3d = None
        self.pred_3d_proba = None
        self.x_test = None
        self.y_test = None

    @abstractmethod
    def _create_model(self, params: dict) -> LGBMClassifier | RandomForestClassifier:
        """Create the model pipeline.

        Returns:
            LGBMClassifier | RandomForestClassifier: The created model.

        """

    @abstractmethod
    def _get_x_format(self, x: np.ndarray) -> np.ndarray | pd.DataFrame:
        """Get the training data in desired format.

        LGBM classifier requires a DataFrame with column names, while a Numpy array is
        sufficient for Random Forest.
        """

    def train(self, cfg: dict[str, Any]) -> None:
        """Train the model based on the dataset configuration.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary.

        """
        # ---- Load training data ----
        cfg_dataset = cfg.get("dataset", [])
        x_list_train, y_list_train, _masks_train, ids_train, feature_names = load_train(
            cfg_dataset=cfg_dataset
        )

        self.feature_names_ = feature_names

        self.x_train, self.y_train, self.groups_train = get_array(
            x_list_train, y_list_train, ids_train, cfg_dataset
        )

        logger.info("Training data shape: %s", self.x_train.shape)

        # ---- Fit the model ----
        self.fit()

    def load_model(self, model_id: str) -> None:
        """Load a pre-trained model from the specified path.

        Args:
            model_id (str): ID of the model to load.

        """
        model_uri = f"models:/{model_id}"
        self.classifier = mlflow.sklearn.load_model(model_uri)
        mlflow.log_param("loaded_model_id", model_uri)
        self.log_shap = True

    @abstractmethod
    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.

        Returns:
            float: The objective value to maximize (e.g., mean F1 score).

        """

    def fit(self) -> None:
        """Fit the model to the data."""
        if self.grid_search_enabled:
            logger.info(
                "Starting hyperparameter optimization with Optuna"
                "(and stratifiedGroupKFold cross-validation)"
            )
            # ----------------------------------------------
            # Start a new optuna study
            # or overwrite existing study with same name
            # ----------------------------------------------
            try:
                optuna.study.delete_study(
                    study_name=self.run_name, storage=OPTUNA_STORAGE
                )
            except KeyError:
                logger.info(
                    "No existing study found with name %s. Creating a new one.",
                    self.run_name,
                )
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
                storage=OPTUNA_STORAGE,
                study_name=self.run_name,
            )

            # ---- Optimize the Hyperparameters ----
            self.study.optimize(
                self._objective, n_trials=self.grid_search_params.get("n_trials", 30)
            )
            self.optuna_results = self.study.trials_dataframe()
            self.best_params_ = self.study.best_params
            logger.info("Best hyperparameters found: %s", self.best_params_)

            # ----------------------------------------------
            # Log Best untuned model
            # ----------------------------------------------
            with mlflow.start_run(nested=True, run_name="Untuned_Model"):
                self.classifier = self._create_model(self.best_params_)
                self.classifier.fit(self._get_x_format(self.x_train), self.y_train)
                mlflow.sklearn.log_model(
                    self.classifier,
                    name=f"{self._get_model_name().lower()}_untuned",
                    input_example=self._get_x_format(self.x_train[:5]),
                )

            # ---- Log Optuna results ----
            log_results_optuna(self.study)
            logger.info("Optuna study results logged to MLflow")

        else:
            # Perform StratifiedGroupKFold cross-validation manually
            logger.info(
                "Performing StratifiedGroupKFold cross-validation with %d folds",
                self.num_folds,
            )
            self._manual_cross_validation()

        self._fine_tune_threshold()

    def _fine_tune_threshold(self) -> None:
        """Fine-tune the classification threshold using TunedThresholdClassifierCV."""
        logger.info("Fine-tuning classification threshold")
        classifier_tuned = TunedThresholdClassifierCV(
            clone(self.classifier),
            scoring="f1_weighted",
            cv=self.num_folds,
            thresholds=self.num_thresholds,
            n_jobs=1,
        )

        # ----------------------------------------------
        # Log Best tuned model
        # ----------------------------------------------
        with mlflow.start_run(nested=True, run_name="Threshold_Tuning"):
            classifier_tuned.fit(self._get_x_format(self.x_train), self.y_train)
            self.classifier = classifier_tuned.estimator_
            self.best_threshold_ = classifier_tuned.best_threshold_
            mlflow.log_param("best_threshold", self.best_threshold_)
            mlflow.sklearn.log_model(
                self.classifier,
                name="model_with_tuned_threshold",
                input_example=self._get_x_format(self.x_train[:5]),
            )

    def _manual_cross_validation(self) -> None:
        """Perform manual cross-validation with overfitting detection."""
        gkf = StratifiedGroupKFold(n_splits=self.num_folds)
        for fold_num, (train_idx, val_idx) in enumerate(
            gkf.split(self.x_train, self.y_train, self.groups_train)
        ):
            with mlflow.start_run(nested=True, run_name=f"Fold_{fold_num}"):
                logger.info("Training fold %d", fold_num)
                self.classifier.fit(
                    self._get_x_format(self.x_train[train_idx]), self.y_train[train_idx]
                )

                # ---- Get training metrics ----
                y_pred_train = self.classifier.predict(
                    self._get_x_format(self.x_train[train_idx])
                )
                self._update_metrics(
                    self.y_train[train_idx],
                    y_pred_train,
                    fold_type="train",
                )

                # ---- Get validation metrics ----
                y_pred_val = self.classifier.predict(
                    self._get_x_format(self.x_train[val_idx])
                )
                self._update_metrics(self.y_train[val_idx], y_pred_val, fold_type="val")

                log_fold_metrics(fold_num, self.metrics_)

    def _update_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_type: str = "val",
    ) -> None:
        """Update metrics for the current fold.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            fold_type (str): Type of fold metrics ('train' or 'val').
            fold_num (int): The fold number being evaluated.

        """
        if self.metrics_ is None:
            self.metrics_ = {
                "f1_macro_train": [],
                "f1_macro_val": [],
                "recall_macro_train": [],
                "recall_macro_val": [],
                "precision_macro_train": [],
                "precision_macro_val": [],
            }

        f1 = f1_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")

        self.metrics_[f"f1_macro_{fold_type}"].append(f1)
        self.metrics_[f"recall_macro_{fold_type}"].append(recall)
        self.metrics_[f"precision_macro_{fold_type}"].append(precision)

    def _predict(self, cfg_postprocess: dict, group: int) -> None:
        """Predict class labels for the input data.

        Args:
            cfg_postprocess (Dict): Configuration for postprocessing steps.
            group (int): The cross-validation group of the subject

        """
        # ----------------------------------------------
        # Predict Proba and binary outcome
        # ----------------------------------------------
        y_img_proba = self.classifier.predict_proba(
            self._get_x_format(self.subject_data.x)
        )[:, 1]
        y_img_pred = self.classifier.predict(
            self._get_x_format(self.subject_data.x)
        ).astype(np.uint8)

        # ---- Reshape to original shape [H,W,D] ----
        pred_3d = np.zeros(self.subject_data.qmaps.shape[:3], dtype=np.float32)
        pred_3d[self.subject_data.mask] = y_img_pred

        pred_3d_proba = np.zeros(self.subject_data.qmaps.shape[:3], dtype=np.float32)
        pred_3d_proba[self.subject_data.mask] = y_img_proba

        self.pred_3d_proba = pred_3d_proba

        # ---- Apply post processing given in the config file ---
        y_pred_postprocess, y_true_post_process = (
            self.subject_data.post_process_predictions(
                pred_3d,
                cfg_postprocess=cfg_postprocess,
            )
        )
        self.pred_3d = y_pred_postprocess

        self.y_true[group].extend(y_true_post_process[self.subject_data.mask])
        self.y_pred[group].extend(self.pred_3d_proba[self.subject_data.mask])

        # ---- Compute lesion-wise metrics ----
        df_global_metrics, df_spatial_metrics = get_all_metrics(
            test=self.pred_3d,
            reference=self.subject_data.gt,
            spatial_mask=self.subject_data.spatial_mask,
            threshold=200,
        )
        df_global_metrics[ID] = self.subject_data.subject_id
        df_spatial_metrics[ID] = self.subject_data.subject_id

        if self.df_metrics is not None:
            self.df_metrics = pd.concat([self.df_metrics, df_global_metrics])
            self.df_spatial_metrics = pd.concat(
                [self.df_spatial_metrics, df_spatial_metrics]
            )
        else:
            self.df_metrics = df_global_metrics
            self.df_spatial_metrics = df_spatial_metrics

        subject_fazekas = self.df_test.loc[
            self.df_test[ID] == self.subject_data.subject_id, FAZEKAS
        ].to_numpy()[0]

        # ---- Log the 3D mask only for Fazekas 6 ----
        max_fazekas = 6
        if subject_fazekas == max_fazekas:
            log_3d_mask(
                self.pred_3d.astype(np.float32),
                self.subject_data.gt.astype(np.float32),
                self.subject_data.subject_id,
            )

    def _load_test_data(
        self, h5_path: Path, cfg_dataset: dict[str, Any], maps_indices: list
    ) -> None:
        """Load the test data and extract features.

        Args:
            h5_path (Path): The path to the h5 file
            cfg_dataset (Dict[str, Any]): The config dict for the features
            maps_indices (list): The list of map indices to load

        """
        subject_data = load_subject(h5_path, cfg_dataset, maps_indices, split=TEST)
        self.subject_data = subject_data

        if self.feature_names_ is None:
            self.feature_names_ = subject_data.feature_names

    def _save_result(self, cfg_dataset: dict[str, Any]) -> None:
        """Save the images to the folder.

        Args:
            cfg_dataset (Dict[str, Any]): The config of the dataset

        """
        # ---- Retrieve saving informations from config file ---
        folder_name = cfg_dataset.get("save_folder", self._get_model_name().lower())
        dataset_path = Path(cfg_dataset.get("path", DATA_DIR))
        save_path = Path(
            dataset_path / self.subject_data.subject_id / "segmentation" / folder_name
        )
        logger.info(
            "Saving %s prediction for subject %s at %s",
            self._get_model_name(),
            self.subject_data.subject_id,
            save_path,
        )
        save_path.mkdir(parents=True, exist_ok=True)

        # ---- Pad arrays to original shape ----
        self.pred_3d = pad_to_original_shape(
            self.pred_3d,
            self.subject_data.original_shape,
            self.subject_data.patch_coords,
        )
        self.pred_3d_proba = pad_to_original_shape(
            self.pred_3d_proba,
            self.subject_data.original_shape,
            self.subject_data.patch_coords,
        )

        # ---- Save predictions in dedicated folder ----
        model_name = self._get_model_name().lower()
        nib.save(
            nib.Nifti1Image(self.pred_3d.astype(np.float32), self.subject_data.affine),
            save_path / f"{self.subject_data.subject_id}_{model_name}_pred.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(
                self.pred_3d_proba.astype(np.float32), self.subject_data.affine
            ),
            save_path
            / f"{self.subject_data.subject_id}_{model_name}_pred_proba.nii.gz",
        )

    def predict_test(self, cfg: dict[str, Any]) -> None:
        """Predict on the test dataset.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary.

        """
        # ---- Retrieve config informations ----
        cfg_dataset = cfg.get("dataset", {})
        dataset_name = cfg_dataset.get("name", "")
        dataset_path = Path(cfg_dataset.get("path", DATA_DIR))
        cfg_postprocess = cfg_dataset.get("postprocess", {})
        csv_name = cfg_dataset.get("csv_name", "dataset_description.csv")
        csv_path = dataset_path / csv_name
        split_to_segment = cfg_dataset.get("split_to_segment", None)

        # ---- Load test CSV and split it into folds for CV ----
        df = load_csv_description(csv_path)
        self.df_test = filter_csv_description(df, split=split_to_segment)
        self.df_test = split_test_set(self.df_test)
        if self.x_test is None:
            self.y_true = {k: [] for k in self.df_test[GROUP].unique()}
            self.y_pred = {k: [] for k in self.df_test[GROUP].unique()}
        is_save = cfg_dataset.get("save", False)

        maps_to_load = cfg_dataset.get("maps", "all")
        maps_indices = (
            [DEFAULT_MAP_ORDER.index(map_name) for map_name in maps_to_load]
            if maps_to_load != "all"
            else list(range(len(DEFAULT_MAP_ORDER)))
        )

        # ----------------------------------------------
        # Perform prediction for test participants
        # ----------------------------------------------
        for _idx, row in tqdm(
            self.df_test.dropna().sort_values(by=FAZEKAS, ascending=False).iterrows(),
            total=self.df_test.dropna().shape[0],
            desc=f"Predicting test data with {self._get_model_name()}",
        ):
            test_path = row[H5_PATH]
            group = row[GROUP]

            if test_path == "nan" or not Path(test_path).exists():
                continue
            test_path = Path(str(test_path))

            # ---- Compute prediction ----
            self._load_test_data(test_path, cfg_dataset, maps_indices)
            self._predict(cfg_postprocess, group)

            # ---- Save results ----
            if is_save:
                self._save_result(cfg_dataset=cfg_dataset)
        if self.df_metrics is None:
            logger.warning("No test predictions were made; skipping result logging.")
            return
        self.df_metrics[DATASET] = dataset_name
        self.df_spatial_metrics[DATASET] = dataset_name

        # ---- Log results to MLflow ----
        logger.info("Logging test results to MLflow")
        self._log_results()

    def _log_results(self) -> None:
        """Log the results of the test predictions to MLflow."""
        if self.df_metrics is not None:
            model_name = self._get_model_name().lower()

            # ---- Merge with datasetd description to have Fazekas ----
            self.df_metrics = self.df_metrics.merge(self.df_test[[ID, FAZEKAS]], on=ID)
            self.df_spatial_metrics = self.df_spatial_metrics.merge(
                self.df_test[[ID, FAZEKAS]], on=ID
            )
            self.df_spatial_metrics[REGION_NAME] = self.df_spatial_metrics[REGION].map(
                SPATIAL_REGIONS_BY_INT
            )
            log_dataframe_as_csv(
                self.df_metrics, f"{model_name}_test_global_metrics.csv", "test_results"
            )
            log_dataframe_as_csv(
                self.df_spatial_metrics,
                f"{model_name}_test_spatial_metrics.csv",
                "test_results",
            )
            logger.info("Saving test plots")
            # Generate and save plots
            generate_test_plots(
                model_name,
                self.df_metrics,
                self.df_spatial_metrics,
                self.y_true,
                self.y_pred,
            )
            logger.info("Logging metrics plots")
            log_metrics_summary(self.df_metrics)

    @abstractmethod
    def _get_model_name(self) -> str:
        """Get the name of the model for logging purposes.

        Returns:
            str: The model name.

        """

    @classmethod
    @abstractmethod
    def from_config(cls, cfg_model: dict[str, Any]) -> "BaseModel":
        """Create a model instance from a configuration dictionary.

        Args:
            cfg_model (Dict[str, Any]): Configuration dictionary for the model.

        Returns:
            BaseModel: An instance of the model.

        """
