# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-12-15 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Implement the class for the model -*-

import logging
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from constants import MAPS_ORDER_ALL, MAPS_ORDER_DIFF
from data_loader import load_subject
from lightgbm import LGBMClassifier
from post_processing import wm_restriction
from tqdm import tqdm
from utils import check_csv, pad_to_original_shape

logger = logging.getLogger(__name__)

file_path = Path(__file__).resolve()


class PredictionModel:
    def __init__(self, is_diff_model: bool) -> None:
        """Initialize the Prediction model."""
        self.is_diff_model = is_diff_model
        self.df_subjects = pd.DataFrame()
        self.csv_basedir = Path()
        self.classifier = LGBMClassifier

        self.load_model()

    def load_model(self) -> None:
        """Load the model."""
        if self.is_diff_model:
            with Path(file_path.parents[1] / "model" / "diff_model.pkl").open(
                "rb"
            ) as file:
                self.classifier = pickle.load(file)  # noqa: S301
            self.modalities = MAPS_ORDER_DIFF
        else:
            with Path(file_path.parents[1] / "model" / "mpm_diff_model.pkl").open(
                "rb"
            ) as file:
                self.classifier = pickle.load(file)  # noqa: S301
            self.modalities = MAPS_ORDER_ALL

    def predict(self, csv_path: Path) -> None:
        """Predict based on csv path.

        Args:
            csv_path (Path): Path to the csv

        """
        self.df_subjects = check_csv(csv_path, self.modalities)

        self.csv_basedir = Path(csv_path).parent

        for _idx, row in tqdm(
            self.df_subjects.iterrows(),
            total=len(self.df_subjects),
            desc="Predicting subjects",
        ):
            self.subject_data = load_subject(
                row, self.modalities, self.csv_basedir, self.is_diff_model
            )
            out_dir = Path(self.csv_basedir / row["OUT_DIR"])
            subject_id = str(row["ID"])
            logger.info("Predicting participant %s", subject_id)
            self._predict()
            logger.info("Saving result to %s", out_dir)
            self._save_result(out_dir, subject_id)
            logger.info("Participant %s completed successfully", subject_id)

    def _predict(self) -> None:
        """Predict class labels for the input data.

        Args:
            cfg_postprocess (Dict): Configuration for postprocessing steps.
            group (int): The cross-validation group of the subject

        """
        y_img_proba = self.classifier.predict_proba(
            pd.DataFrame(self.subject_data.x, columns=self.subject_data.feature_names)
        )[:, 1]
        y_img_pred = self.classifier.predict(
            pd.DataFrame(self.subject_data.x, columns=self.subject_data.feature_names)
        ).astype(np.uint8)

        pred_3d = np.zeros(self.subject_data.qmaps.shape[:3], dtype=np.float32)
        pred_3d[self.subject_data.mask] = y_img_pred

        pred_3d_proba = np.zeros(self.subject_data.qmaps.shape[:3], dtype=np.float32)
        pred_3d_proba[self.subject_data.mask] = y_img_proba

        self.pred_3d_proba = pred_3d_proba
        self.pred_3d = pred_3d

        self.pred_3d, self.pred_3d_proba = wm_restriction(
            self.pred_3d,
            self.pred_3d_proba,
            self.subject_data.wm_mask,
            self.subject_data.csf_mask,
            dilation_iter=2,
            lesion_frac=0.5,
        )

    def _save_result(self, out_dir: Path, subject_id: str) -> None:
        """Save the image to the folder."""
        out_dir.mkdir(parents=True, exist_ok=True)

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
        # Save the predictions
        nib.save(
            nib.Nifti1Image(self.pred_3d.astype(np.float32), self.subject_data.affine),
            out_dir / f"{subject_id}_pred.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(
                self.pred_3d_proba.astype(np.float32), self.subject_data.affine
            ),
            out_dir / f"{subject_id}_pred_proba.nii.gz",
        )
