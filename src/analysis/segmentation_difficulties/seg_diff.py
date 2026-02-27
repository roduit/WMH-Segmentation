# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-19 -*-
# -*- Last revision: 2025-12-03 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Functions to process the results -*-

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from tqdm import tqdm

from data_func.data_class import SubjectData
from data_func.h5_helpers import open_h5
from data_func.post_processing import clean_small_objects
from data_func.utils import crop_array
from misc.constants import (
    GT_FILENAME_PATTERN,
    GT_MPM_REL_DIR,
    ID,
    RESULTS_DIR,
    SPATIAL_REGIONS_BY_INT,
)


def compute_lesion_stats(
    df_dataset: pd.DataFrame, min_lesion_size: int = 5
) -> pd.DataFrame:
    """Compute the stats for each lesion.

    This stats include patient-scale stats and lesions-scale stats. At patient-scale,
    The TP and FN voxels are aggregate and the mean is computed. At lesion-scale, TP and
    FN voxels are aggregate lesion-wise.

    Args:
        df_dataset (pd.DataFrame): The DataFrame containing dataset description
        min_lesion_size (int): Minimum size of a lesion to be considered

    Returns:
        pd.DataFrame: the lesion stats DataFrame

    See Also:
        - :func:`seg_diff.process_subject`: Function that computes the stats for each
          patient.

    """
    # ---- Create CSV paths and check if they exist ----
    result_lesions_path = RESULTS_DIR / "seg_diff" / "seg_diff_lesion_results.csv"
    result_patients_path = RESULTS_DIR / "seg_diff" / "seg_diff_patients_results.csv"

    if result_lesions_path.exists():
        df_results_lesions = pd.read_csv(result_lesions_path)
        df_results_patients = pd.read_csv(result_patients_path)

    else:
        df_results_lesions = []
        df_results_patients = []
        # ------------------------------------------------------------------
        # Process analysis subject-wise
        # ------------------------------------------------------------------
        for _, row in tqdm(df_dataset.iterrows(), total=len(df_dataset)):
            patient_results, subject_results = process_subject(row, min_lesion_size)
            if subject_results is not None:
                df_results_lesions.extend(subject_results)
            if patient_results is not None:
                df_results_patients.extend(patient_results)

        df_results_lesions = pd.DataFrame(df_results_lesions)

        df_results_patients = pd.DataFrame(df_results_patients)

        df_results_lesions.to_csv(result_lesions_path, index=False)
        df_results_patients.to_csv(result_patients_path, index=False)
    return df_results_patients, df_results_lesions


# ======================================================================================
# =====                              HELPER FUNCTIONS                              =====
# ======================================================================================


def process_subject(row: pd.Series, min_lesion_size: int) -> list:
    """Process one subject.

    Args:
        row (pd.Series): the row of df_dataset
        min_lesion_size (int): the minimum size of the lesion to be considered

    Returns:
        list: list of rows

    """
    paths = load_subject_paths(row)
    if paths is None:
        return None, None

    subject_data, gt, pred = load_subject_arrays(paths)

    pred, gt = clean_small_objects(pred, gt, 5, 3)

    label_array, num_features = label(gt)
    if num_features == 0:
        return None, None

    nawm_stats = compute_nawm_stats(subject_data)
    voxel_predicted = np.logical_and(pred, gt)
    voxel_missed = np.logical_and(gt, np.logical_not(pred))
    if np.sum(voxel_missed) > 0 and np.sum(voxel_predicted) > 0:
        patient_scale_dict = [
            {
                **nawm_stats,
                **{
                    k: float(v)
                    for k, v in zip(
                        subject_data.feature_names,
                        subject_data.qmaps[voxel_predicted].mean(axis=0),
                        strict=False,
                    )
                },
                "id": row[ID],
                "detected": True,
            },
            {
                **nawm_stats,
                **{
                    k: float(v)
                    for k, v in zip(
                        subject_data.feature_names,
                        subject_data.qmaps[voxel_missed].mean(axis=0),
                        strict=False,
                    )
                },
                "id": row[ID],
                "detected": False,
            },
        ]
    else:
        patient_scale_dict = None

    lesion_rows = []
    for idx in range(1, num_features + 1):
        comp_stats = process_lesion(
            idx,
            label_array,
            subject_data,
            pred,
            row,
            min_lesion_size,
            nawm_stats,
        )
        if comp_stats is not None:
            lesion_rows.extend(comp_stats)

    return patient_scale_dict, lesion_rows


def load_subject_paths(row: pd.Series) -> dict:
    """Load subejct maps.

    Args:
        row (pd.Series): The subject roe

    Returns:
        dict: dictionnary containing the paths

    """
    subject_dir = Path(row["subject_dir"])
    h5_path = Path(str(row["h5_path"]))

    gt_path = next(iter((subject_dir / GT_MPM_REL_DIR).glob(GT_FILENAME_PATTERN)), None)
    pred_path = next(
        iter((subject_dir / "segmentation" / "lgbm").glob("*pred.nii*")), None
    )

    if pred_path is None or gt_path is None or not h5_path.exists():
        return None

    return {
        "h5": h5_path,
        "gt": gt_path,
        "pred": pred_path,
    }


def load_subject_arrays(paths: dict) -> tuple[SubjectData, np.ndarray, np.ndarray]:
    """Load the subject array.

    Args:
        paths (dict): Dict containing the paths of the needed files
          (subject_data, gt, pred)

    Returns:
        tuple: a tuple containing one array per map

    """
    subject_data = open_h5(paths["h5"])
    subject_data.clean_inputs()

    gt = crop_array(nib.load(paths["gt"]).get_fdata(), subject_data.patch_coords)
    pred = crop_array(nib.load(paths["pred"]).get_fdata(), subject_data.patch_coords)

    return subject_data, gt, pred


def compute_nawm_stats(subject_data: SubjectData) -> dict:
    """Compute Normal Appearing White Matter stats.

    Args:
        subject_data (SubjectData): the SubjectData Class.

    Returns:
        dict: the dict with nawm stats

    """
    wm_mask = subject_data.wm_mask
    nawm_mask = np.logical_and(wm_mask, np.logical_not(subject_data.gt))

    return {
        f"{k}_nawm_mean": float(v)
        for k, v in zip(
            subject_data.feature_names,
            subject_data.qmaps[nawm_mask].mean(axis=0),
            strict=True,
        )
    }


def process_lesion(
    idx: int,
    label_array: np.ndarray,
    subject_data: SubjectData,
    pred: np.ndarray,
    row: pd.Series,
    min_lesion_size: int,
    nawm_stats: dict,
) -> dict:
    """Process one lesion.

    Args:
        idx (int): index of the row
        label_array (np.ndarray): Manual segmentation
        subject_data (SubjectData): SubjectData class
        pred (np.ndarray): prediction array
        pred_proba (np.ndarray): probability prediction array
        row (pd.Series): row of the df_dataset
        min_lesion_size (int): minimum lesion size
        nawm_stats (dict): dict with nawm stats

    Returns:
        dict: the results dict

    """
    component = label_array == idx
    lesion_size = component.sum()

    dominant_region = np.argmax(np.bincount(subject_data.spatial_mask[component]))
    if lesion_size < min_lesion_size or dominant_region == 0:
        return None

    detected_component = np.logical_and(component, pred)
    percentage_detected = np.sum(detected_component) / lesion_size

    return [
        {
            **nawm_stats,
            **{
                k: float(v)
                for k, v in zip(
                    subject_data.feature_names,
                    subject_data.qmaps[component].mean(axis=0),
                    strict=False,
                )
            },
            "lesion_size": float(lesion_size),
            "dominant_region": SPATIAL_REGIONS_BY_INT[dominant_region],
            "id": row[ID],
            "percentage_detected": percentage_detected,
            "lesion_id": idx,
        }
    ]
