# -*- Last revision: 2026-02-09 by Vincent Roduit -*-
import mlflow
import nibabel as nib
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from tqdm import tqdm

from data_func.h5_helpers import open_h5
from data_func.utils import crop_array
from misc.constants import (
    DATA_DIR,
    FEATURES_BEST_MODEL,
    ID,
    MPM_REL_DIR,
    PRED_FILE_PATTERN,
    PRED_PROBA_FILE_PATTERN,
    SHAP_RESULTS_DIR,
    VOXEL_CLASS,
)


def get_subject_voxels(
    subject_id: str, model: str, participant_sample: int
) -> pd.DataFrame:
    """Get a subset of voxels for the shap analysis.

    Args:
        subject_id (str): id of the subject.
        model (str): model name.
        participant_sample (int): number of voxels to select per participant.

    Returns:
        pd.DataFrame: DataFrame containing all voxel informations.

    """
    # ---- Fetch and load participant informations ----
    subject_dir = DATA_DIR / subject_id
    h5_path = next(iter((subject_dir / MPM_REL_DIR).glob("*.h5")), None)
    prediction_path = next(
        iter((subject_dir / "segmentation" / model).glob(PRED_FILE_PATTERN)), None
    )
    prediction_proba_path = next(
        iter((subject_dir / "segmentation" / model).glob(PRED_PROBA_FILE_PATTERN)), None
    )

    subject_data = open_h5(h5_path)
    subject_data.clean_inputs()
    subject_data.compute_features(FEATURES_BEST_MODEL)

    pred = nib.load(prediction_path).get_fdata()
    pred_prob = nib.load(prediction_proba_path).get_fdata()

    pred = crop_array(pred, subject_data.patch_coords)
    pred_prob = crop_array(pred_prob, subject_data.patch_coords)
    n_features = subject_data.x.shape[-1]

    # ---- Flatten arrays to voxels shape ----
    x_flat = subject_data.x.reshape(-1, n_features)  # (n_voxels, n_features)
    y_flat = subject_data.gt.flatten().astype(bool)  # (n_voxels,)
    pred_flat = pred.flatten().astype(bool)  # (n_voxels,)
    wm_mask_flat = subject_data.wm_mask.flatten()  # (n_voxels,)

    # ---- Apply WM mask to filter voxels ----
    x_masked = x_flat[wm_mask_flat]
    y_masked = y_flat[wm_mask_flat]
    pred_masked = pred_flat[wm_mask_flat]

    # ---- Compute voxel class ----
    tp_idx = np.where(y_masked & pred_masked)[0]
    tn_idx = np.where(~y_masked & ~pred_masked)[0]
    fn_idx = np.where(y_masked & ~pred_masked)[0]
    fp_idx = np.where(~y_masked & pred_masked)[0]

    # ---- Subsample voxels ----
    rng = np.random.default_rng(12345)
    num_sample = participant_sample // 4
    if len(tp_idx) == 0 or len(tn_idx) == 0 or len(fn_idx) == 0 or len(fp_idx) == 0:
        return None
    sub_tn = rng.choice(tn_idx, min(num_sample, len(tn_idx)), replace=False)
    sub_tp = rng.choice(tp_idx, min(num_sample, len(tp_idx)), replace=False)
    sub_fn = rng.choice(fn_idx, min(num_sample, len(fn_idx)), replace=False)
    sub_fp = rng.choice(fp_idx, min(num_sample, len(fp_idx)), replace=False)
    selected_idx = np.concatenate([sub_tp, sub_tn, sub_fn, sub_fp])

    # ---- Select subsample of x ----
    x_selected = x_masked[selected_idx]
    y_selected = y_masked[selected_idx]
    p_selected = pred_masked[selected_idx]
    df_x = pd.DataFrame(x_selected, columns=subject_data.feature_names)

    voxel_class = np.full(len(y_selected), "", dtype=object)  # empty string array

    # True Positive: predicted 1, label 1
    voxel_class[(p_selected) & (y_selected)] = "TP"

    # False Negative: predicted 0, label 1
    voxel_class[(~p_selected) & (y_selected)] = "FN"

    # True Negative: predicted 0, label 0
    voxel_class[(~p_selected) & (~y_selected)] = "TN"

    # False Positive: predicted 1, label 0
    voxel_class[(p_selected) & (~y_selected)] = "FP"

    df_x[VOXEL_CLASS] = voxel_class

    df_x[ID] = subject_id

    return df_x


def get_voxels(
    df_dataset: pd.DataFrame,
    model: str = "lgbm_post",
    filename: str = "shap_voxels.csv",
    n_jobs: int = 4,
    participant_sample: int = 2000,
) -> pd.DataFrame:
    """Get voxels subsample from dataset.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing participants information.
        model (str, optional): Name of the model folder. Defaults to "lgbm_post".
        filename (str, optional): Output filename. Defaults to "shap_voxels.csv".
        n_jobs (int, optional): Number of jobs for parallel computation. Defaults to 4.
        participant_sample (int): Number of voxels to select per subject.

    """
    csv_path = SHAP_RESULTS_DIR / "shap_voxels.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    subject_ids = df_dataset[ID].tolist()

    def process_subject(subject_id: str) -> pd.DataFrame:
        return get_subject_voxels(
            subject_id, model=model, participant_sample=participant_sample
        )

    # Run in parallel
    dfs = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(sid)
        for sid in tqdm(subject_ids, desc="Processing subjects")
    )

    # Concatenate all at once
    df_x = pd.DataFrame(pd.concat(dfs, ignore_index=True))

    df_x.to_csv(SHAP_RESULTS_DIR / filename)

    return df_x


def get_explainer(model_id: str) -> shap.TreeExplainer:
    """Get the explaination based on the model id and the voxel subsample.

    Args:
        model_id (str): ID of the model to use.

    Returns:
        shap.TreeExplainer: The SHAP explainer.

    """
    model = mlflow.sklearn.load_model(model_id)
    return shap.TreeExplainer(model)
