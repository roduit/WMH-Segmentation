# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-19 -*-
# -*- Last revision: 2026-02-08 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Functions to compare results with Deep Learning models -*-

import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from analysis.metrics import get_all_metrics
from misc.constants import (
    DATASET,
    FAZEKAS,
    GT_FILENAME_PATTERN,
    ID,
    MODEL,
    RESULTS_DIR,
    SUBJECT_DIR,
)

logger = logging.getLogger(__name__)


def fetch_models_results(
    df: pd.DataFrame,
    csv_results_name: str,
    pred_patterns: dict,
    segmentations_folder: str = "segmentation",
    models: list | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fetch the model results.

    Args:
        df (pd.DataFrame): The DataFrame containing all the model results.
        csv_results_name (str): Name of the csv to save results.
        pred_patterns (dict): The pattern of the prediction name, since it differs
          depending on the model.
        segmentations_folder (str): The name of the folder containing the segmentations.
        models (list | None, optional): The list of models to compute the results.
        n_jobs (int, optional): Number of jobs for parallel computation. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the results.

    """
    # ---- Check if the results CSV already exists and return it if its the case ----
    file_name = RESULTS_DIR / "models_comp" / f"{csv_results_name}.csv"
    if file_name.exists():
        return pd.read_csv(file_name)

    # ------------------------------------------------------------------
    # Parallel computation of the results
    # ------------------------------------------------------------------
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_subject)(
            row,
            pred_patterns,
            segmentations_folder,
            models,
        )
        for row in tqdm(df.itertuples(index=False), total=len(df))
    )

    # ---- Create the final CSV and add extra informations ----
    rows = [df for sublist in results for df in sublist]

    df_metrics = pd.concat(rows, ignore_index=True)
    df_metrics[DATASET] = "Brainlaus"

    df_metrics.to_csv(
        RESULTS_DIR / "models_comp" / f"{csv_results_name}.csv", index=False
    )

    return df_metrics


def process_subject(
    row: pd.Series,
    pred_patterns: dict,
    segmentations_folder: str,
    models: list | None,
) -> list:
    """Process the model's comparisopn for one subject.

    Args:
        row (pd.Series): The series containing all subject informations
        pred_patterns (dict): The pattern of the prediction name, since it differs
          depending on the model.
        segmentations_folder (str): The name of the folder containing the segmentations.
        models (list | None): The list of models to compute results.

    Returns:
        list: a list of DataFrame, containing results for each model.

    """
    # ------------------------------------------------------------------
    # Retrieve subject information from DataFrame
    # ------------------------------------------------------------------
    subject_path = Path(getattr(row, SUBJECT_DIR))
    subject_id = str(getattr(row, ID))
    segmentation_folder = subject_path / segmentations_folder
    rows = []

    # Segmentation folder missing
    if not segmentation_folder.exists():
        logger.warning("Missing segmentation folder for subject %s", subject_id)
        return rows

    # Fetch GT path
    manual_path = next(
        (subject_path / "manual_seg" / "in_MPM").glob(GT_FILENAME_PATTERN),
        None,
    )
    if manual_path is None:
        logger.warning("Missing Manual segmentation for subject %s", subject_id)
        return rows

    gt_orig = nib.load(manual_path).get_fdata()

    # ------------------------------------------------------------------
    # Loop over models to retrieve results
    # ------------------------------------------------------------------
    if models is None:
        models = pred_patterns.keys()
    for model_name in models:
        model_path = Path(segmentation_folder / model_name)
        if not model_path.exists():
            logger.warning(
                "Segmentation missing for model %s (subject %s)", model_name, subject_id
            )
            continue

        # ---- Search for the prediction ----
        pred_path = next(model_path.glob(pred_patterns.get(model_name, "")), None)
        if pred_path is None:
            logger.warning(
                "Prediction missing for model %s (subject %s)", model_name, subject_id
            )
            continue

        # ---- Shiva requires a cropped version of the manual segmentation
        if model_name == "shiva":
            gt_shiva = next(model_path.glob("manual_cropped*.nii*"), None)
            if gt_shiva is None:
                logger.warning("Missing cropped GT for shiva (subject %s)", subject_id)
                continue
            gt = nib.load(gt_shiva).get_fdata()
        else:
            gt = gt_orig

        # ---- Load prediction ----
        prediction = nib.load(pred_path).get_fdata()

        if prediction.shape != gt.shape:
            logger.warning(
                "Subject %s Mismatch in shapes between pred. (%s) and gt (%s)",
                str(subject_id),
                str(prediction.shape),
                str(gt.shape),
            )
            continue

        # ------------------------------------------------------------------
        # Compute results
        # ------------------------------------------------------------------
        df = get_all_metrics(prediction, gt)
        df[ID] = subject_id
        df[FAZEKAS] = getattr(row, FAZEKAS)
        df[MODEL] = model_name

        rows.append(df)

    return rows


def bootstrap_ci(
    values: np.array,
    n_bootstrap: int = 10_000,
    ci: int = 95,
    agg_fn: Callable = np.mean,
    random_state: int = 42,
) -> tuple:
    """Perform bootstrap on an array of values.

    Args:
        values (np.array): Array containing the values to bootstrap.
        n_bootstrap (int, optional): Number of bootstrap to perform. Defaults to 10_000.
        ci (int, optional): The percentile. Defaults to 95.
        agg_fn (Callable, optional): The aggregate function to use. Defaults to np.mean.
        random_state (int | None, optional): The random state of the generator.
          Defaults to 42.

    Returns:
        tuple: tuple containing the upper and lower bound of the bootstrap.

    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)

    boot = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    stats = agg_fn(boot, axis=1)

    alpha = 100 - ci
    lower = np.percentile(stats, alpha / 2)
    upper = np.percentile(stats, 100 - alpha / 2)

    return lower, upper


def clean_summary_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Format the summary table to be Latex ready.

    Args:
        df_metrics (pd.DataFrame): The DataFrame containing the metrics.

    Returns:
        pd.DataFrame: The formated DataFrame.

    """
    df_metrics["Precision OP"] = (
        df_metrics["median_precision_op"].round(3).astype(str)
        + " ["
        + df_metrics["ci_low_op_precision_median"].round(3).astype(str)
        + ", "
        + df_metrics["ci_high_op_precision_median"].round(3).astype(str)
        + "]"
    )

    df_metrics["AP"] = (
        df_metrics["ap_median"].round(3).astype(str)
        + " ["
        + df_metrics["ci_median_low"].round(3).astype(str)
        + ", "
        + df_metrics["ci_median_high"].round(3).astype(str)
        + "]"
    )
    df_metrics["Model"] = df_metrics["Model"].apply(lambda x: x.split("_")[1])

    return df_metrics[["Model", "AP", "Precision OP"]]


def format_table_results(
    df_results: pd.DataFrame,
    cols_to_keep: list,
    cols_rename: dict,
    col_groupby: list,
    model_renames: dict | None = None,
) -> pd.DataFrame:
    """Format the result table for report.

    It formats the DataFrame containing the results of the different trials.

    Args:
        df_results (pd.DataFrame): The original DataFrame.
        cols_to_keep (list): List of columns to keep in the final table.
        col_groupby: (list): Columns to specify where to perform the groupby.
        cols_rename (dict): dict for renaming columns.
        model_renames (dict | None): Dict to rename the model folders to a proper name.

    Returns:
        pd.DataFrame: The formated DataFrame

    See Also:
        - :func:`model_comparison.process_run_results`: Function that creates the table.

    """
    df_results = deepcopy(df_results)
    drop_columns = list({FAZEKAS, DATASET, ID} - set(col_groupby))
    if model_renames is not None:
        df_results["model"] = df_results["model"].apply(lambda x: model_renames[x])

    def mean_ci_fmt(x: list) -> str:
        m = x.mean()
        lo, hi = bootstrap_ci(x)
        return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"

    df_formatted = (
        df_results.drop(columns=drop_columns)
        .groupby(by=col_groupby)
        .apply(lambda g: g.agg(mean_ci_fmt))
    )

    df_formatted = df_formatted[cols_to_keep]

    return df_formatted.rename(columns=cols_rename)
