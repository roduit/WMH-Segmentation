# -*- authors : Vincent Roduit -*-
# -*- date : 2026-01-12 -*-
# -*- Last revision: 2025-01-12 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to calculate the uncertainty correlation-*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

from analysis.uncertainty_quantification.constants import (
    ENTROPY_MEASURES,
    MEASURES_RENAMES,
)
from misc.constants import FAZEKAS


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, random_state: int = 0
) -> dict:
    """Bootsrap the spearman correlation coefficient.

    Args:
        x (np.ndarray): First array.
        y (np.ndarray): Second array.
        n_boot (int, optional): Number of bootstraps. Defaults to 1000.
        random_state (int, optional): The random state. Defaults to 0.

    Returns:
        dict: Dict containing the median the 95 percentile CI and all rho values.

    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)

    rhos = np.empty(n_boot)
    p_values = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        rhos[i] = spearmanr(x[idx], y[idx]).statistic
        p_values[i] = spearmanr(x[idx], y[idx]).pvalue

    ci = np.percentile(rhos, [2.5, 50, 97.5])
    p_values_ci = np.percentile(p_values, [2.5, 50, 97.5])
    return {
        "rho_median": ci[1],
        "ci_95": (ci[0], ci[2]),
        "p_ci_95": (p_values_ci[0], p_values_ci[2]),
        "all_rhos": rhos,
        "p_value_median": np.median(p_values),
        "all_p_values": p_values,
    }


def calculate_correlation(
    df_patient_metrics: pd.DataFrame, segmentation_metric: str
) -> pd.DataFrame:
    """Calculate the correlation for each entropy given a segmentation metric.

    Args:
        df_patient_metrics (pd.DataFrame): DataFrame containing patient metrics.
        segmentation_metric (str): Segmentation metric to use for the correlation.

    Returns:
        pd.DataFrame: DataFrame contaning the correlation values for each entropy
          metric.

    """
    res_list = []
    for entropy_metric in ENTROPY_MEASURES["participant"]:
        for split in df_patient_metrics["split"].unique():
            df_split = df_patient_metrics[df_patient_metrics["split"] == split]
            res = bootstrap_spearman(
                df_split[entropy_metric], df_split[segmentation_metric], n_boot=10000
            )
            res["entropy_measure_latex"] = MEASURES_RENAMES[entropy_metric]
            res["entropy_measure"] = entropy_metric
            res["split"] = split
            res_list.append(res)

    df_corr = pd.DataFrame(res_list).sort_values(by="rho_median", ascending=True)

    df_corr["value"] = df_corr.apply(
        lambda x: f"{x['rho_median']:.2f} [{x['ci_95'][0]:.2f}, {x['ci_95'][1]:.2f}]",
        axis=1,
    )
    df_corr["p_value"] = df_corr.apply(
        lambda x: (
            f"{x['p_value_median']:.2e} [{x['p_ci_95'][0]:.2e}, {x['p_ci_95'][1]:.2e}]"
        ),
        axis=1,
    )
    df_corr["entropy_measure_latex"] = df_corr["entropy_measure_latex"].apply(
        lambda x: f"${x}$"
    )

    return df_corr


def correct_correlation(
    df_patient_metrics: pd.DataFrame, segmentation_metric: str, entropy_metric: str
) -> pd.DataFrame:
    """Correct the correlation between the DSC and the entropy metric.

    This is to remove the Fazekas component.

    Args:
        df_patient_metrics (pd.DataFrame): DataFrame containing the patient metrics.
        segmentation_metric (str): The segmentation metric to use.
        entropy_metric (str): The entropy metric to use for the correlation.

    Returns:
        pd.DataFrame: DataFrame containing the partial correlation coefficients.

    """
    partial_corrs = []
    splits = df_patient_metrics["split"].unique()

    for sp in splits:
        df_split = df_patient_metrics[df_patient_metrics["split"] == sp]

        # Regress avg. mean entropy_of_expected on Fazekas
        x = sm.add_constant(df_split[FAZEKAS])
        res_entropy = sm.OLS(df_split[entropy_metric], x).fit().resid

        # Regress Dice on Fazekas
        res_dice = sm.OLS(df_split[segmentation_metric], x).fit().resid

        # Compute correlation between residuals
        r, p = pearsonr(res_entropy, res_dice)
        partial_corrs.append({"split": sp, "residual": r, "p_value": p})

    return pd.DataFrame(partial_corrs)
