# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-09 -*-
# -*- Last revision: 2026-02-08 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Functions to analyze the lesions differences -*-

import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from viz.viz_utils import MAPS_NAME_FOR_PLOTS

logger = logging.getLogger(__name__)


def cliffs_delta(a: list, b: list) -> float:
    """Calculate the cliff's delta.

    Args:
        a (float): First group
        b (float): Second group

    Returns:
        float: The cliff's delta

    """
    a_array = np.array(a)
    b_array = np.array(b)
    n1 = len(a)
    n2 = len(b)
    greater = sum(x > y for x in a_array for y in b_array)
    smaller = sum(x < y for x in a_array for y in b_array)

    return (greater - smaller) / (n1 * n2)


def compare_qmaps_distributions(
    df_lesions_stats: pd.DataFrame,
    feature_names: pd.DataFrame,
    target_col: str,
    sig_level: float = 0.05,
) -> pd.DataFrame:
    """Compare statistically the distributions of detected and missed lesions.

    Args:
        df_lesions_stats (pd.DataFrame): The DataFrame containing the lesion
          informations.
        feature_names (pd.DataFrame): The names of the features to compute the stats.
          This corresponds to the MRI images of interest.
        target_col (str): The target column to discriminate the two classes.
          Typically, this column contains the missed vs detected information.
        sig_level (float): The significance level. Default to 0.05

    Returns:
        pd.DataFrame: the statistical tests of the features.

    """
    rows = []

    for feat in feature_names:
        # ---- Change the table to a wide format for group analysis ----
        df_wide = df_lesions_stats.pivot_table(
            index="id", columns=target_col, values=feat
        )

        df_wide = df_wide.rename(columns={0: "missed", 1: target_col})
        a = df_wide[target_col].to_numpy()
        b = df_wide["missed"].to_numpy()

        # ---- Wilcoxon test ----
        _wilcoxon_stat, wilcoxon_p_value = wilcoxon(a, b)

        # ---- Effect sizes ----
        cliffs = cliffs_delta(a, b)

        rows.append(
            {
                "Feature": str(feat),
                "Wilcoxon": wilcoxon_p_value,
                r"Cliff's delta ($\delta$)": cliffs,
            }
        )

    df_stats = pd.DataFrame(rows)

    # ---- Perform FDR correction ----
    df_stats["Wilcoxon (FDR)"] = multipletests(df_stats["Wilcoxon"], method="fdr_bh")[1]
    df_stats["significant"] = df_stats["Wilcoxon (FDR)"] < sig_level

    df_stats["Feature"] = df_stats["Feature"].apply(
        lambda x: MAPS_NAME_FOR_PLOTS[str(x)]
    )

    return df_stats


def compute_pca(
    df_lesions_stats: pd.DataFrame,
    feature_cols: list,
    region_col: str,
    n_components: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute the PCA embedding of the lesions.

    Args:
        df_lesions_stats (pd.DataFrame): Lesions DataFrame
        feature_cols (list): The feature columns
        region_col (str): The column containing region information
        n_components (int): Number of PCA components. Default to 2.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: a tuple containing:
          - the cleaned DataFrame
          - the PCA embedding array

    """
    df_clean = df_lesions_stats.dropna(subset=[*feature_cols, region_col]).reset_index(
        drop=True
    )

    x = df_clean[feature_cols].to_numpy()
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_components, random_state=0)
    embeddings = pca.fit_transform(x_scaled)

    logger.info("Explained variances: %s", pca.explained_variance_ratio_)

    return df_clean, embeddings, pca
