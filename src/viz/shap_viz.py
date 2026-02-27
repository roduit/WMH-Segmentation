import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pypalettes import load_cmap

from misc.constants import ID, RESULTS_DIR, VOXEL_CLASS
from viz.viz_utils import set_style

set_style()

logger = logging.getLogger(__name__)


def plot_beeswarm(
    df: pd.DataFrame, explainer: shap.TreeExplainer, palette: str = "Kippenberger"
) -> None:
    """Plot the Beeswarm plot.

    Args:
        df (pd.DataFrame): Voxel DataFrame.
        explainer (shap.TreeExplainer): SHAP Tree Explainer.
        palette (str): The palette to use.

    """
    fig_path = RESULTS_DIR / "shap" / "beeswarm.png"
    logger.info("Plotting Beeswarm Plot")
    df_x_correct = df[df[VOXEL_CLASS].isin(["TP", "TN"])]

    logger.info(
        "Calculating explanation for correct clasess (#voxels: %s).", len(df_x_correct)
    )
    explanation_correct = explainer(df_x_correct.drop([VOXEL_CLASS, ID], axis=1))
    df_x_correct = df_x_correct.sort_values(by=VOXEL_CLASS)

    shap.plots.beeswarm(
        explanation_correct,
        max_display=10,
        color=load_cmap(palette, cmap_type="continuous"),
        show=False,
    )
    plt.savefig(fig_path, bbox_inches="tight")

    logger.info("Beeswarm plot saved successfully at %s", fig_path)


def plot_interaction(
    explanation: shap.TreeExplainer,
    df: pd.DataFrame,
    feat_idx: tuple,
    feat_names: tuple,
    color_by: str = "shap",
) -> None:
    """Plot pairwise feature interactions colored by SHAP value or class.

    Args:
        explanation (shap.TreeExplainer): SHAP TreeExplainer.
        df (pd.DataFrame): DataFrame containing voxel informations.
        feat_idx (tuple): Indices of the features to plot.
        feat_names (tuple): Names of the features to plot.
        color_by (str, optional): Element to color. Defaults to "shap".

    """
    f1, f2, f3 = feat_idx
    f1_name, f2_name, f3_name = feat_names

    # -------------------------
    # Feature values
    # -------------------------
    x1 = explanation[:, f1].data
    x2 = explanation[:, f2].data
    x3 = explanation[:, f3].data

    # -------------------------
    # SHAP interaction score
    # -------------------------
    shap_total = np.sum(explanation.values, axis=1)
    # -------------------------
    # Coloring
    # -------------------------
    if color_by == "shap":
        base = load_cmap("Kippenberger", cmap_type="continuous", reverse=True)

        cmap = LinearSegmentedColormap.from_list(
            "kippenberger_cont",
            base.colors,
            N=256,  # interpolation
        )
        norm = Normalize(vmin=shap_total.min(), vmax=shap_total.max())
    else:
        cmap = load_cmap("CarolMan", cmap_type="continuous", shuffle=True).colors[
            : df[VOXEL_CLASS].nunique()
        ]
        norm = None

    # -------------------------
    # Plotting helper
    # -------------------------
    def scatter(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: np.ndarray,
        ylabel: np.ndarray,
        shap_sum: np.ndarray,
    ) -> None:
        hue = shap_sum if color_by == "shap" else df[VOXEL_CLASS]
        sns.scatterplot(
            x=x,
            y=y,
            hue=hue,
            palette=cmap,
            s=5,
            alpha=0.5,
            ax=ax,
            legend=color_by != "shap",
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    # -------------------------
    # Create figure
    # -------------------------
    _fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    scatter(
        axes[0],
        x1,
        x2,
        f1_name,
        f2_name,
        shap_sum=explanation[:, f1].values + explanation[:, f2].values,  # noqa: PD011
    )
    scatter(
        axes[1],
        x1,
        x3,
        f1_name,
        f3_name,
        shap_sum=explanation[:, f1].values + explanation[:, f3].values,  # noqa: PD011
    )
    scatter(
        axes[2],
        x2,
        x3,
        f2_name,
        f3_name,
        shap_sum=explanation[:, f2].values + explanation[:, f2].values,  # noqa: PD011
    )

    # -------------------------
    # Legends / Colorbar
    # -------------------------
    if color_by == "shap":
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=axes[2], pad=0.03)
        cbar.set_label("SHAP value", fontsize=12)
    else:
        for ax in axes:
            ax.legend(title=color_by, markerscale=2)
    plt.tight_layout()
