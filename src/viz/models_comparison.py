# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-03 -*-
# -*- Last revision: 2025-12-03 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Plot comparison of models -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypalettes import load_cmap

from misc.constants import FAZEKAS
from viz.stats import plot_box_dice
from viz.viz_utils import set_style

set_style()


def plot_model_comparison(
    model_names: list[str],
    ap_means: list[float],
    ci_uppers: list[float],
    ci_lowers: list[float],
    title: str,
) -> plt.Figure:
    """Plot the model comparison based on bootstrap CI intervals.

    Args:
        model_names (list[str]): The model names
        ap_means (list[float]): the mean Average Precision
        ci_uppers (list[float]): The lower CI intervals
        ci_lowers (list[float]): The upper CI intervals
        title (str): The title of the plot

    Returns:
        plt.Figure: The figure

    """

    def get_map_type(name: str) -> str:
        """Get the name of the map from model name.

        Args:
            name (str): The name of the model.

        Returns:
            str: the map name

        """
        prefix = name.split("/", maxsplit=1)[0]
        if prefix.startswith("MPM"):
            return "MPM"
        if prefix.startswith("DIFF"):
            return "DIFF"
        return "ALL"

    map_types = [get_map_type(name) for name in model_names]

    palette = load_cmap(name="Kippenberger").colors
    # Color mapping
    color_map = {
        "MPM": palette[3],
        "DIFF": palette[5],
        "ALL": palette[7],
    }
    colors = np.array([color_map[t] for t in map_types])

    names = [name.split("_")[1] for name in model_names]

    # -----------------------------
    # Convert to numpy + sort by AP
    # -----------------------------
    ap = np.array(ap_means)
    low = np.array(ci_lowers)
    upp = np.array(ci_uppers)
    names = np.array(names)

    order = np.argsort(ap)[::-1]
    ap = ap[order]
    low = low[order]
    upp = upp[order]
    names = names[order]
    colors = colors[order]

    # Compute asymmetric errors
    error_low = ap - low
    error_high = upp - ap

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    y_pos = np.arange(len(ap))

    # Keep track of which modalities we already added to the legend
    plotted = set()

    for i in range(len(ap)):
        modality = map_types[order[i]]  # the modality corresponding to this point
        if modality not in plotted:
            ax.errorbar(
                ap[i],
                y_pos[i],
                xerr=[[error_low[i]], [error_high[i]]],
                fmt="o",
                color=colors[i],  # color for both point + error bar
                capsize=5,
                markersize=8,
                linewidth=2,
                label=modality,  # label only once per modality
            )
            plotted.add(modality)
        else:
            ax.errorbar(
                ap[i],
                y_pos[i],
                xerr=[[error_low[i]], [error_high[i]]],
                fmt="o",
                color=colors[i],
                capsize=5,
                markersize=8,
                linewidth=2,
            )

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Average Precision (AP)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Modality")  # legend shows only one entry per modality
    if title is not None:
        plt.title(title)

    return fig


def compare_models_metrics(
    df_metrics: pd.DataFrame,
    save_dir: str,
    hue: str = "model",
    palette_name: str = "Kippenberger",
) -> dict:
    """Compare the models based on columns of. the DataFrame.

    Args:
        df_metrics (pd.DataFrame): The DataFrame containing the metrics.
        save_dir(str): Name of the folder directory to save.
        hue (str): The column to use for hue arguement
        palette_name (str): Name of the palette to use.

    Returns:
        dict: a dict containing figure for each metric

    """
    skip_cols = [FAZEKAS, "dataset"]
    df_metrics[FAZEKAS] = pd.to_numeric(df_metrics[FAZEKAS])
    for metric in df_metrics.select_dtypes(include=["number"]).columns:
        if metric in skip_cols:
            continue
        df_metrics[metric] = pd.to_numeric(df_metrics[metric])
        fig = plot_box_dice(
            df_metrics,
            FAZEKAS,
            metric,
            hue,
            title=None,
            palette_name=palette_name,
        )

        fig.savefig(save_dir / f"{metric.replace(' ', '_')}.png")
        plt.close()
