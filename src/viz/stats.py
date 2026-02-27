# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-12-02 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: FUnctions to plot stats-*-

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pypalettes import load_cmap

from misc.constants import RESULTS_DIR
from viz.viz_utils import MAPS_NAME_FOR_PLOTS, darken, set_style

set_style()


def plot_lesion_stats(df_lesions: pd.DataFrame, df_total_vol: pd.DataFrame) -> None:
    """Plot histogram and boxplot of lesion volumes.

    Args:
        df_lesions (pd.DataFrame): Tidy DataFrame with lesion volumes and dataset names.
        df_total_vol (pd.DataFrame): Tidy DataFrame with total lesion volume per subject
          and dataset names.

    """
    n_datasets = df_lesions["Dataset"].nunique()
    if n_datasets > len(sns.color_palette("Set3")):
        msg = f"""
        Too many datasets for distinct colors (max {len(sns.color_palette("Set3"))}).
        """
        raise ValueError(msg)
    # Get Set3 colors
    base_colors = sns.color_palette("Set3")[:n_datasets]
    shifted_colors = [darken(c) for c in base_colors]

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(
        x="Dataset", y="Lesions", data=df_lesions, hue="Dataset", palette=base_colors
    )
    plt.title("Lesion Distribution per Dataset")
    plt.ylabel("Lesions Volume (ml)")
    plt.xlabel("Dataset")
    plt.semilogy()

    plt.subplot(1, 2, 2)
    sns.boxplot(
        x="Dataset",
        y="total_lesion_volume",
        data=df_total_vol,
        hue="Dataset",
        palette=base_colors,
    )
    sns.stripplot(
        x="Dataset",
        y="total_lesion_volume",
        data=df_total_vol,
        hue="Dataset",
        palette=shifted_colors,
        alpha=1,
        jitter=0.2,
        size=7,
        edgecolor="black",
        dodge=True,
    )
    plt.title("Total Lesion Volume per Subject")
    plt.ylabel("Total Volume per Subject (ml)")
    plt.xlabel("Dataset")
    plt.semilogy()

    plt.tight_layout()
    plt.show()


def plot_box_dice(
    df_metrics: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str | None = None,
    palette_name: str = "Kippenberger",
    strip: bool = True,
) -> plt.Figure:
    """Plot boxplot with optional stripplot overlay.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing metrics for each test subject.
        x (str): The column name for x-axis.
        y (str): The column name for y-axis.
        hue (str): The column name for hue grouping.
        title (str): Title of the plot.
        palette_name (str, optional): The palette to use. Defaults to "Blues".
        strip (bool, optional): Bool to wether plot strip or not. Defaults to True.

    Returns:
        plt.Figure: The figure

    """
    n_colors = df_metrics[hue].nunique()
    palette = load_cmap(name=palette_name, cmap_type="discrete").colors[:n_colors]
    palette_darker = [darken(c) for c in palette]

    plt.figure(figsize=(10, 6))
    df_metrics[hue] = df_metrics[hue].astype("category")
    # --- Draw boxplot ---
    box_ax = sns.boxplot(
        data=df_metrics,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
    )

    # Save legend handles/labels from the boxplot
    handles, labels = box_ax.get_legend_handles_labels()

    # --- Overlay stripplot ---
    if strip:
        sns.stripplot(
            data=df_metrics,
            x=x,
            y=y,
            hue=hue,
            palette=palette_darker,
            dodge=True,
            jitter=True,
            alpha=0.9,
            legend=False,
        )

    # --- Clean up ---
    plt.ylabel(y.upper(), fontsize=14)
    plt.xlabel(x.capitalize(), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(axis="y")

    # Re-add legend from boxplot only
    plt.legend(handles, labels, title=hue)

    if title is not None:
        plt.title(title)

    return plt.gcf()


def plot_box_model_comparison(
    df_metrics: pd.DataFrame, palette_name: str = "Blues"
) -> None:
    """Plot boxplot comparing Dice scores of different models.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing metrics for each test subject.
        palette_name (str, optional): Name of seaborn color palette.
          Defaults to 'Blues'.

    """
    n_models = df_metrics["model"].nunique()
    palette = sns.color_palette(palette_name, n_colors=n_models)

    if n_models > len(palette):
        msg = f"""
        Too many models for distinct colors (max {len(sns.color_palette(palette))}).
        """
        raise ValueError(msg)

    darkened_palette = [darken(c) for c in palette]

    plt.figure(figsize=(15, 6))
    sns.boxplot(
        data=df_metrics, x="dataset", y="global_dice", hue="model", palette=palette
    )
    sns.stripplot(
        data=df_metrics,
        x="dataset",
        y="global_dice",
        hue="model",
        palette=darkened_palette,
        dodge=True,
        jitter=True,
        alpha=0.7,
    )
    plt.title("Distribution of Global Dice Scores on Test Set")
    plt.ylabel("Global Dice Score")
    plt.xlabel("Dataset")
    plt.grid(axis="y")

    plt.savefig(RESULTS_DIR / "model_comparison_dice_scores.png")

    plt.show()


def plot_lesions_volumes(df: pd.DataFrame, algo: str, logscale: str = "none") -> None:
    """Plot boxplot of lesion volumes for each dataset.

    Args:
        df (pd.DataFrame): DataFrame containing metrics for each test subject.
        algo (str): Name of the algorithm (for title).
        logscale (str, optional): Scale for x-axis ('log', 'semilog', 'none').
          Defaults to 'none'.

    """
    volumes = df["total_volume"].astype(float).to_numpy()
    dices = df["global_dice"].astype(float).to_numpy()

    # Plot dice distribution in term of lesion volumes
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=volumes, y=dices, alpha=0.5)
    plt.xlabel("Total Volume (mm³)")
    plt.ylabel("Global Dice Score")
    plt.title(f"Global Dice Scores vs. Total Volume ({algo})")
    if logscale == "x":
        plt.semilogx()
    elif logscale == "xy":
        plt.loglog()
    elif logscale == "y":
        plt.semilogy()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()


def plot_proc_effect(df: pd.DataFrame, palette: str, metric_name: str) -> plt.Figure:
    """Plot the effect of post processing function.

    Args:
        df (pd.DataFrame): The dataframe containing both original and post process
          results
        palette (str): The palette color
        metric_name (str): The name of the metric

    Returns:
        plt.Figure: the figure

    """
    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x=f"{metric_name}_pre",
        y=f"{metric_name}_post",
        hue="Fazekas",
        palette=palette,
        alpha=0.7,
    )

    # Diagonal line
    mn = min(df[f"{metric_name}_pre"].min(), df[f"{metric_name}_post"].min())
    mx = max(df[f"{metric_name}_pre"].max(), df[f"{metric_name}_post"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="black")

    plt.xlabel("Before")
    plt.ylabel("After")
    plt.title(f"{metric_name}: Before vs After (aligned, colored by Fazekas)")
    plt.legend(title="Fazekas")
    plt.tight_layout()

    return plt.gcf()


def plot_qmaps_detection(
    df_lesions_stats: pd.DataFrame,
    feature_cols: list,
    detected_col: str,
    file_name: str,
    palette: str,
) -> None:
    """Plot the qmaps difference between detected and missed lesions.

    Args:
        df_lesions_stats (pd.DataFrame): The DataFrame containing lesions informations.
        feature_cols (list): The list of feature columns
        detected_col (str): The boolean column detected / missed
        file_name (str): The name of the save to save.
        palette (str): The name of the palette to use.

    """
    df_norm = df_lesions_stats.copy()

    # Z-score per feature
    df_norm[feature_cols] = df_norm[feature_cols].apply(
        lambda x: (x - x.mean()) / x.std()
    )

    # Melt to long format for seaborn
    df_long = df_norm.melt(
        id_vars=["id", detected_col],
        value_vars=feature_cols,
        var_name="feature",
        value_name="value",
    )
    df_long["feature"] = df_long["feature"].apply(lambda x: MAPS_NAME_FOR_PLOTS[str(x)])
    df_long[detected_col] = df_long[detected_col].apply(
        lambda x: "Detected" if int(x) == 1 else "Missed"
    )
    df_long = df_long.rename(columns={"feature": "Feature", "value": "Z-scored value"})

    # Load palette and pick first and last (or symmetric) colors
    cmap = load_cmap(palette, cmap_type="discrete")
    selected_colors = list(np.array(cmap.colors)[[3, -3]])

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    sns.boxplot(
        data=df_long,
        x="Feature",
        y="Z-scored value",
        hue=detected_col,
        palette=selected_colors,
        ax=ax,
        vert=True,
    )
    ax.tick_params(axis="both", which="major", labelsize=14)
    plt.xlabel("Feature", fontsize=16)
    plt.ylabel("Z-scored value", fontsize=16)
    ax.legend(fontsize=14)

    # Save figure
    save_dir = RESULTS_DIR / "seg_diff"
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / f"{file_name}.png")


def plot_embedding(
    df_lesions: pd.DataFrame,
    embeddings: np.ndarray,
    region_col: str,
    palette: str,
    reverse_palette: bool = True,
) -> None:
    """Plot the Umap embedding of the lesions.

    Args:
        df_lesions (pd.DataFrame): The DataFrame containing the lesions.
        embeddings (np.ndarray): Embeddings array.
        region_col (str): Column containing the region informations.
        detected_col (str): Column containing the detection information.
        palette (str): Name of the palette to choose.
        reverse_palette (bool): Boolean to reverse the order of colors.

    """
    unique_regions = df_lesions[region_col].unique()
    detection_classes = df_lesions["detection_class"].unique()

    df_lesions["lesion_size_log"] = np.log1p(df_lesions["lesion_size"])

    vmin, vmax = np.percentile(df_lesions["lesion_size_log"], [5, 95])

    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = load_cmap(palette, reverse=reverse_palette, cmap_type="continuous")

    _fig, axes = plt.subplots(
        len(detection_classes),
        len(unique_regions),
        figsize=(5 * len(unique_regions), 2 * len(detection_classes)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    x_min, y_min = embeddings.min(axis=0)
    x_max, y_max = embeddings.max(axis=0)
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)

    for det_idx, detection_class in enumerate(detection_classes):
        for region_idx, region in enumerate(unique_regions):
            df_reg = df_lesions[df_lesions[region_col] == region]
            idx = df_reg["detection_class"] == detection_class
            axes[det_idx, region_idx].scatter(
                embeddings[df_reg[idx].index, 0],
                embeddings[df_reg[idx].index, 1],
                c=df_reg.loc[df_reg[idx].index, "lesion_size_log"],
                cmap=cmap,
                norm=norm,
                s=30,
            )
            axes[det_idx, 0].set_ylabel("PCA-2", fontsize=14)
            axes[0, region_idx].set_title(f"Region: {region}", fontsize=16)
            axes[det_idx, region_idx].set_xlim(x_min - x_margin, x_max + x_margin)
            axes[det_idx, region_idx].set_ylim(y_min - y_margin, y_max + x_margin)
            axes[det_idx, region_idx].tick_params(
                axis="both", which="major", labelsize=12
            )
            if det_idx == len(detection_classes) - 1:
                axes[det_idx, region_idx].set_xlabel("PCA-1", fontsize=14)
        axes[det_idx, region_idx].yaxis.set_label_position("right")
        axes[det_idx, region_idx].set_ylabel(
            detection_class, va="center", labelpad=15, fontsize=14
        )
    cbar = plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[:, region_idx],  #  both rows
        pad=0.03,  # distance from axes
    )
    cbar.set_label("log(lesion size + 1)", labelpad=15, fontsize=14)

    save_dir = RESULTS_DIR / "seg_diff"
    save_dir.mkdir(exist_ok=True)

    plt.savefig(save_dir / "lesions_embedding.png")


def plot_pca_loadings(
    loadings: pd.DataFrame,
    pcs: tuple = ("PC1", "PC2"),
    palette: str = "Kippenberger",
) -> None:
    """PCA Loadings.

    Args:
        loadings (pd.DataFrame): _description_
        pcs (tuple, optional): _description_. Defaults to ("PC1", "PC2").
        palette (str): desc
        reverse_palette (bool): True to reverse the palette provided.

    """
    _fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.1},
        constrained_layout=True,
    )

    cmap = load_cmap(palette, cmap_type="continuous", reverse=True)

    # Normalize loadings
    norm = mcolors.Normalize(vmin=loadings.min(axis=None), vmax=loadings.max(axis=None))

    for ax, pc in zip(axes, pcs, strict=True):
        df = loadings[[pc]].reset_index()
        df.columns = ["Feature", "Loading"]
        df["Feature"] = df["Feature"].apply(lambda x: MAPS_NAME_FOR_PLOTS[str(x)])
        df["abs"] = df["Loading"].abs()
        df = df.sort_values("abs", ascending=False)
        df["sign"] = df["Loading"] > 0
        colors = [cmap(norm(v)) for v in df["Loading"]]

        sns.barplot(
            data=df,
            y="Loading",
            x="Feature",
            hue="Feature",
            palette=colors,
            orient="v",
            legend=False,
            ax=ax,
        )
        ax.set_title(pc, fontsize=16)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlabel("")
        ax.set_ylabel("PCA Loading", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), pad=0.03, ax=axes)
    cbar.set_label("PCA Loading Scale", labelpad=15, fontsize=14)

    save_dir = RESULTS_DIR / "seg_diff"
    save_dir.mkdir(exist_ok=True)

    plt.savefig(save_dir / "pca_loadings.png")
