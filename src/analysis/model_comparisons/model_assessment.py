# -*- authors : Vincent Roduit -*-
# -*- date : 2025-02-08 -*-
# -*- Last revision: 2025-08- by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Functions to assess model performances between different trials -*-

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import average_precision_score, precision_recall_curve

from misc.constants import (
    DATASET,
    FAZEKAS,
    ID,
)
from mlflow_func.mlflow_logger import log_dataframe_as_csv
from viz.models_comparison import plot_model_comparison

logger = logging.getLogger(__name__)


def compute_model_scores(
    y_true: dict,
    y_pred: dict,
    recall_op: float = 0.7,
    precision_op: float = 0.7,
    n_recall_points: int = 200,
    alpha: float = 0.05,
) -> dict:
    """Compute model scores with confidence intervals.

    Args:
        y_true (dict): Keys = split id, values = ground truth labels
        y_pred (dict): Keys = split id, values = predicted scores
        recall_op (float): Operating point on recall curve
        precision_op (float): Operating point on precision curve
        n_recall_points (int): Number of points for recall interpolation
        alpha (float): Significance level (default 0.05 → 95% CI)

    Returns:
        dict: Aggregated statistics

    """
    # --- Containers ---
    ap_scores = []
    all_pr_interp = []
    all_precisions = []
    all_recalls = []
    recalls_at_op = []

    recall_grid = np.linspace(0.0, 1.0, n_recall_points)

    # ---- Per-split metrics ----
    for yt, yp in zip(y_true.values(), y_pred.values(), strict=True):
        ap_scores.append(average_precision_score(yt, yp, pos_label=1))

        precision, recall, _ = precision_recall_curve(yt, yp)
        pr_interp = np.interp(recall_grid, recall[::-1], precision[::-1])
        all_pr_interp.append(pr_interp)
        all_precisions.append(precision)
        all_recalls.append(recall)

    for p, r in zip(all_precisions, all_recalls, strict=True):
        idx = np.argmin(np.abs(p - precision_op))
        recalls_at_op.append(r[idx])

    recalls_at_op = np.asarray(recalls_at_op)
    n = len(recalls_at_op)
    ap_scores = np.asarray(ap_scores)
    all_pr_interp = np.asarray(all_pr_interp)

    n = len(ap_scores)
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)

    # ------------------------------------------------------------------
    # Average Precision statistics
    # ------------------------------------------------------------------
    ap_mean = ap_scores.mean()
    ap_std = ap_scores.std(ddof=1)

    ap_half_width = t_crit * ap_std / np.sqrt(n)
    ap_ci = (ap_mean - ap_half_width, ap_mean + ap_half_width)

    ap_median = np.median(ap_scores)
    ap_mad = np.median(np.abs(ap_scores - ap_median))
    ap_mad_scaled = 1.4826 * ap_mad

    ap_half_width_robust = 1.96 * ap_mad_scaled / np.sqrt(n)
    ap_ci_robust = (
        ap_median - ap_half_width_robust,
        ap_median + ap_half_width_robust,
    )

    # ------------------------------------------------------------------
    # Precision-Recall curve statistics
    # ------------------------------------------------------------------
    mean_precision = all_pr_interp.mean(axis=0)
    std_precision = all_pr_interp.std(axis=0)

    pr_half_width = t_crit * std_precision / np.sqrt(n)
    pr_ci = (
        mean_precision - pr_half_width,
        mean_precision + pr_half_width,
    )

    median_precision = np.median(all_pr_interp, axis=0)
    mad_precision = np.median(np.abs(all_pr_interp - median_precision), axis=0)
    mad_scaled = 1.4826 * mad_precision

    pr_half_width_robust = 1.96 * mad_scaled / np.sqrt(n)
    pr_ci_robust = (
        median_precision - pr_half_width_robust,
        median_precision + pr_half_width_robust,
    )

    # ------------------------------------------------------------------
    # Operating point statistics (recall = recall_op)
    # ------------------------------------------------------------------
    idx = np.argmin(np.abs(recall_grid - recall_op))
    precisions_op = all_pr_interp[:, idx]

    mean_op_recall = precisions_op.mean()
    std_op = precisions_op.std(ddof=1)

    op_half_width = t_crit * std_op / np.sqrt(n)
    op_ci_recall = (mean_op_recall - op_half_width, mean_op_recall + op_half_width)

    median_op_recall = np.median(precisions_op)
    mad_op = np.median(np.abs(precisions_op - median_op_recall))
    mad_op_scaled = 1.4826 * mad_op

    op_half_width_robust = 1.96 * mad_op_scaled / np.sqrt(n)
    op_ci_robust_recall = (
        median_op_recall - op_half_width_robust,
        median_op_recall + op_half_width_robust,
    )

    # ------------------------------------------------------------------
    return {
        # AP
        "ap_mean_stats": (ap_mean, *ap_ci),
        "ap_median_stats": (ap_median, *ap_ci_robust),
        # PR curve
        "recall_grid": recall_grid,
        "pr_mean_stats": (mean_precision, *pr_ci),
        "pr_median_stats": (median_precision, *pr_ci_robust),
        # Operating points
        "precision_at_recall": recall_op,
        "precision_op_mean_stats": (mean_op_recall, *op_ci_recall),
        "precision_op_median_stats": (median_op_recall, *op_ci_robust_recall),
    }


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the dataframe.

    Args:
        df (pd.DataFrame): The original Dataframe

    Returns:
        pd.DataFrame: The stats of the dataframe

    """
    stats = df.drop([ID, DATASET, FAZEKAS], axis=1).agg(["mean", "std"])
    # ---- Create a multi-index on columns: (metric, stat) ----
    stats = pd.concat({col: stats[col] for col in stats.columns}, axis=1)
    return stats


def plot_pr_curve(
    ax: plt.Axes,
    recall: np.ndarray,
    precision: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    label: str,
    fill: bool,
) -> plt.Axes:
    """Pplot pr curve partially.

    Args:
        ax (plt.Axes): The axes of the figure
        recall (np.ndarray): The recall array
        precision (np.ndarray): The precision array
        ci_low (np.ndarray): The low CI array
        ci_high (np.ndarray): The high CI array
        label (str): the label
        fill (bool): True to fill with CI

    Returns:
        plt.Axes: The updated axes

    """
    ax.plot(recall, precision, label=label)
    if fill:
        ax.fill_between(recall, ci_low, ci_high, alpha=0.2)


def finalize_plot(ax: plt.Axes, title: str | None = None) -> None:
    """Finalize the plot by setting title and labels.

    Args:
        ax (plt.Axes): The axes
        title (str | None): Optional title name of the plot.

    """
    ax.axvline(x=0.7, label="Recall OP", linestyle="--", color="gray")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)

    if title is not None:
        ax.set_title(title)


def finalize_pr_plots(
    ax_pr_all: plt.Axes,
    ax_pr_median_all: plt.Axes,
    fig_pr_all: plt.Figure,
    fig_pr_median_all: plt.Figure,
    fig_pr_type: dict,
    fig_pr_median_type: dict,
    map_types: list,
) -> None:
    """Finalize plots and log it to Mlflow.

    Args:
        ax_pr_all (plt.Axes): Axes of pr curves with all combinations and mean AP curves
        ax_pr_median_all (plt.Axes): Axes of pr curves with all combinations and
          median AP curves
        fig_pr_all (plt.Figure): Figure of all combinations with mean AP
        fig_pr_median_all (plt.Figure): Figure of all combinations with median AP
        fig_pr_type (dict): dict containing the figures for type of combinations
          (mean AP)
        fig_pr_median_type (dict): dict containing the figures for type of combinations
          (median AP)
        map_types (list): the list of different map configurations (All, Diff, MPM)

    """
    finalize_plot(ax_pr_all)
    finalize_plot(ax_pr_median_all)

    mlflow.log_figure(fig_pr_all, "fig/PR_all_mean.png")
    mlflow.log_figure(fig_pr_median_all, "fig/PR_all_median.png")

    for map_type in map_types:
        finalize_plot(fig_pr_type[map_type][1])
        mlflow.log_figure(fig_pr_type[map_type][0], f"fig/PR_mean_{map_type}.png")

        finalize_plot(fig_pr_median_type[map_type][1])
        mlflow.log_figure(
            fig_pr_median_type[map_type][0], f"fig/PR_median_{map_type}.png"
        )


def finalize_model_comparison_plots(metrics_store: dict, fill: bool) -> None:
    """Finalize and log plots to Mlflow.

    Args:
        metrics_store (dict): dict containing arrays of informations
        fill (bool): True to fill confidence intervals

    """
    if fill:
        title_mean = "Model comparison with mean AP and 95% CI"
        title_median = "Model comparison with median AP and 95% CI"
    else:
        title_mean = "Model comparison with mean AP"
        title_median = "Model comparison with median AP"

    fig1 = plot_model_comparison(
        metrics_store["model_names"],
        metrics_store["ap_means"],
        metrics_store["ci_means_uppers"],
        metrics_store["ci_means_lowers"],
        title=title_mean,
    )
    mlflow.log_figure(fig1, "fig/model_comparison_mean.png")

    fig2 = plot_model_comparison(
        metrics_store["model_names"],
        metrics_store["ap_medians"],
        metrics_store["ci_medians_uppers"],
        metrics_store["ci_medians_lowers"],
        title=title_median,
    )
    mlflow.log_figure(fig2, "fig/model_comparison_median.png")


def get_map_type(name: str) -> str:
    """Get the map type from the run name.

    Args:
        name (str): the run name

    Returns:
        str: the map type

    """
    prefix = name.split("/", maxsplit=1)[0]
    if prefix.startswith("MPM"):
        return "MPM"
    if prefix.startswith("DIFF"):
        return "DIFF"
    return "ALL"


def process_single_result(
    res: dict,
    fill: bool,
    ax_pr_all: plt.Axes,
    ax_pr_type: dict,
    ax_pr_median_all: plt.Axes,
    ax_pr_median_type: dict,
    metrics_store: dict,
) -> None:
    """Process a single result.

    Args:
        res (dict): The result dict
        fill (bool): True to fill CI in figures
        ax_pr_all (plt.Axes): The axes of the PR curve
        ax_pr_type (dict): The axes of the PR curve for different map combinations
        ax_pr_median_all (plt.Axes): The axes of the PR curve and median AP
        ax_pr_median_type (dict): The axes of the PR curve for different map
          combinations for median AP
        metrics_store (dict): the dict storing the metrics

    """
    name = res["name"]
    mtype = get_map_type(name)
    logger.info("Model Name %s", name)

    metrics_store["model_names"].append(name)

    recall = res["recall_grid"]

    # ---- PR MEAN CURVE ----
    mean_precision, low_ci_pr, high_ci_pr = res["pr_mean_stats"]
    plot_pr_curve(ax_pr_all, recall, mean_precision, low_ci_pr, high_ci_pr, name, fill)
    plot_pr_curve(
        ax_pr_type[mtype], recall, mean_precision, low_ci_pr, high_ci_pr, name, fill
    )

    # ---- PR MEDIAN CURVE ----
    median_precision, low_ci_med, high_ci_med = res["pr_median_stats"]
    plot_pr_curve(
        ax_pr_median_all, recall, median_precision, low_ci_med, high_ci_med, name, fill
    )
    plot_pr_curve(
        ax_pr_median_type[mtype],
        recall,
        median_precision,
        low_ci_med,
        high_ci_med,
        name,
        fill,
    )

    # ---- AP stats ----
    ap_mean, ci_low, ci_high = res["ap_mean_stats"]
    metrics_store["ap_means"].append(ap_mean)
    metrics_store["ci_means_lowers"].append(ci_low)
    metrics_store["ci_means_uppers"].append(ci_high)

    ap_med, med_low, med_high = res["ap_median_stats"]
    metrics_store["ap_medians"].append(ap_med)
    metrics_store["ci_medians_lowers"].append(med_low)
    metrics_store["ci_medians_uppers"].append(med_high)

    mean_precision_op, ci_low_op_precision, ci_high_op_precision = res[
        "precision_op_mean_stats"
    ]
    median_precision_op, ci_low_op_precision_median, ci_high_op_precision_median = res[
        "precision_op_median_stats"
    ]

    # ---- CSV row ----
    metrics_store["df_pr_metrics"].append(
        {
            "Model": name,
            "map_type": mtype,
            "ap_mean": ap_mean,
            "ci_mean_lower": ci_low,
            "ci_mean_upper": ci_high,
            "ap_median": ap_med,
            "ci_median_low": med_low,
            "ci_median_high": med_high,
            "mean_precision_op": mean_precision_op,
            "ci_low_op_precision": ci_low_op_precision,
            "ci_high_op_precision": ci_high_op_precision,
            "median_precision_op": median_precision_op,
            "ci_low_op_precision_median": ci_low_op_precision_median,
            "ci_high_op_precision_median": ci_high_op_precision_median,
        }
    )
    metrics_store["all_stats"].append(res["stats"])


def process_run_results(results: dict, fill: bool) -> None:
    """Process the results for the different runs.

    Args:
        results (dict): The dictionnary containing the results
        fill (bool): True to fill ci in the plots.

    """
    map_types = ["MPM", "DIFF", "ALL"]

    # Create all figures
    fig_pr_all, ax_pr_all = plt.subplots(figsize=(15, 9))
    fig_pr_median_all, ax_pr_median_all = plt.subplots(figsize=(15, 9))

    fig_pr_type = {t: plt.subplots(figsize=(15, 9)) for t in map_types}
    fig_pr_median_type = {t: plt.subplots(figsize=(15, 9)) for t in map_types}

    ax_pr_type = {t: fig_pr_type[t][1] for t in map_types}
    ax_pr_median_type = {t: fig_pr_median_type[t][1] for t in map_types}

    # Storage buckets
    metrics_store = {
        "model_names": [],
        "ap_means": [],
        "ci_means_lowers": [],
        "ci_means_uppers": [],
        "ap_medians": [],
        "ci_medians_lowers": [],
        "ci_medians_uppers": [],
        "op_recall_means": [],
        "ci_op_recall_lowers": [],
        "ci_op_recall_uppers": [],
        "op_precision_means": [],
        "ci_op_precision_lowers": [],
        "ci_op_precision_uppers": [],
        "df_pr_metrics": [],
        "all_stats": [],
    }

    # ---- Process each result item ----
    for res in results:
        process_single_result(
            res=res,
            fill=fill,
            ax_pr_all=ax_pr_all,
            ax_pr_type=ax_pr_type,
            ax_pr_median_all=ax_pr_median_all,
            ax_pr_median_type=ax_pr_median_type,
            metrics_store=metrics_store,
        )

    # ---- Convert lists to DataFrames ----
    df_pr_metrics = pd.DataFrame(metrics_store["df_pr_metrics"])
    df_combined_stats = pd.concat(metrics_store["all_stats"])

    # ---- Finalize and log all PR plots -----
    finalize_pr_plots(
        ax_pr_all=ax_pr_all,
        ax_pr_median_all=ax_pr_median_all,
        fig_pr_all=fig_pr_all,
        fig_pr_median_all=fig_pr_median_all,
        fig_pr_type=fig_pr_type,
        fig_pr_median_type=fig_pr_median_type,
        map_types=map_types,
    )

    # ---- Model comparison plots ----
    finalize_model_comparison_plots(metrics_store, fill)

    # ---- Save CSVs ----
    log_dataframe_as_csv(df_pr_metrics, "pr_metrics.csv", "csv")
    log_dataframe_as_csv(df_combined_stats, "combined_stats.csv", "csv")
