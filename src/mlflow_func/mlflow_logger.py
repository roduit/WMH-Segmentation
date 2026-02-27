# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-06 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Function to log artifcats to Mlflow -*-

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from misc.constants import DATASET, FAZEKAS, ID, REGION_NAME
from viz.stats import plot_box_dice
from viz.viewer import plot_3dview


def log_dataframe_as_csv(df: pd.DataFrame, filename: str, artifact_path: str) -> None:
    """Log a DataFrame as a CSV file in MLflow.

    Args:
        df (pd.DataFrame): The DataFrame to log.
        filename (str): The name of the CSV file.
        artifact_path (str): The artifact path in MLflow.

    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        csv_path = Path(tmpdirname) / filename
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(local_path=csv_path, artifact_path=artifact_path)


def log_results_optuna(study: optuna.study.Study) -> None:
    """Log the Optuna study results as artifacts in MLflow."""
    log_dataframe_as_csv(
        df=study.trials_dataframe(),
        filename="optuna_study.csv",
        artifact_path="optuna",
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Log the optimization history plot
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{tmpdirname}/optimization_history.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/optimization_history.html",
            artifact_path="optuna",
        )

        # Log contour plot
        fig = optuna.visualization.plot_contour(study)
        fig.write_html(f"{tmpdirname}/contour_plot.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/contour_plot.html", artifact_path="optuna"
        )

        # Log EDF plot
        fig = optuna.visualization.plot_edf(study)
        fig.write_html(f"{tmpdirname}/edf_plot.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/edf_plot.html", artifact_path="optuna"
        )

        # Log parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{tmpdirname}/parallel_coordinate_plot.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/parallel_coordinate_plot.html",
            artifact_path="optuna",
        )

        # Plot parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{tmpdirname}/param_importances.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/param_importances.html",
            artifact_path="optuna",
        )

        # Plot Rank
        fig = optuna.visualization.plot_rank(study)
        fig.write_html(f"{tmpdirname}/rank.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/rank.html", artifact_path="optuna"
        )

        # Plot Slice
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f"{tmpdirname}/slice.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/slice.html", artifact_path="optuna"
        )

        # Plot timeline
        fig = optuna.visualization.plot_timeline(study)
        fig.write_html(f"{tmpdirname}/timeline.html")
        mlflow.log_artifact(
            local_path=f"{tmpdirname}/timeline.html", artifact_path="optuna"
        )


def generate_test_plots(
    model_name: str,
    df_metrics: pd.DataFrame,
    df_spatial_metrics: pd.DataFrame,
    y_true: dict,
    y_pred: dict,
) -> None:
    """Generate and log test result plots and metrics to MLflow.

    Args:
        model_name (str): Name of the model for plot titles.
        df_metrics (pd.DataFrame): DataFrame containing subject-level metrics.
        df_spatial_metrics (pd.DataFrame): DataFrame containing spatial metrics.
        y_true (list): True labels for ROC/PR curves.
        y_pred (list): Predicted probabilities for ROC/PR curves.

    """

    def log_boxplot(
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        title_prefix: str,
        subfolder: str,
        tmpdir: Path,
    ) -> None:
        """Generate, save, and log a single boxplot.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            hue (str): Column name for hue.
            title_prefix (str): Prefix for the plot title.
            subfolder (str): Subfolder in MLflow to log the artifact.
            tmpdir (Path): Temporary directory for saving the plot.

        """
        plot_box_dice(
            df_metrics=df,
            x=x,
            y=y,
            hue=hue,
            title=f"{title_prefix}: {y} ({model_name.upper()})",
        )
        filename = (
            f"{model_name}_{title_prefix.lower().replace(' ', '_')}_{y}_{x}_boxplot.png"
        )
        path = tmpdir / filename
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(path), artifact_path=subfolder)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)

        # --- Subject-level metrics ---
        for metric in df_metrics.select_dtypes(include=["number"]).columns:
            log_boxplot(
                df_metrics,
                FAZEKAS,
                metric,
                FAZEKAS,
                "Subject Metric",
                "test_results/plots/fazekas_metrics",
                tmpdir,
            )
            log_boxplot(
                df_metrics,
                "dataset",
                metric,
                FAZEKAS,
                "Subject Metric",
                "test_results/plots/dataset_metrics",
                tmpdir,
            )

        # --- Spatial metrics ---
        for metric in df_spatial_metrics.select_dtypes(include=["number"]).columns:
            log_boxplot(
                df_spatial_metrics,
                FAZEKAS,
                metric,
                REGION_NAME,
                "Spatial Metric",
                "test_results/plots/fazekas_spatial_metrics",
                tmpdir,
            )
            log_boxplot(
                df_spatial_metrics,
                "dataset",
                metric,
                REGION_NAME,
                "Spatial Metric",
                "test_results/plots/dataset_spatial_metrics",
                tmpdir,
            )

            # PR curve
            pr_aucs = []
            for y_true_group, y_pred_group in zip(
                y_true.values(), y_pred.values(), strict=True
            ):
                if len(y_pred_group) == 0:
                    continue
                precision, recall, _ = precision_recall_curve(
                    y_true_group, y_pred_group, pos_label=1
                )
                pr_auc = auc(recall, precision)
                pr_aucs.append(pr_auc)
            mlflow.log_metric("precision_mean", float(np.mean(precision)))
            mlflow.log_metric("recall_mean", float(np.mean(recall)))
            mlflow.log_metric("pr_auc_mean", float(np.mean(pr_aucs)))
            mlflow.log_metric("pr_auc_std", float(np.std(pr_aucs)))

        # Save raw curve data
        np.savez_compressed(tmpdir / "metrics.npz", y_true=y_true, y_pred=y_pred)
        mlflow.log_artifact(str(tmpdir / "metrics.npz"), artifact_path="data")


def log_overfitting_curves(tmpdirname: str, metrics: dict, model_name: str) -> None:
    """Generate and log learning curves showing train/val performance.

    Args:
        tmpdirname (str): Temporary directory for saving plots.
        metrics (dict): Dictionary containing training and validation metrics.
        model_name (str): Name of the model for plot titles.

    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics_to_plot = [
        ("f1_macro", "F1 Score"),
        ("recall_macro", "Recall"),
        ("precision_macro", "Precision"),
    ]

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        train_key = f"{metric_key}_train"
        val_key = f"{metric_key}_val"

        if train_key in metrics and val_key in metrics:
            folds = range(1, len(metrics[train_key]) + 1)
            axes[idx].plot(folds, metrics[train_key], "o-", label="Train")
            axes[idx].plot(folds, metrics[val_key], "s-", label="Validation")
            axes[idx].fill_between(
                folds,
                metrics[train_key],
                metrics[val_key],
                alpha=0.2,
                color="red",
                label="Overfitting Gap",
            )
            axes[idx].set_xlabel("Fold")
            axes[idx].set_ylabel(metric_label)
            axes[idx].set_title(f"{metric_label} - Train vs Validation")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1])

    plt.tight_layout()
    plot_path = Path(tmpdirname) / f"{model_name}_overfitting_curves.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    mlflow.log_artifact(
        local_path=plot_path,
        artifact_path="training/overfitting_analysis",
    )
    plt.close(fig)


def log_fold_metrics(fold_num: int, metrics: dict) -> None:
    """Log metrics for the current fold to MLflow as continuous metrics (one per type).

    Args:
        fold_num (int): The current fold number.
        metrics (dict): Dictionary containing training and validation metrics.

    """
    # Log train/val metrics with step = fold_num
    mlflow.log_metric("f1_macro_train", metrics["f1_macro_train"][-1], step=fold_num)
    mlflow.log_metric("f1_macro_val", metrics["f1_macro_val"][-1], step=fold_num)
    mlflow.log_metric(
        "recall_macro_train", metrics["recall_macro_train"][-1], step=fold_num
    )
    mlflow.log_metric(
        "recall_macro_val", metrics["recall_macro_val"][-1], step=fold_num
    )
    mlflow.log_metric(
        "precision_macro_train", metrics["precision_macro_train"][-1], step=fold_num
    )
    mlflow.log_metric(
        "precision_macro_val", metrics["precision_macro_val"][-1], step=fold_num
    )

    # Calculate and log overfitting gaps
    f1_gap = metrics["f1_macro_train"][-1] - metrics["f1_macro_val"][-1]
    recall_gap = metrics["recall_macro_train"][-1] - metrics["recall_macro_val"][-1]
    precision_gap = (
        metrics["precision_macro_train"][-1] - metrics["precision_macro_val"][-1]
    )

    mlflow.log_metric("f1_macro_gap", f1_gap, step=fold_num)
    mlflow.log_metric("recall_macro_gap", recall_gap, step=fold_num)
    mlflow.log_metric("precision_macro_gap", precision_gap, step=fold_num)


def log_3d_mask(prediction: np.ndarray, labels: np.ndarray, subject_id: str) -> None:
    """Log 3D visualization of prediction and labels to MLflow.

    Args:
        prediction (np.ndarray): 3D array of predicted segmentation.
        labels (np.ndarray): 3D array of ground truth segmentation.
        subject_id (str): Identifier for the subject.

    """
    fig = plot_3dview(prediction, labels, subject_id)

    with tempfile.TemporaryDirectory() as tmpdirname:
        plot_path = Path(tmpdirname) / f"{subject_id}_3d_mask_visualization.html"
        fig.write_html(plot_path)
        mlflow.log_artifact(
            local_path=plot_path,
            artifact_path=f"test_results/3d_visualizations/{subject_id}",
        )


def log_metrics_summary(df_metrics: pd.DataFrame) -> None:
    """Log metric results to Mlflow.

    Args:
        df_metrics (pd.DataFrame): The DataFrame containing the metrics.

    """
    df_metrics = df_metrics.drop(columns=[FAZEKAS, DATASET, ID])
    for col in df_metrics.columns:
        mean_val = np.mean(df_metrics[col])
        mlflow.log_metric(col, mean_val)
