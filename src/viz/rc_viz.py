# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-30 -*-
# -*- Last revision: 2026-02-09 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: RC plot functions-*-

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pypalettes import load_cmap
from scipy.stats import pearsonr, t

from analysis.metrics import weighted_dice_coefficient
from analysis.uncertainty_quantification.constants import (
    ENTROPY_MEASURES,
    MEASURES_RENAMES,
    SCALES,
)
from analysis.uncertainty_quantification.utils import (
    bootstrap_mean_ci,
    calculate_best_measures,
    get_ci,
    load_frac_retained,
    load_rc_h5,
)
from data_func.h5_helpers import open_h5
from data_func.utils import crop_array
from misc.constants import DATA_DIR, RESULTS_DIR
from viz.viz_utils import set_style

set_style()


def load_curves(h5_path: str, scale: str, entropy_measure: str) -> tuple:
    """Load observed, random, and ideal curves from an h5 file and remove NaNs.

    Args:
        h5_path (str): The h5 path.
        scale (str): The scale.
        entropy_measure (str): The entropy measure.

    Returns:
        tuple: A tuole containing:
            - np.ndarray: observation values
            - np.ndarray: aucs values

            for observation, ideal and random

    """
    obs_curve, aucs_obs = load_rc_h5(h5_path, scale, entropy_measure, "obs")
    obs_curve = obs_curve if scale == "participant" else np.mean(obs_curve, axis=0)
    obs_curve = obs_curve[~np.isnan(obs_curve)]

    rand_curve, aucs_rand = load_rc_h5(h5_path, scale, entropy_measure, "rand")
    rand_curve = (
        np.mean(rand_curve, axis=0)
        if scale != "participant"
        else rand_curve[~np.isnan(rand_curve)]
    )

    ideal_curve, aucs_ideal = load_rc_h5(h5_path, scale, entropy_measure, "ideal")
    ideal_curve = (
        np.mean(ideal_curve, axis=0)
        if scale != "participant"
        else ideal_curve[~np.isnan(ideal_curve)]
    )

    return obs_curve, aucs_obs, rand_curve, aucs_rand, ideal_curve, aucs_ideal


def plot_entropy_measure(
    ax: plt.Axes,
    x_values: np.ndarray,
    obs_curve: np.ndarray,
    aucs_obs: np.ndarray,
    measure_name: str,
    color: str,
    scale: str,
) -> str:
    """Plot observed curve and compute bootstrap statistics.

    Args:
        ax (plt.Axes): Matplotlib axes.
        x_values (np.ndarray): x values.
        obs_curve (np.ndarray): Observation values.
        aucs_obs (np.ndarray): AUCs values.
        measure_name (str): The name of the measure.
        color (str): The color to use.
        scale (str): The scale.

    Returns:
        str: String with the AUC stats.

    """
    ax.plot(x_values, obs_curve, label=f"${measure_name}$", color=color)
    if scale == "participant":
        mean_auc, ci_low, ci_high = get_ci(aucs_obs)
    else:
        mean_auc, ci_low, ci_high, _ = bootstrap_mean_ci(aucs_obs)
    return f"{mean_auc:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def plot_baselines(
    ax: plt.Axes,
    x_values: np.ndarray,
    random_values: np.ndarray,
    ideal_values: np.ndarray,
    random_aucs: np.ndarray,
    ideal_aucs: np.ndarray,
    scale: str,
) -> dict:
    """Plot mean random and ideal curves and compute statistics.

    Args:
        ax (plt.Axes): Matplotlib axes.
        x_values (np.ndarray): x values.
        random_values (np.ndarray): Random Values.
        ideal_values (np.ndarray): Ideal Values.
        random_aucs (np.ndarray): Random AUCs.
        ideal_aucs (np.ndarray): Ideal AUCs.
        scale (str): the scale to plot (voxel, lesion, patient).

    Returns:
        dict: Dictionary with stats for ranom and ideal.

    """
    mean_rand = np.mean(np.vstack(random_values), axis=0)
    mean_ideal = np.mean(np.vstack(ideal_values), axis=0)
    if scale == "participant":
        mean_rand_auc, ci_low_r, ci_high_r = get_ci(
            np.mean(np.vstack(random_aucs), axis=0)
        )
        mean_ideal_auc, ci_low_i, ci_high_i = get_ci(
            np.mean(np.vstack(ideal_aucs), axis=0)
        )

    else:
        mean_rand_auc, ci_low_r, ci_high_r, _ = bootstrap_mean_ci(
            np.mean(np.vstack(random_aucs), axis=0)
        )
        mean_ideal_auc, ci_low_i, ci_high_i, _ = bootstrap_mean_ci(
            np.mean(np.vstack(ideal_aucs), axis=0)
        )

    ax.plot(x_values, mean_rand, "--", label="Random", color="black")
    ax.plot(x_values, mean_ideal, "--", label="Ideal", color="gray")

    return {
        "Random": f"{mean_rand_auc:.3f} [{ci_low_r:.3f}, {ci_high_r:.3f}]",
        "Ideal": f"{mean_ideal_auc:.3f} [{ci_low_i:.3f}, {ci_high_i:.3f}]",
    }


def plot_rc_curves(
    h5_paths: list,
    segmentation_metric: str,
    scales: list = SCALES,
    entropy_measures: dict = ENTROPY_MEASURES,
    measures_renames: dict = MEASURES_RENAMES,
    palette: str = "Kippenberger",
) -> pd.DataFrame:
    """Plot the retention curves.

    Args:
        h5_paths (list): The list of h5 path.
        segmentation_metric (str): The segmentation metric to use.
        scales (list): The list of SCALES. Default to SCALES.
        entropy_measures (dict): The dict of entropy measures.
          Default to entropy_meaures.
        measures_renames (dict): The dict of measure names for plotting.
          Default to MEASURES_RENAMES.
        palette (str): The color palette. Default to Kippenberger.

    Returns:
        pd.DataFrame: DataFrame containing AUCs stats.

    """
    nrows, ncols = len(h5_paths), len(scales)
    _fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        sharex=False,
        sharey=False,
    )
    results = []
    cmap = load_cmap(palette, cmap_type="discrete")
    colors = cmap.colors

    for i, h5_path in enumerate(h5_paths):
        fracs_retained = load_frac_retained(h5_path)
        set_name = "Train" if "train" in Path(h5_path).name else "Test"

        for j, scale in enumerate(scales):
            ax = axes[i, j]
            random_values, random_aucs = [], []
            ideal_values, ideal_aucs = [], []

            for cnt, measure in enumerate(entropy_measures[scale]):
                obs_curve, aucs_obs, rand_curve, aucs_rand, ideal_curve, aucs_ideal = (
                    load_curves(h5_path, scale, measure)
                )
                x_values = (
                    np.linspace(0, 1, len(obs_curve))
                    if scale == "participant"
                    else fracs_retained
                )
                results_val = plot_entropy_measure(
                    ax,
                    x_values,
                    obs_curve,
                    aucs_obs,
                    measures_renames[measure],
                    colors[cnt % len(colors)],
                    scale,
                )
                results.append(
                    {
                        "Split": set_name,
                        "Measure": measures_renames[measure],
                        "Scale": scale,
                        "Value": results_val,
                    }
                )

                random_values.append(rand_curve)
                random_aucs.append(aucs_rand)
                ideal_values.append(ideal_curve)
                ideal_aucs.append(aucs_ideal)

            baseline_results = plot_baselines(
                ax,
                x_values,
                random_values,
                ideal_values,
                random_aucs,
                ideal_aucs,
                scale,
            )
            for k, v in baseline_results.items():
                results.append(
                    {"Split": set_name, "Measure": k, "Scale": scale, "Value": v}
                )
            ax.set_xlabel(f"Fraction of retained {scale}s", fontsize=14, labelpad=15)
            if j == 0:
                ax.set_ylabel(set_name, fontsize=14)
            if i == len(h5_paths) - 1:
                ax.legend(loc="upper right", bbox_to_anchor=(1.6, 1.7), fontsize=14)
            if i == 0:
                ax.set_title(f"{segmentation_metric} ({scale})", fontsize=14)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75, hspace=0.5, wspace=0.8)
    return pd.DataFrame(results)


def plot_pearson_corr(
    entropy: str,
    df_patient_metrics: pd.DataFrame,
    segmentation_metric: str,
    palette: str,
) -> None:
    """Plot pearson correlation between a segmentation metric and entropy measure.

    Args:
        entropy (str): Name of the entropy column.
        df_corr (pd.DataFrame): Correlation DataFrame.
        df_patient_metrics (pd.DataFrame): Patient metrics.
        segmentation_metric (str, optional): The segmentation metric to use.
        palette (str): Name of the palette to use.

    """
    cmap = load_cmap(palette, cmap_type="discrete")
    colors = list(np.array(cmap.colors)[[3, -3]])
    plt.figure(figsize=(6, 5))
    # Scatter points
    sns.scatterplot(
        df_patient_metrics,
        x=entropy,
        y=segmentation_metric,
        hue="split",
        palette=colors,
    )
    for idx, split in enumerate(df_patient_metrics["split"].unique()):
        df_split = df_patient_metrics[df_patient_metrics["split"] == split]

        x = df_split[entropy].to_numpy()
        y = df_split[segmentation_metric].to_numpy()

        # ---- Linear fit ----
        coef = np.polyfit(x, y, 1)
        y_hat = np.polyval(coef, x)

        # ---- Residuals ----
        n = len(x)
        residuals = y - y_hat
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))

        # ---- Confidence interval (95%) ----
        x_mean = np.mean(x)
        t_val = t.ppf(0.975, n - 2)

        ci = (
            t_val
            * s_err
            * np.sqrt(1 / n + (x - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
        )

        y_lower = y_hat - ci
        y_upper = y_hat + ci
        order = np.argsort(x)
        plt.plot(
            x[order],
            y_hat[order],
            linewidth=2,
            color=colors[idx],
            label=f"y={coef[1]:.1f}{coef[0]:.1f}x",
        )

        plt.fill_between(
            x[order], y_lower[order], y_upper[order], alpha=0.2, color=colors[idx]
        )

    # ---- Labels and title ----
    plt.xlabel(rf"${MEASURES_RENAMES[entropy]}$", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(segmentation_metric, fontsize=16)

    # ---- Adjust limits ----
    plt.xlim(
        df_patient_metrics[entropy].min() * 0.95,
        df_patient_metrics[entropy].max() * 1.05,
    )
    plt.ylim(0, 1.05)
    plt.legend(fontsize=16)
    plt.tight_layout()

    save_dir = RESULTS_DIR / "uq"
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "pearson_corr.png")


def plot_partial_corr_with_ci(
    entropy: str,
    df_patient_metrics: pd.DataFrame,
    dice_col: str = "DSC",
    cov_col: str = "Fazekas",
    split_col: str = "split",
    n_boot: int = 10000,
    ci: int = 95,
    palette: str = "Kippenberger",
) -> None:
    """Plot partial correlation between entropy and Dice controlling for Fazekas.

    Args:
        entropy (str): The name of the entropy to use.
        df_patient_metrics (pd.DataFrame): DataFrame containing the patient metrics.
        dice_col (str, optional): Name of col containing the Dice. Defaults to "DSC".
        cov_col (str, optional): Name of the covariate col. Defaults to "Fazekas".
        split_col (str, optional): Name of the col to split. Defaults to "split".
        n_boot (int, optional): Number of bootstrap to perform. Defaults to 10000.
        ci (int, optional): Quantile to use. Defaults to 95.
        palette (str, optional): Name of the palette to use. Defaults to "Kippenberger".

    """

    def partial_corr_boot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        cov_col: str,
        n_boot: int = 10000,
        ci: int = 95,
    ) -> tuple[float, float, float, float, float]:
        x = sm.add_constant(df[cov_col])
        res_x = sm.OLS(df[x_col], x).fit().resid
        res_y = sm.OLS(df[y_col], x).fit().resid

        r, _ = pearsonr(res_x, res_y)

        # ---- Bootstrap CI ----
        boot_r = []
        n = len(res_x)
        for _ in range(n_boot):
            generator = np.random.default_rng(12345)
            idx = generator.choice(n, n, replace=True)
            r_boot, _ = pearsonr(res_x[idx], res_y[idx])
            boot_r.append(r_boot)

        lower = np.percentile(boot_r, (100 - ci) / 2)
        upper = np.percentile(boot_r, 100 - (100 - ci) / 2)

        return r, lower, upper, res_x, res_y

    colors = load_cmap(palette, cmap_type="discrete").colors
    colors = list(np.array(colors)[[3, -3]])

    plt.figure(figsize=(6, 5))

    for idx, sp in enumerate(df_patient_metrics[split_col].unique()):
        df_split = df_patient_metrics[df_patient_metrics[split_col] == sp]
        df_split.index = np.arange(len(df_split))
        _r, _lower_r, _upper_r, res_x, res_y = partial_corr_boot(
            df_split, entropy, dice_col, cov_col, n_boot, ci
        )

        # ---- Plot scatter points ----
        plt.scatter(
            res_x, res_y, alpha=0.6, color=colors[idx], label=f"{sp.capitalize()}"
        )

        # ---- Regression line + 95% CI ----
        x_fit = sm.add_constant(res_x)
        model = sm.OLS(res_y, x_fit).fit()
        x_vals = np.linspace(res_x.min(), res_x.max(), 100)
        x_pred = sm.add_constant(x_vals)
        y_pred = model.predict(x_pred)

        pred_ci = model.get_prediction(x_pred).conf_int(alpha=0.05)
        y_lower, y_upper = pred_ci[:, 0], pred_ci[:, 1]

        _beta0, beta1 = model.params

        label = rf"$y={beta1:.2f}x$"

        plt.plot(x_vals, y_pred, color=colors[idx], linewidth=2, label=label)
        plt.fill_between(x_vals, y_lower, y_upper, color=colors[idx], alpha=0.2)

    # ---- Plot origin lines ----
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel(
        rf"${MEASURES_RENAMES[entropy]}$ residuals (controlling {cov_col})", fontsize=16
    )
    plt.ylabel(f"{dice_col} residuals (controlling {cov_col})", fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    save_dir = RESULTS_DIR / "uq"
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "partial_corr.png")


def plot_maps(  # noqa: PLR0915
    df_data: pd.DataFrame,
    model: str,
    segmentation_metric: str,
    patient_unc_stats: tuple = (0.47, 0.65),
) -> None:
    """Plot uncertainty map at 3 scales (voxel, lesion, patient).

    Args:
        df_data (pd.DataFrame): Data containing needed information to plot.
        model (str): The model to choose the segmentation.
        segmentation_metric (str): The segmentation metric to use.
        patient_unc_stats (tuple, optional): The patient uncertainty stats (min,max) to
          adjust the scale. Defaults to (0.47, 0.65).

    """
    n_rows = len(df_data)
    _fig, axes = plt.subplots(
        n_rows,
        5,
        figsize=(10, 2.2 * n_rows),
    )
    cmap_gt = ListedColormap(["#00b0be"])
    cmap_pred = ListedColormap(["#e45c5c"])

    # ---- Normalize patient value ----
    norm = mcolors.Normalize(vmin=patient_unc_stats[0], vmax=patient_unc_stats[1])
    cmap = cm.inferno

    for idx, row in df_data.iterrows():
        patient_id = row["id"]
        slice_idx = row["slice_idx"]

        # ---- Fetch participant prediction ----
        pred_path = next(
            iter((DATA_DIR / patient_id / "segmentation" / model).glob("*pred.nii.gz")),
            None,
        )
        pred_proba_path = next(
            iter(
                (DATA_DIR / patient_id / "segmentation" / model).glob(
                    "*pred_proba.nii.gz"
                )
            ),
            None,
        )
        flair_path = next(
            iter(
                (DATA_DIR / patient_id / "corrected" / "FLAIR" / "in_MPM").glob(
                    "*masked*.nii*"
                )
            ),
            None,
        )
        h5_path = next(
            iter((DATA_DIR / patient_id / "corrected" / "MPM").glob("*.h5")),
            None,
        )

        pred = nib.load(pred_path)
        pred_proba = nib.load(pred_proba_path)
        flair = nib.load(flair_path)
        subject_data = open_h5(h5_path)
        subject_data.clean_inputs()

        pred_array = pred.get_fdata()
        pred_proba_array = pred_proba.get_fdata()
        flair_array = flair.get_fdata()

        manual_array = subject_data.gt
        pred_array = crop_array(pred_array, subject_data.patch_coords)
        pred_proba_array = crop_array(pred_proba_array, subject_data.patch_coords)
        flair_array = crop_array(flair_array, subject_data.patch_coords)

        dice = weighted_dice_coefficient(manual_array, pred_array)

        voxel_unc_mask, lesion_unc, patient_uq = calculate_best_measures(
            pred_array, pred_proba_array
        )
        slice_idx = slice_idx - subject_data.patch_coords[2]
        patient_color = cmap(norm(patient_uq))

        # --------------------------------------------------
        # Plot Participant UQ at different scales
        # --------------------------------------------------
        # ---- Voxel-scale ----
        axes[idx, 1].imshow(flair_array[:, slice_idx, :], cmap="gray")
        axes[idx, 1].axis("off")
        axes[idx, 1].imshow(
            np.ma.masked_where(
                pred_array[:, slice_idx, :] == 0, pred_array[:, slice_idx, :]
            ),
            cmap=cmap_gt,
            alpha=0.9,
        )
        axes[idx, 0].imshow(flair_array[:, slice_idx, :], cmap="gray")

        axes[idx, 0].imshow(
            np.ma.masked_where(
                manual_array[:, slice_idx, :] == 0, manual_array[:, slice_idx, :]
            ),
            cmap=cmap_pred,
            alpha=1,
        )
        axes[idx, 2].imshow(voxel_unc_mask[:, slice_idx, :], cmap=cmap)
        axes[idx, 2].axis("off")

        # ---- Lesion-scale ----
        axes[idx, 3].imshow(lesion_unc[:, slice_idx, :], cmap=cmap)
        axes[idx, 3].axis("off")

        # ---- Patient-scale ----
        axes[idx, 4].imshow(np.zeros_like(voxel_unc_mask[:, slice_idx, :]), cmap="gray")
        axes[idx, 4].text(
            voxel_unc_mask.shape[2] // 2,
            voxel_unc_mask.shape[1] // 2,
            f"{patient_uq:.2f}",
            color=patient_color,
            fontsize=14,
            ha="center",
            va="center",
        )
        axes[idx, 4].axis("off")
        axes[idx, 0].set_yticks([])
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_ylabel(
            f"{patient_id}\n({segmentation_metric}={dice:.3f})", labelpad=2
        )
        # ---- Add title ---
        if idx == 0:
            axes[idx, 0].set_title("Ground Truth")
            axes[idx, 1].set_title("Prediction")
            axes[idx, 2].set_title("Voxel-scale")
            axes[idx, 3].set_title("Lesion-scale")
            axes[idx, 4].set_title("Participant-scale")

    plt.gca().axes.yaxis.set_ticklabels([])

    save_dir = RESULTS_DIR / "uq"
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "uq_maps.png")
