# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-12-02 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Plot functions for data analysis -*-

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.ndimage import label
from tqdm import tqdm

from data_func.h5_helpers import open_h5
from data_func.utils import crop_array
from misc.constants import (
    DATA_DIR,
    FAZEKAS,
    H5_PATH,
    ID,
    NEUROM_LABELS,
    REGION,
    RESULTS_DIR,
    SPLIT,
    SUBJECT_EXAMPLE_H5,
)
from viz.viz_utils import MAPS_NAME_FOR_PLOTS, darken, set_style

set_style()


def plot_representative_slices(
    df_dataset: pd.DataFrame, subjects: list, slice_indices: list
) -> None:
    """Plot representative slices for each Fazekas score with WMH overlay.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing dataset information.
        subjects (list): List of PR numbers to plot.
        slice_indices (list): list of slice index to plot
        equalize (bool, optional): Whether to apply histogram equalization.
          Defaults to True.

    """
    df_dataset = df_dataset.dropna()
    _fig, axes = plt.subplots(2, len(subjects), figsize=(10, 4))
    for idx, (subject_id, slice_idx) in enumerate(
        zip(subjects, slice_indices, strict=True)
    ):
        flair_path = next(
            iter(
                (DATA_DIR / subject_id / "corrected" / "FLAIR" / "in_MPM").glob(
                    "*masked_from_MPRAGE_in_MPM.nii.gz"
                )
            ),
            None,
        )
        gt_path = next(
            iter(
                (DATA_DIR / subject_id / "manual_seg" / "in_MPM").glob(
                    "*_trilinear_in_MPM_thr.nii.gz"
                )
            ),
            None,
        )
        flair = nib.load(flair_path).get_fdata()
        gt = nib.load(gt_path).get_fdata().astype(int)
        flair_slice = flair[:, slice_idx, :]
        gt_slice = gt[:, slice_idx, :]

        background = np.zeros_like(flair_slice)

        cmap = ListedColormap(
            [
                "#00000000",  # fully transparent
                "#C1447EFF",  # opaque red
            ]
        )

        axes[0, idx].imshow(flair_slice, cmap="gray")

        axes[1, idx].imshow(background, cmap="gray")
        axes[1, idx].imshow(flair_slice, cmap="gray", alpha=0.6)
        axes[1, idx].imshow(gt_slice, cmap=cmap, vmin=0, vmax=1)
        axes[0, idx].axis("off")
        axes[1, idx].axis("off")
        axes[0, idx].set_title(f"Fazekas {idx}")

        plt.xticks([])  # hide x-axis ticks
        plt.yticks([])  # hide y-axis ticks

    plt.tight_layout()


def plot_processing_img() -> None:
    """Visualize processing effect on images."""
    subject_data = open_h5(SUBJECT_EXAMPLE_H5)

    raw_data = deepcopy(subject_data.qmaps)
    subject_data.clean_inputs()
    clean_data = deepcopy(subject_data.qmaps)
    subject_data.equalize_histogram()
    equ_data = deepcopy(subject_data.qmaps)
    raw_data = crop_array(raw_data, subject_data.patch_coords)

    slice_idx = np.argmax(np.sum(subject_data.gt, axis=(0, 2)))
    plt.figure(figsize=(20, 8))
    offset = 10
    for map_idx, map_name in enumerate(subject_data.feature_names):
        plt.subplot(3, 10, map_idx + 1)
        plt.imshow(raw_data[:, slice_idx, :, map_idx], cmap="gray")
        plt.title(map_name, fontsize=16)
        if map_idx == 0:
            plt.ylabel("Raw", fontsize=16)
        plt.xticks([])  # hide x-axis ticks
        plt.yticks([])  # hide y-axis ticks
        plt.subplot(3, 10, map_idx + 1 + offset)
        plt.imshow(clean_data[:, slice_idx, :, map_idx], cmap="gray")
        if map_idx == 0:
            plt.ylabel("Clean", fontsize=16)
        plt.xticks([])  # hide x-axis ticks
        plt.yticks([])  # hide y-axis ticks
        plt.subplot(3, 10, map_idx + 1 + 2 * offset)
        plt.imshow(equ_data[:, slice_idx, :, map_idx], cmap="gray")
        if map_idx == 0:
            plt.ylabel("Equalization", fontsize=16)
        plt.xticks([])  # hide x-axis ticks
        plt.yticks([])  # hide y-axis ticks

    plt.suptitle("Effect of the Different Processings")
    plt.tight_layout()


def plot_map_example() -> None:
    """Visualize processing effect on images."""
    subject_data = open_h5(SUBJECT_EXAMPLE_H5)

    raw_data = deepcopy(subject_data.qmaps)
    subject_data.clean_inputs()
    raw_data = crop_array(raw_data, subject_data.patch_coords)

    slice_idx = np.argmax(np.sum(subject_data.gt, axis=(0, 2)))
    slice_idx = 85

    unique_labels = np.unique(subject_data.neuromorphic_mask[:, slice_idx, :])

    rng = np.random.default_rng(42)  # stable colors
    colors = rng.random((len(unique_labels), 3))

    # Optional: force background (0) to black
    colors[unique_labels == 0] = [0, 0, 0]

    cmap = ListedColormap(colors)
    label_to_idx = {neurom_label: i for i, neurom_label in enumerate(unique_labels)}

    indexed_seg = np.vectorize(label_to_idx.get)(
        subject_data.neuromorphic_mask[:, slice_idx, :]
    )
    plt.figure(figsize=(12, 8))
    for map_idx, map_name in enumerate(subject_data.feature_names):
        plt.subplot(2, 6, map_idx + 1)
        plt.imshow(raw_data[:, slice_idx, :, map_idx], cmap="gray")
        plt.title(MAPS_NAME_FOR_PLOTS[map_name], fontsize=16)
        plt.xticks([])  # hide x-axis ticks
        plt.yticks([])  # hide y-axis ticks
    plt.subplot(2, 6, 11)
    plt.imshow(np.ma.masked_where(indexed_seg == 0, indexed_seg), cmap=cmap)
    plt.xticks([])  # hide x-axis ticks
    plt.yticks([])  # hide y-axis ticks
    plt.title("Neurom. mask")
    plt.subplot(2, 6, 12)
    plt.imshow(
        np.ma.masked_where(subject_data.gt == 0, subject_data.gt)[:, slice_idx, :]
    )
    plt.xticks([])  # hide x-axis ticks
    plt.yticks([])  # hide y-axis ticks
    plt.title("Manual segmentation")
    plt.tight_layout()


def plot_processing_hist() -> None:
    """Visualize the effect of processing on histogramm distribution."""
    subject_data = open_h5(SUBJECT_EXAMPLE_H5)

    raw_data = deepcopy(subject_data.qmaps)
    subject_data.clean_inputs()
    clean_data = deepcopy(subject_data.qmaps)
    subject_data.equalize_histogram()
    equ_data = deepcopy(subject_data.qmaps)
    raw_data = crop_array(raw_data, subject_data.patch_coords)

    plt.figure(figsize=(20, 5))
    offset = 10
    for map_idx, map_name in enumerate(subject_data.feature_names):
        plt.subplot(3, 10, map_idx + 1)
        sns.histplot(
            raw_data[..., map_idx][subject_data.mask.astype(bool)], stat="percent"
        )
        plt.title(map_name)
        if map_idx == 0:
            plt.ylabel("Raw")
        plt.subplot(3, 10, map_idx + 1 + offset)
        sns.histplot(
            clean_data[..., map_idx][subject_data.mask.astype(bool)], stat="percent"
        )
        if map_idx == 0:
            plt.ylabel("Clean")
        plt.subplot(3, 10, map_idx + 1 + 2 * offset)
        sns.histplot(
            equ_data[..., map_idx][subject_data.mask.astype(bool)], stat="percent"
        )
        if map_idx == 0:
            plt.ylabel("Equalization")

    plt.suptitle("Effect of the Different Processings on Histogramm distribution")
    plt.tight_layout()


def compute_wm_masks_stats(df_dataset: pd.DataFrame) -> dict:
    """Compute lesion statistics for segmentation masks.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing dataset information.

    Returns:
        dict: Dictionary containing lesion statistics.

    """
    total_wmh_synthseg = []
    total_wmh_neuromorphic = []
    lesions_stats = []
    inv_map_neuromorphic = {v: k for k, v in NEUROM_LABELS.items()}

    for _idx, row in tqdm(
        df_dataset.dropna().iterrows(), total=df_dataset.dropna().shape[0]
    ):
        h5_path = Path(row[H5_PATH])
        subject_id = row[ID]
        fazekas = row[FAZEKAS]

        subject_data = open_h5(Path(h5_path))
        neuromorphic_mask = subject_data.neuromorphic_mask

        labels = subject_data.gt

        if np.sum(labels) == 0:
            lesions_stats.append(
                {
                    "id": subject_id,
                    FAZEKAS: fazekas,
                    "lesion_size": 0,
                    "dominant_region_synthseg": "No region",
                    "dominant_region_neuromorphic": "No region",
                }
            )
            continue

        region_wmh_neuromorphic = neuromorphic_mask[labels.astype(bool)]
        region_names_neuromorphic = [
            inv_map_neuromorphic.get(int(val), int(val))
            for val in region_wmh_neuromorphic
        ]
        total_wmh_neuromorphic.extend(region_names_neuromorphic)

        labeled_pred, n_pred = label(labels.astype(bool))
        for lesion_idx in range(1, n_pred + 1):
            lesion_mask = np.array(labeled_pred == lesion_idx)
            lesion_size = lesion_mask.sum()
            lesion_regions_neuromorphic = neuromorphic_mask[lesion_mask]
            unique, counts = np.unique(lesion_regions_neuromorphic, return_counts=True)
            dominant_region = unique[np.argmax(counts)]
            dominant_region_name_neuromorphic = inv_map_neuromorphic.get(
                int(dominant_region), "Unknown"
            )

            lesions_stats.append(
                {
                    "id": subject_id,
                    FAZEKAS: fazekas,
                    "lesion_size": lesion_size,
                    "dominant_region_neuromorphic": dominant_region_name_neuromorphic,
                }
            )

    return total_wmh_synthseg, total_wmh_neuromorphic, lesions_stats


def plot_mask_analysis(df_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot analysis of WMH masks.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing dataset information.

    Return:
        tuple[pd.DataFrame, pd.DataFrame]: a tuple containing:
            - df_dataset_lesions: lesions statistics per participant
            - df_wmh_grouped: statistics per Neurom. label region.

    """
    region_dist_path = RESULTS_DIR / "general" / "wmh_mask_region_distribution.csv"
    lesions_stats_path = RESULTS_DIR / "general" / "wmh_lesions_stats.csv"
    if not region_dist_path.exists() or not lesions_stats_path.exists():
        total_wmh_synthseg, total_wmh_neuromorphic, lesions_stats = (
            compute_wm_masks_stats(df_dataset)
        )

        df_dataset_lesions = pd.DataFrame(lesions_stats)

        values_synthseg, counts_synthseg = np.unique(
            total_wmh_synthseg, return_counts=True
        )
        values_neuromorphic, counts_neuromorphic = np.unique(
            total_wmh_neuromorphic, return_counts=True
        )

        df_wmh_distribution = pd.DataFrame(
            {
                REGION: values_synthseg,
                "count": counts_synthseg,
                "mask_type": "synthseg",
            }
        )

        df_wmh_distribution = pd.concat(
            [
                df_wmh_distribution,
                pd.DataFrame(
                    {
                        REGION: values_neuromorphic,
                        "count": counts_neuromorphic,
                        "mask_type": "neuromorphic",
                    }
                ),
            ]
        )

        df_wmh_distribution["grouped_region"] = df_wmh_distribution[REGION].replace(
            {"Right ", "Left "}, regex=True, value=""
        )

        df_wmh_grouped = (
            df_wmh_distribution.groupby(["grouped_region", "mask_type"])
            .agg({"count": "sum"})
            .reset_index()
        )
        df_wmh_grouped["percentage"] = (
            df_wmh_grouped["count"]
            / df_wmh_grouped.groupby("mask_type")["count"].transform("sum")
            * 100
        )
        df_wmh_grouped.to_csv(region_dist_path, index=False)
        df_dataset_lesions.to_csv(lesions_stats_path, index=False)
    else:
        df_wmh_grouped = pd.read_csv(region_dist_path)
        df_dataset_lesions = pd.read_csv(lesions_stats_path)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        df_wmh_grouped.query("mask_type == 'neuromorphic'").sort_values(
            by="percentage", ascending=False
        )[:10],
        x="grouped_region",
        y="percentage",
    )
    plt.xlabel("Label Value", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.semilogy()
    plt.tight_layout()
    plt.grid(alpha=0.4)

    return df_dataset_lesions, df_wmh_grouped


def plot_lesion_size_distribution(
    df_lesions: pd.DataFrame, palette: str = "Blues", strip: bool = False
) -> None:
    """Plot lesion size distribution by Fazekas score.

    Args:
        df_lesions (pd.DataFrame): DataFrame containing lesion statistics.
        palette (str, optional): Name of seaborn color palette. Defaults to 'Blues'.
        strip (bool, optional): Whether to overlay stripplot. Defaults to False.

    """
    plt.figure(figsize=(12, 6))
    df_lesions = deepcopy(df_lesions)
    df_lesions[FAZEKAS] = df_lesions[FAZEKAS].astype(str)
    # logarithmic bins
    sns.boxplot(
        data=df_lesions, y="lesion_size", x=FAZEKAS, hue=FAZEKAS, palette=palette
    )
    if strip:
        sns.stripplot(
            data=df_lesions,
            y="lesion_size",
            x=FAZEKAS,
            hue=FAZEKAS,
            palette=[
                darken(c)
                for c in sns.color_palette(
                    palette, n_colors=df_lesions[FAZEKAS].nunique()
                )
            ],
            alpha=0.7,
            jitter=0.2,
            size=5,
            edgecolor="white",
        )
    plt.semilogy()
    plt.xlabel("Fazekas Score")
    plt.ylabel("Lesion Size (mm³)")
    plt.title("Distribution of Lesion Sizes by Fazekas Score")


def plot_volume_fazekas(
    df_dataset_lesions: pd.DataFrame, palette: str = "blues"
) -> None:
    """Plot total lesion volume per subject by Fazekas score.

    Args:
        df_dataset_lesions (pd.DataFrame): DataFrame containing lesion statistics.
        palette (str, optional): Name of seaborn color palette. Defaults to 'blues'.

    """
    df_subjects_lesion_sum = (
        df_dataset_lesions.groupby("id")
        .agg({"lesion_size": "sum", FAZEKAS: "first"})
        .reset_index()
    )

    # Get Set3 colors
    n_fazekas = df_subjects_lesion_sum[FAZEKAS].nunique()
    base_colors = sns.color_palette(palette, n_colors=n_fazekas)[:n_fazekas]
    shifted_colors = [darken(c) for c in base_colors]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_subjects_lesion_sum.sort_values(by=FAZEKAS),
        x=FAZEKAS,
        y="lesion_size",
        hue=FAZEKAS,
        palette=base_colors,
        legend=False,
    )
    sns.stripplot(
        data=df_subjects_lesion_sum.sort_values(by=FAZEKAS),
        x=FAZEKAS,
        y="lesion_size",
        hue=FAZEKAS,
        palette=shifted_colors,
        alpha=0.7,
        jitter=0.2,
        size=5,
        edgecolor="white",
        legend=False,
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Fazekas Score", fontsize=16)
    plt.ylabel(r"Total Volume of Lesions [$mm^3$]", fontsize=16)


def plot_distribution_data(df_dataset: pd.DataFrame) -> None:
    """Plot distribution of Fazekas scores in the dataset.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing dataset information.

    """
    df_dataset = df_dataset.query("max_grade < 3")
    df_dataset = df_dataset.dropna(subset="h5_path")
    fazekas = df_dataset[FAZEKAS].unique()
    max_count = df_dataset.groupby([SPLIT, FAZEKAS]).size().max()

    plt.figure(figsize=(8, 6))
    ax = sns.countplot(
        x=FAZEKAS, data=df_dataset, order=fazekas, hue=SPLIT, palette="Set2"
    )
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.title("Distribution of Fazekas Scores in Brainlaus Dataset")
    plt.xlabel("Fazekas Score")
    plt.yticks(np.arange(0, max_count, step=5))
    plt.ylabel("Number of Subjects")
