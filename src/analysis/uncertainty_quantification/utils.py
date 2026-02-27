"""Update code from https://github.com/Medical-Image-Analysis-Laboratory/MS_WML_uncs.

Credits: Medical Image Analysis Laboratory
Github: https://github.com/Medical-Image-Analysis-Laboratory/MS_WML_uncs
Citation: @misc{molchanova2024structuralbased,
                title={Structural-Based Uncertainty in Deep Learning Across Anatomical
                        Scales: Analysis in White Matter Lesion Segmentation
                        },
                author={Nataliia Molchanova and Vatsal Raina and Andrey Malinin and
                        Francesco La Rosa and Adrien Depeursinge and Mark Gales and
                        Cristina Granziera and Henning Muller and Mara Graziani and
                        Meritxell Bach Cuadra
                        },
                  year={2024},
                  eprint={2311.08931},
                  archivePrefix={arXiv},
                  primaryClass={cs.CV}
                }
"""
# -*- authors : Medical Image Analysis Laboratory -*-
# -*- date : 2024-10-23 -*-
# -*- Last revision: 2025-12-24 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Transform functions-*-

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from scipy import ndimage

from analysis.uncertainty_quantification.voxel_uncertainty_measures import (
    entropy_of_expected,
)


def intersection_over_union(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU for 2 binary masks.

    Args:
        mask1 (np.ndarray): First binary mask. Shape [H,W,D]
        mask2 (np.ndarray): econd binary mask. Shape [H,W,D]

    Returns:
        float: The intersection over Union

    """
    den = np.sum(mask1 + mask2 - mask1 * mask2)
    return 0.0 if den == 0 else np.sum(mask1 * mask2) / den


def get_cc_mask(binary_mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """Get a labeled mask from a binary one.

    Args:
        binary_mask(np.ndarray): The binaray mask.
        connectivity (int): The connectivity to use in the sutructural element.
          Default to 2.

    Returns:
        np.ndarray: The labeled binary mask.

    """
    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    return ndimage.label(binary_mask, structure=struct_el)[0]


def load_rc_h5(
    h5_path: Path,
    scale: Literal["voxel", "lesion", "participant"],
    entropy_measure: str,
    curve: str,
) -> tuple:
    """Load RC values and aucs from h5 file.

    Args:
        h5_path (Path): The h5 path.
        scale (Literal[voxel, lesion, patient]): The scale. Can be either voxel, lesion
          or patient.
        entropy_measure (str): The entropy measure to load.
        curve (str): The curve to load

    Returns:
        tuple: tuple containing the values and the aucs.

    """
    with h5py.File(h5_path, "r") as f:
        return f[f"{scale}/entropy/{entropy_measure}/{curve}/values"][:], f[
            f"{scale}/entropy/{entropy_measure}/{curve}/aucs"
        ][:]


def load_frac_retained(h5_path: Path) -> np.ndarray:
    """Load the fraction retained array.

    Args:
        h5_path (Path): The h5 path.

    Returns:
        np.ndarray: The fraction retained array.

    """
    with h5py.File(h5_path, "r") as f:
        return f["fracs_retained"][:]


def bootstrap_mean_ci(
    data: np.ndarray, n_boot: int = 10000, ci: int = 95, random_state: int = 0
) -> tuple:
    """Bootsrap the AUC.

    Args:
        data (np.ndarray): Array of aucs.
        n_boot (int, optional): Number of bootstraps. Defaults to 10000.
        ci (int, optional): Confidence Interval value. Defaults to 95.
        random_state (int, optional): Random State. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - mean AUC
            - lower AUC
            - higher AUC
            - bootstrap AUC

    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)

    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = sample.mean()

    mean, lower, upper = get_ci(boot_means, ci)

    return mean, lower, upper, boot_means


def get_ci(means: np.ndarray, ci: int = 95) -> tuple:
    """Get the confidence interval for a mean array.

    Args:
        means (np.array): The means
        ci (int, optional): The confidence interval. Defaults to 95.

    Returns:
        tuple: The mean, lower and upper bounds.

    """
    alpha = (100 - ci) / 2
    lower = np.percentile(means, alpha)
    upper = np.percentile(means, 100 - alpha)

    return means.mean(), lower, upper


def calculate_best_measures(
    pred_array: np.ndarray, pred_proba_array: np.ndarray, connectivity: int = 2
) -> tuple:
    """Calculate the best measures for mask generation.

    This function is meant to retrieve the best measure at different scales, for
      plotting purpose.

    Args:
        pred_array (np.ndarray): The binary prediction array.
        pred_proba_array (np.ndarray): The probability prediction array.
        connectivity (int): The connectivity to use in the sutructural element.
          Default to 2.

    Returns:
        tuple: tuple containing voxel, lesion and patient uncertainty measures.

    """
    voxel_unc_mask = entropy_of_expected(
        np.stack([pred_proba_array, 1 - pred_proba_array], axis=-1)
    )

    lesion_unc = np.zeros_like(pred_array)
    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, num_features = ndimage.label(pred_array, structure=struct_el)

    lesions_uq = []
    for i in range(1, num_features + 1):
        component = labels == i
        mean_uq = np.mean(voxel_unc_mask[component])
        lesion_unc[component] = mean_uq
        lesions_uq.append(mean_uq)

    patient_uq = np.mean(lesions_uq)

    return voxel_unc_mask, lesion_unc, patient_uq
