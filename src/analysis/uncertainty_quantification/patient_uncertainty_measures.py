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
# -*- Description: Functions to calculate patient uncertainty measures-*-

import numpy as np
import pandas as pd

from analysis.uncertainty_quantification.lesion_uncertainty_measures import (
    les_uncs_measures as evaluated_uncertainty_measures,
)
from analysis.uncertainty_quantification.voxel_uncertainty_measures import (
    voxels_uncertainty,
)


def lesion_scale_measures(subject_lesion_uncs: pd.DataFrame) -> dict:
    """Compute the lesion scale measures.

    Args:
        subject_lesion_uncs (pd.DataFrame): lesion uncertainties for a subject.

    Returns:
        dict: Dictionary with computed uncertainty measures

    """

    def scale(arr: np.ndarray) -> np.ndarray:
        if np.max(arr) != np.min(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        return arr

    result = {}
    measures = set(evaluated_uncertainty_measures).intersection(
        set(subject_lesion_uncs.columns)
    )
    for measure in list(measures):
        # without scaling (looking at the distribution on the dataset level)
        result["avg. " + measure] = subject_lesion_uncs[measure].mean()
        # scaling subject wise
        result["avg. scaled " + measure] = scale(subject_lesion_uncs[measure]).mean()
    return result


def voxel_scale_measures(pred_probs: np.ndarray, _brain_mask: np.ndarray) -> dict:
    """Compute the voxel-scale measures.

    Args:
        pred_probs (np.ndarray): Prediction probabilities. Shape [H,W,D,C].
        pred (np.ndarray): The binary prediction mask. Shape [H,W,D,C]
        brain_mask (np.ndarray): Mask of the brain. Shape [H,W,D].

    Returns:
        dict: Dictionary with computed uncertainty measures.

    """

    def scale(arr: np.ndarray) -> np.ndarray:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    uncs_maps: dict = voxels_uncertainty(pred_probs)

    result = {}

    for k, v in uncs_maps.items():
        result["avg. " + k] = np.mean(v * _brain_mask)
        result["avg. scaled " + k] = np.mean(scale(v) * _brain_mask)

    return result


def patient_uncertainty(
    pred_probs: np.ndarray,
    brain_mask: np.ndarray,
    subject_lesion_uncs: pd.DataFrame,
) -> dict:
    """Compute patient-level uncertainties.

    Args:
        pred_probs (np.ndarray): Prediction probabilities. Shape [H,W,D,C].
        pred (np.ndarray): The binary prediction mask. Shape [H,W,D]
        brain_mask (np.ndarray): Mask of the brain. Shape [H,W,D].
        subject_lesion_uncs (pd.DataFrame): lesion uncertainties for a subject.

    Returns:
        dict: Dictionary with computed uncertainty measures

    """
    result = voxel_scale_measures(pred_probs, brain_mask)
    result.update(lesion_scale_measures(subject_lesion_uncs))
    return result
