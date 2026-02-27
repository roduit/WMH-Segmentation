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
# -*- Description: Functions to calculate voxel uncertainty measures-*-

import numpy as np


def renyi_entropy_of_expected(prob_array: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Calculate Renyi entropy.

    Renyi entropy is a generalised version of Shannon, where the two are equal
    if alpha=1.

    Args:
        prob_array (np.ndarray): Probability array. Shape [H,W,D,C].
        alpha (float, optional): Entropy order. Defaults to 0.8.

    Returns:
        np.ndarray: Entropy array. Shape [H,W,D]

    """
    scale = 1.0 / (1.0 - alpha)
    return scale * np.log(np.sum(prob_array**alpha, axis=-1))


def entropy_of_expected(prob_array: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Calculate the entropy of expected.

    Args:
        prob_array (np.ndarray): Probability array. Shape [H,W,D,C].
        epsilon (float, optional): Stability factor. Defaults to 1e-10.

    Returns:
        np.ndarray: Entropy array. Shape [H,W,D]

    """
    log_probs = -np.log(prob_array + epsilon)
    return np.sum(prob_array * log_probs, axis=-1)


def voxels_uncertainty(prob_array: np.ndarray, epsilon: float = 1e-10) -> dict:
    """Compute voxel scale uncertainty measures.

    Args:
        prob_array (np.ndarray): Probability array. Shape [H,W,D,C].
        epsilon (float, optional): Stability factor. Defaults to 1e-10.

    Returns:
        dict: Dictionary of uncertainties.

    """
    mean_probs = prob_array
    conf = np.max(mean_probs, axis=-1)

    eoe = entropy_of_expected(prob_array, epsilon)

    return {
        "neg_confidence": -1 * conf,
        "entropy_of_expected": eoe,
    }
