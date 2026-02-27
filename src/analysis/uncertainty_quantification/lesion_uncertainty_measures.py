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
# -*- Description: Functions to calculate lesion uncertainty measures-*-

from functools import partial

import numpy as np
from joblib import Parallel, delayed

vox_uncs_measures = ["neg_confidence", "entropy_of_expected"]
les_uncs_measures = []
for vum in vox_uncs_measures:
    if vum != "neg_confidence":
        les_uncs_measures.append(f"logsum {vum}")
    les_uncs_measures.append(f"mean {vum}")


def single_lesion_uncertainty(
    cc_mask: np.ndarray, vox_unc_maps: dict, epsilon: float = 1e-10
) -> dict:
    """Compute different uncertainty measures for a single connected component.

    The uncertainty measures are:
        - Mean from vox uncertainty (M. Dojat)
        - Log-sum from vox uncertainty (T. Alber)
        - Detection disagreement uncertainty (N. Molchanova)

    Args:
        cc_mask (np.ndarray): The labeled manual segmentation. Each connected component
          (i.e. lesion) should have a different label. Shape [H,W,D].
        vox_unc_maps (dict): The dictionary containing the voxel uncertainty maps.
        epsilon (float, optional): Stability factor. Defaults to 1e-10.

    Returns:
        dict: Dictionary of lesion uncertainty measures names and values.

    """
    res = {}

    # ---- Voxel-scale based uncertainties ----
    for unc_measure, vox_unc_map in vox_unc_maps.items():
        # ---- Log is not defined for negative values ----
        if unc_measure == "neg_confidence":
            res.update(
                {
                    f"mean {unc_measure}": np.sum(vox_unc_map * cc_mask)
                    / np.sum(cc_mask),
                }
            )
        else:
            res.update(
                {
                    f"mean {unc_measure}": np.sum(vox_unc_map * cc_mask)
                    / np.sum(cc_mask),
                    f"logsum {unc_measure}": np.sum(
                        np.sum(np.log(vox_unc_map[cc_mask == 1] + epsilon))
                    ),
                }
            )

    return res


def lesions_uncertainty(
    y_pred_multi: np.ndarray, vox_unc_maps: dict, n_jobs: int | None = None
) -> dict:
    """Parallel evaluation of all lesion scale uncertainty maps for one subject.

    Args:
        y_pred_multi (np.ndarray): Prediction array with different label
          for each lesion (i.e. connected component).
        vox_unc_maps (dict): dict with keys being names of voxel-scale uncertainty
          measures and values being corresponding 3D uncertainty maps of shape [H, W, D]
        n_jobs (int | None, optional): Number of job to use. Defaults to None.

    Returns:
        dict: dictionary containing lesions uncertainty for different measures.

    """

    def ld_to_dl(ld: list) -> dict:
        keys = list(ld[0].keys())
        dl = dict(zip(keys, [[] for _ in keys], strict=False))
        for el in ld:
            for k in keys:
                v = el[k]
                dl[k].append(v)
        return dl

    cc_labels = np.unique(y_pred_multi)
    cc_labels = cc_labels[cc_labels != 0.0]

    # -------------------------------
    # Parallel process of lesions
    # -------------------------------
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        process = partial(single_lesion_uncertainty, vox_unc_maps=vox_unc_maps)
        les_uncs_list = parallel_backend(
            delayed(process)(cc_mask=(y_pred_multi == cc_label).astype("float"))
            for cc_label in cc_labels
        )

    if not les_uncs_list:
        return dict(
            zip(les_uncs_measures, [[] for _ in les_uncs_measures], strict=False)
        )

    return ld_to_dl(les_uncs_list)


def lesions_uncertainty_maps(
    y_pred_multi: np.ndarray, vox_unc_maps: dict, n_jobs: int | None = None
) -> dict:
    """Parallel evaluation of all lesion scale uncertainty maps for one subject.

    This function computes the lesion uncertainty map for a given voxel uncertainty map.

    Args:
        y_pred_multi (np.ndarray): labeled lesion mask aka instance segmentation mask.
        vox_unc_maps (np.ndarray): Voxel scale lesion uncertainty mask.
        n_jobs (int | None, optional): Number of parallel workers. Defaults to None.

    Returns:
        dict: dictionary with lesion uncertainty maps for each measure

    """
    cc_labels = np.unique(y_pred_multi)
    cc_labels = cc_labels[cc_labels != 0.0]
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        process = partial(single_lesion_uncertainty, vox_unc_maps=vox_unc_maps)
        les_uncs_list = parallel_backend(
            delayed(process)(cc_mask=(y_pred_multi == cc_label).astype("float"))
            for cc_label in cc_labels
        )  # returns lists of dictionaries, but need dictionary of lists

    if les_uncs_list:
        results = dict(
            zip(
                les_uncs_measures,
                [np.zeros_like(y_pred_multi, dtype="float") for _ in les_uncs_measures],
                strict=True,
            )
        )
        # for each measure of uncertainty create a lesion uncertainty map
        for cc_label, uncs_dict in zip(cc_labels, les_uncs_list, strict=True):
            for unc_name in les_uncs_measures:
                results[unc_name] += uncs_dict[unc_name] * (
                    y_pred_multi == cc_label
                ).astype("float")
        return results

    return None
