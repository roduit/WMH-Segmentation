# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Post processing functions -*-

import numpy as np
from scipy.ndimage import binary_dilation, label
from skimage.morphology import remove_small_objects


def clean_small_objects(pred_mask: np.ndarray, object_size: int) -> np.ndarray:
    """Clean manual segmentation and prediction from small objects.

    Args:
        pred_mask (np.ndarray): The mask prediction
        object_size (int): The minimum size of objects

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - the corrected prediction mask
            - the corrected manual segmentation

    """
    return remove_small_objects(
        pred_mask.astype(bool), max_size=object_size - 1
    ).astype(np.uint8)


def wm_restriction(
    pred_mask: np.ndarray,
    pred_proba_mask: np.ndarray,
    wm_mask: np.ndarray,
    csf_mask: np.ndarray,
    dilation_iter: int,
    lesion_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict lesion to White Matter only.

    Args:
        pred_mask (np.ndarray): The mask prediction
        pred_proba_mask (np.ndarray): The probability mask
        wm_mask (np.ndarray): The white matter mask
        csf_mask (np.ndarray): The csf mask
        dilation_iter (int): The number of iteration to dilate the white matter mask
        lesion_frac (float): The minimum fraction of the lesion to be inside the wm mask
          to be kept

    Returns:
        np.ndarray: the processed prediction mask

    """
    wm_dilated = binary_dilation(wm_mask.astype(bool), iterations=dilation_iter)
    wm_dilated = np.logical_and(wm_dilated, np.logical_not(csf_mask))

    labels, num_features = label(pred_mask)
    mask_post_process = np.zeros_like(pred_mask)
    mask_proba_post_process = np.zeros_like(pred_proba_mask)
    for i in range(1, num_features + 1):
        component = labels == i
        voxels_in_wm = np.logical_and(component, wm_dilated).sum()
        if voxels_in_wm > 0:
            if voxels_in_wm / component.sum() <= lesion_frac:
                mask_proba_post_process[component] = 0
                continue
            mask_post_process = np.logical_or(mask_post_process, component)

    return mask_post_process, mask_proba_post_process


def remove_csf_overlap(
    pred_mask: np.ndarray, csf_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the overlap with the csf mask.

    Args:
        pred_mask (np.ndarray): The mask prediction
        gt_mask (np.ndarray): The ground truth prediction
        csf_mask (np.ndarray): The csf mask

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - the corrected prediction mask
            - the corrected manual segmentation

    """
    return np.logical_and(pred_mask, np.logical_not(csf_mask))


def clean_pv_area(
    pred_mask: np.ndarray, csf_mask: np.ndarray, dilation_iter: int, lesion_frac: float
) -> np.ndarray:
    """Clean the periventricular area.

    Args:
        pred_mask (np.ndarray): The prediction mask
        csf_mask (np.ndarray): The csf mask
        dilation_iter (int): the number of dilation iteration
        lesion_frac (float): the fraction of the lesion to be inside the csf in order to
          remove it

    Returns:
        np.ndarray: The corrected prediction mask

    """
    labels, num_features = label(pred_mask)
    pred_mask_post = np.zeros_like(pred_mask)

    extended_csf = binary_dilation(csf_mask, iterations=dilation_iter)
    for i in range(1, num_features + 1):
        component = labels == i
        voxels_in_csf = np.logical_and(component, extended_csf).sum()
        if voxels_in_csf / component.sum() > lesion_frac:
            continue

        pred_mask_post = np.logical_or(pred_mask_post, component)

    return pred_mask_post
