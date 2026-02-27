# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Post processing functions -*-

import numpy as np
from scipy.ndimage import binary_dilation, label
from skimage.morphology import remove_small_objects


def clean_small_objects(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    object_size: int = 5,
    connectivity: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean gt segmentation and prediction from small objects.

    Args:
        pred_mask (np.ndarray): The mask prediction.
        gt_mask (np.ndarray): The ground truth prediction.
        object_size (int): The minimum size of objects. Default to 5.
        connectivity (int): The connectivity for the binary pattern. Default to 2.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - the corrected prediction mask
            - the corrected gt segmentation

    """
    pred_mask_post = remove_small_objects(
        pred_mask.astype(bool), min_size=object_size, connectivity=connectivity
    ).astype(np.uint8)
    if gt_mask is not None:
        gt_mask_post = remove_small_objects(
            gt_mask.astype(bool), min_size=object_size, connectivity=connectivity
        ).astype(np.uint8)

        return pred_mask_post, gt_mask_post

    return pred_mask_post


def wm_restriction(
    pred_mask: np.ndarray,
    pred_proba: np.ndarray,
    wm_mask: np.ndarray,
    csf_mask: np.ndarray,
    dilation_iter: int,
    lesion_frac: float,
) -> np.ndarray:
    """Restrict lesion to White Matter only.

    Args:
        pred_mask (np.ndarray): The mask prediction
        pred_proba (np.ndarray): The probability mask
        wm_mask (np.ndarray): The white matter mask
        csf_mask (np.ndarray): The csf mask
        dilation_iter (int): The number of iteration to dilate the white matter mask
        lesion_frac (float): The minimum fraction of the lesion to be inside the wm mask
          to be kept

    Returns:
        np.ndarray: the processed prediction mask

    """
    # ---- Dilate WM mask to overcome imperfection of Neurom mask ----
    wm_dilated = binary_dilation(wm_mask.astype(bool), iterations=dilation_iter)
    wm_dilated = np.logical_and(wm_dilated, np.logical_not(csf_mask))

    labels, num_features = label(pred_mask)
    mask_post_process = np.zeros_like(pred_mask)

    # Process lesions and remove if voxel fraction outside WM > lesion_frac ----
    for i in range(1, num_features + 1):
        component = labels == i
        voxels_in_wm = np.logical_and(component, wm_dilated).sum()
        if voxels_in_wm > 0:
            if voxels_in_wm / component.sum() < lesion_frac:
                continue
            mask_post_process = np.logical_or(mask_post_process, component)

    pred_proba[~np.logical_or(wm_dilated, mask_post_process)] = 0

    return mask_post_process, pred_proba
