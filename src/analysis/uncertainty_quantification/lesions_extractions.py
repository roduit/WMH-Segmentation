# -*- authors : Medical Image Analysis Laboratory -*-
# -*- date : 2024-10-23 -*-
# -*- Last revision: 2025-12-24 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to extract the lesion type (TP;FP;TN;FN)-*-

from functools import partial

import numpy as np
from joblib import Parallel, delayed

from analysis.uncertainty_quantification.utils import (
    get_cc_mask,
    intersection_over_union,
)


def get_lesion_types(
    y_pred: np.ndarray,
    y: np.ndarray,
    iou_threshold: float = 0.25,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Get lesion types.

    A Lesion type is either TP,FP,FN or TN. In this context, only TP of FP are relevant.

    Args:
        y_pred (np.ndarray): Prediction array.
        y (np.ndarray): Ground truth array.
        iou_threshold (float, optional): Threhold to consider a lesion as tp.
          Defaults to 0.25.
        n_jobs (int, optional): number of parallel processes. Defaults to None.

    Returns:
        np.array: Array containing tp or fp values for each lesion.

    """
    gt_multi_mask = get_cc_mask(y)
    pred_multi_mask = get_cc_mask(y_pred)
    n_pred_labels = len(np.unique(pred_multi_mask))

    def get_tp_fp(label_pred: int, gt_multi: np.ndarray, pred_multi: np.ndarray) -> str:
        lesion_pred = (pred_multi == label_pred).astype(float)
        max_iou = 0.0
        for label_gt in np.unique(gt_multi * lesion_pred):
            if label_gt != 0.0:
                mask_label_gt = (gt_multi == label_gt).astype(int)
                iou = intersection_over_union(lesion_pred, mask_label_gt)
                max_iou = max(max_iou, iou)
        return "tp" if max_iou >= iou_threshold else "fp"

    process_tp_fn = partial(
        get_tp_fp, gt_multi=gt_multi_mask, pred_multi=pred_multi_mask
    )

    with Parallel(n_jobs=n_jobs) as parallel:
        return parallel(
            delayed(process_tp_fn)(label) for label in range(1, n_pred_labels)
        )
