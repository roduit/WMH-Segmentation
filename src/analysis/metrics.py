#    Copyright 2020 Division of Medical Image Computing, German Cancer Research
#    Center (DKFZ), Heidelberg, Germany.
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# -*- authors : DKFZ -*-
# -*- date : 2020 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Metric functions -*-

import logging
import math

import numpy as np
import pandas as pd
from medpy import metric
from scipy.ndimage import binary_dilation
from scipy.spatial import distance
from surface_distance import compute_surface_distances

from misc.constants import DATASET, FAZEKAS, REGION, REGION_NAME

logger = logging.getLogger(__name__)


def build_weighted_map(
    mask: np.ndarray, weights: tuple, structure: np.ndarray | None = None
) -> np.ndarray:
    """Build a weighted map using successive dilations.

    Args:
        mask (np.ndarray): The mask to dilate. Shape [H,W,D]
        weights (tuple): The weights for the successive dilations.
        structure (np.ndarray | None, optional): The binary structure to use.
          Defaults to None.

    Returns:
        np.ndarray: The mask containing weights for the succesive dilations.
          Shape [H,W,D]

    """
    weighted = np.zeros_like(mask, dtype=np.float32)

    current = mask.astype(bool)
    weighted[current] = weights[1]

    for w in weights[1:]:
        dilated = binary_dilation(current, structure=structure)
        shell = dilated & (~current)
        weighted[shell] = w
        current = dilated

    return weighted


def weighted_dice_coefficient(
    test: np.ndarray,
    reference: np.ndarray,
    threshold: float | None = None,  # noqa: ARG001
    weights: tuple = (1.0, 0.7, 0.5, 0.3),
    structure: np.ndarray | None = None,
    eps: float = 1e-6,
) -> float:
    """Weighted Dice Coefficient (WDC) from the paper.

    Args:
        test (np.ndarray): The prediction mask. Shape [H,W,D]
        reference (np.ndarray): The ground truth mask. Shape [H,W,D]
        threshold (float | None): The threshold to use. This argument is not use in this
          function but needed to be consistent with the other metrics.
        weights (tuple, optional): The weights of the successive dilations.
          Defaults to (1.0, 0.7, 0.5, 0.3).
        structure (np.ndarray | None, optional): The binary structure to use for
          dilation. Defaults to None.
        eps (float, optional): Stability factor. Defaults to 1e-6.

    Returns:
        float: The Weighted Dice Coefficient

    References:
        .. [1] https://link.springer.com/article/10.1007/s10278-025-01535-1


    """
    s_x = build_weighted_map(reference, weights, structure)
    s_y = build_weighted_map(test, weights, structure)

    intersection = np.minimum(s_x, s_y).sum()
    normalization = s_x.sum() + s_y.sum()

    return (2 * intersection + eps) / (normalization + eps)


def assert_shape(test: np.ndarray, reference: np.ndarray) -> None:
    """Assert the shape of two arrays.

    Args:
        test (np.ndarray): test array
        reference (np.ndarray): reference array

    Raises:
        AssertionError: if shapes do not match

    """
    if test.shape != reference.shape:
        msg = f"Shape mismatch: {test.shape} and {reference.shape}"
        raise ValueError(msg)


class ConfusionMatrix:
    def __init__(
        self,
        test: np.ndarray = None,
        reference: np.ndarray = None,
        voxel_spacing: tuple = (1.0, 1.0, 1.0),
        threshold: float | None = None,
    ) -> None:
        """Confusion matrix for binary classification.

        Args:
            test (np.ndarray, optional): The predicted segmentation. Defaults to None.
            reference (np.ndarray, optional): The ground truth. Defaults to None.
            voxel_spacing (tuple, optional): The voxel spacing, given as a tuple.
              Defaults to None.
            threshold (float, optional): The thresold used for small lesions (in mm).
              Defaults to None.

        """
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_small = None
        self.reference_full = None
        self.test_empty = None
        self.test_small = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)
        self.set_voxel_spacing(voxel_spacing)
        self.set_threshold(threshold)

    def set_test(self, test: np.ndarray) -> None:
        """Set the test array.

        Args:
            test (np.ndarray): The predicted segmentation.

        """
        self.test = test
        self.reset()

    def set_reference(self, reference: np.ndarray) -> None:
        """Set the reference array.

        Args:
            reference (np.ndarray): The ground truth segmentation.

        """
        self.reference = reference
        self.reset()

    def set_voxel_spacing(self, voxel_spacing: tuple) -> None:
        """Set the voxel spacing.

        Args:
            voxel_spacing (tuple): The voxel spacing

        """
        self.voxel_spacing = voxel_spacing
        self.reset()

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold.

        Args:
            threshold (float): The threshold value

        """
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset the ConfusionMatrix statistics."""
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_small = None
        self.reference_full = None
        self.test_empty = None
        self.test_small = None
        self.test_full = None

    def compute(self) -> None:
        """Compute the confusion matrix.

        Raises:
            ValueError: Raised if 'test' or 'reference' is None.

        """
        if self.test is None or self.reference is None:
            msg = "'test' and 'reference' must both be set to compute confusion matrix."
            raise ValueError(msg)

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

        if isinstance(self.threshold, float):
            # compute volume of test and reference
            voxel_volume = math.prod(self.voxel_spacing)
            volume_tes = (self.tp + self.fp) * voxel_volume * 0.001
            volume_ref = (self.tp + self.fn) * voxel_volume * 0.001
            self.test_small = not volume_tes > self.threshold
            self.reference_small = not volume_ref > self.threshold
        else:
            self.test_small = self.test_empty
            self.reference_small = self.reference_empty

    def get_matrix(self) -> tuple[int, int, int, int]:
        """Get the confusion matrix values.

        Retunrs:
            tuple[int, int, int, int]: (tp, fp, tn, fn)
        """
        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self) -> int:
        """Get the size of the confusion matrix.

        Returns:
            int: size of the confusion matrix.

        """
        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self) -> tuple[bool, bool, bool, bool]:
        """Get existence flags for test and reference.

        Returns:
            tuple[bool, bool, bool, bool]: a tuple containing
                - test_empty (bool): True if test is empty
                - test_full (bool): True if test is full
                - reference_empty (bool): True if reference is empty
                - reference_full (bool): True if reference is full

        """
        for case in (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        ):
            if case is None:
                self.compute()
                break

        return (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        )

    def get_thresholded(self) -> tuple[bool, bool]:
        """Get thresholded existence flags.

        Returns:
            tuple[bool, bool]: (test_small, reference_small)

        """
        for case in (self.test_small, self.reference_small):
            if case is None:
                self.compute()
                break

        return self.test_small, self.reference_small


def dice_orig(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute the original Dice similarity coefficient.

    The Dice coefficient (also called F1 score) measures the overlap between
    predicted and reference segmentations. Formula: 2TP / (2TP + FP + FN).
    Range: [0, 1] where 1 is perfect overlap and 0 is no overlap.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if both segmentations are
          empty. Defaults to False.

    Returns:
        float: Dice coefficient in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        _test_full,
        reference_empty,
        _reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        return 1.0

    return float(2.0 * tp / (2 * tp + fp + fn))


def dice_th(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    nan_for_nonexisting: bool = False,
    threshold: float | None = None,
) -> float:
    """Compute the Dice similarity coefficient with thresholding.

    Similar to dice_orig but handles small lesions based on volume threshold.
    Objects smaller than the threshold are excluded from the calculation.
    Formula: 2TP / (2TP + FP + FN + 1e-8).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
         Defaults to False.
        threshold (float, optional): Volume threshold in ml for small lesions. Defaults
          to None.

    Returns:
        float: Thresholded Dice coefficient in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(2.0 * tp / (2 * tp + fp + fn + 1e-8))


def jaccard_th(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute the Jaccard index (Intersection over Union) with thresholding.

    The Jaccard index measures the ratio of intersection to union of two sets.
    Formula: TP / (TP + FP + FN + 1e-8). Excludes small lesions based on threshold.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.

    Returns:
        float: Jaccard index in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, threshold=threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(tp / (tp + fp + fn + 1e-8))


def precision(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute precision (positive predictive value).

    Precision measures the fraction of predicted positives that are actually correct.
    Formula: TP / (TP + FP + 1e-8). High precision means few false positives.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.

    Returns:
        float: Precision in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, _fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(tp / (tp + fp + 1e-8))


def sensitivity(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute sensitivity (recall, true positive rate).

    Sensitivity measures the fraction of actual positives correctly identified.
    Formula: TP / (TP + FN + 1e-8). High sensitivity means few false negatives.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.

    Returns:
        float: Sensitivity in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, _fp, _tn, fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(tp / (tp + fn + 1e-8))


def recall(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute recall (synonym for sensitivity).

    Recall measures the fraction of actual positives correctly identified.
    Formula: TP / (TP + FN + 1e-8). Equivalent to sensitivity.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.

    Returns:
        float: Recall in range [0, 1].

    """
    return sensitivity(
        test,
        reference,
        confusion_matrix,
        threshold,
        voxel_spacing,
        nan_for_nonexisting,
    )


def specificity(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute specificity (true negative rate).

    Specificity measures the fraction of actual negatives correctly identified.
    Formula: TN / (TN + FP). High specificity means few false positives.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if reference is fully positive.
          Defaults to True.

    Returns:
        float: Specificity in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    _tp, fp, tn, _fn = confusion_matrix.get_matrix()
    (
        _test_empty,
        _test_full,
        _reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(tn / (tn + fp))


def accuracy(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> float:
    """Compute accuracy (fraction of correct predictions).

    Accuracy measures the fraction of all predictions (both positive and negative)
      that are correct.
    Formula: (TP + TN) / (TP + FP + FN + TN). Useful for balanced datasets.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        float: Accuracy in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
    beta: float = 1.0,
) -> float:
    """Compute F-beta score (harmonic mean of precision and recall).

    The F-score is the weighted harmonic mean of precision and recall.
    Formula: (1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP).
    Default beta=1.0 gives equal weight to precision and recall (F1 score).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.
        beta (float, optional): Beta parameter controlling precision/recall tradeoff.
          Defaults to 1.0.

    Returns:
        float: F-score in range [0, 1].

    """
    precision_ = precision(
        test, reference, confusion_matrix, threshold, voxel_spacing, nan_for_nonexisting
    )
    recall_ = recall(
        test, reference, confusion_matrix, threshold, voxel_spacing, nan_for_nonexisting
    )

    return (
        (1 + beta * beta)
        * precision_
        * recall_
        / ((beta * beta * precision_) + recall_ + 1e-8)
    )


def false_positive_rate(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute false positive rate (1 - specificity).

    False positive rate measures the fraction of actual negatives incorrectly predicted
      as positive.
    Formula: FP / (FP + TN) = 1 - Specificity. Useful for ROC curve analysis.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if reference is fully positive.
          Defaults to True.

    Returns:
        float: False positive rate in range [0, 1].

    """
    return 1 - specificity(
        test, reference, confusion_matrix, voxel_spacing, threshold, nan_for_nonexisting
    )


def false_omission_rate(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute false omission rate (FN / (TN + FN)).

    False omission rate measures the fraction of predicted negatives that are actually
      incorrect.
    Formula: FN / (FN + TN). Complement of negative predictive value.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if prediction is fully
          positive. Defaults to True.

    Returns:
        float: False omission rate in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    _tp, _fp, tn, fn = confusion_matrix.get_matrix()
    (
        _test_empty,
        test_full,
        _reference_empty,
        _reference_full,
    ) = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(fn / (fn + tn))


def false_negative_rate(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute false negative rate (1 - sensitivity).

    False negative rate measures the fraction of actual positives incorrectly predicted
      as negative.
    Formula: FN / (TP + FN) = 1 - Sensitivity. Useful for ROC curve analysis.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if reference is fully negative.
          Defaults to True.

    Returns:
        float: False negative rate in range [0, 1].

    """
    return 1 - sensitivity(
        test, reference, confusion_matrix, threshold, voxel_spacing, nan_for_nonexisting
    )


def true_negative_rate(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    nan_for_nonexisting: bool = False,
    threshold: float | None = None,  # noqa: ARG001
) -> float:
    """Compute true negative rate (synonym for specificity).

    True negative rate measures the fraction of actual negatives correctly identified.
    Formula: TN / (TN + FP). Equivalent to specificity.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        nan_for_nonexisting (bool, optional): Return NaN if reference is fully positive.
          Defaults to True.
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        float: True negative rate in range [0, 1].

    """
    return specificity(
        test, reference, confusion_matrix, voxel_spacing, nan_for_nonexisting
    )


def false_discovery_rate(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute false discovery rate (1 - precision).

    False discovery rate measures the fraction of predicted positives that are actually
      incorrect.
    Formula: FP / (TP + FP) = 1 - Precision. Lower values indicate better precision.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.

    Returns:
        float: False discovery rate in range [0, 1].

    """
    return 1 - precision(
        test, reference, confusion_matrix, threshold, voxel_spacing, nan_for_nonexisting
    )


def negative_predictive_value(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    threshold: float | None = None,  # noqa: ARG001
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute negative predictive value (1 - false omission rate).

    Negative predictive value measures the fraction of predicted negatives that are
      actually correct.
    Formula: TN / (TN + FN) = 1 - False Omission Rate. Complement of false omission
      rate.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if prediction is fully
          positive. Defaults to True.

    Returns:
        float: Negative predictive value in range [0, 1].

    """
    return 1 - false_omission_rate(
        test, reference, confusion_matrix, nan_for_nonexisting
    )


def total_positives_test(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Compute total positives in prediction (TP + FP).

    Counts the total number of voxels predicted as positive.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: Total number of predicted positive voxels.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, _fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Compute total negatives in prediction (TN + FN).

    Counts the total number of voxels predicted as negative.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: Total number of predicted negative voxels.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    _tp, _fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Compute total positives in reference (TP + FN).

    Counts the total number of voxels that are positive in ground truth.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: Total number of positive voxels in ground truth.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, _fp, _tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Compute total negatives in reference (TN + FP).

    Counts the total number of voxels that are negative in ground truth.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: Total number of negative voxels in ground truth.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    _tp, fp, tn, _fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance_95(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
    connectivity: int = 1,
) -> float:
    """Compute the 95th percentile Hausdorff distance between surfaces.

    The Hausdorff distance measures the maximum distance between surface boundaries.
    The 95th percentile is used to reduce sensitivity to outliers. Smaller distances
    indicate better spatial overlap. Returns NaN or 0 for small lesions.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.
        connectivity (int, optional): Connectivity for distance computation (1 or 2).
          Defaults to 1.

    Returns:
        float: 95th percentile Hausdorff distance in mm.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0

    return metric.hd95(
        confusion_matrix.test,
        confusion_matrix.reference,
        confusion_matrix.voxel_spacing,
        connectivity,
    )


def avg_surface_distance(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
    connectivity: int = 1,
) -> float:
    """Compute average surface distance (directed).

    Measures the average distance from prediction surface to reference surface.
    This is directional: it only measures how far prediction deviates from reference.
    Use avg_surface_distance_symmetric for symmetric distance.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.
        connectivity (int, optional): Connectivity for distance computation (1 or 2).
          Defaults to 1.

    Returns:
        float: Average surface distance in mm.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0
    try:
        return metric.asd(
            confusion_matrix.test,
            confusion_matrix.reference,
            confusion_matrix.voxel_spacing,
            connectivity,
        )
    except OSError:
        return float("NaN")


def avg_surface_distance_symmetric(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
    connectivity: int = 1,
) -> float:
    """Compute symmetric average surface distance.

    Measures the average distance between prediction and reference surfaces in both
      directions. More fair metric than directed distance as it penalizes both over
      and under segmentation equally.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.
        connectivity (int, optional): Connectivity for distance computation (1 or 2).
          Defaults to 1.

    Returns:
        float: Symmetric average surface distance in mm.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, confusion_matrix.voxel_spacing, connectivity)


def compute_surface_dice_at_tolerance_list(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
    tolerance_list: list | None = None,
) -> list:
    """Compute surface DICE coefficients at multiple tolerance levels.

    Computes the surface DICE coefficient (not volumetric DICE) at specified tolerances.
    Measures overlap of two surfaces where a surface element counts as overlapping when
    the closest distance to the other surface is ≤ tolerance. More robust than
    volumetric DICE for surface-based metrics.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.
        tolerance_list (list, optional): Tolerances in mm for surface overlap.
          Defaults to [2, 5, 10].

    Returns:
        list: Surface DICE coefficients for each tolerance level, range [0, 1].

    """
    if tolerance_list is None:
        tolerance_list = [2, 5, 10]
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return [float("NaN") for x in tolerance_list]
        return [0 for x in tolerance_list]

    test, reference = confusion_matrix.test, confusion_matrix.reference
    surface_distances = compute_surface_distances(
        test, reference, confusion_matrix.voxel_spacing
    )
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    surfel_areas_pred_sum = np.sum(surfel_areas_pred)
    surfel_areas_gt_sum = np.sum(surfel_areas_gt)

    return [
        (
            (
                np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance])
                + np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance])
            )
            / (surfel_areas_gt_sum + surfel_areas_pred_sum)
        )
        for tolerance in tolerance_list
    ]


def compute_surface_jaccard_at_tolerance_list(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
    tolerance_list: list | None = None,
) -> list:
    """Compute surface Jaccard indices (IoU) at multiple tolerance levels.

    Computes the surface Jaccard index (intersection over union) at specified
      tolerances. Similar to compute_surface_dice_at_tolerance_list but uses Jaccard
      metric instead of DICE. Surface-based metric measuring boundary overlap rather
      than volume overlap.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.
        tolerance_list (list, optional): Tolerances in mm for surface overlap.
          Defaults to None.

    Returns:
        list: Surface Jaccard indices for each tolerance level, range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small or test_small:
        if nan_for_nonexisting:
            return [float("NaN") for x in tolerance_list]
        return [0 for x in tolerance_list]

    test, reference = confusion_matrix.test, confusion_matrix.reference
    surface_distances = compute_surface_distances(
        test, reference, confusion_matrix.voxel_spacing
    )
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    surfel_areas_pred_sum = np.sum(surfel_areas_pred)
    surfel_areas_gt_sum = np.sum(surfel_areas_gt)

    return [
        (
            np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance])
            / (
                surfel_areas_gt_sum
                + surfel_areas_pred_sum
                - np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance])
            )
        )
        for tolerance in tolerance_list
    ]


def malahanobis_distance(
    test: np.ndarray | None = None, reference: np.ndarray | None = None
) -> float:
    """Compute Mahalanobis distance between two segmentations.

    Mahalanobis distance accounts for correlations between variables and uses
      the inverse covariance matrix to scale distances. Useful for multivariate
      comparison.

    Args:
        test (np.ndarray, optional): Predicted segmentation (flattened).
          Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation (flattened).
          Defaults to None.

    Returns:
        float: Mahalanobis distance between segmentations.

    """
    covariance = np.cov([test.ravel(), reference.ravel()])
    cov_inv = np.linalg.inv(covariance)  # inv. covariance matrix
    return distance.mahalanobis(test, reference, cov_inv)


def volume_test(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> float:
    """Compute volume of predicted segmentation in mm³.

    Calculates the total volume of voxels predicted as positive.
    Formula: (TP + FP) * voxel_volume * 0.001 (converts to mm³).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        float: Volume in mm³.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, _fn = confusion_matrix.get_matrix()

    voxel_volume = math.prod(confusion_matrix.voxel_spacing)

    return (tp + fp) * voxel_volume * 0.001


def volume_reference(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> float:
    """Compute volume of ground truth segmentation in mm³.

    Calculates the total volume of voxels that are positive in ground truth.
    Formula: (TP + FN) * voxel_volume * 0.001 (converts to mm³).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        float: Volume in mm³.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, _fp, _tn, fn = confusion_matrix.get_matrix()

    voxel_volume = math.prod(confusion_matrix.voxel_spacing)

    return float((tp + fn) * voxel_volume * 0.001)


def abs_volume_difference(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute absolute volume difference between prediction and reference in mm³.

    Measures how much the predicted volume deviates from ground truth volume,
      ignoring sign.
    Formula: |volume_test - volume_reference| in mm³.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if both segmentations
          are too small. Defaults to True.

    Returns:
        float: Absolute volume difference in mm³.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    voxel_volume = math.prod(confusion_matrix.voxel_spacing)

    if reference_small and test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(abs((tp + fn) - (tp + fp)) * voxel_volume * 0.001)


def volume_difference(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute signed volume difference between prediction and reference in mm³.

    Measures how much the predicted volume deviates from ground truth volume,
      preserving sign.
    Positive values indicate over-segmentation, negative indicate under-segmentation.
    Formula: (volume_test - volume_reference) in mm³.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if both segmentations
          are too small. Defaults to True.

    Returns:
        float: Signed volume difference in mm³.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    voxel_volume = math.prod(confusion_matrix.voxel_spacing)

    if reference_small and test_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(((tp + fn) - (tp + fp)) * voxel_volume * 0.001)


def rel_volume_difference(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute relative volume difference between prediction and reference.

    Measures the volume difference normalized by reference volume.
    Formula: |volume_test - volume_reference| / volume_reference.
    Range: [0, ∞] where 0 is perfect agreement.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if reference is too small.
          Defaults to True.

    Returns:
        float: Relative volume difference (dimensionless).

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    _test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(abs((tp + fn) - (tp + fp)) / (tp + fn))


def volumetric_similarity(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute volumetric similarity between prediction and reference.

    Measures similarity between segmentation volumes, penalizing both over and under
      segmentation.
    Formula: 1 - |FN - FP| / (2TP + FP + FN).
    Range: [0, 1] where 1 indicates perfect volumetric agreement.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if segmentations are too small.
          Defaults to False.

    Returns:
        float: Volumetric similarity in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, fn = confusion_matrix.get_matrix()
    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small ^ (reference_small and test_small):
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return float(1 - (abs(fn - fp) / (2 * tp + fp + fn + 1e-8)))


def detection(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Compute correct classification (image-level binary label).

    Returns 1 if prediction and reference agree on lesion presence
      (both large or both small),
    0 otherwise. Used for image-level classification metrics.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: 1 if classifications match (both have/lack lesion), 0 otherwise.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if (reference_small and test_small) or (
        not reference_small and not test_small
    ):  # images classified true positive
        return 1
    return 0


def detection_tp(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Detect true positives at image level.

    Returns 1 if both prediction and reference correctly identify presence of
      large lesion.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: 1 if both predict large lesion presence, 0 otherwise.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if not reference_small and not test_small:  # images classified true positive
        return 1
    return 0


def detection_tn(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Detect true negatives at image level.

    Returns 1 if both prediction and reference correctly identify absence of large
      lesion (both small/empty).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: 1 if both predict small/no lesion, 0 otherwise.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small and test_small:  # images classified true negative
        return 1
    return 0


def detection_fp(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Detect false positives at image level.

    Returns 1 if prediction incorrectly identifies large lesion when reference
      has small/no lesion.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: 1 if prediction has large lesion but reference doesn't, 0 otherwise.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small and not test_small:  # images classified false positive
        return 1
    return 0


def detection_fn(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Detect false negatives at image level.

    Returns 1 if prediction incorrectly identifies small/no lesion when reference
      has large lesion.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: 1 if reference has large lesion but prediction doesn't, 0 otherwise.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    test_small, reference_small = confusion_matrix.get_thresholded()

    if not reference_small and test_small:  # images classified false negative
        return 1
    return 0


def ldr(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int | float | None:
    """Compute Lesion Detection Rate (LDR).

    Binary indicator for whether lesion is successfully detected at image level.
    Returns 1 if prediction volume exceeds threshold, 0 if below threshold, NaN if
      reference is too small.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml for lesion detection.
          Defaults to None.

    Returns:
        int or float: 1 if lesion detected, 0 if not detected, NaN if reference
          too small, None if uncertain.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, _tn, _fn = confusion_matrix.get_matrix()

    test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small:
        return float("NaN")
    if (tp + fp) > confusion_matrix.threshold:
        return 1
    if test_small:
        return 0
    return None


def class_imbalance_alpha(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute class imbalance ratio (positive class prevalence).

    Measures the fraction of positive voxels in the entire image including both
    positive and negative regions. Useful for understanding dataset balance.
    Only for comparison within datasets with constant voxel counts.
    Formula: (TP + FN) / (TP + FP + TN + FN).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if reference is empty.
          Defaults to True.

    Returns:
        float: Class imbalance ratio (fraction of positive voxels) in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        _test_empty,
        _test_full,
        reference_empty,
        _reference_full,
    ) = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return (tp + fn) / (tn + fp + tn + fn)


def relative_tp(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    nan_for_nonexisting: bool = False,
) -> float:
    """Compute relative true positive rate (TP normalized by all voxels).

    Fraction of all voxels in the image that are correctly predicted as positive.
    Formula: TP / (TP + TN + FN + FP) = TP / total_voxels.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.
        nan_for_nonexisting (bool, optional): Return NaN if reference is too small.
          Defaults to True.

    Returns:
        float: Relative TP rate in range [0, 1].

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    _test_small, reference_small = confusion_matrix.get_thresholded()

    if reference_small:
        if nan_for_nonexisting:
            return float("NaN")
        return 0.0

    return tp / (tp + tn + fn + fp)


def tp(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> int:
    """Compute true positive count (absolute number of correct positive voxels).

    Returns the total number of voxels correctly predicted as positive.

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: Number of true positive voxels.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    tp, _fp, _tn, _fn = confusion_matrix.get_matrix()

    return tp


def fp(
    test: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    confusion_matrix: ConfusionMatrix | None = None,
    voxel_spacing: tuple | None = None,
    threshold: float | None = None,
) -> int:
    """Compute false positive count.

    Returns the total number of voxels incorrectly predicted as positive (false alarms).

    Args:
        test (np.ndarray, optional): Predicted segmentation. Defaults to None.
        reference (np.ndarray, optional): Ground truth segmentation. Defaults to None.
        confusion_matrix (ConfusionMatrix, optional): Pre-computed confusion matrix.
          Defaults to None.
        voxel_spacing (tuple, optional): Voxel spacing in mm. Defaults to (1.0,1.0,1.0).
        threshold (float, optional): Volume threshold in ml. Defaults to None.

    Returns:
        int: Number of false positive voxels.

    """
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference, voxel_spacing, threshold)

    _tp, fp, _tn, _fn = confusion_matrix.get_matrix()

    return fp


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice original": dice_orig,
    "Dice": dice_th,
    "WDC": weighted_dice_coefficient,
    "Surface Dice Variable": compute_surface_dice_at_tolerance_list,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Volume Relative Difference": rel_volume_difference,
    "Volumetric Similarity": volumetric_similarity,
}


def get_metrics(
    test: np.ndarray,
    reference: np.ndarray,
    threshold: float = 200,
) -> pd.DataFrame:
    """Compute all metrics for a single subject and return as a DataFrame.

    Args:
        test (np.ndarray): The predicted binary segmentation.
        reference (np.ndarray): The ground truth binary segmentation.
        threshold (float, optional): Volume threshold in ml for small object handling.
          Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame containing all computed metrics.

    """
    # Compute all metrics and print results
    results = {}
    for metric_name, metric_function in ALL_METRICS.items():
        results[metric_name] = metric_function(
            test=test.astype(bool),
            reference=reference.astype(bool),
            threshold=threshold,
        )

    return pd.DataFrame([results])


def clean_metric_cols(df_metric: pd.DataFrame) -> pd.DataFrame:
    """Clean metric DataFrame column names for better readability.

    Args:
        df_metric (pd.DataFrame): DataFrame containing metric results.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.

    """
    df_metric_cleaned = df_metric.copy()
    for metric_col in df_metric_cleaned.columns:
        if (
            df_metric_cleaned[metric_col].dtype == "float64"
            or df_metric_cleaned[metric_col].dtype == "int64"
        ):
            continue
        if metric_col not in [FAZEKAS, REGION_NAME, DATASET, REGION]:
            # metric is a list of values - check if first value is actually a list
            first_val = df_metric_cleaned[metric_col].iloc[0]
            if isinstance(first_val, (list, tuple, np.ndarray)):
                tolerance_list = [2, 5, 10]
                for idx, tol in enumerate(tolerance_list):
                    df_metric_cleaned[f"{metric_col} tol {tol}"] = df_metric_cleaned[
                        metric_col
                    ].apply(
                        lambda x, i=idx: (
                            x[i] if isinstance(x, (list, tuple, np.ndarray)) else x
                        )
                    )
                df_metric_cleaned = df_metric_cleaned.drop(columns=[metric_col])
    return df_metric_cleaned


def get_all_metrics(
    test: np.ndarray,
    reference: np.ndarray,
    spatial_mask: np.ndarray = None,
    threshold: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all metrics for a single subject and return as a DataFrame.

    Args:
        test (np.ndarray): The predicted binary segmentation.
        reference (np.ndarray): The ground truth binary segmentation.
        spatial_mask (np.ndarray): The spatial mask defining different regions.
        voxel_spacing (tuple, optional): Voxel spacing in mm.
          Defaults to (1.0, 1.0, 1.0).
        threshold (float, optional): Volume threshold in ml for small object handling.
          Defaults to 200.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: a tuple containing:
            - DataFrame with global metrics
            - DataFrame with spatially resolved metrics

    """
    df_spatial_metrics = pd.DataFrame()
    df_global_metrics = get_metrics(test=test, reference=reference, threshold=threshold)
    if spatial_mask is not None:
        for region in np.unique(spatial_mask):
            if region == 0:
                continue
            region_mask = spatial_mask == region
            pred_region = np.logical_and(test, region_mask)
            gt_region = np.logical_and(reference, region_mask)

            df_metrics = get_metrics(
                test=pred_region,
                reference=gt_region,
                threshold=None,
            )

            if not df_metrics.empty:
                df_metrics[REGION] = region
                df_spatial_metrics = pd.concat(
                    [df_spatial_metrics, df_metrics], ignore_index=True
                )

        return clean_metric_cols(df_global_metrics), clean_metric_cols(
            df_spatial_metrics
        )

    return clean_metric_cols(df_global_metrics)
