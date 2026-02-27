# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-27 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Class for handling data -*-

import logging

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import exposure

from data_func.feature_extractors import (
    compute_gradient_features,
    compute_lbp_features,
    compute_log_features,
    compute_neig_features,
    compute_wm_energy,
)
from data_func.post_processing import (
    clean_small_objects,
    wm_restriction,
)
from data_func.utils import crop_array
from misc.constants import (
    CSF_INT_LABELS,
    SPATIAL_REGIONS_BY_NAME,
    TEST,
    WM_LABELS,
)

logger = logging.getLogger(__name__)


class SubjectData:
    def __init__(
        self,
        qmaps: np.ndarray,
        gt: np.ndarray,
        affine: np.ndarray,
        brain_mask: np.ndarray,
        neuromorphic_mask: np.ndarray,
        lobe_mask: np.ndarray,
        subject_id: str,
        feature_names: list,
        original_shape: tuple,
    ) -> None:
        """Initialize SubjectData.

        Args:
            qmaps (np.ndarray): MRI images array. Shape [H,W,D,N], with N the number of
              MRI images. If using all images, N=10 (diffusion+relaxometry).
            gt (np.ndarray): The ground truth mask. Shape [H,W,D]
            affine (np.ndarray): Affine transformation matrix.
            brain_mask (np.ndarray): Brain mask array. Shape [H,W,D]
            neuromorphic_mask (np.ndarray): Neuromorphic mask array. Shape [H,W,D]
            lobe_mask (np.ndarray): Lobe mask array. Shape [H,W,D]
            subject_id (str): Subject ID.
            feature_names (list): List of raw feature names (i.e. MRI image names).
            original_shape (tuple): Original shape of the data.

        Attributes:
            x (np.ndarray): Feature array. Shape [n_voxels, n_features]
            y (np.ndarray): Target array. Shape [n_features]
            patch_coords (tuple): Patch coordinates of the brain, used to remove
              unecessary background. Format (x_min, x_max, y_min, y_max, z_min, z_max).
            mask (np.ndarray): Array used to mask voxels. Default to brain mask.
              Shape [H,W,D]
            spatial_mask (np.ndarray): Spatial mask defining 3 regions (SWM, PWM, DWM).
              Shape [H,W,D]
            wm_mask (np.ndarray): White Matter mask. Shape [H,W,D]
            csf_mask: CSF mask. Shape [H,W,D]

        """
        self.qmaps = qmaps
        self.gt = gt
        self.affine = affine
        self.brain_mask = brain_mask
        self.lobe_mask = lobe_mask
        self.neuromorphic_mask = neuromorphic_mask
        self.subject_id = subject_id
        self.feature_names = feature_names
        self.original_shape = original_shape

        self.patch_coords = None
        self.x = None
        self.y = None
        self.mask = brain_mask
        self.spatial_mask = np.zeros_like(self.gt)
        self.wm_mask = np.zeros_like(self.gt)
        self.csf_mask = np.zeros_like(self.gt)

        self.initialize_masks()

    def clean_inputs(self) -> None:
        """Clean the input maps.

        The cleaning process includes:
        - Forcing all voxels outside the brain_mask to 0
        - Removing NaN values by replacing them with 0
        - Cropping the maps, gt, and masks to the bounding box of the brain mask.

        """
        # ---- Force all voxels outside the brain_mask to 0 ----
        self.qmaps[~self.brain_mask.astype(bool)] = 0

        # ---- remove NaNs if any ----
        self.qmaps = np.nan_to_num(self.qmaps, nan=0.0)

        # ---- Force negative values to 0 ----
        self.qmaps[self.qmaps < 0] = 0

        # ---- Change brain_mask type to boolean ----
        self.brain_mask = self.brain_mask.astype(bool)

        # ---- Crop the original shape to the size of the brain_mask ----
        coords = np.argwhere(self.brain_mask)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        patch_bounding_box = (x_min, x_max, y_min, y_max, z_min, z_max)

        # ---- Crop arrays to the brain ----
        self.qmaps = crop_array(self.qmaps, patch_bounding_box)
        self.gt = crop_array(self.gt, patch_bounding_box)
        self.brain_mask = crop_array(self.brain_mask, patch_bounding_box)
        self.neuromorphic_mask = crop_array(self.neuromorphic_mask, patch_bounding_box)
        self.spatial_mask = crop_array(self.spatial_mask, patch_bounding_box)
        self.mask = crop_array(self.mask, patch_bounding_box)
        self.wm_mask = crop_array(self.wm_mask, patch_bounding_box)
        self.csf_mask = crop_array(self.csf_mask, patch_bounding_box)
        self.lobe_mask = crop_array(self.lobe_mask, patch_bounding_box)
        self.patch_coords = patch_bounding_box

    def equalize_histogram(self) -> None:
        """Equalize histogram of each map channel within the brain_mask.

        The equalization is based on adaptative histogram equalization.

        References:
            .. [1] http://tog.acm.org/resources/GraphicsGems/
            .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
            .. [3] https://scikit-image.org/docs/0.25.x/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html

        """
        # ---- Rescale each qmap between its 5th and 95th percentile -----
        rescaled_maps = [
            np.clip(
                self.qmaps[..., i],
                np.percentile(self.qmaps[..., i], 5),
                np.percentile(self.qmaps[..., i], 95),
            )
            for i in range(self.qmaps.shape[-1])
        ]

        # ---- Normalize each map between 0 and 1 ----
        im_orig = [
            (rescaled_map - rescaled_map.min())
            / (rescaled_map.max() - rescaled_map.min())
            for rescaled_map in rescaled_maps
        ]

        # ---- Apply adaptive histogram equalization to each map ----
        equalized_qmap = [exposure.equalize_adapthist(im) for im in im_orig]

        # ---- Stack back into a single array ----
        self.qmaps = np.stack(equalized_qmap, axis=-1)

    def reduce_voxels_to_wm(self, split: str) -> np.ndarray:
        """Reduce the input quantitative maps to only white matter voxels.

        This function reduces the voxels to white matter only. But this is only done for
        the train set. It is not applied to the test in order to allow the model to find
        lesions everywhere since the neuromorphometric mask is not perfect.

        Args:
            split (str): data split type (train/test)

        Returns:
            np.ndarray: mask with only white matter voxels

        """
        # ---- don't apply the function to test set ----
        if split == TEST:
            return

        # ---- Update the mask to the WM mask ----
        self.mask = self.wm_mask

    def apply_processing(
        self,
        split: str,
        processing_list: list[dict],
    ) -> None:
        """Process the input maps based on the specified processing type.

        At the moment, processings available are cleaning, histogram equalization and
        reducing the voxel to WM only (only for training).

        Args:
            split (str): data split type (train/test)
            processing_list (list): list of processing types to apply.

        """
        # ---- Perform at least the cleaning operation ----
        if processing_list is None:
            processing_list = ["cleaning"]

        # --------------------------------------------------
        # Compute Processing
        # --------------------------------------------------
        for proc in processing_list:
            proc_name = proc.get("name")
            match proc_name:
                case "cleaning":
                    self.clean_inputs()

                case "histogram_equalization":
                    self.equalize_histogram()

                case "wm_voxels_only":
                    self.reduce_voxels_to_wm(split)

                case _:
                    msg = f"Unknown processing type: {proc_name}"
                    raise ValueError(msg)

    def compute_features(self, cfg_features: list) -> None:
        """Compute features based on a list of config dict.

        Args:
            cfg_features (dict): The config dict of features.

        Raises:
            ValueError: Raise error for unknown features.

        """
        features = None
        added_feature_names = []
        feature_names = []

        # --------------------------------------------------
        # Compute features
        # --------------------------------------------------
        for feature_cfg in cfg_features:
            feature_name = feature_cfg.get("name")
            cfg_feature_char = feature_cfg.get("characteristics", {})
            match feature_name:
                case "neighbor_features":
                    window_size = cfg_feature_char.get("window_size", 3)
                    added_features, added_feature_names = compute_neig_features(
                        self.qmaps, window_size, self.feature_names
                    )  # (H,W,D,2)

                case "gradient":
                    added_features, added_feature_names = compute_gradient_features(
                        self.qmaps, self.feature_names
                    )  # (H,W,D)

                case "LoG":
                    sigma = cfg_feature_char.get("sigma", 1)
                    added_features, added_feature_names = compute_log_features(
                        self.qmaps, self.feature_names, sigma
                    )  # (H,W,D)

                case "LBP":
                    added_features, added_feature_names = compute_lbp_features(
                        self.qmaps, self.feature_names
                    )  # (H,W,D)

                case "wm_ratio":
                    added_features, added_feature_names = compute_wm_energy(
                        self.qmaps, self.feature_names, self.wm_mask
                    )  # (H,W,D)

                case "lobe_info":
                    added_features = np.stack([self.lobe_mask], axis=-1)  # (H,W,D)
                    added_feature_names = ["lobe_info"]

                case "wm_region":
                    added_features = np.stack([self.spatial_mask], axis=-1)  # (H,W,D)
                    added_feature_names = ["wm_region"]

                case _:
                    msg = f"Unknown feature type: {feature_name}"
                    raise ValueError(msg)

            # ---- Accumulate features and feature names ----
            feature_names.extend(added_feature_names)
            features = (
                np.concatenate([features, added_features], -1)
                if added_features is not None and features is not None
                else added_features
            )

        # ---- Update the feature vector and corresponding feature names ----
        self.feature_names.extend(feature_names)
        self.x = (
            np.concatenate([self.qmaps, features], -1)
            if features is not None
            else self.qmaps
        )
        self.x = np.array(self.x)

    def mask_data(self) -> None:
        """Mask the data based on the provided mask.

        By default, voxels are masked to the brain only. To change the behavior, update
        the attribute `mask` before using the function.

        Args:
            subject_data (SubjectData): the input SubjectData object

        """
        self.mask = self.mask.astype(bool)
        self.x = self.x[self.mask]
        self.y = self.gt[self.mask]

    def initialize_masks(self) -> None:
        """Initialize different masks from the neuromorphometric mask."""
        self.csf_mask = np.isin(self.neuromorphic_mask, CSF_INT_LABELS)
        self.wm_mask = np.isin(self.neuromorphic_mask, WM_LABELS)

        # --------------------------------------------------
        # Region Definitions
        # --------------------------------------------------
        csf_dilation = binary_dilation(self.csf_mask, iterations=8)
        wm_erosion = binary_erosion(self.wm_mask, iterations=2)

        pvwm = np.asarray(self.wm_mask & csf_dilation & ~self.csf_mask)
        swm = np.asarray(self.wm_mask & ~wm_erosion & ~pvwm)
        dwm = np.asarray(self.wm_mask & ~pvwm & ~swm)

        pvwm_mask = pvwm.astype(np.uint8) * SPATIAL_REGIONS_BY_NAME["PVWM"]
        swm_mask = swm.astype(np.uint8) * SPATIAL_REGIONS_BY_NAME["SWM"]
        dwm_mask = dwm.astype(np.uint8) * SPATIAL_REGIONS_BY_NAME["DWM"]

        # ---- Spatial Mask ----
        self.spatial_mask = pvwm_mask + swm_mask + dwm_mask

    def post_process_predictions(
        self,
        pred_mask: np.ndarray,
        cfg_postprocess: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Post-process the predicted mask based on the provided configuration.

        The post-processing comprises: remove small objects (both on gt and prediction)
        and restricting predictions to WM only.

        Args:
            pred_mask (np.ndarray): The predicted mask to be post-processed.
            cfg_postprocess (list, optional): List of post-processing steps to apply.
              Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The post-processed predicted mask.
                - The post-processed gt mask restricted to WM.

        """
        if cfg_postprocess is None:
            return pred_mask, self.gt

        pred_mask_post = pred_mask.copy()
        gt_mask_post = self.gt.copy()

        # --------------------------------------------------
        # Apply post processing operations
        # --------------------------------------------------
        for proc in cfg_postprocess:
            proc_name = proc.get("name")
            match proc_name:
                case "remove_small_objects":
                    object_size = proc.get("size", 5)
                    pred_mask_post, gt_mask_post = clean_small_objects(
                        pred_mask_post, gt_mask_post, object_size
                    )

                case "wm_restriction":
                    dilation_iter = proc.get("dilation_iter", 1)
                    lesion_frac = proc.get("dilation_iter", 0.5)
                    pred_mask_post = wm_restriction(
                        pred_mask_post,
                        self.wm_mask,
                        self.csf_mask,
                        dilation_iter,
                        lesion_frac,
                    )

                case _:
                    msg = f"Unknown post-processing type: {proc_name}"
                    raise ValueError(msg)

        return pred_mask_post, gt_mask_post
