# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Class for handling data -*-

import logging

import numpy as np
from constants import (
    CSF_INT_LABELS,
    FEATURES_ALL,
    FEATURES_DIFF,
    LABEL_TO_LOBE,
    LOBE_TO_ID,
    SPATIAL_REGIONS_BY_NAME,
    WM_LABELS,
)
from feature_extractors import (
    compute_gradient_features,
    compute_lbp_features,
    compute_log_features,
    compute_neig_features,
    compute_wm_ratio,
)
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from utils import crop_array

logger = logging.getLogger(__name__)


class SubjectData:
    def __init__(
        self,
        qmaps: np.ndarray,
        affine: np.ndarray,
        neuromorphic_mask: np.ndarray,
        feature_names: list,
        is_diff_model: bool,
    ) -> None:
        """Initialize SubjectData.

        Args:
            qmaps (np.ndarray): Quantitative maps array.
            affine (np.ndarray): Affine transformation matrix.
            neuromorphic_mask (np.ndarray): Neuromorphic mask array.
            is_diff_model (bool): True if using Diffusion only model.
            feature_names (list): List of feature names.

        Attributes:
            x (np.ndarray): Feature array.
            mask (np.ndarray): Array used to mask voxels. Default to brain mask.
            spatial_mask (np.ndarray): Spatial mask defining 3 regions (SWM, PWM, DWM).
            wm_mask (np.ndarray): White Matter mask.
            csf_mask: CSF mask.
            original_shape (tuple): Original shape of the data.
            patch_coords (np.ndarray): Patch coordinates of the brain. Used to remove
              unecessary background.

        """
        self.qmaps = qmaps
        self.affine = affine
        self.neuromorphic_mask = neuromorphic_mask
        self.feature_names = feature_names

        self.patch_coords = None
        self.x = None
        self.original_shape = self.qmaps.shape[:3]
        self.mask = np.ones(self.original_shape)
        self.spatial_mask = np.zeros_like(self.original_shape)
        self.wm_mask = np.zeros_like(self.original_shape)
        self.csf_mask = np.zeros_like(self.original_shape)

        features_to_compute = FEATURES_DIFF if is_diff_model else FEATURES_ALL

        self.initialize_masks()
        self.clean_inputs()
        self.compute_features(features_to_compute)
        self.mask_data()

    def clean_inputs(self) -> None:
        """Clean the input quantitative maps.

        The cleaning process includes:
        - Forcing all voxels outside the brain_mask to 0
        - Removing NaN values by replacing them with 0
        - Cropping the quantitative maps, labels, and masks to the bounding box of
        the brain_mask

        """
        # Force all voxels outside the brain_mask to 0
        self.qmaps[~self.brain_mask.astype(bool)] = 0

        # remove NaNs if any
        self.qmaps = np.nan_to_num(self.qmaps, nan=0.0)

        # Force negative values to 0
        self.qmaps[self.qmaps < 0] = 0

        # Change brain_mask type to boolean
        self.brain_mask = self.brain_mask.astype(bool)

        # Crop the original shape to the size of the brain_mask
        coords = np.argwhere(self.brain_mask)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        patch_bounding_box = (x_min, x_max, y_min, y_max, z_min, z_max)

        # Crop arrays to the brain
        self.qmaps = crop_array(self.qmaps, patch_bounding_box)
        self.brain_mask = crop_array(self.brain_mask, patch_bounding_box)
        self.neuromorphic_mask = crop_array(self.neuromorphic_mask, patch_bounding_box)
        self.spatial_mask = crop_array(self.spatial_mask, patch_bounding_box)
        self.mask = crop_array(self.mask, patch_bounding_box)
        self.wm_mask = crop_array(self.wm_mask, patch_bounding_box)
        self.csf_mask = crop_array(self.csf_mask, patch_bounding_box)
        self.lobe_mask = crop_array(self.lobe_mask, patch_bounding_box)
        self.patch_coords = patch_bounding_box

    def compute_features(self, features_to_compute: list) -> None:
        """Compute features based on a config dict.

        Args:
            features_to_compute (list): The list of config features to use.

        Raises:
            ValueError: Raise error for unknown features.

        """
        features = None
        added_feature_names = []
        feature_names = []

        for feature_cfg in features_to_compute:
            feature_name = feature_cfg.get("name")
            cfg_feature_char = feature_cfg.get("characteristics", {})

            match feature_name:
                case "neighbor_features":
                    window_size = cfg_feature_char.get("window_size", 3)
                    added_features, added_feature_names = compute_neig_features(
                        self.qmaps, window_size, self.feature_names
                    )  # (N_voxels, 2*C)
                # Add other feature types here as needed
                case "gradient":
                    added_features, added_feature_names = compute_gradient_features(
                        self.qmaps, self.feature_names
                    )  # (N_voxels, C)
                case "LoG":
                    sigma = cfg_feature_char.get("sigma", 1)
                    added_features, added_feature_names = compute_log_features(
                        self.qmaps, self.feature_names, sigma
                    )  # (N_voxels, C)
                case "LBP":
                    added_features, added_feature_names = compute_lbp_features(
                        self.qmaps, self.feature_names
                    )  # (N_voxels, C)
                case "wm_ratio":
                    added_features, added_feature_names = compute_wm_ratio(
                        self.qmaps, self.feature_names, self.wm_mask
                    )  # (N_voxels, C)
                case "lobe_info":
                    added_features = np.stack(
                        [self.lobe_mask], axis=-1
                    )  # (N_voxels, 2)
                    added_feature_names = ["lobe_region"]
                case "wm_region":
                    added_features = np.stack(
                        [self.spatial_mask], axis=-1
                    )  # (N_voxels, 2)
                    added_feature_names = ["spatial_wm"]
                case _:
                    msg = f"Unknown feature type: {feature_name}"
                    raise ValueError(msg)

            # Accumulate features and feature names
            feature_names.extend(added_feature_names)
            features = (
                np.concatenate([features, added_features], -1)
                if added_features is not None and features is not None
                else added_features
            )

        # Update the feature vector and corresponding feature names
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

        Returns:
            subject_data (SubjectData): masked SubjectData object

        """
        self.mask = self.mask.astype(bool)
        self.x = self.x[self.mask]

    def compute_lobe_mask(self) -> None:
        """Create a label mask from the neuromorphometric mask."""
        data = self.neuromorphic_mask.astype(int)
        # Create an array that maps label ID → lobe, default 0
        max_label = data.max()
        lut = np.zeros(max_label + 1, dtype=np.uint8)

        for lab, lobe_id in LABEL_TO_LOBE.items():
            lut[lab] = lobe_id

        # Apply lookup table to whole image
        seed_mask = lut[data]

        distances = []
        for lobe_id in sorted(LOBE_TO_ID.values()):
            inv_mask = seed_mask != lobe_id
            dist = distance_transform_edt(inv_mask)
            distances.append(dist)

        distance_stack = np.stack(distances)
        nearest_lobe = np.argmin(distance_stack, axis=0) + 1

        lobar_assign = np.zeros_like(data, dtype=np.uint8)
        wm_mask = np.isin(data, WM_LABELS)

        # Assign each WM voxel the lobe with shortest distance
        lobar_assign[wm_mask] = nearest_lobe[wm_mask]

        # Preserve the original cortical lobe seeds
        lobar_assign[seed_mask > 0] = seed_mask[seed_mask > 0]

        self.lobe_mask = lobar_assign

    def initialize_masks(self) -> None:
        """Initialize different masks from the neuromorphometric mask.

        This includes:

        1. Spatial mask is defined in 3 different regions:
            - periventricular white matter (PVWM)
            - superficial white matter (SWM)
            - deep white matter (DWM)

        2. CSF mask: mask of the cerebrospinal fluid

        3. wm_mask: mask of the white matter

        """
        self.csf_mask = np.isin(self.neuromorphic_mask, CSF_INT_LABELS)
        self.wm_mask = np.isin(self.neuromorphic_mask, WM_LABELS)
        self.brain_mask = (self.neuromorphic_mask > 0).astype(int)
        csf_dilation = binary_dilation(self.csf_mask, iterations=8)
        wm_erosion = binary_erosion(self.wm_mask, iterations=2)

        # Periventricular WM
        pvwm_mask = (
            np.array(self.wm_mask & csf_dilation & ~self.csf_mask).astype(np.uint8)
            * SPATIAL_REGIONS_BY_NAME["PVWM"]
        )

        # Superficial WM
        swm_mask = (
            np.array(self.wm_mask & ~wm_erosion & ~pvwm_mask).astype(np.uint8)
            * SPATIAL_REGIONS_BY_NAME["SWM"]
        )

        # Deep WM
        dwm_mask = (
            np.array(wm_erosion & ~pvwm_mask & ~swm_mask).astype(np.uint8)
            * SPATIAL_REGIONS_BY_NAME["DWM"]
        )

        self.spatial_mask = pvwm_mask + swm_mask + dwm_mask

        self.compute_lobe_mask()
