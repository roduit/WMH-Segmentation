# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-11-04 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Loading functions -*-

import logging

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, uniform_filter

logger = logging.getLogger(__name__)


def compute_neig_features(qmap: np.ndarray, window_size: int, map_names: list) -> tuple:
    """Compute neighboring feature for quantitative maps.

    Neighboring features consist of applying a uniform filter with a certain window
    size. It gives then for each voxel a context of its neighborood. The mean and the
    standard deviation of the neighborood is returned.

    Args:
        qmap (np.ndarray): The MPM images.
        window_size (int, optional): Size of the window for neighborhood features.
        map_names (list): List of map names to compute.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Computed features array.
            - list: Added feature names.

    """
    # Compute neighborhood features
    feats = []
    added_feature_names = []
    for c in range(len(map_names)):
        img_c = qmap[..., c]
        mean_c = uniform_filter(img_c, size=window_size)
        var = uniform_filter(img_c**2, size=window_size) - mean_c**2
        std_c = np.sqrt(np.clip(var, 0, None) + 1e-8)
        feats.append(mean_c)
        feats.append(std_c)
        added_feature_names.extend([f"{map_names[c]}_mean", f"{map_names[c]}_std"])
    neigh_feats = np.stack(feats, axis=-1)  # (N_voxels, 2*C)

    return neigh_feats, added_feature_names


def compute_gradient_features(qmap: np.ndarray, map_names: list) -> tuple:
    """Compute the gradient for each map.

    Args:
        qmap (np.ndarray): the qmap
        map_names (list): the map names

    Returns:
        np.ndarray: the gradient magnitude

    """
    added_features_names = []
    grad_mag = np.zeros(qmap.shape)
    for c in range(qmap.shape[-1]):
        # Compute gradient along x, y, z
        gx, gy, gz = np.gradient(qmap[..., c])

        # Gradient magnitude
        grad_mag[..., c] = np.sqrt(gx**2 + gy**2 + gz**2)
        added_features_names.append(f"{map_names[c]}_gradient_magnitude")

    return grad_mag, added_features_names


def compute_log_features(qmap: np.ndarray, map_names: list, sigma: int) -> tuple:
    """Compute the laplacian feature for each map.

    Args:
        qmap (np.ndarray): the qmap
        map_names (list): the map names
        sigma (int): the sigma for the laplacian
    Returns:
        np.ndarray: the laplacian

    """
    added_features_names = []
    laplacian_features = []
    for c in range(qmap.shape[-1]):
        log = ndimage.gaussian_laplace(qmap[..., c], sigma=sigma)

        # Laplacian
        laplacian_features.append(log)
        added_features_names.extend([f"{map_names[c]}_LoG"])

    return np.stack(laplacian_features, axis=-1), added_features_names


def lbp_3d(volume: np.ndarray, radius: int = 1) -> np.ndarray:
    """Compute 3D Local Binary Pattern (LBP) for a 3D volume.

    Args:
        volume (np.ndarray): 3D input volume.
        radius (int): Radius for LBP computation.

    Returns:
        np.ndarray: 3D array of LBP codes.

    """
    z, y, x = volume.shape
    lbp = np.zeros_like(volume, dtype=np.uint32)

    # Define 3D neighborhood offsets (26 neighbors in 3x3x3 cube)
    offsets = [
        (i, j, k)
        for i in [-radius, 0, radius]
        for j in [-radius, 0, radius]
        for k in [-radius, 0, radius]
        if not (i == 0 and j == 0 and k == 0)
    ]

    # Center region
    center = volume[radius:-radius, radius:-radius, radius:-radius]
    code = np.zeros_like(center, dtype=np.uint32)

    for bit, (dz, dy, dx) in enumerate(offsets):
        neighbor = volume[
            radius + dz : z - radius + dz,
            radius + dy : y - radius + dy,
            radius + dx : x - radius + dx,
        ]

        # Compare and cast to integer before bitwise ops
        bitmask = (neighbor >= center).astype(np.uint32)
        code |= bitmask << bit

    lbp[radius:-radius, radius:-radius, radius:-radius] = code
    return lbp


def compute_lbp_features(qmap: np.ndarray, map_names: list, radius: int = 1) -> tuple:
    """Compute the LBP features, given a qmap.

    Args:
        qmap (np.ndarray): the qmap
        map_names (list): the map names
        radius (int): the radius for LBP computation

    Returns:
        np.ndarray: the LBP features

    """
    added_features_names = []
    lbp_features = []
    for c in range(qmap.shape[-1]):
        lbp = lbp_3d(qmap[..., c], radius=radius)

        # LBP
        lbp_features.append(lbp)
        added_features_names.append(f"{map_names[c]}_lbp")

    return np.stack(lbp_features, axis=-1), added_features_names


def compute_wm_ratio(
    qmaps: np.ndarray, map_names: list, wm_mask: np.ndarray
) -> tuple[np.ndarray, list]:
    """Compute the energy intensity between the mean WM value and the voxel.

    Args:
        qmaps (np.ndarray): The MPM images.
        map_names (list): List of map names.
        wm_mask (np.ndarray): The white matter mask.

    Returns:
        tuple[np.ndarray, list]: a tuple containing:
            - np.ndarray: The difference features array.
            - list: Added feature names.

    """
    added_feature_names = [f"{map_name}_wm_energy" for map_name in map_names]
    ratio = np.zeros_like(qmaps, dtype=np.float32)

    for c in range(qmaps.shape[-1]):
        img_c = qmaps[..., c]
        wm_mask_eroded = binary_erosion(wm_mask, iterations=4)
        wm_values = img_c[wm_mask_eroded > 0]

        if wm_values.size > 0:
            # Compute 5th and 95th percentiles
            p5, p95 = np.percentile(wm_values, [2.5, 97.5])
            # Keep only values within [5th, 95th] percentile
            wm_values_trimmed = wm_values[(wm_values >= p5) & (wm_values <= p95)]

            if wm_values_trimmed.size > 0:
                wm_mean = np.mean(wm_values_trimmed)
                wm_std = np.std(wm_values_trimmed)
                ratio[..., c][wm_mask] = ((img_c[wm_mask] - wm_mean) / wm_std) ** 2

    return ratio, added_feature_names
