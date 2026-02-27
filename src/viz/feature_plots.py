# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2026-02-18 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to plot features-*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from pypalettes import load_cmap

from data_func.h5_helpers import open_h5
from misc.constants import (
    FEATURES_EXAMPLE,
    SUBJECT_EXAMPLE_H5,
)
from viz.viz_utils import set_style

set_style()


def plot_features(
    h5_path: Path = SUBJECT_EXAMPLE_H5,
    features: list = FEATURES_EXAMPLE,
    slice_idx: int = 180,
) -> None:
    """Plot example feature maps from a subject H5 file.

    Args:
        h5_path (Path, optional): Path of h5 file of participant.
          Defaults to SUBJECT_EXAMPLE_H5.
        features (dict, optional): list of features to plot
        slice_idx (int, optional): Index of the slice to compute. Defaults to 180.

    """
    # ---- Plot on MT (index 5) metric ----
    subject_data = open_h5(h5_path=h5_path, qmaps_to_load=[5])
    subject_data.compute_features(features)
    cmap = ListedColormap(load_cmap("Kippenberger", cmap_type="continuous").colors)
    selected_features = [feat for feat in subject_data.feature_names if "MT_" in feat]
    feature_names = [f.get("name") for f in features]
    spatial_features = list(set({"lobe_info", "wm_region"}) & set(feature_names))
    selected_features.extend(spatial_features)
    features_idx = [
        subject_data.feature_names.index(feat) for feat in selected_features
    ]

    # ---- Redefine names for better plotting result ----
    feature_names = {
        "MT_mean": r"$X^{\mu}$",
        "MT_std": r"$X^{\sigma}$",
        "MT_gradient_magnitude": r"$X^{\nabla}$",
        "MT_LoG": r"$X^{LoG}$",
        "MT_lbp": r"$X^{LBP}$",
        "MT_wm_energy": r"$X^{WMIE}$",
        "lobe_info": "Lobe region",
        "wm_region": "WM region",
    }

    # --- Plot selected features ----
    plt.figure(figsize=(8, 7))
    for i, feat_idx in enumerate(features_idx):
        if selected_features[i] in spatial_features or selected_features[i] == "MT_lbp":
            data_to_plot = np.ma.masked_where(
                subject_data.x[:, slice_idx, :, feat_idx] == 0,
                subject_data.x[:, slice_idx, :, feat_idx],
            )
        else:
            data_to_plot = subject_data.x[:, slice_idx, :, feat_idx]
        plt.subplot(2, int(len(features_idx) / 2), i + 1)
        plt.imshow(
            data_to_plot,
            cmap="gray" if selected_features[i] not in spatial_features else cmap,
        )
        plt.title(rf"{feature_names[selected_features[i]]}", fontsize=16)
        plt.axis("off")
    plt.tight_layout()
