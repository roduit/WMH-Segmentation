# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-13 -*-
# -*- Last revision: 2025-12-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Loading functions -*-

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from data_func.data_class import SubjectData
from data_func.h5_helpers import open_h5
from data_func.utils import filter_csv_description, load_csv_description
from misc.constants import (
    DATA_DIR,
    DEFAULT_MAP_ORDER,
)

logger = logging.getLogger(__name__)


def load_subject(
    h5_path: str,
    cfg_dataset: dict,
    maps_indices: list,
    split: Literal["train", "test", "val"],
) -> SubjectData:
    """Load participant data from HDF5 file.

    Args:
        h5_path (str): Path to the HDF5 file.
        cfg_dataset (dict): Configuration for feature extraction.
        maps_indices (list): List of indices for the maps to load.
        split (str): Data split type ('train', 'val', 'test').

    Returns:
        SubjectData: Loaded participant data including features and labels.

    """
    cfg_features = cfg_dataset.get("features", [])
    processing_list = cfg_dataset.get("preprocessing", ["cleaning"])

    # ---- Load participant ----
    subject_data = open_h5(Path(h5_path), qmaps_to_load=maps_indices)

    # ---- Apply pre-processing ----
    subject_data.apply_processing(split, processing_list)

    # ---- Compute features ----
    subject_data.compute_features(cfg_features)

    # ---- Mask data to either brain or WM mask ----
    subject_data.mask_data()

    return subject_data


def load_train(cfg_dataset: dict) -> tuple[list, list, list, list, list]:
    """Load all training data from HDF5 files.

    Args:
        cfg_dataset (dict): Dataset configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - x_list_train (list): List of feature arrays for training data.
            - y_list_train (list): List of labels arrays for training data.
            - masks_train (list): List of masks for training data.
            - ids_train (list): List of participant IDs for training data.
            - feature_names (list): List of feature names used.

    """
    # ---- Retrieve informations from config file ----
    path = Path(cfg_dataset.get("path", DATA_DIR))
    csv_name = cfg_dataset.get("csv_name", "dataset_description.csv")
    maps_to_load = cfg_dataset.get("maps", "all")
    min_fazekas = cfg_dataset.get("min_fazekas", 3)
    max_qc = cfg_dataset.get("max_qc", 3)
    maps_indices = (
        [DEFAULT_MAP_ORDER.index(map_name) for map_name in maps_to_load]
        if maps_to_load != "all"
        else list(range(len(DEFAULT_MAP_ORDER)))
    )

    csv_path = path / csv_name

    df = load_csv_description(csv_path)
    df_train = filter_csv_description(
        df, split="train", max_qc_grade=max_qc, min_fazekas=min_fazekas
    )

    x_list_train, y_list_train, masks_train, ids_train = [], [], [], []

    # --------------------------------------------------
    # Loop over training data
    # --------------------------------------------------
    for _idx, row in tqdm(
        df_train.iterrows(), total=len(df_train), desc="Loading training data"
    ):
        h5_path = str(row["h5_path"])

        subject_data = load_subject(h5_path, cfg_dataset, maps_indices, split="Train")

        # ---- Append to lists ----
        x_list_train.append(subject_data.x)
        y_list_train.append(subject_data.y)
        masks_train.append(subject_data.brain_mask)
        ids_train.append(subject_data.subject_id)

    return (
        x_list_train,
        y_list_train,
        masks_train,
        ids_train,
        subject_data.feature_names,
    )


def get_array(x_list: list, y_list: list, ids_list: list, cfg_dataset: dict) -> tuple:
    """Convert lists of arrays into single concatenated arrays and create groups.

    In order to avoid leaking voxels from the same participant into train and validation
    splits, groups are created.

    This function also performs sampling of the orignal dataset to restrict the number
    of voxels. This is done by using StratifiedShuffleSplit.

    Args:
        x_list (list): List of feature arrays.
        y_list (list): List of labels arrays.
        ids_list (list): List of participant IDs.
        cfg_dataset (dict): Dataset configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - x_array (np.ndarray): Concatenated feature array.
            - y_array (np.ndarray): Concatenated labels array.
            - groups (np.ndarray): Array of participant IDs.

    References:
        .. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

    """
    # ---- Concatenate and group voxels per participant ----
    x_array = np.concatenate(x_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)
    groups = np.concatenate(
        [
            np.repeat(id_, len(y_sub))
            for id_, y_sub in zip(ids_list, y_list, strict=False)
        ]
    )

    # ---- If the number of voxels is specified, reduce the size of the dataset ----
    size = cfg_dataset.get("size")
    if size is not None:
        size = int(float(size))
        if size < x_array.shape[0]:
            # ---- Randomly sample a subset of the data ----
            sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=42)
            indices, _ = next(sss.split(x_array, y_array))
            x_array = x_array[indices]
            y_array = y_array[indices]
            groups = groups[indices]

    logger.info("Dataset size after processing: %d samples", x_array.shape[0])
    logger.info(
        "Class distribution: %s",
        dict(zip(*np.unique(y_array, return_counts=True), strict=False)),
    )
    return x_array, y_array, groups
