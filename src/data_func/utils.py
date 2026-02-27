# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-13 -*-
# -*- Last revision: 2025-10-29 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Util functions for data processing -*-

import logging
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from misc.constants import FAZEKAS, H5_PATH, ID, SPLIT

logger = logging.getLogger(__name__)


def load_csv_description(csv_path: Path) -> pd.DataFrame:
    """Load csv description file.

    Args:
        csv_path (Path): Path to the CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist.

    Returns:
        pd.DataFrame: Loaded DataFrame from the CSV file.

    """
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        msg = f"CSV file not found: {csv_path}"
        raise FileNotFoundError(msg)

    # Load the dataframe
    return pd.read_csv(csv_path)


def filter_csv_description(
    df_dataset: pd.DataFrame,
    split: Literal["train", "test"] | None,
    max_qc_grade: int | None = None,
    min_fazekas: int | None = None,
) -> pd.DataFrame:
    """Filter the original DataFrame.

    Args:
        df_dataset (pd.DataFrame): The original DataFrame
        split (str | None): Split name
        max_qc_grade (int): Max Quality control value to filter the DataFrame.
          Default to 3.
        min_fazekas (int): Minimum Fazekas score. Default to 0

    Returns:
        pd.DataFrame: The filtered DataFrame

    """
    df_dataset_processed = deepcopy(df_dataset)
    df_dataset_processed = df_dataset_processed.dropna(subset=[ID, FAZEKAS, H5_PATH])

    df_dataset_processed[SPLIT] = df_dataset_processed[SPLIT].apply(
        lambda x: str(x).lower()
    )

    # ---- Filter QC ----
    if max_qc_grade is not None:
        df_dataset_processed = df_dataset_processed.query("max_grade < @max_qc_grade")

    # ---- Filter by split ----
    if split is not None:
        df_dataset_processed = df_dataset_processed.query("split == @split")

    # ---- Filter Fazekas ----
    if min_fazekas is not None:
        df_dataset_processed = df_dataset_processed.query("Fazekas >= @min_fazekas")

    return df_dataset_processed


def crop_array(array: np.ndarray, bounding_box: tuple) -> np.ndarray:
    """Crop the input array to the specified bounding box.

    Args:
        array (np.ndarray): the input array to be cropped
        bounding_box (tuple): bounding box coordinates
          (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        np.ndarray: cropped array

    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    return array[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


def pad_to_original_shape(
    pred_crop: np.ndarray, original_shape: tuple, patch_coords: tuple
) -> np.ndarray:
    """Pad the cropped prediction back to the original shape.

    Args:
        pred_crop (np.ndarray): Cropped prediction array.
        original_shape (tuple): Original shape of the data.
        patch_coords (tuple): Coordinates of the patch used for cropping.

    Returns:
        np.ndarray: Padded prediction array with the original shape.

    """
    pred_full = np.zeros(original_shape, dtype=pred_crop.dtype)
    patch_x_min, patch_x_max, patch_y_min, patch_y_max, patch_z_min, patch_z_max = (
        patch_coords
    )
    pred_full[
        patch_x_min : patch_x_max + 1,
        patch_y_min : patch_y_max + 1,
        patch_z_min : patch_z_max + 1,
    ] = pred_crop
    return pred_full
