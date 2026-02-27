# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2026-02-24 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Util functions -*-


import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from constants import FILE_NAME_DICT, NEUROM_FILENAME_PATTERN

logger = logging.getLogger(__name__)


def create_csv(
    input_path: Path, modalities: list, temp_dir: TemporaryDirectory
) -> Path | None:
    """Create the CSV file from an input directory.

    Args:
        input_path (Path): The input directory containing the modalities.
        modalities (list): The list of modalities (Diff or All maps).
        temp_dir (TemporaryDirectory): Temporary folder to save the csv.

    """
    logger.info("Creating CSV file")
    data = {}
    founded_files = []
    temp_dir_path = Path(temp_dir.name)
    csv_path = temp_dir_path / "data.csv"
    for modality in sorted(modalities, key=str.lower):
        file_path_list = list(input_path.glob(FILE_NAME_DICT[modality]))
        if len(file_path_list) == 0:
            logger.critical("Missing file ! (%s)", modality)
            return None

        if len(file_path_list) == 1:
            file_path = file_path_list[0]
        else:
            filtered_paths = [
                path for path in file_path_list if path not in founded_files
            ]
            file_path = filtered_paths[0]
            if len(filtered_paths) > 1:
                logger.warning(
                    "More than one file match the expected pattern (%s)",
                    FILE_NAME_DICT[modality],
                )
                logger.warning("Taking the first path: %s", file_path)
        logger.info("%s found (%s)", modality, file_path)
        data[modality] = [file_path]
        founded_files.append(file_path)

    neurom_path = next(iter(input_path.glob(NEUROM_FILENAME_PATTERN)), None)
    if neurom_path is None:
        logger.critical("Missing Neurom mask !")
        return None

    logger.info("Neuromorphometric file found ! (%s)", neurom_path)
    match = re.search(r"PR\d{5}", neurom_path.name)
    data["MASK"] = [neurom_path]
    data["OUT_DIR"] = [str(input_path)]
    data["ID"] = match.group()

    df_data = pd.DataFrame.from_dict(data)
    df_data.to_csv(csv_path, index=False)

    logger.info("CSV created successfully !")
    return csv_path


def check_csv(csv_path: Path, modalities: list) -> pd.DataFrame:
    """Check the input csv.

    The check consists of looking at the required columns.

    Args:
        csv_path (Path): Path to the csv
        modalities (list): Modalities used in the model

    Raises:
        ValueError: Raise Error in case of missing column.

    Returns:
        pd.DataFrame: The processed DataFrame

    """
    df_processed = pd.read_csv(csv_path, index_col=False)
    df_processed.columns = [col.upper() for col in df_processed.columns]

    missing = [c for c in modalities if c.upper() not in df_processed.columns]

    if missing:
        msg = f"Missing {missing} columns in CSV."
        raise ValueError(msg)

    return df_processed


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


def set_logging(
    level: int = logging.INFO,
    module_name: str | None = None,
    file_output: Path | None = None,
) -> logging.Logger:
    """Set the logging level for the application.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        module_name (str): Name of the module for which to set the logger.
        file_output (Path, optional): If provided, logs will be written to this file.

    Returns:
        logger: Configured logger instance.

    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=file_output,
        filemode="a" if file_output else None,
    )
    # Set specific logger levels if needed
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    logging.getLogger("nibabel").setLevel(logging.WARNING)

    return logger
