# -*- authors : Vincent Roduit -*-
# -*- date : 2024-07-02 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Script to save participant data to HDF5 files -*-

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_func.h5_helpers import save_to_h5
from misc.constants import (
    DATA_CSV,
    DATA_DIR,
    H5_PATH,
    SUBJECT_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="save_h5.log",
    filemode="a",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Main entry point for the script.

    Based on a CSV provided, save data as H5 files.
    """
    parser = ArgumentParser(description="Script to save maps to HDF5 files")
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        help="Path to the dataset folder",
        default=DATA_DIR,
    )
    parser.add_argument(
        "--overwrite",
        "-ow",
        action="store_true",
        help="Overwrite existing HDF5 files",
    )
    parser.add_argument(
        "--dry-run",
        "-l",
        action="store_true",
        help="Perform a dry run without copying files",
    )

    args = parser.parse_args()
    args = vars(args)

    dir_path = Path(args["dir"])
    overwrite = args["overwrite"]
    dry_run = args["dry_run"]

    if not dir_path.exists():
        msg = f"Dataset folder not found: {dir_path}"
        raise FileNotFoundError(msg)

    if not (dir_path / DATA_CSV).exists():
        logger.warning("Missing Dataset descprtion csv !")
        sys.exit(1)

    if dry_run:
        logger.info("Dry run enabled. No files will be copied.")
    df_dataset = pd.read_csv(dir_path / DATA_CSV)

    subjects = [f.name for f in dir_path.iterdir() if f.is_dir()]
    logger.info("Found %d participants in the dataset folder.", len(subjects))
    updated_rows = []

    # --------------------------------------------------
    # Save each participant as H5 file
    # --------------------------------------------------
    for _idx, row in tqdm(df_dataset.iterrows(), total=len(df_dataset)):
        subject_path = Path(row[SUBJECT_DIR])
        created_path = save_to_h5(subject_path, overwrite=overwrite, dry_run=dry_run)

        if created_path is not None:
            row[H5_PATH] = created_path
    if not dry_run:
        df_dataset.to_csv(dir_path / DATA_CSV, index=False)
