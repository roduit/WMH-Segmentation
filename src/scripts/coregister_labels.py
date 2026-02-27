# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-19 -*-
# -*- Last revision: 2025-11-19 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Script to coregister labels from T1w to MPM -*-

import logging
from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces import fsl
from tqdm import tqdm

from misc.constants import (
    DATA_CSV,
    GT_MPM_REL_DIR,
    GT_T1W_REL_DIR,
    MPM_REL_DIR,
)

# ---- Configure logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="coregister_labels.log",
    filemode="a",
)
logger = logging.getLogger(__name__)


def coregister_labels(
    df_dataset: pd.DataFrame,
    interpolation: str,
    threshold: float,
    mt_pattern: str = "sPR*_MT.nii",
    gt_pattern: str = "*.nii",
    matrix_pattern: str = "*_matrix_MPRAGE_to_MPM.mat",
) -> None:
    """Coregister labels from T1w space to MPM space.

    FLIRT is using to perform the registration. The interpolationcan be either sinc,
    nearest neigbors or trilinear. We recommand using nearest neigbors.

    Args:
        df_dataset (pd.DataFrame): The DataFrame containing participant informations
        interpolation (str): The interpolation to apply
        threshold (float): threshold to apply to the created mask
        mt_pattern (str, optional): Pattern of the MT file. Defaults to "sPR*_MT.nii".
        gt_pattern (str, optional): Pattern of the GT file. Defaults to "*.nii".
        matrix_pattern (str, optional): Pattern of the matrix file.
          Defaults to "*_matrix_MPRAGE_to_MPM.mat".

    References:
        .. [1] https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FLIRT(2f)UserGuide.html

    """
    for subject_dir in tqdm(df_dataset["subject_dir"], total=df_dataset.shape[0]):
        subject_dir_path = Path(subject_dir)

        # ---- Fetch paths to data ----
        mt_path = next(iter((subject_dir_path / MPM_REL_DIR).glob(mt_pattern)), None)
        gt_path = next(iter((subject_dir_path / GT_T1W_REL_DIR).glob(gt_pattern)), None)

        matrix_path = next(
            iter((subject_dir_path / MPM_REL_DIR).glob(matrix_pattern)), None
        )

        # ---- Check if all files exists ----
        if gt_path is not None and mt_path is not None and matrix_path is not None:
            output_matrix_path = (
                subject_dir_path / GT_MPM_REL_DIR / f"{gt_path.stem}_T1w_to_MPM.mat"
            )

            out_file = (
                subject_dir_path
                / GT_MPM_REL_DIR
                / f"{gt_path.stem}_{interpolation}_in_MPM.nii.gz"
            )

            # --------------------------------------------------
            # Run FSL
            # --------------------------------------------------
            flirt = fsl.FLIRT()
            flirt.inputs.in_file = gt_path
            flirt.inputs.out_file = out_file
            flirt.inputs.reference = mt_path
            flirt.inputs.in_matrix_file = matrix_path
            flirt.inputs.apply_xfm = True
            flirt.inputs.interp = interpolation
            flirt.inputs.out_matrix_file = output_matrix_path
            flirt.run()

            # ---- Load the registered mask to threshold ----
            mask = nib.load(out_file)
            mask_array = mask.get_fdata()
            mask_array = np.where(mask_array >= threshold, 1, 0)
            # Threshold the label
            out_file = (
                subject_dir_path
                / MPM_REL_DIR
                / f"{gt_path.stem}_{interpolation}_in_MPM_thr.nii.gz"
            )
            nib.save(
                nib.Nifti1Image(mask_array.astype(np.float32), mask.affine), out_file
            )

        else:
            logger.warning("missing file for participant %s", subject_dir_path.name)


if __name__ == "__main__":
    """Main entry point for the script.

    Perform the label registration from T1w space to MPM space.
    """
    parser = ArgumentParser(description="Script to save maps to HDF5 files")
    parser.add_argument(
        "--csv",
        "-d",
        type=Path,
        help="Path to the dataset folder",
        default=DATA_CSV,
    )
    parser.add_argument(
        "--interp",
        "-i",
        type=str,
        help="Interpolation for the registration",
        default="trilinear",
    )
    parser.add_argument(
        "--threshold",
        "-thr",
        type=float,
        help="threshold to apply to the coregister mask",
        default=0.5,
    )

    args = parser.parse_args()
    args = vars(args)

    dir_path = Path(args["csv"])
    interpolation = str(args["interp"])
    threshold = float(args["threshold"])

    if not dir_path.exists():
        msg = f"Dataset folder not found: {dir_path}"
        raise FileNotFoundError(msg)

    df_dataset = pd.read_csv(dir_path)
    coregister_labels(df_dataset, interpolation, threshold)
