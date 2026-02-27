# -*- authors : Vincent Roduit -*-
# -*- date : 2024-10-20 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Functions to create the complete BrainLaus dataset -*-

import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces import fsl
from scipy.ndimage import distance_transform_edt
from tqdm.notebook import tqdm

from data_func.h5_helpers import check_maps, get_maps, open_h5
from misc.constants import (
    DATA_CSV,
    DATA_DIR,
    LOBE_TO_ID,
    MPM_REL_DIR,
    ORIGINAL_FLAIR_PATH,
    ORIGINAL_FREESURFER_PATH,
    ORIGINAL_MRAGE_TO_MPM_PATH,
    ORIGINAL_NEUROM_PATH,
    ORIGNAL_MPM_PATH,
    PROJECT_ROOT,
    QC_DIFFUSION_PATH,
    QC_MPM_PATH,
)

logger = logging.getLogger(__name__)


def copy_mpm_to_brainlaus(
    df_dataset: pd.DataFrame,
    mpm_dir: Path = ORIGNAL_MPM_PATH,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Copy MPM images to Brainlaus directory, checking for completeness.

    Args:
        df_dataset (pd.DataFrame): BrainLaus dataframe
        mpm_dir (Path): Path to the directory containing subject folders with
          MPM and diffusion maps. Default to ORIGNAL_MPM_PATH.
        dry_run (bool): If True, only log the actions without performing them.
          Defaults to False.
        overwrite (bool): If True, overwrite existing directories in Brainlaus.
          Defaults to False.

    """
    mpm_dirs = sorted([d for d in mpm_dir.iterdir() if d.is_dir()])
    mpm_pr_names = [d.name.split("_")[0].replace("sPR", "PR") for d in mpm_dirs]

    for _idx, row in tqdm(
        df_dataset.iterrows(),
        total=df_dataset.shape[0],
        desc="Copying MPM maps to BrainLaus",
    ):
        subject_dir = Path(row["subject_dir"])
        subject_id = row["id"]

        if subject_id in mpm_pr_names:
            mpm_subject_dir = next(
                d
                for d in mpm_dirs
                if d.name.split("_")[0].replace("sPR", "PR") == subject_id
            )
            if not (subject_dir / MPM_REL_DIR).exists() or overwrite:
                logger.info(
                    "Preparing to copy MPM maps for subject %s from %s to %s",
                    subject_id,
                    mpm_subject_dir,
                    subject_dir / MPM_REL_DIR,
                )
            else:
                logger.info(
                    "MPM maps for subject %s already exist at %s, skipping.",
                    subject_id,
                    subject_dir / MPM_REL_DIR,
                )
                continue

            file_paths = list(mpm_subject_dir.glob("*.nii*"))

            map_types, maps = get_maps(file_paths)

            if check_maps(map_types, maps, mpm_subject_dir, file_paths):
                if dry_run:
                    logger.info(
                        "Dry run: would copy subject %s to %s",
                        mpm_subject_dir.name,
                        DATA_DIR / subject_dir / MPM_REL_DIR,
                    )
                else:
                    logger.info("Processing subject %s", mpm_subject_dir.name)
                    # copy folder contents to brain laus folder
                    out_path = DATA_DIR / subject_dir / MPM_REL_DIR
                    out_path.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(mpm_subject_dir, out_path, dirs_exist_ok=True)
                    logger.info(
                        "Copied MPM maps of subject %s to %s",
                        mpm_subject_dir.name,
                        out_path,
                    )


def copy_ssm_to_brainlaus(
    df_dataset: pd.DataFrame,
    ssm_dir: Path = ORIGINAL_FREESURFER_PATH,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Copy SynthSeg masks to BrainLaus dataset directory.

    Args:
        df_dataset (pd.DataFrame): BrainLaus dataframe
        ssm_dir (Path): Directory containing SynthSeg masks.
          Default to ORIGINAL_FREESURFER_PATH.
        dry_run (bool): If True, only log actions without performing them.
        overwrite (bool): If True, overwrite existing files in destination.

    """
    synthseg_masks = list(ssm_dir.glob("*_synthseg_T1.nii"))

    count = 0

    logger.info("Found %d SynthSeg masks in %s", len(synthseg_masks), ssm_dir)

    for _, row in tqdm(df_dataset.iterrows(), total=df_dataset.shape[0]):
        subject_id = row["id"]
        subject_dir = Path(row["subject_dir"])

        if subject_id in [
            mask.stem.split("_")[0].replace("sPR", "PR") for mask in synthseg_masks
        ]:
            ssm_path = next(
                mask
                for mask in synthseg_masks
                if mask.stem.split("_")[0].replace("sPR", "PR") == subject_id
            )
            dest_path = (
                subject_dir
                / "masks"
                / "synthseg"
                / "MPRAGE"
                / f"{ssm_path.stem}_in_MPRAGE.nii"
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            count += 1

            if dest_path.exists() and not overwrite:
                logger.info("File %s already exists. Skipping copy.", dest_path)
                continue

            if dry_run:
                logger.info("[Dry Run] Would copy %s to %s", ssm_path, dest_path)
            else:
                shutil.copy(ssm_path, dest_path)
                logger.info("Copied %s to %s", ssm_path, dest_path)
        else:
            logger.warning(
                "SynthSeg mask for subject %s not found in %s", subject_id, ssm_dir
            )

    logger.info("Copied %d SynthSeg masks to BrainLaus dataset.", count)


def copy_mpm_mat_to_brainlaus(
    df_dataset: pd.DataFrame,
    mat_path: Path = ORIGINAL_MRAGE_TO_MPM_PATH,
    dry_run: bool = True,
    overwrite: bool = False,
) -> None:
    """Copy MPM .mat files to BrainLaus dataset structure.

    Args:
        df_dataset (pd.DataFrame): BrainLaus dataframe
        mat_path (Path): Path to the directory containing MPM .mat files.
          Default to ORIGINAL_MRAGE_TO_MPM_PATH.
        dry_run (bool): If True, only simulate the copying without actual
          file operations.
        overwrite (bool): If True, overwrite existing files in the destination.

    """
    mpm_mat_dirs = sorted([d for d in mat_path.iterdir() if d.is_dir()])

    count = 0

    for _, row in tqdm(
        df_dataset.sort_values(by="id").iterrows(),
        total=df_dataset.shape[0],
        desc="Copying MPM .mat files to BrainLaus",
    ):
        subject_dir = Path(row["subject_dir"])
        subject_id = row["id"]

        if subject_id in [
            d.name.split("_")[0].replace("sPR", "PR") for d in mpm_mat_dirs
        ]:
            mpm_mat_subject_dir = next(
                d
                for d in mpm_mat_dirs
                if d.name.split("_")[0].replace("sPR", "PR") == subject_id
            )
            mprage_to_mpm_mat = next(
                iter(mpm_mat_subject_dir.glob("*MPRAGE_to_MPM.mat")), None
            )
            if mprage_to_mpm_mat is None:
                logger.warning(
                    "No MPRAGE_to_MPM.mat file found for subject %s in %s",
                    subject_id,
                    mpm_mat_subject_dir,
                )
                continue
            count += 1
            dest_dir = subject_dir / MPM_REL_DIR
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / mprage_to_mpm_mat.name
            if not dest_path.exists() or overwrite:
                if dry_run:
                    logger.info(
                        "[Dry Run] Would copy %s to %s", mprage_to_mpm_mat, dest_path
                    )
                else:
                    shutil.copy2(mprage_to_mpm_mat, dest_path)
                    logger.info("Copied %s to %s", mprage_to_mpm_mat, dest_path)
            else:
                logger.info(
                    "MPM .mat file for subject %s already exists at %s, skipping.",
                    subject_id,
                    dest_path,
                )
        else:
            logger.warning(
                "No MPM .mat directory found for subject %s in %s", subject_id, mat_path
            )

    logger.info("Total MPM .mat files copied: %d", count)


def transform_ssm_to_mpm_space(
    df_dataset: pd.DataFrame, dry_run: bool = False, overwrite: bool = False
) -> None:
    """Transform SynthSeg masks from MPRAGE space to MPM space using FSL FLIRT.

    Args:
        df_dataset (pd.DataFrame): Path to the subject directory.
        dry_run (bool): If True, only simulate the transformation without
          actual file operations.
        overwrite (bool): If True, overwrite existing files in the destination.

    """
    for _, row in tqdm(
        df_dataset.iterrows(),
        total=df_dataset.shape[0],
        desc="Transforming SynthSeg masks to MPM space",
    ):
        subject_id = row["id"]
        subject_dir = Path(row["subject_dir"])
        mpm_path = subject_dir / MPM_REL_DIR
        mprage_mask_path = subject_dir / "masks" / "synthseg" / "MPRAGE"
        synthseg_mask_mprage = next(
            iter(mprage_mask_path.glob("*_synthseg_T1_in_MPRAGE.nii")), None
        )
        mprage_to_mpm_mat = next(iter(mpm_path.glob("*_MPRAGE_to_MPM.mat")), None)
        reference_mpm = next(iter(mpm_path.glob("*_MT.nii")), None)

        if (
            synthseg_mask_mprage is None
            or mprage_to_mpm_mat is None
            or reference_mpm is None
        ):
            logger.warning("Missing files for subject %s:", subject_id)
            if synthseg_mask_mprage is None:
                logger.warning("  - SynthSeg mask not found in MPRAGE space.")
            if mprage_to_mpm_mat is None:
                logger.warning("  - MPRAGE to MPM transformation matrix not found.")
            if reference_mpm is None:
                logger.warning("  - Reference MPM not found.")

        else:
            mask_mpm_dir = subject_dir / "masks" / "synthseg" / "MPM"
            mask_mpm_dir.mkdir(parents=True, exist_ok=True)
            synthseg_mask_mpm = (
                mask_mpm_dir
                / f"{synthseg_mask_mprage.name.replace('_in_MPRAGE', '_in_MPM')}"
            )
            if synthseg_mask_mpm.exists() and not overwrite:
                logger.info(
                    "SynthSeg mask for subject %s already exists at %s,"
                    " skipping transformation.",
                    subject_id,
                    synthseg_mask_mpm,
                )
                continue
            if dry_run:
                logger.info(
                    "[Dry Run] Would transform %s to MPM space and save to %s",
                    synthseg_mask_mprage,
                    synthseg_mask_mpm,
                )
                continue

            # Perform the transformation using FSL FLIRT
            flirt = fsl.FLIRT()
            flirt.inputs.in_file = synthseg_mask_mprage  # type: ignore[attr-defined]
            flirt.inputs.reference = reference_mpm  # type: ignore[attr-defined]
            flirt.inputs.out_file = synthseg_mask_mpm  # type: ignore[attr-defined]
            flirt.inputs.interp = "nearestneighbour"  # type: ignore[attr-defined]
            flirt.inputs.output_type = "NIFTI"
            flirt.inputs.in_matrix_file = mprage_to_mpm_mat  # type: ignore[attr-defined]
            flirt.inputs.apply_xfm = True  # type: ignore[attr-defined]
            flirt.inputs.out_matrix_file = (
                synthseg_mask_mpm.parent / f"{synthseg_mask_mpm.name}.mat"
            )  # type: ignore[attr-defined]

            flirt.run()  # type: ignore[attr-defined]
            logger.info(
                "SynthSeg mask for subject %s transformed to MPM space and saved to %s",
                subject_id,
                synthseg_mask_mpm,
            )


def copy_neurom_to_brainlaus(
    df_dataset: pd.DataFrame,
    neuromorphic_path: Path = ORIGINAL_NEUROM_PATH,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Copy SynthSeg masks to BrainLaus dataset directory.

    Args:
        df_dataset (pd.DataFrame): BrainLaus dataframe
        neuromorphic_path (Path): Directory containing SynthSeg masks.
        dry_run (bool): If True, only log actions without performing them.
        overwrite (bool): If True, overwrite existing files in destination.

    """
    neuromorphic_folders = list(neuromorphic_path.iterdir())
    folder_names = [f.name.split("_")[0] for f in neuromorphic_folders]

    for subject_id in df_dataset["id"].to_numpy():
        if subject_id in folder_names:
            neurom_folder = (
                neuromorphic_path / neuromorphic_folders[folder_names.index(subject_id)]
            )
            neurom_mask_path = next(iter(neurom_folder.rglob("*.nii*")), None)
            neurom_mask_dest = (
                DATA_DIR
                / subject_id
                / "masks"
                / "neuromorphics"
                / "MPM"
                / f"{subject_id}_neuromorphics_MPM.nii"
            )
            if neurom_mask_path is None:
                logger.warning("Missing Mask for subject %s", subject_id)
                continue
            if neurom_mask_dest.exists() and not overwrite:
                logger.info("File %s already exists. Skipping copy.", neurom_mask_dest)
                continue
            if dry_run:
                logger.info(
                    "[Dry Run] Would copy %s to %s", neurom_mask_path, neurom_mask_dest
                )
            else:
                neurom_mask_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(neurom_mask_path, neurom_mask_dest)
                logger.info("Copied %s to %s", neurom_mask_path, neurom_mask_dest)

        else:
            logger.warning("Missing Folder for subject %s", subject_id)


def add_qc_info_to_df(df_dataset: pd.DataFrame) -> None:
    """Add the quality information to the DataFrame and save it.

    Args:
        df_dataset (pd.DataFrame): The description DataFrame.

    """
    # Load DataFrames
    df_quality_diff = pd.read_csv(QC_DIFFUSION_PATH)
    df_quality_mpm = pd.read_csv(QC_MPM_PATH)

    # Fetch IDs
    df_quality_diff["id"] = df_quality_diff["Names"].apply(
        lambda x: x.split("_")[0].replace("sPR", "PR")
    )
    df_quality_mpm["id"] = df_quality_mpm["Names"].apply(
        lambda x: x.split("_")[0].replace("sPR", "PR")
    )

    # Merge DataFrame togethers and compute the max_grade
    df_dataset = df_dataset.merge(
        right=df_quality_mpm[["id", "QC_MT", "QC_R1", "QC_R2s"]], how="left"
    )
    df_dataset = df_dataset.merge(right=df_quality_diff[["id", "QC"]], how="left")
    df_dataset["max_grade"] = df_dataset[["QC_MT", "QC_R1", "QC_R2s", "QC"]].max(axis=1)

    # Save Dataframe
    df_dataset.to_csv(DATA_CSV, index=False)


def copy_flair_mpm_to_brainlaus(
    df_dataset: pd.DataFrame,
    flairmpm_path: Path = ORIGINAL_FLAIR_PATH,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Copy the FLAIR in MPM space to Brainlaus dataset.

    Args:
        df_dataset (pd.DataFrame): _description_
        flairmpm_path (Path, optional): The path where the original files are.
          Defaults to ORIGINAL_FLAIR_PATH.
        dry_run (bool, optional): True to make a dry rune. Defaults to False.
        overwrite (bool, optional): True to overwrite existing data. Defaults to False.

    """
    flair_folders = list(flairmpm_path.iterdir())
    folder_names = [f.name.split("_")[0].replace("sPR", "PR") for f in flair_folders]

    for subject_id in tqdm(df_dataset["id"].to_numpy()):
        if subject_id in folder_names:
            flair_folder = (
                flairmpm_path / flair_folders[folder_names.index(subject_id)] / "MPRAGE"
            )
            flair_path = next(
                iter(flair_folder.rglob("*MPRAGE_masked_in_MPM.nii*")), None
            )
            if flair_path is None:
                logger.warning("Missing file for %s", subject_id)
                continue
            flair_dest_path = (
                DATA_DIR / subject_id / "corrected" / "T1w" / "in_MPM" / flair_path.name
            )
            if flair_path is None:
                logger.warning("Missing FLAIR for subject %s", subject_id)
                continue
            if flair_dest_path.exists() and not overwrite:
                logger.info("File %s already exists. Skipping copy", flair_dest_path)
                continue
            if dry_run:
                logger.info(
                    "[Dry Run] Would copy %s to %s", flair_path, flair_dest_path
                )
            else:
                flair_dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(flair_path, flair_dest_path)

                logger.info("copied %s to %s", flair_path, flair_dest_path)
        else:
            logger.warning("Missing folder for subject %s", subject_id)


def create_meta_neurom() -> None:
    """Create the new metadata for the Neuromorphometric mask."""
    df_neurom_labels = pd.read_excel(
        PROJECT_ROOT / "data" / "neuromorphometric" / "neurom_labels.xlsx"
    )
    df_lobe_regions = pd.read_csv(
        PROJECT_ROOT / "data" / "neuromorphometric" / "lobe_regions.csv"
    )
    df_subcortical_labels = pd.read_csv(
        PROJECT_ROOT / "data" / "neuromorphometric" / "subcortical_labels.csv"
    )

    def get_region(value: int, df: pd.DataFrame) -> str:
        for col in df.columns:
            if value in df[col].to_numpy():
                return col
        return "other"

    new_rows = []
    for _idx, row in df_neurom_labels.iterrows():
        neurom_index = row["Neurom_index"]
        row["lobe_region"] = get_region(int(neurom_index), df_lobe_regions)

        new_rows.append(row)

    df_neurom_new = pd.DataFrame(new_rows)

    df_neurom_new["new_category"] = df_neurom_new["category"].apply(
        lambda x: "white matter" if x == "subcortical" else x
    )
    df_neurom_new["new_neurom_index"] = df_neurom_new["Neurom_index"].apply(
        lambda x: (
            44 if x in df_subcortical_labels["subcortical_labels"].to_numpy() else x
        )
    )

    df_neurom_new["lobe_index"] = df_neurom_new["lobe_region"].map(LOBE_TO_ID)

    df_neurom_new.to_excel(PROJECT_ROOT / "data" / "neurom_labels.xlsx")


def compute_lobe_mask(
    data: np.ndarray, label_to_lobe: dict, lobe_to_id: dict, wm_labels: list
) -> np.ndarray:
    """Create a label mask from the neuromorphometric mask.

    Args:
        data (np.ndarray): the neurom mask
        label_to_lobe (dict): a mapping assigning index to lobe
        lobe_to_id (dict): a mapping assigning for each str an int label
        wm_labels (list): list of wm labels

    Returns:
        np.ndarray: the lobe mask

    """
    data = data.astype(int)
    # Create an array that maps label ID → lobe, default 0
    max_label = data.max()
    lut = np.zeros(max_label + 1, dtype=np.uint8)

    for lab, lobe_id in label_to_lobe.items():
        lut[lab] = lobe_id

    # Apply lookup table to whole image
    seed_mask = lut[data]

    distances = []
    for lobe_id in sorted(lobe_to_id.values()):
        inv_mask = seed_mask != lobe_id
        dist = distance_transform_edt(inv_mask)
        distances.append(dist)

    distance_stack = np.stack(distances[1:])
    nearest_lobe = np.argmin(distance_stack, axis=0) + 1

    lobar_assign = np.zeros_like(data, dtype=np.uint8)
    wm_mask = np.isin(data, wm_labels)

    # Assign each WM voxel the lobe with shortest distance
    lobar_assign[wm_mask] = nearest_lobe[wm_mask]

    # Preserve the original cortical lobe seeds
    lobar_assign[seed_mask > 0] = seed_mask[seed_mask > 0]

    return lobar_assign


def create_lobe_masks(
    df_dataset: pd.DataFrame, dry_run: bool = False, overwrite: bool = False
) -> None:
    """Create the lobe masks for the dataset.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing data informations
        dry_run (bool, optional): True to dry run. Defaults to False.
        overwrite (bool, optional): True to overwrite. Defaults to False.

    """
    df_neurom = pd.read_excel(PROJECT_ROOT / "data" / "neurom_labels_new.xlsx")
    df_neurom["lobe_index"] = df_neurom["lobe_index"].fillna(0)

    # Define your lobe → ID mapping
    lobe_to_id = {
        "frontal": 1,
        "parietal": 2,
        "temporal": 3,
        "occipital": 4,
        "other": 0,
    }

    # Label → lobe ID (for **cortical** labels only)
    label_to_lobe = dict(
        zip(df_neurom["Neurom_index"], df_neurom["lobe_index"], strict=True)
    )
    wm_labels = df_neurom.loc[
        df_neurom["new_category"] == "white matter", "Neurom_index"
    ].to_numpy()

    for _idx, row in tqdm(
        df_dataset.dropna(subset="h5_path").iterrows(),
        total=len(df_dataset.dropna(subset="h5_path")),
    ):
        h5_path = Path(row["h5_path"])
        subject_dir = Path(row["subject_dir"])
        subject_id = row["id"]
        lobe_mask_path = (
            subject_dir / "masks" / "lobe" / "MPM" / f"lobe_mask_{subject_id}.nii"
        )

        if lobe_mask_path.exists() and not overwrite:
            logger.info("Lobe mask already exists, skipping %s", subject_id)
            continue

        if dry_run:
            logger.info(
                "[Dry Run] Would create a lobe mask and save: %s", lobe_mask_path
            )

        else:
            subject_data = open_h5(h5_path)
            lobe_mask = compute_lobe_mask(
                subject_data.neuromorphic_mask, label_to_lobe, lobe_to_id, wm_labels
            )
            lobe_mask_path.parent.mkdir(exist_ok=True, parents=True)
            nib.save(
                nib.Nifti1Image(
                    lobe_mask.astype(np.uint16), affine=subject_data.affine
                ),
                lobe_mask_path,
            )
