# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-17 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Helper functions to build H5 files -*-

import logging
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from data_func.data_class import SubjectData
from misc.constants import (
    BRAIN_MASK_PATTERN,
    DEFAULT_MAP_ORDER,
    GT_FILENAME_PATTERN,
    GT_MPM_REL_DIR,
    LOBE_MASK_REL_DIR,
    LOBE_PATTERN,
    MAPS_TO_KEEP,
    MPM_REL_DIR,
    NEUROM_PATTERN,
    NEUROM_REL_DIR,
    PATTERN_EXTRACTION,
)

logger = logging.getLogger(__name__)


def get_maps(maps_path: list) -> tuple[list, list]:
    """Get maps and their types from a list of file paths.

    Args:
        maps_path (list): List of file paths to the maps.

    Returns:
        tuple: A tuple containing:
            - map_types (list): List of map types.
            - maps (list): List of file paths to the maps.

    """
    pairs = []
    for pattern in list(MAPS_TO_KEEP.keys()):
        for img in maps_path:
            if img.match(pattern):
                map_type = re.findall(PATTERN_EXTRACTION, str(img))
                map_type = map_type[-1] if map_type else None
                pairs.append((map_type, img))

    pairs.sort(key=lambda x: x[0])  # sort alphabetically by map type

    map_types, maps = zip(*pairs, strict=False) if pairs else ([], [])

    return list(map_types), list(maps)


def check_maps(
    map_types: list, maps: list, subject_path: Path, files_path: list
) -> bool:
    """Check if the maps are in the expected order and if all required maps are present.

    Args:
        map_types (list): List of map types.
        maps (list): List of file paths to the maps.
        subject_path (Path): Path to the subject directory.
        files_path (list): List of file paths to the maps.

    Returns:
        bool: True if the maps are in the expected order and all required maps
          are present, False otherwise.

    """
    if map_types != DEFAULT_MAP_ORDER:
        logger.warning(
            "Subject %s has unexpected qmaps maps: %s.", subject_path.name, map_types
        )

        missing_maps = set(MAPS_TO_KEEP.keys()) - {
            pattern
            for img in files_path
            for pattern in list(MAPS_TO_KEEP.keys())
            if pattern.replace("*", "") in str(img)
        }
        if len(maps) != len(list(MAPS_TO_KEEP.keys())):
            logger.warning("(found %d/%d).", len(maps), len(MAPS_TO_KEEP))
            logger.warning("Missing maps: %s", missing_maps)
        else:
            logger.warning(
                "Subject %s has all required qmaps maps but in unexpected order.",
                subject_path.name,
            )

        return False

    return True


def save_to_h5(
    subject_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    brain_mask_pattern: str = BRAIN_MASK_PATTERN,
    neurom_mask_pattern: str = NEUROM_PATTERN,
    lobe_mask_pattern: str = LOBE_PATTERN,
) -> None | Path:
    """Save qmaps images and labels to an HDF5 file.

    Args:
        subject_path (Path): Path to the subject directory.
        overwrite (bool): If True, overwrite existing HDF5 file. Defaults to False.
        dry_run (bool): If True, only log the actions without performing them.
            Defaults to False.
        brain_mask_pattern (str): Pattern of the brain mask file.
          Default to BRAIN_MASK_PATTERN
        neurom_mask_pattern (str): Pattern of the neurom mask file.
          Default to NEUROM_PATTERN
        lobe_mask_pattern (str): Pattern of the lobe mask file.
          Default to LOBE_PATTERN

    Returns:
        None | Path: Path to the created HDF5 file, or None if not created

    """
    h5_created_path = None
    if not (subject_path / GT_MPM_REL_DIR).exists():
        logger.warning("Missing manual segmentation folder for %s", subject_path.name)
        return None

    mpm_folder = subject_path / MPM_REL_DIR

    out_path = mpm_folder / f"{subject_path.name}_mpm.h5"

    if out_path.exists() and not overwrite:
        logger.info("HDF5 file already exists for %s, skipping.", subject_path.name)
        return None

    # ---- Fetch required files ----
    files_path = list(mpm_folder.glob("*.nii*"))
    brain_mask_path = next(iter((mpm_folder).glob(brain_mask_pattern)), None)

    neuromorphic_mask_path = next(
        iter((subject_path / NEUROM_REL_DIR).glob(neurom_mask_pattern)), None
    )

    lobe_mask_path = next(
        iter((subject_path / LOBE_MASK_REL_DIR).glob(lobe_mask_pattern)), None
    )

    required_files = [brain_mask_path, neuromorphic_mask_path, lobe_mask_path]

    # ---- Handle missing files ----
    if any(f is None for f in required_files):
        if neuromorphic_mask_path is None:
            logger.warning(
                "Missing the neurom masks for %s",
                subject_path.name,
            )
        else:
            logger.warning(
                "Missing MPM data for %s",
                subject_path.name,
            )
        return None

    map_types, maps = get_maps(files_path)

    # ---- Check if data is consistent and build H5 file ----
    if check_maps(map_types, maps, subject_path, files_path):
        # Load images and labels
        imgs = [nib.load(m).get_fdata(dtype=np.float32) for m in maps]
        stacked = np.stack(imgs, axis=-1)  # (X, Y, Z, C)
        brain_mask = nib.load(brain_mask_path).get_fdata(dtype=np.float32)
        neuromorphic_mask = nib.load(neuromorphic_mask_path).get_fdata(dtype=np.float32)
        lobe_mask = nib.load(lobe_mask_path).get_fdata(dtype=np.float32)
        label_path = next(
            iter((subject_path / GT_MPM_REL_DIR).glob(GT_FILENAME_PATTERN))
        )
        gt = np.asarray(nib.load(label_path).get_fdata(dtype=np.float32))
        affine = np.asarray(nib.load(label_path).affine)

        # ---- Save to HDF5 ----
        if dry_run:
            logger.info(
                "Dry run: would save %s with shape %s and labels shape %s to %s",
                subject_path.name,
                stacked.shape,
                gt.shape,
                out_path,
            )
            return None
        with h5py.File(out_path, "w") as f:
            f.create_dataset("qmaps", data=stacked, compression="gzip")
            f.create_dataset("gt", data=gt, compression="gzip")
            f.create_dataset("brain_mask", data=brain_mask, compression="gzip")
            f.create_dataset(
                "neuromorphic_mask", data=neuromorphic_mask, compression="gzip"
            )
            f.create_dataset("affine", data=affine)
            f.create_dataset("lobe_mask", data=lobe_mask, compression="gzip")
            f.attrs["subject_id"] = subject_path.name
            f.attrs["maps_names"] = map_types

        logger.info(
            "Saved %s with shape %s and labels shape %s to %s",
            subject_path.name,
            stacked.shape,
            gt.shape,
            out_path,
        )
        h5_created_path = out_path

    return h5_created_path


def open_h5(h5_path: Path, qmaps_to_load: list | str = "all") -> SubjectData:
    """Open an HDF5 file and load qmaps images and labels.

    Args:
        h5_path (Path): Path to the HDF5 file.
        qmaps_to_load (list | str, optional): List of qmaps indices to load
          or "all". Defaults to "all".

    Returns:
        SubjectData: SubjectData object containing the loaded data.

    """
    # ---- Load MRI images based on the list. If "all" provided, load all images ----
    if qmaps_to_load == "all":
        qmaps_to_load = list(range(len(MAPS_TO_KEEP)))
    elif isinstance(qmaps_to_load, list):
        if not all(
            isinstance(i, int) and 0 <= i < len(MAPS_TO_KEEP) for i in qmaps_to_load
        ):
            msg = (
                "qmaps_to_load must be a list of integers between 0 and %d (inclusive)."
            )
            raise ValueError(msg % (len(MAPS_TO_KEEP) - 1))
    else:
        msg = "qmaps_to_load must be 'all' or a list of integers."
        raise ValueError(msg)

    if not h5_path.exists():
        msg = f"HDF5 file not found: {h5_path}"
        raise FileNotFoundError(msg)

    qmaps_to_load = sorted(qmaps_to_load)

    # ---- Load subject and return SubjectData class ----
    with h5py.File(h5_path, "r") as f:
        qmaps = f["qmaps"][:, :, :, qmaps_to_load]
        brain_mask = f["brain_mask"][:]
        neuromorphic_mask = f["neuromorphic_mask"][:]
        lobe_mask = f["lobe_mask"][:]
        gt = f["gt"][:]
        affine = f["affine"][:]
        subject_id = f.attrs["subject_id"]
        maps_names = f.attrs["maps_names"]
        # return filtered maps_names
        maps_names = [maps_names[i] for i in qmaps_to_load]

    return SubjectData(
        qmaps=qmaps,
        gt=gt,
        affine=affine,
        brain_mask=brain_mask,
        neuromorphic_mask=neuromorphic_mask,
        lobe_mask=lobe_mask,
        subject_id=subject_id,
        feature_names=maps_names,
        original_shape=qmaps.shape[:-1],
    )
