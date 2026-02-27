# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-12-15 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to load data -*-

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from data_class import SubjectData


def load_subject(
    row: pd.Series, modalities: list, basedir: str, is_diff_model: bool
) -> SubjectData:
    """Load subject from pandas serie containing path to different modalities.

    Args:
        row (pd.Series): Row of the DataFrame.
        modalities (list): list of modalities
        basedir (str): the base path of the data
        is_diff_model (bool): True if using diffusion model.

    Returns:
        SubjectData: The Data class of the subject

    """
    qmaps = []
    for modality in modalities:
        modality_path = Path(basedir / str(row[modality.upper()]))
        if not modality_path.exists():
            msg = f"File {modality_path} does not exists"
            raise FileNotFoundError(msg)

        file = nib.load(filename=modality_path)
        qmaps.append(file.get_fdata())

    qmaps = np.stack(qmaps, axis=-1)

    neurom_path = Path(basedir / str(row["MASK"]))
    neurom_mask = nib.load(neurom_path)
    affine = neurom_mask.affine
    neurom_mask_array = neurom_mask.get_fdata()

    return SubjectData(
        qmaps=qmaps,
        affine=affine,
        neuromorphic_mask=neurom_mask_array,
        feature_names=modalities,
        is_diff_model=is_diff_model,
    )
