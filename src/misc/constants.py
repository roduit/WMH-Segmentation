# -*- authors : Vincent Roduit -*-
# -*- date : 2024-07-02 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Constants used in the WMH segmentation project -*-

import os
import re
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

# ======================================================================================
# =====                          DIRECTORIES INFORMATIONS                          =====
# ======================================================================================

# Find project root by looking for .git directory
current_path = Path(__file__).resolve()
PROJECT_ROOT = current_path

while not (PROJECT_ROOT / ".git").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        msg = "No .git directory found in any parent directories."
        raise FileNotFoundError(msg)
    PROJECT_ROOT = PROJECT_ROOT.parent

# Try to load .env from project root
if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")
else:
    msg = "No .env file found in project root or current directory."
    raise FileNotFoundError(msg)

# BRAINLAUS related directories
DATA_DIR = Path(str(os.getenv("DATA_PATH")))
DATA_CSV = DATA_DIR / "dataset_description_qmri.csv"
HDF5_CSV = DATA_DIR / "h5_paths.csv"

ORIGNAL_MPM_PATH = Path(os.getenv("ORIGNAL_MPM_PATH"))
ORIGINAL_FREESURFER_PATH = Path(os.getenv("ORIGINAL_FREESURFER_PATH"))
ORIGINAL_MRAGE_TO_MPM_PATH = Path(os.getenv("ORIGINAL_MRAGE_TO_MPM_PATH"))
ORIGINAL_NEUROM_PATH = Path(os.getenv("ORIGINAL_NEUROM_PATH"))
ORIGINAL_FLAIR_PATH = Path(os.getenv("ORIGINAL_FLAIR_PATH"))

# Create folder for results
PROJECT_DATA = Path(PROJECT_ROOT / "data")
RESULTS_DIR = Path(PROJECT_DATA / "results")
GENERAL_RESULTS_DIR = Path(RESULTS_DIR / "general")
SEG_DIFF_RESULTS_DIR = Path(RESULTS_DIR / "seg_diff")
MODELS_COMP_RESULTS_DIR = Path(RESULTS_DIR / "models_comp")
SHAP_RESULTS_DIR = Path(RESULTS_DIR / "shap")
UQ_RESULTS_DIR = Path(RESULTS_DIR / "uq")
SEG_GRADING_DIR = Path(RESULTS_DIR / "segmentation_grading")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for d in [
    PROJECT_DATA,
    RESULTS_DIR,
    GENERAL_RESULTS_DIR,
    SEG_DIFF_RESULTS_DIR,
    MODELS_COMP_RESULTS_DIR,
    SHAP_RESULTS_DIR,
    UQ_RESULTS_DIR,
    SEG_GRADING_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Create folder for OPTUNA
OPTUNA_STORAGE_PATH = PROJECT_ROOT.parent / "optuna"
OPTUNA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
OPTUNA_STORAGE_PATH = OPTUNA_STORAGE_PATH / "db.sqlite3"
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_STORAGE_PATH}"

# Create tmp fodler for dev
QC_DIFFUSION_PATH = PROJECT_DATA / "quality_check" / "qc_diffusion.csv"
QC_MPM_PATH = PROJECT_DATA / "quality_check" / "qc_mpm.csv"

# ======================================================================================
# =====                               MPM CONSTANTS                                =====
# ======================================================================================

MAPS_TO_KEEP = {
    "*MT_masked.nii.gz": "MT",
    "*R1_masked.nii.gz": "R1",
    "*R2s_OLS_masked.nii.gz": "R2s",
    "*A_masked.nii.gz": "A",
    "*ISOVF_in_MPM.nii.gz": "ISOVF",
    "*MD_in_MPM.nii.gz": "MD",
    "*FA_in_MPM.nii.gz": "FA",
    "*ICVF_in_MPM.nii.gz": "ICVF",
    "*OD_in_MPM.nii.gz": "OD",
    "*g_ratio.nii": "g_ratio",
}
DEFAULT_MAP_ORDER = [
    "A",
    "FA",
    "ICVF",
    "ISOVF",
    "MD",
    "MT",
    "OD",
    "R1",
    "R2s_OLS",
    "g_ratio",
]

PATTERN_EXTRACTION = re.compile(
    r"_(A|MT|R1|R2s_OLS|FA|MD|ICVF|OD|ISOVF|g_ratio)(?=_|\.nii|\.nii\.gz)",
    flags=re.IGNORECASE,
)

MPM_REL_DIR = Path("corrected/MPM")
GT_MPM_REL_DIR = Path("manual_seg/in_MPM")
GT_T1W_REL_DIR = Path("manual_seg/in_T1w")
NEUROM_REL_DIR = Path("masks/neuromorphics/MPM")
LOBE_MASK_REL_DIR = Path("masks/lobe/MPM")

GT_FILENAME_PATTERN = "manual_*_trilinear_in_MPM_thr.nii.gz"
BRAIN_MASK_PATTERN = "*c2_c3_mask.nii.gz"
NEUROM_PATTERN = "*_neuromorphics_MPM.nii"
LOBE_PATTERN = "lobe_mask_*.nii"
PRED_FILE_PATTERN = "*pred.nii.gz"
PRED_PROBA_FILE_PATTERN = "*pred_proba.nii.gz"

# ======================================================================================
# =====                             GENERAL CONSTANTS                              =====
# ======================================================================================
N_CORES = joblib.cpu_count(only_physical_cores=True)
N_DIM_LGB = 2
N_DIM_RF = 3

# ======================================================================================
# =====                        NEUROMORPHOMETRIC CONSTANTS                         =====
# ======================================================================================

df_neurom_labels = pd.read_excel(
    PROJECT_ROOT / "data" / "neuromorphometric" / "neurom_labels.xlsx"
)

NEUROM_LABELS = dict(
    df_neurom_labels[["Label_plot", "Neurom_index"]].to_numpy().tolist()
)

CSF_LABELS = [
    "3rd Ventricle",
    "CSF",
    "Right Inf Lat Vent",
    "Left Inf Lat Vent",
    "Right Lateral Ventricle",
    "Left Lateral Ventricle",
]
CSF_INT_LABELS = [NEUROM_LABELS[part] for part in CSF_LABELS]

# Define cortical labels
CORTICAL_LABELS = df_neurom_labels.loc[
    df_neurom_labels["category"] == "cortex", "Neurom_index"
].to_list()

# Define wm labels
WM_LABELS = df_neurom_labels.loc[
    df_neurom_labels["category"] == "white matter", "Neurom_index"
].to_list()

WM_LABELS_EXT = df_neurom_labels.loc[
    df_neurom_labels["new_category"] == "white matter", "Neurom_index"
].to_numpy()

# Define spatial regions (superficial WM, deep WM and periventricual WM)
SPATIAL_REGIONS_BY_INT = {0: "other", 1: "PVWM", 2: "SWM", 3: "DWM"}
SPATIAL_REGIONS_BY_NAME = {"other": 0, "PVWM": 1, "SWM": 2, "DWM": 3}

# Lobe assignment
LOBE_TO_ID = {"frontal": 1, "parietal": 2, "temporal": 3, "occipital": 4, "other": 0}

# ======================================================================================
# =====                             EXAMPLES CONSTANTS                             =====
# ======================================================================================

FEATURES_EXAMPLE = [
    {"name": "neighbor_features", "characteristics": {"window_size": 3}},
    {"name": "gradient"},
    {"name": "LoG", "characteristics": {"sigma": 2.0}},
    {"name": "LBP"},
    {"name": "wm_ratio"},
    {"name": "lobe_info"},
    {"name": "wm_region"},
]

FEATURES_BEST_MODEL = [
    {"name": "neighbor_features", "characteristics": {"window_size": 3}},
    {"name": "gradient"},
    {"name": "LoG", "characteristics": {"sigma": 2.0}},
    {"name": "LBP"},
    {"name": "wm_ratio"},
]

SUBJECT_EXAMPLE_H5 = DATA_DIR / "PR05740" / MPM_REL_DIR / "PR05740_mpm.h5"

# ======================================================================================
# =====                               NAME CONSTANTS                               =====
# ======================================================================================
FAZEKAS = "Fazekas"
SPLIT = "split"
SUBJECT_ID = "subject_id"
REGION = "region"
DATASET = "dataset"
TRAIN = "train"
TEST = "test"
VAL = "val"
REGION_NAME = "region_name"
DATASET = "dataset"
SUBJECT_DIR = "subject_dir"
MODEL = "model"
H5_PATH = "h5_path"
ID = "id"
GROUP = "group"
VOXEL_CLASS = "voxel_class"
