# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Constants for deployment -*-

from pathlib import Path

import pandas as pd

MAPS_ORDER_DIFF = [
    "A",
    "FA",
    "ICVF",
    "ISOVF",
    "MD",
    "OD",
]

MAPS_ORDER_ALL = [
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
FILE_NAME_DICT = {
    "A": "*_A*",
    "FA": "*_FA*",
    "g_ratio": "*_g_ratio*",
    "ICVF": "*_ICVF*",
    "ISOVF": "*_ISOVF*",
    "MD": "*_MD*",
    "MT": "*_MT*",
    "OD": "*_OD*",
    "R1": "*_R1*",
    "R2s_OLS": "*_R2s_OLS*",
}
NEUROM_FILENAME_PATTERN = "label_*"
current_path = Path(__file__).resolve()
PROJECT_ROOT = current_path.parents[1]

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

# Define spatial regions (superficial WM, deep WM and periventricual WM)
SPATIAL_REGIONS_BY_INT = {0: "other", 1: "PVWM", 2: "SWM", 3: "DWM"}
SPATIAL_REGIONS_BY_NAME = {"other": 0, "PVWM": 1, "SWM": 2, "DWM": 3}

# Lobe assignment
LOBE_TO_ID = {"frontal": 1, "parietal": 2, "temporal": 3, "occipital": 4, "other": 5}

# Label → lobe ID (for **cortical** labels only)
LABEL_TO_LOBE = dict(
    zip(
        df_neurom_labels["Neurom_index"],
        df_neurom_labels["lobe_region"].map(LOBE_TO_ID),
        strict=False,
    )
)
FEATURES_DIFF = [
    {"name": "neighbor_features", "characteristics": {"window_size": 3}},
    {"name": "gradient"},
    {"name": "LoG", "characteristics": {"sigma": 2.0}},
    {"name": "LBP"},
    {"name": "wm_ratio"},
    {"name": "lobe_info"},
    {"name": "wm_region"},
]

FEATURES_ALL = [
    {"name": "neighbor_features", "characteristics": {"window_size": 3}},
    {"name": "gradient"},
    {"name": "LoG", "characteristics": {"sigma": 2.0}},
    {"name": "LBP"},
    {"name": "wm_ratio"},
]
