import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")  # Use non interactive backend
sys.path.append(str(Path(__file__).resolve().parents[1]))


from analysis.model_comparisons.model_comparison import fetch_models_results
from misc.constants import DATA_CSV

df_dataset = pd.read_csv(Path(DATA_CSV))
df_dataset = df_dataset.query("split == 'test'")
pred_patterns = {
    "MDGRU": "*_MDGRU.nii.gz",
    "NNUNET": "*_NNUNET.nii.gz",
    "PGS": "*_PGS.nii.gz",
    "samseg": "*_pred.nii.gz",
    "segcsvd": "thr_*.nii.gz",
    "shiva": "*_pred.nii.gz",
    "whitenet": "*.nii.gz",
    "lgbm_post": "*_pred.nii.gz",
}
model_renames = {
    "MDGRU": "MD-GRU",
    "NNUNET": "nnUnet",
    "PGS": "PGS",
    "samseg": "Samseg",
    "segcsvd": "segcsvd",
    "shiva": "Shiva",
    "whitenet": "WHITE-Net",
    "lgbm_post": "LGBM",
}
df_metrics = fetch_models_results(
    df_dataset,
    csv_results_name="final_results",
    segmentations_folder="segmentation",
    pred_patterns=pred_patterns,
    n_jobs=16,
)
