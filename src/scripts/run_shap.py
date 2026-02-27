# -*- authors : Vincent Roduit -*-
# -*- date : 2026-02-14 -*-
# -*- Last revision: 2026-02-14 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Script to run SHAP analysis-*-


# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-20 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to run the segmentation-*-

# Import libraries
import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from analysis.shap_analysis.utils import get_explainer, get_voxels
from misc.constants import DATA_CSV, ID, SHAP_RESULTS_DIR, SPLIT, TEST, VOXEL_CLASS
from misc.utils import set_logging
from viz.shap_viz import plot_beeswarm, plot_interaction

mpl.use("Agg")  # Use non interactive backend
import mlflow

# Disable proxy
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def main(args: argparse.Namespace) -> None:
    """Serve as main function to run the model training and evaluation.

    Args:
        args (argparse.Namespace): Command line arguments.

    """
    logger_level = args.logger_level
    logger_file = args.logger_file
    model_name = args.model_name
    model_id = args.model_id
    n_jobs = args.n_jobs
    csv_path = Path(args.csv_path)
    split = args.split
    module_name = Path(__file__).stem
    # Set logging
    logger = set_logging(
        level=logger_level, module_name=module_name, file_output=logger_file
    )
    logger.info("Starting SHAP analysis")

    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    if not csv_path.exists():
        logger.warning("No CSV found at %s", csv_path)
        sys.exit()

    logger.info("Loading dataset (%s)", csv_path)
    df_dataset = pd.read_csv(csv_path)
    df_dataset = df_dataset[df_dataset[SPLIT] == split]
    df_x = get_voxels(df_dataset, model_name, n_jobs=n_jobs)

    explainer = get_explainer(model_id)

    plot_beeswarm(df_x, explainer)

    index_od = list(df_x.columns).index("OD")
    index_icvf = list(df_x.columns).index("ICVF")
    index_md = list(df_x.columns).index("MD")

    df_subsample = df_x.sample(frac=0.4)
    df_subsample = df_subsample.sort_values(by=VOXEL_CLASS)

    logger.info(
        "Calculating Explanation for subsample voxels (#voxels = %s)", len(df_subsample)
    )
    explanation_subsample = explainer(df_subsample.drop([VOXEL_CLASS, ID], axis=1))

    interaction_path = SHAP_RESULTS_DIR / "voxels_interaction.png"
    logger.info("Plotting interaction plot at %s", interaction_path)
    plot_interaction(
        explanation_subsample,
        df_subsample,
        [index_od, index_icvf, index_md],
        ["OD", "ICVF", "MD"],
        color_by="shap",
    )
    plt.savefig(interaction_path)

    interaction_path = SHAP_RESULTS_DIR / "voxels_interaction_errors.png"
    logger.info("Plotting interaction plot at %s", interaction_path)
    plot_interaction(
        explanation_subsample,
        df_subsample,
        [index_od, index_icvf, index_md],
        ["OD", "ICVF", "MD"],
        color_by="detection",
    )
    plt.savefig(interaction_path)

    df_subsample["shap_mean"] = explanation_subsample.values.sum(axis=1)  # noqa: PD011

    df_stats = (
        df_subsample[["ICVF", "OD", "MD", "shap_mean", VOXEL_CLASS]]
        .groupby(VOXEL_CLASS)
        .describe()
    )

    df_stats.T.to_csv(SHAP_RESULTS_DIR / "voxels_stats.csv", index=True)

    logger.info("Saved stats CSV in %s", SHAP_RESULTS_DIR / "voxels_stats.csv")


if __name__ == "__main__":
    # Use argument
    parser = argparse.ArgumentParser(description="Run SHAP analysis")
    parser.add_argument(
        "--model_id", type=str, default="models:/m-fcbe48bed03f4861a5b98c950c88ada2"
    )
    parser.add_argument("--csv_path", type=str, default=DATA_CSV)
    parser.add_argument("--model_name", type=str, default="lgbm_post")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--split", type=str, default=TEST)
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--logger_file", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
