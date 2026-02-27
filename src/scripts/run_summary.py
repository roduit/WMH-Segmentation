# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-01 -*-
# -*- Last revision: 2025-11-04 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Run the summary of models-*-

# Import libraries
import argparse
import os
from pathlib import Path

import matplotlib as mpl
import mlflow
import numpy as np
import pandas as pd
from mlflow.client import MlflowClient

mpl.use("Agg")  # Use non interactive backend
from analysis.model_comparisons.model_assessment import (
    compute_model_scores,
    process_run_results,
)
from misc.constants import DATASET, FAZEKAS, ID
from misc.utils import read_yml, set_logging, set_seed

# Disable proxy
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def find_and_load_test_metrics(
    run_id: str, pattern: str = "_test_global_metrics.csv"
) -> str:
    """Find and download the test metrics CSV stored under test_results/.

    Args:
        run_id (str): The run id
        pattern (str, optional): The pattern to search.
          Defaults to "_test_global_metrics.csv".

    Raises:
        FileNotFoundError: Raise error if no file was founded.

    Returns:
        str: The path to the artifact to load

    """
    client = mlflow.tracking.MlflowClient()

    # ---- List only inside test_results ----
    artifacts = client.list_artifacts(run_id, "test_results")

    # ---- Find the matching file ----
    for f in artifacts:
        if f.path.endswith(pattern):
            return client.download_artifacts(run_id, f.path)

    msg = f"No file ending with '{pattern}'"
    raise FileNotFoundError(msg)


def summary_model(run_id: str, client: MlflowClient) -> np.ndarray:
    """Load model and compute summary.

    Args:
        run_id (str): Run identifier.
        client (MlflowClient): MLflow client to interact with the tracking server.

    Returns:
        np.ndarray: Summary data.

    """
    # ---- Placeholder for model loading and summary computation ----
    local_path_array = client.download_artifacts(run_id=run_id, path="data/metrics.npz")
    local_path_df = find_and_load_test_metrics(run_id)
    return np.load(local_path_array, allow_pickle=True), pd.read_csv(local_path_df)


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the dataframe.

    Args:
        df (pd.DataFrame): The original Dataframe

    Returns:
        pd.DataFrame: The stats of the dataframe

    """
    stats = df.drop([FAZEKAS, DATASET, ID], axis=1).agg(["mean", "std"])
    stats = pd.concat({col: stats[col] for col in stats.columns}, axis=1)
    return stats


def process_one_run(run_info: dict) -> dict:
    """Process one run and return results.

    Args:
        run_info (dict): Dict containing informations (id, name) of the run

    Returns:
        dict: contains results for the run

    """
    client = MlflowClient()

    run_id = run_info.get("id")
    run_name = run_info.get("name", "UnnamedRun")

    data, df_results = summary_model(run_id=run_id, client=client)

    df_stats = summarize_df(df_results)
    df_stats = pd.concat({run_name: df_stats}, axis=0)

    y_true, y_pred = data["y_true"].item(), data["y_pred"].item()

    res = compute_model_scores(y_true, y_pred)
    res["name"] = run_name
    res["stats"] = df_stats
    return res


def process_runs(runs: list[dict], fill: bool) -> None:
    """Process runs and generate plots.

    Args:
        runs (list[dict]): The runs information
        fill (bool): True to fill CI in plots.

    """
    results = []
    for run in runs:
        result = process_one_run(run)
        results.append(result)

    process_run_results(results, fill)


def main(args: argparse.Namespace) -> None:
    """Serve as main function to run the model training and evaluation.

    Args:
        args (argparse.Namespace): Command line arguments.

    """
    cfg_file = Path(args.cfg)
    seed = args.seed
    logger_level = args.logger_level
    logger_file = args.logger_file
    module_name = Path(__file__).stem
    # Set logging
    logger = set_logging(
        level=logger_level, module_name=module_name, file_output=logger_file
    )
    logger.info("Using configuration file: %s", str(cfg_file))
    logger.info("Using seed: %d", seed)
    set_seed(seed=seed)

    ## CONFIGURATION
    cfg = read_yml(cfg_file=cfg_file)
    experiment = cfg.get("experiment", "Default")
    name = cfg.get("name", "debug")
    task = cfg.get("task", "")
    fill = cfg.get("fill", False)
    list(cfg.get("run_ids", []))

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment)

    run_name = f"{experiment}:{name}:{task}"
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info("Started MLflow run with ID: %s", run_id)
        runs = cfg.get("runs")

        process_runs(runs, fill)

        logger.info("Summary run completed.")


if __name__ == "__main__":
    # Use argument
    parser = argparse.ArgumentParser(description="Run model summary computation")
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--logger_file", type=str, default=None)
    parser.add_argument("--fill", type=bool, default=False)

    args = parser.parse_args()

    main(args=args)
