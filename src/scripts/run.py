# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-20 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to run the segmentation-*-

# Import libraries
import argparse
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import mlflow

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="surface_distance"
)

mpl.use("Agg")  # Use non interactive backend

from misc.utils import read_yml, set_logging, set_seed  # noqa: E402
from models.model_loaders import choose_model  # noqa: E402

# Disable proxy
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


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
    maps = cfg.get("maps", "")
    features = cfg.get("feat", "")
    processing = cfg.get("proc", "")
    cfg_model = cfg.get("model", {})
    description = cfg.get("description", "")
    cfg.get("dataset", [])

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment)
    mlflow.sklearn.autolog(log_models=False, log_datasets=False)

    run_name = f"{maps}:{features}:{processing}"
    with mlflow.start_run(run_name=run_name, description=description):
        run_id = mlflow.active_run().info.run_id

        mlflow.log_artifact(str(cfg_file), artifact_path="config")

        logger.info("Loading model: %s", cfg_model.get("name", "Unknown"))
        model = choose_model(cfg_model=cfg_model)
        model.set_run_name(run_name=run_name)

        if cfg_model.get("model_id", None):
            model_id = str(cfg_model["model_id"])
            logger.info("Loading pre-trained model from: %s", str(model_id))
            model.load_model(model_id=model_id)
        else:
            logger.info("Starting training and evaluation")
            # Train the model
            model.train(cfg)

        logger.info("Predicting on test set")
        # Predict on the test set
        model.predict_test(cfg)

        logger.info("Run ID: %s successfully logged", run_id)


if __name__ == "__main__":
    # Use argument
    parser = argparse.ArgumentParser(description="Run model computation")
    parser.add_argument(
        "--cfg",
        type=str,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--logger_file", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
