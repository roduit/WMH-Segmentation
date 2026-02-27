# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-15 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Entry script to predict segmentation -*-

# Import libraries
import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from constants import MAPS_ORDER_ALL, MAPS_ORDER_DIFF
from model import PredictionModel
from utils import create_csv, set_logging

file_path = str(Path(__file__).resolve())


def main(args: argparse.Namespace) -> None:
    """Serve as main function to run the model training and evaluation.

    Args:
        args (argparse.Namespace): Command line arguments.

    """
    logger_level = args.logger_level
    logger_file = args.logger_file
    module_name = Path(__file__).stem
    logger = set_logging(
        level=logger_level, module_name=module_name, file_output=logger_file
    )
    logger.info("Starting Prediction")
    temp_dir = tempfile.TemporaryDirectory()

    is_diff_model = args.diff

    if args.csv is None:
        input_path = Path(args.input_dir)
        modalities = MAPS_ORDER_DIFF if is_diff_model else MAPS_ORDER_ALL
        csv_path = create_csv(input_path, modalities, temp_dir)

        if csv_path is None:
            sys.exit()
    else:
        csv_path = Path(args.csv)
    model = PredictionModel(is_diff_model=is_diff_model)

    model.predict(csv_path)

    temp_dir.cleanup()


if __name__ == "__main__":
    # Use argument
    parser = argparse.ArgumentParser(description="Run model computation")
    parser.add_argument(
        "--diff", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--csv", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--logger_file", type=str, default=None)

    args = parser.parse_args()

    main(args=args)
