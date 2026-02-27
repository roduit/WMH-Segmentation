# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-20 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to run the Uncertainty Quantification-*-

import argparse
import time
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

mpl.use("Agg")

from analysis.uncertainty_quantification.constants import SCALES
from analysis.uncertainty_quantification.error_retention_curves import (
    lesion_scale_lppv_rc,
    patient_scale_rc,
    voxel_scale_metrics,
    voxel_scale_rc,
)
from analysis.uncertainty_quantification.lesion_uncertainty_measures import (
    lesions_uncertainty,
)
from analysis.uncertainty_quantification.lesions_extractions import get_lesion_types
from analysis.uncertainty_quantification.patient_uncertainty_measures import (
    patient_uncertainty,
)
from analysis.uncertainty_quantification.utils import get_cc_mask
from analysis.uncertainty_quantification.voxel_uncertainty_measures import (
    voxels_uncertainty,
)
from data_func.h5_helpers import open_h5
from data_func.post_processing import clean_small_objects
from misc.constants import DATA_CSV, UQ_RESULTS_DIR
from misc.utils import set_logging
from viz.rc_viz import plot_rc_curves

entropy_measures = {
    "voxel": ["neg_confidence", "entropy_of_expected"],
    "lesion": [
        "mean neg_confidence",
        "mean entropy_of_expected",
        "logsum entropy_of_expected",
    ],
    "participant": [
        "avg. neg_confidence",
        "avg. scaled neg_confidence",
        "avg. entropy_of_expected",
        "avg. scaled entropy_of_expected",
        "avg. mean neg_confidence",
        "avg. scaled mean neg_confidence",
        "avg. mean entropy_of_expected",
        "avg. scaled mean entropy_of_expected",
        "avg. logsum entropy_of_expected",
        "avg. scaled logsum entropy_of_expected",
    ],
}


def init_h5(path: Path, n_subjects: int, n_fracs: int, n_bootstraps: int) -> None:
    """Init the h5 file.

    Args:
        path (Path): Path to save the file.
        n_subjects (int): Number of subjects
        n_fracs (int): Number of fracs iterations.
        n_bootstraps (int): Number of bootstrap iterations.

    """
    with h5py.File(path, "w") as f:
        f.create_dataset("fracs_retained", data=np.linspace(0, 1, n_fracs))

        f.create_dataset(
            "meta/subject_ids",
            shape=(n_subjects,),
            dtype=h5py.string_dtype(),
        )
        for scale in SCALES:
            for entropy_measure in entropy_measures[scale]:
                for curve in ["obs", "rand", "ideal"]:
                    if scale != "participant":
                        f.create_dataset(
                            f"{scale}/entropy/{entropy_measure}/{curve}/values",
                            shape=(n_subjects, n_fracs),
                            dtype=np.float32,
                            compression="gzip",
                        )
                        f.create_dataset(
                            f"{scale}/entropy/{entropy_measure}/{curve}/aucs",
                            shape=n_subjects,
                            dtype=np.float32,
                            compression="gzip",
                        )
                    else:
                        f.create_dataset(
                            f"{scale}/entropy/{entropy_measure}/{curve}/values",
                            shape=n_subjects,
                            dtype=np.float32,
                            compression="gzip",
                        )
                        f.create_dataset(
                            f"{scale}/entropy/{entropy_measure}/{curve}/aucs",
                            shape=n_bootstraps,
                            dtype=np.float32,
                            compression="gzip",
                        )


def compute_voxel_rc(
    h5f: h5py.File,
    idx: int,
    probas: np.ndarray,
    pred: np.ndarray,
    manual: np.ndarray,
    fracs_retained: np.ndarray,
    n_jobs: int | None,
    segmentation_metric: str,
) -> dict:
    """Compute retention curve at voxel level.

    Args:
        h5f (h5py.File): The h5 file.
        idx (int): The index of the subject.
        probas (np.ndarray): The prediction probability map.
        pred (np.ndarray): The thresholded prediction map.
        manual (np.ndarray): The manual segmentation.
        fracs_retained (np.ndarray): The array of fraction retained.
        n_jobs (int | None): Number of workers for parallelization.
        segmentation_metric (str): The segmentation metric to use in analysis.

    Returns:
        dict: Dictionary containing the uncertainty maps.

    """
    # voxel uncertainty
    voxel_unc = voxels_uncertainty(probas)

    # voxel RC
    for unc_name, unc_values in voxel_unc.items():
        (auc_obs, rc_obs), (auc_rand, rc_rand), (auc_ideal, rc_ideal) = voxel_scale_rc(
            pred.flatten(),
            manual.flatten(),
            unc_values.flatten(),
            segmentation_metric,
            fracs_retained,
            n_jobs=n_jobs,
        )

        h5f[f"voxel/entropy/{unc_name}/obs/values"][idx] = rc_obs
        h5f[f"voxel/entropy/{unc_name}/rand/values"][idx] = rc_rand
        h5f[f"voxel/entropy/{unc_name}/ideal/values"][idx] = rc_ideal

        h5f[f"voxel/entropy/{unc_name}/obs/aucs"][idx] = auc_obs
        h5f[f"voxel/entropy/{unc_name}/rand/aucs"][idx] = auc_rand
        h5f[f"voxel/entropy/{unc_name}/ideal/aucs"][idx] = auc_ideal

    return voxel_unc


def compute_lesion_rc(
    h5f: h5py.File,
    idx: int,
    voxel_unc: np.ndarray,
    pred: np.ndarray,
    manual: np.ndarray,
    fracs_retained: np.ndarray,
    n_jobs: int | None,
) -> dict:
    """Compute retention curve at lesion level.

    Args:
        h5f (h5py.File): The h5 file.
        idx (int): The index of the subject.
        voxel_unc (dict): The prediction probability map.
        pred (np.ndarray): The thresholded prediction map.
        manual (np.ndarray): The manual segmentation.
        fracs_retained (np.ndarray): The array of fraction retained.
        n_jobs (int | None): The dictionary containing the uncertainty maps at voxel
          level.

    Returns:
        dict: Dictionary containing the uncertainty maps.

    """
    lesion_types = get_lesion_types(pred, manual, n_jobs=n_jobs)
    pred_multi = get_cc_mask(pred)
    lesion_unc = lesions_uncertainty(
        pred_multi,
        voxel_unc,
        n_jobs=n_jobs,
    )

    for unc_name, unc_values in lesion_unc.items():
        (auc_obs, rc_obs), (auc_rand, rc_rand), (auc_ideal, rc_ideal) = (
            lesion_scale_lppv_rc(
                np.asarray(unc_values),
                np.asarray(lesion_types),
                fracs_retained,
            )
        )

        h5f[f"lesion/entropy/{unc_name}/obs/values"][idx] = rc_obs
        h5f[f"lesion/entropy/{unc_name}/rand/values"][idx] = rc_rand
        h5f[f"lesion/entropy/{unc_name}/ideal/values"][idx] = rc_ideal

        h5f[f"lesion/entropy/{unc_name}/obs/aucs"][idx] = auc_obs
        h5f[f"lesion/entropy/{unc_name}/rand/aucs"][idx] = auc_rand
        h5f[f"lesion/entropy/{unc_name}/ideal/aucs"][idx] = auc_ideal

    return lesion_unc


def compute_patient_uq(
    probas: np.ndarray, brain_mask: np.ndarray, lesion_unc: dict
) -> dict:
    """Compute patient uncertainty quantities.

    Args:
        probas (np.ndarray): The prediction probability map.
        brain_mask (np.ndarray): The brain mask of the subject.
        lesion_unc (dict): The dictionary containing the list of lesion uncertainties.

    Returns:
        dict: Dictionary containing the patient uncertainties.

    """
    return patient_uncertainty(probas, brain_mask, pd.DataFrame(lesion_unc))


def compute_patient_rc(
    h5f: h5py.File,
    n_subjects: int,
    patient_metrics: list,
    patient_unc_dict: dict,
    n_bootstraps: int | None,
    n_jobs: int | None,
    segmentation_metric: str,
) -> None:
    """Compute the patient retention curve.

    Args:
        h5f (h5py.File): The h5 file.
        n_subjects (int): The number of subjects.
        patient_metrics (list): The metrics for each patient.
        patient_unc_dict (dict): The uncertainty dict.
        n_bootstraps (int | None): The bumber of bootstrap to perform.
        n_jobs (int | None): Number of workers for the parallelisation.
        segmentation_metric (str): The patient metric to use in the analysis.

    """
    df_patient_metrics = pd.DataFrame(patient_metrics)
    for unc_name, unc_values in patient_unc_dict.items():
        (
            (auc_obs, rc_obs),
            (auc_rand, rc_rand),
            (auc_ideal, rc_ideal),
        ) = patient_scale_rc(
            np.array(unc_values),
            df_patient_metrics[segmentation_metric].to_numpy(),
            n_random=20,
            n_bootstrap=n_bootstraps,
            n_jobs=n_jobs,
        )

        def pad_to_length(x: np.ndarray, n: int, fill: float = np.nan) -> np.ndarray:
            out = np.full(n, fill)
            out[: len(x)] = x
            return out

        rc_obs_array = pad_to_length(rc_obs, n_subjects)
        rc_rand_array = pad_to_length(rc_rand, n_subjects)
        rc_ideal_array = pad_to_length(rc_ideal, n_subjects)

        h5f[f"participant/entropy/{unc_name}/obs/values"][:] = rc_obs_array
        h5f[f"participant/entropy/{unc_name}/rand/values"][:] = rc_rand_array
        h5f[f"participant/entropy/{unc_name}/ideal/values"][:] = rc_ideal_array

        h5f[f"participant/entropy/{unc_name}/obs/aucs"][:] = auc_obs
        h5f[f"participant/entropy/{unc_name}/rand/aucs"][:] = auc_rand
        h5f[f"participant/entropy/{unc_name}/ideal/aucs"][:] = auc_ideal


def compute_retention_curves(
    path: Path,
    df: pd.DataFrame,
    fracs_retained: np.ndarray,
    n_jobs: int | None,
    n_bootstraps: int | None,
    model: str,
    segmentation_metric: str,
) -> pd.DataFrame:
    """Compute the retention curves for a given DataFrame.

    Args:
        path (Path): The path to the h5 file.
        df (pd.DataFrame): DataFrame containing subject informations.
        fracs_retained (np.ndarray): The array of fraction retained.
        n_jobs (int | None): Number of workers for parallelization.
        n_bootstraps (int | None): Number of bootstrap operations to perform.
        model (str): Name of the model to compute rc from.
        segmentation_metric (str): The segmentation metric to use.

    Returns:
        pd.DataFrame: DataFrame containing patient-scale informations.

    """
    h5f = h5py.File(path, "a")

    patient_unc_dict = {k: [] for k in entropy_measures["participant"]}
    patient_metrics = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        subject_id = row["id"]
        subject_path = Path(row["subject_dir"])
        h5_path = Path(str(row["h5_path"]))

        if not h5_path.exists():
            continue

        subject_data = open_h5(h5_path)

        manual = np.asarray(
            nib.load(
                next((subject_path / "manual_seg/in_MPM").glob("*trilinear*thr*.nii*"))
            ).dataobj
        )
        pred = np.asarray(
            nib.load(
                next((subject_path / "segmentation" / model).glob("*pred.nii*"))
            ).dataobj
        )
        pred_proba = np.asarray(
            nib.load(
                next((subject_path / "segmentation" / model).glob("*pred_proba.nii*"))
            ).dataobj
        )

        pred, manual = clean_small_objects(pred, manual)

        if manual.sum() == 0 or pred.sum() == 0:
            continue

        probas = np.stack([pred_proba, 1 - pred_proba], axis=-1)
        # Voxel RC
        voxel_unc = compute_voxel_rc(
            h5f, idx, probas, pred, manual, fracs_retained, n_jobs, segmentation_metric
        )

        # Lesion RC
        lesion_unc = compute_lesion_rc(
            h5f, idx, voxel_unc, pred, manual, fracs_retained, n_jobs
        )

        # Patient UQ
        patient_unc = compute_patient_uq(probas, subject_data.brain_mask, lesion_unc)

        patient_metrics.append(voxel_scale_metrics(pred.flatten(), manual.flatten()))
        patient_metrics[-1]["id"] = subject_data.subject_id
        for unc_name, unc_values in patient_unc.items():
            patient_metrics[-1][unc_name] = unc_values
            patient_unc_dict[unc_name].append(unc_values)

        # Add subject ids to meta
        h5f["meta/subject_ids"][idx] = subject_id

    compute_patient_rc(
        h5f,
        len(df),
        patient_metrics,
        patient_unc_dict,
        n_bootstraps,
        n_jobs,
        segmentation_metric,
    )

    h5f.close()

    return pd.DataFrame(patient_metrics)


def main(args: argparse) -> None:
    """Compute Retention Curves.

    Args:
        args (argparse): The arugments

    """
    t0 = time.time()
    csv_path = Path(args.dataset_csv_path)
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else None
    n_bootstraps = int(args.n_bootstraps)
    segmentation_metric = str(args.segmentation_metric)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test_rc.h5"
    train_path = out_dir / "train_rc.h5"

    model = args.model

    logger.info("Loading csv...")
    df = pd.read_csv(csv_path)
    df_test = df.query("split == 'test' and max_grade < 3").dropna(subset="h5_path")
    df_train = df.query("split == 'train' and max_grade < 3").dropna(subset="h5_path")

    msg = f"""Dataset split:
                Train: {len(df_train)} subjects
                Test: {len(df_test)} subjects"""
    logger.info(msg)
    df_test.index = np.arange(len(df_test))
    df_train.index = np.arange(len(df_train))

    fracs_retained = np.linspace(0.0, 1.0, 400)

    logger.info("Initializing train H5 file")
    init_h5(train_path, len(df_train), len(fracs_retained), n_bootstraps)

    logger.info("Initializing test H5 file")
    init_h5(test_path, len(df_test), len(fracs_retained), n_bootstraps)

    logger.info("Computing train retention curves")
    df_patient_metrics_train = compute_retention_curves(
        train_path,
        df_train,
        fracs_retained,
        n_jobs,
        n_bootstraps,
        model,
        segmentation_metric,
    )
    logger.info("Succes computing train retention curves")

    logger.info("Computing test retention curves")
    df_patient_metrics_test = compute_retention_curves(
        test_path,
        df_test,
        fracs_retained,
        n_jobs,
        n_bootstraps,
        model,
        segmentation_metric,
    )
    logger.info("Succes computing test retention curves")

    logger.info("Plotting results")
    plot_rc_curves([train_path, test_path], segmentation_metric)
    img_path = out_dir / "rc.png"
    plt.savefig(img_path)
    logger.info("Image result saved at %s", img_path)

    patient_metric_path = out_dir / "patient_metrics.csv"
    logger.info("Saving patient-scale DataFrame to %s", patient_metric_path)
    df_patient_metrics = pd.concat([df_patient_metrics_train, df_patient_metrics_test])
    df_patient_metrics.to_csv(patient_metric_path, index=False)

    elapsed = time.time() - t0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    formatted = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    logger.info("Script terminated with succes in %s", formatted)


if __name__ == "__main__":
    # Use argument
    parser = argparse.ArgumentParser(description="Run model computation")
    parser.add_argument(
        "--csv_path",
        type=str,
    )
    parser.add_argument("--out_dir", type=str, default=UQ_RESULTS_DIR)
    parser.add_argument("--dataset_csv_path", type=str, default=DATA_CSV)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--logger_file", type=str, default=None)
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    parser.add_argument("--model", type=str, default="lgbm")
    parser.add_argument("--segmentation_metric", type=str, default="DSC")

    args = parser.parse_args()

    module_name = Path(__file__).stem
    logger_level = args.logger_level
    logger_file = args.logger_file
    # Set logging
    logger = set_logging(
        level=logger_level, module_name=module_name, file_output=logger_file
    )

    main(args=args)
