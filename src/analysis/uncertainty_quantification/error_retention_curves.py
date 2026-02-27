"""Update code from https://github.com/Medical-Image-Analysis-Laboratory/MS_WML_uncs.

Credits: Medical Image Analysis Laboratory
Github: https://github.com/Medical-Image-Analysis-Laboratory/MS_WML_uncs
Citation: @misc{molchanova2024structuralbased,
                title={Structural-Based Uncertainty in Deep Learning Across Anatomical
                        Scales: Analysis in White Matter Lesion Segmentation
                        },
                author={Nataliia Molchanova and Vatsal Raina and Andrey Malinin and
                        Francesco La Rosa and Adrien Depeursinge and Mark Gales and
                        Cristina Granziera and Henning Muller and Mara Graziani and
                        Meritxell Bach Cuadra
                        },
                  year={2024},
                  eprint={2311.08931},
                  archivePrefix={arXiv},
                  primaryClass={cs.CV}
                }
"""
# -*- authors : Medical Image Analysis Laboratory -*-
# -*- date : 2024-10-23 -*-
# -*- Last revision: 2025-12-24 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to calculate lesion uncertainty measures-*-

import logging
from collections import Counter
from copy import deepcopy
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from sklearn import metrics

logger = logging.getLogger(__name__)


def voxel_scale_metrics(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001) -> dict:
    """Compute voxel-scale metrics (DSC, nDSC, WDC, TPR, TDR).

    Args:
        y_pred (np.ndarray): Prediction array. Shape [H,W,D]
        y (np.ndarray): Ground truth array. Shape [H,W,D]
        r (float, optional): effective lesion load
          (for more info see Raina et al., 2023). Defaults to 0.001.

    Returns:
        dict: Dictionary with the corresponding metrics.

    """
    if np.sum(y) + np.sum(y_pred) == 0:
        return {"DSC": 1.0, "nDSC": 1.0, "TPR": 1.0, "TDR": 1.0, "FNR": 0.0}
    k = (
        (1 - r) * np.sum(y) / (r * (len(y.flatten()) - np.sum(y)))
        if np.sum(y) != 0.0
        else 1.0
    )
    tp = np.sum(y * y_pred)
    fp = np.sum(y_pred[y == 0])
    fn = np.sum(y[y_pred == 0])
    return {
        "DSC": 2.0 * tp / (2.0 * tp + fp + fn),
        "nDSC": 2.0 * tp / (2.0 * tp + fp * k + fn),
        "TPR": tp / (tp + fn),
        "TDR": tp / (tp + fp),
    }


def voxel_scale_rc(
    y_pred: np.ndarray,
    y: np.ndarray,
    uncertainties: np.ndarray,
    segmentation_metric: str,
    fracs_retained: np.ndarray | None = None,
    n_jobs: int | None = None,
) -> tuple:
    """Compute error retention curve values and nDSC-AAC on voxel-scale.

    Args:
        y_pred (np.ndarray): Prediction array. Shape [H,W,D]
        y (np.ndarray): Ground truth array. Shape [H,W,D]
        uncertainties (np.ndarray): uncertainty values for the corresponding voxels.
          Shape [H,W,D]
        segmentation_metric (str): The segmentation metric to use (e.g. WDC)
        fracs_retained (np.ndarray | None, optional):fractions of retained voxels at
          each iteration ( x-axis values of the error-RC ). Defaults to None.
        n_jobs (int | None, optional): Number of parallel processes. Default to None.

    Returns:
        tuple: tuple (area under the curve, y-axis values) for observation, random and
          ideal scenarios.

    """
    if fracs_retained is None:
        fracs_retained = np.linspace(0.0, 1.0, 400)

    def compute_metric(
        frac_: float, preds_: np.ndarray, gts_: np.ndarray, n_: int, metric_name_: str
    ) -> float:
        pos = int(n_ * frac_)
        curr_preds = preds_ if pos == n_ else np.concatenate([preds_[:pos], gts_[pos:]])
        return voxel_scale_metrics(y=gts_, y_pred=curr_preds)[metric_name_]

    def compute_rc(ordering: np.ndarray) -> tuple:
        gts = y[ordering].copy()
        preds = y_pred[ordering].copy()

        process = partial(
            compute_metric,
            preds_=preds,
            gts_=gts,
            n_=len(gts),
            metric_name_=segmentation_metric,
        )

        with Parallel(n_jobs=n_jobs) as parallel_backend:
            scores = parallel_backend(delayed(process)(frac) for frac in fracs_retained)

        auc = metrics.auc(fracs_retained, np.asarray(scores))
        return auc, np.asarray(scores)

    # --------------------------------------------------
    # Observed RC
    # --------------------------------------------------
    ordering_obs = uncertainties.argsort()
    auc_obs, rc_obs = compute_rc(ordering_obs)

    # --------------------------------------------------
    # Random RC
    # --------------------------------------------------
    rng = np.random.default_rng(1234)
    ordering_rand = rng.permutation(len(y))
    auc_rand, rc_rand = compute_rc(ordering_rand)

    # --------------------------------------------------
    # Ideal RC
    # --------------------------------------------------
    # TP, TN ==> uncertainty 0
    # FP, FN ==> uncertainty 1
    ideal_uncertainty = np.zeros_like(y, dtype=float)
    ideal_uncertainty[y_pred != y] = 1.0

    ordering_ideal = ideal_uncertainty.argsort()
    auc_ideal, rc_ideal = compute_rc(ordering_ideal)

    return (
        (auc_obs, rc_obs),
        (auc_rand, rc_rand),
        (auc_ideal, rc_ideal),
    )


def lesion_scale_lppv_rc(
    lesion_uncertainties: np.ndarray,
    lesion_types: np.ndarray,
    fracs_retained: np.ndarray,
) -> tuple:
    """Compute lesion-scale LPPV-RC and LPPV-AUC (observed, random, ideal).

    Args:
        lesion_uncertainties (np.ndarray): Array of uncertainties for each lesion.
        lesion_types (np.ndarray): Corresponding array with lesion type (tp,fp)
        fracs_retained (np.ndarray): fractions of retained voxels at
          each iteration ( x-axis values of the error-RC

    Returns:
        tuple: tuple (area under the curve, y-axis values) for observation, random and
          ideal scenarios.

    """

    def compute_lppv(types_: np.ndarray) -> float:
        counter = Counter(types_)
        tp = counter.get("tp", 0)
        fp = counter.get("fp", 0)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def compute_rc(ordering: np.ndarray) -> tuple:
        lesion_type_all = lesion_types[ordering][::-1].copy()

        metric_values = [compute_lppv(lesion_type_all)]

        # ---- reject most uncertain lesions ----
        for i_l, lesion_type in enumerate(lesion_type_all):
            if lesion_type == "fp":
                lesion_type_all[i_l] = "tn"
            metric_values.append(compute_lppv(lesion_type_all))

        n_lesions = len(lesion_types)
        spline_interpolator = interp1d(
            x=[_ / n_lesions for _ in range(n_lesions + 1)],
            y=metric_values[::-1],
            kind="slinear",
            fill_value="extrapolate",
        )
        metric_values_interp = spline_interpolator(fracs_retained)
        auc = metrics.auc(fracs_retained, metric_values_interp)
        return auc, metric_values_interp

    if len(lesion_uncertainties) != len(lesion_types):
        msg = f"""Mismatch of sizes between
        lesion_uncertainties ({lesion_uncertainties}) and
        lesion_types ({lesion_types})
        """
        raise ValueError(msg)

    # -------------------------
    # Observed RC
    # -------------------------
    ordering_obs = lesion_uncertainties.argsort()
    auc_obs, rc_obs = compute_rc(ordering_obs)

    # -------------------------
    # Random RC
    # -------------------------
    rng = np.random.default_rng(1234)
    ordering_rand = rng.permutation(len(lesion_types))
    auc_rand, rc_rand = compute_rc(ordering_rand)

    # -------------------------
    # Ideal RC
    # -------------------------
    ideal_uncertainties = np.zeros_like(lesion_uncertainties, dtype=float)
    ideal_uncertainties[lesion_types == "fp"] = 1.0

    ordering_ideal = ideal_uncertainties.argsort()
    auc_ideal, rc_ideal = compute_rc(ordering_ideal)

    return (
        (auc_obs, rc_obs),
        (auc_rand, rc_rand),
        (auc_ideal, rc_ideal),
    )


def patient_scale_rc(
    uncs_sample: np.ndarray,
    metric_sample: np.ndarray,
    replace_with: float = 1.0,
    n_random: int = 1,
    n_bootstrap: int | None = None,
    n_jobs: int | None = None,
    seed: int = 1234,
) -> tuple:
    """Compute patient-scale RC with optional bootstrap confidence intervals.

    Args:
        uncs_sample (np.ndarray): uncertainty values for each patient.
        metric_sample (np.ndarray): metric values for each patient.
        replace_with (float): replacement metric value.
        n_random (int): number of random permutations.
        n_bootstrap (int | None): number of bootstrap samples. Default to None.
        n_jobs (int | None): Number of parallel jobs. Default to None.
        seed (int): RNG seed.

    Returns:
        Without bootstrap:
            ((auc_obs, rc_obs),
             (auc_rand, rc_rand),
             (auc_ideal, rc_ideal))

        With bootstrap:
            (
              ((auc_obs, rc_obs), (auc_ci_obs,)),
              ((auc_rand, rc_rand), (auc_ci_rand,)),
              ((auc_ideal, rc_ideal), (auc_ci_ideal,))
            )

    """
    rng = np.random.default_rng(seed)

    def compute_rc(metrics_: np.ndarray, ordering: np.ndarray) -> tuple:
        metric_copy = deepcopy(metrics_[ordering])
        rc = [metric_copy.mean()]

        for idx in range(len(metric_copy) - 1, -1, -1):
            metric_copy[idx] = replace_with
            rc.append(metric_copy.mean())

        fracs = np.linspace(0, 1, len(rc))
        auc = metrics.auc(fracs, rc[::-1])
        return auc, np.asarray(rc[::-1])

    # -------------------------
    # Observed RC
    # -------------------------
    auc_obs, rc_obs = compute_rc(metric_sample, np.argsort(uncs_sample))

    # -------------------------
    # Random RC
    # -------------------------
    auc_rand_all, rc_rand_all = [], []
    for _ in range(n_random):
        ordering = rng.permutation(len(metric_sample))
        auc_r, rc_r = compute_rc(metric_sample, ordering)
        auc_rand_all.append(auc_r)
        rc_rand_all.append(rc_r)

    auc_rand = float(np.mean(auc_rand_all))
    rc_rand = np.mean(rc_rand_all, axis=0)

    # -------------------------
    # Ideal RC
    # -------------------------
    auc_ideal, rc_ideal = compute_rc(metric_sample, np.argsort(metric_sample)[::-1])

    # -------------------------
    # Bootstrap
    # -------------------------
    if n_bootstrap is None:
        return (
            (auc_obs, rc_obs),
            (auc_rand, rc_rand),
            (auc_ideal, rc_ideal),
        )

    def bootstrap_once(
        rng_seed: int,
        uncs: np.ndarray,
        metrics_: np.ndarray,
        n_random: int,
    ) -> tuple:
        rng = np.random.default_rng(rng_seed)
        n = len(metrics_)

        idx_bs = rng.integers(0, n, size=n)
        uncs_bs = uncs[idx_bs]
        metrics_bs = metrics_[idx_bs]

        # ---- Observed ----
        auc_obs, _ = compute_rc(metrics_bs, np.argsort(uncs_bs))

        # ---- Random ----
        auc_rand_all = []
        for _ in range(n_random):
            ordering = rng.permutation(n)
            auc_r, _ = compute_rc(metrics_bs, ordering)
            auc_rand_all.append(auc_r)

        auc_rand = float(np.mean(auc_rand_all))

        # ---- Ideal ----
        auc_ideal, _ = compute_rc(metrics_bs, np.argsort(metrics_bs)[::-1])

        return auc_obs, auc_rand, auc_ideal

    auc_obs_bs = []
    auc_rand_bs = []
    auc_ideal_bs = []

    seeds = rng.integers(0, 2**32 - 1, size=n_bootstrap)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(bootstrap_once)(
            seed,
            uncs_sample,
            metric_sample,
            n_random,
        )
        for seed in seeds
    )

    auc_obs_bs, auc_rand_bs, auc_ideal_bs = map(np.asarray, zip(*results, strict=True))

    return (
        (auc_obs_bs, rc_obs),
        (auc_rand_bs, rc_rand),
        (auc_ideal_bs, rc_ideal),
    )
