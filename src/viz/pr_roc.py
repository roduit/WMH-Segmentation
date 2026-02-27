# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-12-02 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: ROC and PR plot functions-*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from viz.viz_utils import set_style

set_style()


def plot_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Plot the Precison-Recall curve.

    Args:
        y_true (np.ndarray): the true labels
        y_proba (np.ndarray): the predicted probabilities

    Returns:
        _tuple: a tuple containing:
            - precision (np.ndarray): the precision array
            - recall (np.ndarray): the recall array
            - thresholds (np.ndarray): the thresholds array
            - pr_auc (float): the area under the curve

    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=1)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")

    return np.array(precision), np.array(recall), np.array(thresholds), float(pr_auc)


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Plot the Receiver Operating Characteristic curve.

    Args:
        y_true (np.ndarray): the true labels
        y_proba (np.ndarray): the predicted probabilities

    Returns:
        _tuple: a tuple containing:
            - fpr (np.ndarray): the false positive array
            - tpr (np.ndarray): the true positive array
            - thresholds (np.ndarray): the thresholds array
            -roc_auc (float): the area under the curve

    """
    roc_auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, linestyle="--", label="ROC curve")
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    return fpr, tpr, thresholds, roc_auc
