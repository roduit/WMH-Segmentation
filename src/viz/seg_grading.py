# -*- authors : Vincent Roduit -*-
# -*- date : 2025-12-02 -*-
# -*- Last revision: 2025-12-02 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: FUnctions to plot segmentation grading-*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pypalettes import load_cmap

from misc.constants import SEG_GRADING_DIR


def raters_boxplot(df_long: pd.DataFrame) -> None:
    """Plot boxplot differences for the 2 raters.

    Args:
        df_long (pd.DataFrame): Long DataFrame containing raters grading.

    """
    cmap = load_cmap(
        "Kippenberger",
        keep=[
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ],
    ).colors
    cmap_darken = load_cmap(
        "Kippenberger",
        keep=[
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ],
    ).colors
    sns.boxplot(data=df_long, x="type", y="grade", hue="rater", palette=cmap)
    sns.stripplot(
        data=df_long,
        x="type",
        y="grade",
        hue="rater",
        dodge=True,
        jitter=True,
        palette=cmap_darken,
        legend=False,
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel="Type", fontsize=14)
    plt.ylabel(ylabel="Grade", fontsize=14)
    plt.legend()
    plt.savefig(SEG_GRADING_DIR / "raters_boxplot.png")
    plt.show()


def raters_diff(df_long: pd.DataFrame) -> None:
    """Plot rater differences.

    Args:
        df_long (pd.DataFrame): Long DataFrame containing raters grading.

    """
    cmap = load_cmap(
        "Kippenberger",
        keep=[
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ],
    ).colors
    plt.figure(figsize=(10, 4))
    for idx, rater in enumerate(["rater1", "rater2"]):
        plt.subplot(1, 2, idx + 1)
        df_subset = df_long[df_long.rater == rater]
        for case_id in df_subset.id.unique():
            vals = df_subset[df_subset.id == case_id].sort_values("type")["grade"]
            plt.plot(["manual", "pred"], vals, color="gray", alpha=0.3)
        sns.pointplot(
            data=df_subset, x="type", y="grade", errorbar="sd", color=cmap[idx]
        )
        plt.title(f"Paired ratings for {rater}", fontsize=16)
        plt.ylabel("Rating", fontsize=14)
        plt.xlabel("Type", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    plt.savefig(SEG_GRADING_DIR / "raters_diff.png")
    plt.show()
