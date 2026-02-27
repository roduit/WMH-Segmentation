# -*- authors : Vincent Roduit -*-
# -*- date : 2026-02-27 -*-
# -*- Last revision: 2026-02-27 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Functions to compare segmentation between gt and prediction -*-

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from misc.constants import SEG_GRADING_DIR


def compare_seg_grading(df_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the segmentation between gt and pred and return mixed model summary.

    Args:
        df_dataset (pd.DataFrame): Dataset description.

    Returns:
        tuple[pd.DataFrame,pd.DataFrame]: tuple containing:
            - Mixed model summary
            - Results of the ratings from the 2 raters.

    """
    df_rater1 = pd.read_csv(SEG_GRADING_DIR / "segmentation_grading_rater1.csv")
    df_rater2 = pd.read_csv(SEG_GRADING_DIR / "segmentation_grading_rater2.csv")
    df_gradings = pd.read_csv(SEG_GRADING_DIR / "segmentation_grading.csv")

    df_rater1 = df_rater1.assign(rater="rater1")
    df_rater2 = df_rater2.assign(rater="rater2")

    df_all = pd.concat([df_rater1, df_rater2], ignore_index=True)
    df = df_gradings.merge(df_all)

    df["manual_grade"] = np.where(
        df["manual_seg_id"] == 1, df["grade_seg_1"], df["grade_seg_2"]
    )

    df["pred_grade"] = np.where(
        df["pred_id"] == 1, df["grade_seg_1"], df["grade_seg_2"]
    )
    df_long = df.melt(
        id_vars=["id", "rater"],
        value_vars=["manual_grade", "pred_grade"],
        var_name="type",
        value_name="grade",
    )

    df_long["type"] = df_long["type"].str.replace("_grade", "")
    df_long["type"] = df_long["type"].astype("category")
    df_long["rater"] = df_long["rater"].astype("category")
    df_long["id"] = df_long["id"].astype("category")

    df_long = df_long.merge(df_dataset)
    model = smf.mixedlm(
        "grade ~ type + C(rater) + Fazekas", df_long, groups=df_long["id"]
    )
    return model.fit(), df_long
