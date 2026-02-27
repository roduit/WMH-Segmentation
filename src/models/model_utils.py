# -*- authors : Vincent Roduit -*-
# -*- date : 2025-11-11 -*-
# -*- Last revision: 2025-11-11 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Helper functions for Deep Learning model -*-


import pandas as pd
from sklearn.model_selection import StratifiedKFold

from misc.constants import FAZEKAS, GROUP


def split_test_set(df_test: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Split the original test set into k folds to build confidence intervals.

    Args:
        df_test (pd.DataFrame): The original DataFrame
        n_splits (int, optional): The number of splits. Defaults to 5.

    Returns:
        pd.DataFrame: The processed DataFrame

    """
    df_test[GROUP] = -1  # initialize

    x = df_test.drop(columns=[FAZEKAS])
    y = df_test[FAZEKAS]

    n_splits = min(len(x), n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for split_number, (_, idx) in enumerate(skf.split(x, y)):
        df_test.iloc[idx, df_test.columns.get_loc(GROUP)] = split_number

    return df_test
