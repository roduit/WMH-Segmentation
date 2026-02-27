# -*- authors : Vincent Roduit -*-
# -*- date : 2026-01-12 -*-
# -*- Last revision: 2025-01-12 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Constants for the Retention curves module-*-

SCALES = ["voxel", "lesion", "participant"]
CURVES = ["obs", "random", "ideal"]

# ---- Define the different entropy measures tested ----
ENTROPY_MEASURES = {
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

# ---- Rename the entropy measures for better plotting result ----
MEASURES_RENAMES = {
    "neg_confidence": r"NC_i",
    "entropy_of_expected": r"H_i",
    "mean neg_confidence": r"\overline{NC_l}",
    "mean entropy_of_expected": r"\overline{H_l}",
    "logsum entropy_of_expected": r"\overline{H_l^*}",
    "avg. neg_confidence": r"\overline{NC_B}",
    "avg. scaled neg_confidence": r"\overline{\widetilde{NC}_B}",
    "avg. entropy_of_expected": r"\overline{H_B}",
    "avg. scaled entropy_of_expected": r"\overline{\widetilde{H}_B}",
    "avg. mean neg_confidence": r"\overline{NC_P}",
    "avg. scaled mean neg_confidence": r"\overline{\widetilde{NC}_P}",
    "avg. mean entropy_of_expected": r"\overline{H_P}",
    "avg. scaled mean entropy_of_expected": r"\overline{\widetilde{H}_P}",
    "avg. logsum entropy_of_expected": r"\overline{H_P^*}",
    "avg. scaled logsum entropy_of_expected": r"\overline{\widetilde{H_P^*}}",
}
