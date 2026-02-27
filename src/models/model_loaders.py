# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-20 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7 -*-
# -*- Description: Functions to load the proper model for image segmentation -*-

from models.lgboost import LgbClassifier
from models.random_forest import RfClassifier


def choose_model(cfg_model: dict) -> LgbClassifier | RfClassifier:
    """Load the model based on the configuration.

    Args:
        cfg_model (dict): Model configuration dictionary.

    Returns:
        model: The loaded model instance.

    """
    model_name = cfg_model.get("name")

    if model_name == "RandomForest":
        model = RfClassifier.from_config(cfg_model=cfg_model.get("config", {}))
    elif model_name == "LGB":
        model = LgbClassifier.from_config(cfg_model=cfg_model.get("config", {}))
    else:
        msg = f"Model {model_name} not recognized."
        raise ValueError(msg)

    return model
