<div align="center">
<img src="../resources/logos.png" alt="Logo Inselspital", width=600>
</div>

# inAGE laboratory - Inselspital, Bern
# LTS5 laboratory - EPFL, Lausanne

## SRC module
This module contains all the code developed in this project. It is organised as follow:

```
src/
├─analysis/
│ ├─model_comparisons/
│ ├─segmentation_difficulties/
│ ├─shap_analysis/
│ └─uncertainty_quantification/
├─data_func/
├─misc/
├─mlflow_func/
├─models/
├─notebooks/
├─scripts/
├─viz/
└─README.md
```
- [analysis](./analysis/): folder containing code to analyze results. This include the [comparision](./analysis/model_comparisons/) of the model with DL tools, [analysis of segmentation difficulties](./analysis/segmentation_difficulties/), [model's interpretability](./analysis/shap_analysis) and [uncertainty quantification](./analysis/uncertainty_quantification/).
- [data_func](./data_func/): contains all files related to data, including feature computation, data loadings, post processing, and files related to h5.
- [misc](./misc/): common functions and constants
- [mlflow_func](./mlflow_func/): Functions related to MLflow
- [models](./models/): files handling model classes.
- [notebooks](./notebooks/): Folder containing summary notebooks
- [scripts](./scripts/): Folder containing main running scripts, including script to [run model training](./scripts/run.py).
- [viz](./viz/): module handling visualization of the results.

# Scripts
They are different scripts and their function is described in this section.

1. [coregister_labels](./scripts/coregister_labels.py): Coregister labels from original space (T1w) to MPM space.
2. [run_shap.py](./scripts/run_shap.py): Run the SHAP analysis
3. [run_summary.py](./scripts/run_summary.py): Run the model comparision (see also [the config file](../config/model/lgb/summary.yml))
4. [run_uq.py](./scripts/run_uq.py): Run the Uncertainty Quantification analysis
5. [run.py](./scripts/run.py): Run the training of a model
6. [save_h5.py](./scripts/save_h5.py): Create the H5 file.
