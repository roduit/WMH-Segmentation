<div align="center">
<img src="./resources/logos.png" alt="Logo Inselspital", width=600>
</div>

# Beyond FLAIR: Machine Learning Segmentation of White Matter Hyperintensities Using Diffusion and Relaxometry MRI

## inAGE laboratory - Inselspital, Bern
## LTS5 laboratory - EPFL, Lausanne
---

## Table of Contents

- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
    - [Mlflow](#mlflow)
    - [Optuna](#optuna)
- [Data](#data)
- [Credits](#credits)

# Abstract

# Repository structure
The top-level structure of the repository is the following:

```
wmh_thesis/
├─config/
├─data/
├─deployment/
├─model_testing/
├─others/
├─resources/
├─src/
├─.env
├─.gitignore
├─.pre-commit-config.yaml
├─LICENSE
├─README.md
├─pyproject.toml
└─requirements.txt
```

A dedicated README is placed in the different folders if needed.
## Main folders
- [config](./config): contains example of configuration files to train the model
- [data](./data/): contains additional data needed for the project as well as results.
- [model_testing](./model_testing/): scripts used for testing other models for comparison.
- [others](./others/): other scripts, for pre-commit for instance.
- [resources](./resources/): contains figures for readme.
- [src](./src/): contains all the code developped in the project.

## Main files
- [.pre-commit-config.yaml](.pre-commit-config.yaml): Config file for the pre-commit
- [pyproject.toml](.pyproject.toml): Config file forthe project, ruff linter and general setup.

# Getting Started
To use this project, first install the dependencies in a conda (or pip) environments.
```
conda create -n wmh_segmentation python=3.13.7
conda activate wmh_segmentation
pip install -r requirements.txt
```

Then set up the project:
```
pip install -e .
```

## Mlflow
[Mlflow](https://mlflow.org) is used to track experiments. To use the code developed in this project you should start an mlflow server:

```
mlflow server --backend-store-uri sqlite:///mlruns.db --port 8080
```

## Optuna
[Optuna](https://optuna.org) is used to fine-tuned the hyperparameters. To use the code developed in this project you should start an optuna server:

```
 optuna-dashboard sqlite:///db.sqlite3 --port 8081
```

# Data
This section presents the data format and structure used in this project. In case you want to use this code, we highly recommend to use the same structure. The folder is organized by participant with the following structure.
```
├─PARTICIPANT_ID1/
│  ├─corrected/
│  │ ├─FLAIR/
│  │ │ ├─file1.nii
│  │ │ ├─...
│  │ │ └─fileX.nii
│  │ │─MPM/
│  │ │   ├─FA.nii
│  │ │   ├─...
│  │ │   └─MT.nii
│  │ └─T1w/
│  │   ├─in_MPM/
│  │   │ ├─file1_in_MPM.nii
│  │   │ ├─...
│  │   │ └─fileX_in_MPM.nii
│  │   ├─file1.nii
│  │   ├─...
│  │   └─fileX.nii
│  ├─manual_seg/
│  │ ├─in_FLAIR/
│  │ │ └─manual_in_FLAIR.nii.gz
│  │ ├─in_MPM/
│  │ │ └─manual_in_MPM.nii.gz
│  │ └─in_T1w/
│  │   └─manual_IN_T1w.nii
│  └─masks/
│    └─neuromorphics/
│      └─MPM/
│        └─neuromorphics_MPM.nii
├─PARTICIPANT_ID2/
├─ ...
└─PARTICIPANT_IDX/
```
For simplicity, a [h5](https://docs.h5py.org/en/stable/high/file.html) file is created for each participant ([script to create the file](./src/scripts/save_h5.py)). Each file contains the following informations:

- **qmaps**: MPM and diffusion MRI metrics.
- **gt**: manual segmentation
- **brain_mask**: a brain mask
- **neuromorphic_mask**: The neuromorphometric mask
- **affine**: The affine matrix of the Nifti file
- **lobe_mask**: The lobe parecellation mask.
- **subjet_id**: The participant ID
- **maps_names** = The name of the MRI metrics (in the same order as they are placed in qmaps)

> [!WARNING]
> You should specify the path of your data in a .env at the root of the repository. The variable should be named `DATA_PATH`

# Credits
This repository was developed at inAGE lab as a Master Thesis by Vincent Roduit (vincent.roduit@epfl.ch) under the supervision of Prof. Bogdan Draganski (bogdan.draganski@insel.ch), Prof. Jean-Philippe Thiran (jean-philippe.thiran@epfl.ch) and Dr. Aurélie Bussy (aureliee.bussy@gmail.com).
