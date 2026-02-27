<div align="center">
<img src="../resources/logos.png" alt="Logo Inselspital", width=600>
</div>

# inAGE laboratory - Inselspital, Bern
# LTS5 laboratory - EPFL, Lausanne

## Deployment module
---
This module provides the minimum necessary information to produce an apptainer image to run the model developped. The folder is organised as follow:
```
deployment/
├─data/
├─src/
├─Dockerfile
├─README.md
├─app.def
├─docker-compose.yml
└─requirements.txt
```
- [data](./data/): contains necessary data to build the model, including lobe regions, neuromorphometric labels and sample data.
- [src](./src/): contains the source code to produce the image.
- [Dockerfile](./Dockerfile): config file for Docker
- [app.def](app.def): config file for apptainer.
- [docker-compose.yml](./docker-compose.yml): docker-compose file.

# Build the aptainer image
To build the Apptainer image, run the following command:

```
apptainer build wmh_segmenter.sif app.def
```

# Use the image
## Data prerequisites
In the current version, we require data to be aligned in the same space, with the same voxel size.
### Data as folder
To be able to use the image you need to provide data in the following format:

<center>

| Name                   | File Name Pattern |
|------------------------|-------------------|
| PD*                    | \*\_A*           |
| FA                     | \*\_FA*          |
| ICVF                   | \*\_ICVF*        |
| ISOVF                  | \*\_ISOVF*       |
| MD                     | \*\_MD*          |
| MT                     | \*\_MT*          |
| OD                     | \*\_OD*          |
| R1                     | \*\_R1*          |
| R2*                    | \*\_R2s_OLS\*    |
| g-ratio                | \*\_g_ratio\*     |
| Neuromorphometric mask | label\_\*         |

</center>

> [!TIP]
> If you want to use the Diffusion-only model, you only need to provide PD*, FA, ICVF, ISOVF, MD and OD.

## Data as CSV
Alternatively, you can use the image by providing a csv file with columns:
I, A, FA, ICVF, ISOVF, MD, MT, OD, R1, R2s_OLS, g_ratio, MASK and OUT_DIR, where **ID** is the participant id, **MASK** is the neuromorphometric mask and **OUT_DIR** is the output directory where to save the results and each row correspond to a subjet.

> [!NOTE]
> A CSV example is provided [here](./data/sample_data/samle_data.csv).

> [!TIP]
> This approach is recommended if you want to predict mutliple participants.

# Run the image
This section gives examples on how to run the provided image.

## Function arguments:

- --input_dir: Argument to specify the path of the input directory
- --csv: Argument to specify the path of the CSV file
- --logger_level: Arguement to specify the level of the logs (INFO, WARNING, CRITICAL,...)
- --logger_file: Arguement to specify a file where to save the logs.

> [!IMPORTANT]
> When running with Apptainer, the folder containing all data should be bind to the image via --bind argument.

## Folder case

To use the image on your data, use the following command:
```
apptainer run --bind /path/to/data:/data path/to/wmh_segmenter.sif --input_dir /data
```
To use the Diffusion only, specify by using argument `--diff`:
```
apptainer run --bind /path/to/data:/data path/to/wmh_segmenter.sif --input_dir /data --diff
```

## CSV case
If you want to run the image with a given CSV, run the following:
```
apptainer run --bind /path/to/data:/data path/to/wmh_segmenter.sif --csv path/to/data.csv
```
