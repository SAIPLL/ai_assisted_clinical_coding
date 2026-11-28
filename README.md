# Exploring Trade-Offs in Human-AI Clinical Coding: Assign, Recommend, or Assist?

This repository accompanies the paper *"Exploring Trade-Offs in Human-AI Clinical Coding: Assign, Recommend, or Assist?"* (link TBD).

Each of the nine approaches introduced in the paper is in its own directory.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Obtain the MDACE files. Note: the official MDACE repository is no longer available. We used a local copy of it (commit `bf438d0`). If you need access to MDACE, please contact the [MDACE authors](https://aclanthology.org/2023.acl-long.416/).
    1. Follow the instructions in the [official MDACE repository](https://github.com/3mcloud/MDACE) to obtain MDACE's ICD-10 inpatient annotations, which is a file named `Inpatient-ICD-10.parquet`
    2. Put the annotation file under the directory `resources/mdace/`
    3. Download the three `MDace-ev-*` split files used by MDACE from [here](https://github.com/3mcloud/MDACE/tree/main/splits/Inpatient)
    4. Put the three split files under the directory `resources/mdace/`
3. Run the following commands to create the environment, download the raw MIMIC files from PhysioNet, and prepare the datasets. Downloading requires `wget` and credentialed access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/), and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/).

```bash
uv sync
bash prepare_data/download.sh <physionet-username>
uv run python prepare_data/prepare_mdace.py
uv run python prepare_data/prepare_mimiciv_icd10.py
```

## Datasets

The prepared datasets are saved in `artifacts/`. See [prepare_data/README.md](prepare_data/README.md) for file paths, preprocessing details, and schemas.

Note: files under `resources/` and `artifacts/` contain MIMIC data. Per the PhysioNet data use agreement, do not share them.

## Running the approaches

The dependencies required by individual approaches may be incompatible. For this reason, we use a separate virtual environment for each approach. We recommend following the same practice rather than using a single shared environment.

To run a specific approach, navigate to the corresponding project subdirectory and follow the instructions in its README.

## Development

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run `uv run ruff check . && uv run ruff format .`

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
