# Exploring Trade-Offs in Human-AI Clinical Coding: Assign, Recommend, or Assist?

This repository accompanies the paper *"Exploring Trade-Offs in Human-AI Clinical Coding: Assign, Recommend, or Assist?"* (link TBD).

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Obtain the MDACE files. The official MDACE repository is no longer available. We used a local copy of it (commit `bf438d0`). If you need access to MDACE, please contact the [MDACE authors](https://aclanthology.org/2023.acl-long.416/).
    1. Obtain MDACE's ICD-10 inpatient annotations (`Inpatient-ICD-10.parquet`) by following the [official MDACE repository](https://github.com/3mcloud/MDACE).
    2. Put the annotation file under `resources/mdace/`.
    3. Download the three `MDace-ev-*` split files from [here](https://github.com/3mcloud/MDACE/tree/main/splits/Inpatient).
    4. Put the split files under `resources/mdace/`.
3. Run the following commands to create the environment, download the raw data, and prepare the datasets.

    ```bash
    uv sync
    bash prepare_data/download.sh <physionet-username>
    uv run python prepare_data/prepare_mdace.py
    uv run python prepare_data/prepare_mimiciv_icd10.py
    ```

    The download script downloads the MIMIC files from PhysioNet and [RoBERTa-base-PM-M3-Voc](https://github.com/facebookresearch/bio-lm), which several approaches use. It requires `wget` and credentialed access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/), and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/).
4. (Optional) Generate the entity text of MIMIC-IV ICD-10 with [Entity-Coding](https://github.com/jd4501/Entity-Coding/blob/main/docs/training.md). Put the entity-only `train.parquet`, `val.parquet`, and `test.parquet` under `resources/entity_text/`. Only approaches that use entity text (e.g., prime automation) need this step.

## Datasets

The prepared datasets are saved in `artifacts/`. See [prepare_data/README.md](prepare_data/README.md) for file paths, preprocessing details, and schemas.

Note: files under `resources/` and `artifacts/` contain MIMIC data. Per the PhysioNet data use agreement, do not share them.

## Running the approaches

Each of the nine approaches in the paper has its own directory, README, and virtual environment, so you can run any approach without installing the dependencies of the others. Complete the [setup](#setup) first, then follow the README in the approach's directory.

## Development

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run `uv run ruff check . && uv run ruff format .`.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
