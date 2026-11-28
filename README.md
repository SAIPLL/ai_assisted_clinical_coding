# Comprehensive Exploration of Human-AI Collaboration in Clinical Coding

## Overview

This repository accompanies the paper *"Exploring Trade-Offs in Human-AI Clinical Coding: Assign, Recommend, or Assist?"* (link TBD).

Each of the nine approaches introduced in the paper is in its own directory.

## How to run this codebase

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Create the environment and install dependencies `uv sync`
3. Download the raw MIMIC files from PhysioNet into `resources/` `bash prepare_data/download.sh <physionet-username>` (requires `wget` and credentialed access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/), and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/))

Note: files under `resources/` and `artifacts/` contain MIMIC data. Per the PhysioNet data use agreement, do not share them.

### Preparing MDACE
Note: the official MDACE repository is no longer available. We used a local copy of it (commit `bf438d0`). If you need access to MDACE, please contact the [MDACE authors](https://aclanthology.org/2023.acl-long.416/).

1. Follow the instructions in the [official MDACE repository](https://github.com/3mcloud/MDACE) to obtain MDACE's ICD-10 inpatient annotations, which is a file named `Inpatient-ICD-10.parquet`
2. Put the annotation file under the directory `resources/mdace/`
3. Download the three `MDace-ev-*` split files used by MDACE from [here](https://github.com/3mcloud/MDACE/tree/main/splits/Inpatient)
4. Put the three split files under the directory `resources/mdace/`
5. Run `uv run python prepare_data/prepare_mdace.py` to reformat MDACE in a schema supported by this codebase.

The formatted dataset can be found in `artifacts/mdace/mdace.feather`. Schema:

| Column Name        | Data Type         | Description                                               |
|--------------------|-------------------|-----------------------------------------------------------|
| split              | string            | Data partition: "train", "val", or "test"                 |
| case_id            | string            | Unique identifier for each specific case                  |
| note_id            | string            | Unique identifier for the specific clinical note          |
| note_type          | string            | The category of note (e.g., discharge summary, radiology) |
| note_text          | string            | The raw text content of the clinical note                 |
| annotation_texts   | object            | List of text spans used as evidence for the codes         |
| annotation_indices | object            | Character start/end positions for the evidence spans      |
| annotation_codes   | object            | List of associated ICD-10-CM/ICD-10-PCS codes             |
| sort_time          | datetime64[ns]    | Timestamp indicating when the note was recorded           |

### Preparing MIMIC-IV
1. Run `uv run python prepare_data/prepare_mimiciv_icd10.py` to build MIMIC-IV ICD-10.

The preprocessing follows [Edin et al. (2023)](https://github.com/JoakimEdin/medical-coding-reproducibility), except that infrequent codes are kept. Admissions are randomly split by patient into train (55%), val (20%), and test (25%) sets.

The formatted dataset can be found in `artifacts/mimiciv_icd10/mimiciv_icd10_all_sorted.feather`. A version in which every code is replaced by its parent code (refer to the paper for definition) can be found in `artifacts/mimiciv_icd10/mimiciv_icd10_all_parents_sorted.feather`. Schema:

| Column Name       | Data Type | Description                                                         |
|-------------------|-----------|---------------------------------------------------------------------|
| note_id           | string    | Unique identifier for the specific discharge summary                |
| subject_id        | int       | Unique identifier for the patient                                   |
| _id               | int       | Unique identifier for the admission                                 |
| note_type         | string    | The category of note (i.e., DS for discharge summary)               |
| note_seq          | int       | Sequence number of the note                                         |
| charttime         | string    | Timestamp indicating when the note was charted                      |
| storetime         | string    | Timestamp indicating when the note was stored                       |
| text              | string    | The preprocessed text content of the discharge summary              |
| icd10_proc        | object    | List of ICD-10-PCS codes, sorted alphabetically                     |
| icd10_diag        | object    | List of ICD-10-CM codes, sorted alphabetically                      |
| target            | object    | List of all codes (`icd10_proc` followed by `icd10_diag`)           |
| num_words         | int       | Number of words in `text`                                           |
| num_targets       | int       | Number of codes in `target`                                         |
| has_primary_diag  | bool      | Whether `target` has any code that is ever a primary diagnosis      |
| has_primary_proc  | bool      | Whether `target` has any code that is ever a primary procedure      |
| split             | string    | Data partition: "train", "val", or "test"                           |
| icd10_diag_sorted | object    | List of ICD-10-CM codes, sorted by MIMIC-IV sequence number         |
| icd10_proc_sorted | object    | List of ICD-10-PCS codes, sorted by MIMIC-IV sequence number        |

### Running the AI-assisted coding approaches

The dependencies required by individual approaches may be incompatible. For this reason, we use a separate virtual environment for each approach. We recommend following the same practice rather than using a single shared environment.

To run a specific approach, navigate to the corresponding project subdirectory and follow the instructions in its README.

### Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run `uv run ruff check . && uv run ruff format .`
