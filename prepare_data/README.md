# Datasets

## MDACE

Saved in `artifacts/mdace/mdace.feather`.

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

## MIMIC-IV ICD-10

Saved in `artifacts/mimiciv_icd10/mimiciv_icd10_all_sorted.feather`. In `artifacts/mimiciv_icd10/mimiciv_icd10_all_parents_sorted.feather`, every code is replaced by its parent code (see the paper for the definition).

The preprocessing follows [Edin et al. (2023)](https://github.com/JoakimEdin/medical-coding-reproducibility), except that infrequent codes are kept. Admissions are randomly split by patient into train (55%), val (20%), and test (25%) sets.

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
