import os

import pandas as pd
from progress import step

# Define input/output data dir
DATA_DIR = "resources/mimiciv"
OUTPUT_DIR = "artifacts/mimiciv_icd10"

# Column names used by the output dataset
ID_COLUMN = "_id"
SUBJECT_ID_COLUMN = "subject_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SPLIT_COLUMN = "split"

# Split ratios and seed of the patient-level random split
TRAIN_RATIO = 0.55
VAL_RATIO = 0.20
SPLIT_SEED = 10


def reformat_icd10(codes, is_diag):
    """
    Put a period in the right place because the MIMIC-IV data files exclude them.
    Diagnosis codes have a period after the first three characters; procedure codes have none.
    """
    codes = codes.str.replace(".", "", regex=False)
    if not is_diag:
        return codes
    return codes.str[:3] + "." + codes.str[3:]


def get_icd10_parent_codes(codes):
    """
    Given a list of full codes, return a list of corresponding parent codes while preserving order
    """
    result = []
    for c in codes:
        if len(c) == 7 and "." not in c:  # Procedure code
            result.append(c[:3])
        else:
            result.append(c.split(".")[0])  # Diagnosis code
    return result


def read_codes(filename, is_diag):
    """Read the ICD-10 codes of a MIMIC-IV code file (diagnoses_icd or procedures_icd)"""
    df = pd.read_csv(f"{DATA_DIR}/{filename}", dtype={"icd_code": str})
    df = df[df["icd_version"] == 10].dropna(subset=["icd_code"]).copy()
    df["icd_code"] = reformat_icd10(df["icd_code"], is_diag=is_diag)
    return df


def group_codes(df, column_name):
    """Group the codes of each admission into a sorted list"""
    df = df.rename(columns={"hadm_id": ID_COLUMN})
    df = df.drop_duplicates(subset=[ID_COLUMN, "icd_code"])
    return (
        df.groupby(ID_COLUMN)["icd_code"]
        .agg(lambda codes: sorted(codes))
        .rename(column_name)
        .reset_index()
    )


def preprocess_text(text):
    """Lowercase, keep alphanumerics, and remove standalone numbers"""
    text = text.str.lower()
    text = text.str.replace(r"[^A-Za-z0-9]+", " ", regex=True)
    text = text.str.replace(r"(\s\d+)+\s", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    return text.str.strip()


def prepare_data(diag_df, proc_df):
    """Build the MIMIC-IV ICD-10 dataset, without removing infrequent codes"""
    with step("Loading discharge summaries"):
        notes = pd.read_csv(f"{DATA_DIR}/discharge.csv.gz")
    notes = notes.rename(columns={"hadm_id": ID_COLUMN})
    notes = notes.dropna(subset=[TEXT_COLUMN])
    notes = notes.drop_duplicates(subset=[ID_COLUMN, TEXT_COLUMN])

    df = notes.merge(group_codes(proc_df, "icd10_proc"), on=ID_COLUMN, how="left")
    df = df.merge(group_codes(diag_df, "icd10_diag"), on=ID_COLUMN, how="left")

    # Remove notes with no codes and convert NaNs to empty lists
    df = df.dropna(subset=["icd10_proc", "icd10_diag"], how="all")
    for column in ["icd10_proc", "icd10_diag"]:
        df[column] = df[column].apply(
            lambda codes: codes if isinstance(codes, list) else []
        )

    df[TARGET_COLUMN] = df["icd10_proc"] + df["icd10_diag"]
    df = df[df[TARGET_COLUMN].apply(len) > 0].reset_index(drop=True)

    with step("Preprocessing text"):
        df[TEXT_COLUMN] = preprocess_text(df[TEXT_COLUMN])
    df["num_words"] = df[TEXT_COLUMN].str.count(" ") + 1
    df["num_targets"] = df[TARGET_COLUMN].apply(len)
    return df


def get_splits(df):
    """Randomly split patients so that train/val/test hold ~55%/20%/25% of the admissions"""
    subjects = df.groupby(SUBJECT_ID_COLUMN).size().reset_index(name="row_count")
    subjects = subjects.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
    cumulative_rows = subjects["row_count"].cumsum()
    train_threshold = df.shape[0] * TRAIN_RATIO
    val_threshold = df.shape[0] * (TRAIN_RATIO + VAL_RATIO)
    subjects[SPLIT_COLUMN] = cumulative_rows.apply(
        lambda rows: (
            "train"
            if rows <= train_threshold
            else "val"
            if rows <= val_threshold
            else "test"
        )
    )
    splits = df[[ID_COLUMN, SUBJECT_ID_COLUMN]].merge(
        subjects, on=SUBJECT_ID_COLUMN, how="left"
    )
    return splits[[ID_COLUMN, SPLIT_COLUMN]]


def sort_codes(df, diag_df, proc_df, splits):
    """Order the codes of each admission by their MIMIC-IV sequence number (i.e., coding priority)"""
    sorted_codes = {}
    primary_codes = {}
    for code_type, seq_df in [("diag", diag_df), ("proc", proc_df)]:
        seq_df = seq_df[seq_df["hadm_id"].isin(df[ID_COLUMN])]
        primary_codes[code_type] = set(seq_df[seq_df["seq_num"] == 1]["icd_code"])
        sorted_codes[code_type] = (
            seq_df.sort_values(by=["hadm_id", "seq_num"])
            .groupby("hadm_id")["icd_code"]
            .agg(list)
            .rename(f"icd10_{code_type}_sorted")
            .reset_index()
            .rename(columns={"hadm_id": ID_COLUMN})
        )

    # Whether the admission contains a code that is the primary code of any admission
    for code_type in ["diag", "proc"]:
        df[f"has_primary_{code_type}"] = ~df[TARGET_COLUMN].map(
            primary_codes[code_type].isdisjoint
        )
    df = df.merge(splits, on=ID_COLUMN, how="inner")

    def filter_codes(row, code_type):
        codes, ordered = row[f"icd10_{code_type}"], row[f"icd10_{code_type}_sorted"]
        if not isinstance(ordered, list):  # This admission has no `code_type` codes
            return codes
        assert set(codes) <= set(ordered), (
            f"Admission {row[ID_COLUMN]} has {code_type} codes without a sequence number"
        )
        return [code for code in ordered if code in codes]

    for code_type in ["diag", "proc"]:
        df = df.merge(sorted_codes[code_type], on=ID_COLUMN, how="left")
        df[f"icd10_{code_type}_sorted"] = df.apply(
            filter_codes, axis=1, code_type=code_type
        )
    return df


def to_parent_codes(df):
    """Replace every code by its parent (category) code"""
    df = df.copy()
    for column in [
        TARGET_COLUMN,
        "icd10_diag",
        "icd10_proc",
        "icd10_diag_sorted",
        "icd10_proc_sorted",
    ]:
        df[column] = df[column].apply(get_icd10_parent_codes)
    return df


def main():
    with step("Loading MIMIC-IV codes"):
        diag_df = read_codes("diagnoses_icd.csv.gz", is_diag=True)
        proc_df = read_codes("procedures_icd.csv.gz", is_diag=False)

    dataset = prepare_data(diag_df, proc_df)
    with step("Splitting patients and sorting codes by sequence number"):
        dataset = sort_codes(dataset, diag_df, proc_df, get_splits(dataset))

    with step("Exporting"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dataset.to_feather(f"{OUTPUT_DIR}/mimiciv_icd10_all_sorted.feather")
        to_parent_codes(dataset).to_feather(
            f"{OUTPUT_DIR}/mimiciv_icd10_all_parents_sorted.feather"
        )
    print(
        f"MIMIC-IV ICD-10 is processed and exported to {OUTPUT_DIR}/mimiciv_icd10_all_sorted.feather and {OUTPUT_DIR}/mimiciv_icd10_all_parents_sorted.feather!"
    )
    print(dataset)


if __name__ == "__main__":
    main()
