import os

import pandas as pd
from progress import step

# Define input/output data dir
DATA_DIR = "resources/mdace"
OUTPUT_DIR = "artifacts/mdace"

# Original column names from the input CSV files
ORIGINAL_CASE_ID = "hadm_id"
ORIGINAL_NOTE_ID = "note_id"
ORIGINAL_NOTE_TYPE = "category"
ORIGINAL_NOTE_TEXT = "text"
ORIGINAL_ANNOTATION_TEXTS = "covered_text"
ORIGINAL_ANNOTATION_CODES = "code"
ORIGINAL_START_POS = "begin"
ORIGINAL_END_POS = "end"
ORIGINAL_CHART_DATE = "CHARTDATE"
ORIGINAL_CHART_TIME = "CHARTTIME"

# Column names of the output dataset
FINAL_SPLIT = "split"
FINAL_CASE_ID = "case_id"
FINAL_SUBJECT_ID = "subject_id"
FINAL_NOTE_ID = "note_id"
FINAL_NOTE_TYPE = "note_type"
FINAL_NOTE_TEXT = "note_text"
FINAL_ANNOTATION_TEXTS = "annotation_texts"
FINAL_ANNOTATION_INDICES = "annotation_indices"
FINAL_ANNOTATION_CODES = "annotation_codes"
FINAL_SORT_TIME = "sort_time"


def prepare_data():
    with step("Loading MDACE annotations"):
        df = pd.read_parquet(f"{DATA_DIR}/Inpatient-ICD-10.parquet")

    # Rename columns for clarity and consistency
    df.rename(
        columns={
            ORIGINAL_CASE_ID: FINAL_CASE_ID,
            ORIGINAL_NOTE_ID: FINAL_NOTE_ID,
            ORIGINAL_NOTE_TYPE: FINAL_NOTE_TYPE,
            ORIGINAL_NOTE_TEXT: FINAL_NOTE_TEXT,
            ORIGINAL_ANNOTATION_TEXTS: FINAL_ANNOTATION_TEXTS,
            ORIGINAL_ANNOTATION_CODES: FINAL_ANNOTATION_CODES,
        },
        inplace=True,
    )

    # Rename doc type values
    df[FINAL_NOTE_TYPE] = df[FINAL_NOTE_TYPE].str.lower()

    # Group data by note ID and aggregate annotations into lists
    grouped_df = (
        df.groupby([FINAL_CASE_ID, FINAL_NOTE_ID])
        .agg(
            {
                FINAL_NOTE_TEXT: "first",
                FINAL_NOTE_TYPE: "first",
                FINAL_ANNOTATION_TEXTS: list,
                FINAL_ANNOTATION_CODES: list,
                ORIGINAL_START_POS: list,
                ORIGINAL_END_POS: list,
            }
        )
        .reset_index()
    )

    # Create the "annotation_indices" column as a list of (start, end) tuples
    grouped_df[FINAL_ANNOTATION_INDICES] = grouped_df.apply(
        lambda row: tuple(zip(row[ORIGINAL_START_POS], row[ORIGINAL_END_POS])), axis=1
    )
    # We no longer need the individual start and end position columns
    grouped_df.drop(columns=[ORIGINAL_START_POS, ORIGINAL_END_POS], inplace=True)

    # Get splits
    for split in ["train", "val", "test"]:
        case_ids = pd.read_csv(f"{DATA_DIR}/MDace-ev-{split}.csv", header=None)[0]
        grouped_df.loc[grouped_df[FINAL_CASE_ID].isin(case_ids), FINAL_SPLIT] = split
    assert grouped_df[FINAL_SPLIT].notna().all(), (
        "Some entries are not assigned to a valid split"
    )

    with step("Loading note timestamps from NOTEEVENTS"):
        notes = pd.read_csv(
            f"{DATA_DIR}/NOTEEVENTS.csv.gz",
            usecols=["ROW_ID", ORIGINAL_CHART_DATE, ORIGINAL_CHART_TIME],
            parse_dates=[ORIGINAL_CHART_DATE, ORIGINAL_CHART_TIME],
            low_memory=False,
        )
    grouped_df = pd.merge(
        grouped_df, notes, how="left", left_on=FINAL_NOTE_ID, right_on="ROW_ID"
    )

    # Use the chart time if available, otherwise the end of the chart date
    end_of_day = grouped_df[ORIGINAL_CHART_DATE] + pd.Timedelta(
        hours=23, minutes=59, seconds=59
    )
    grouped_df[FINAL_SORT_TIME] = grouped_df[ORIGINAL_CHART_TIME].fillna(end_of_day)

    # Build final df
    grouped_df[FINAL_SUBJECT_ID] = pd.NA
    final_df = grouped_df[
        [
            FINAL_SPLIT,
            FINAL_CASE_ID,
            FINAL_SUBJECT_ID,
            FINAL_NOTE_ID,
            FINAL_NOTE_TYPE,
            FINAL_NOTE_TEXT,
            FINAL_ANNOTATION_TEXTS,
            FINAL_ANNOTATION_INDICES,
            FINAL_ANNOTATION_CODES,
            FINAL_SORT_TIME,
        ]
    ].astype(
        {
            FINAL_SPLIT: "string",
            FINAL_CASE_ID: "string",
            FINAL_SUBJECT_ID: "string",
            FINAL_NOTE_ID: "string",
            FINAL_NOTE_TYPE: "string",
            FINAL_NOTE_TEXT: "string",
            FINAL_ANNOTATION_TEXTS: "object",
            FINAL_ANNOTATION_INDICES: "object",
            FINAL_ANNOTATION_CODES: "object",
            FINAL_SORT_TIME: "datetime64[ns]",
        }
    )

    return final_df


def main():
    dataset = prepare_data().reset_index(drop=True)

    with step("Exporting"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dataset.to_feather(f"{OUTPUT_DIR}/mdace.feather")
    print(f"MDACE is processed and exported to {OUTPUT_DIR}/mdace.feather!")
    print(dataset)


if __name__ == "__main__":
    main()
