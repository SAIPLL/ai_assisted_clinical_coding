import os
import sys

import numpy as np
import pandas as pd

# Shared data preparation code in the repository root
sys.path.insert(0, "../prepare_data")
from format_entity_text import format_entity_text, read_entity_text
from progress import step

ENTITY_TEXT_DIR = "../resources/entity_text"
MIMICIV_PATH = "../artifacts/mimiciv_icd10/mimiciv_icd10_all_sorted.feather"
OUTPUT_DIR = "artifacts/mimiciv_icd10"

SECTIONS = ("diagnoses", "hospital_course", "patient_instructions")
TYPES = ("abnormal_finding", "disorder", "procedure")


def first_code(codes):
    """The primary diagnosis code (PDC), i.e., the first code by sequence number"""
    return [codes[0]] if len(codes) else np.nan


def prepare_plm_icd(df):
    """Train on all diagnosis codes that are a PDC of some admission; evaluate on the PDC only"""
    unique_pdcs = set(df["icd10_diag_sorted"].apply(first_code).dropna().explode())

    def retain_pdcs_only(codes):
        codes = [code for code in codes if code in unique_pdcs]
        return codes if len(codes) else np.nan

    df["icd10_diag"] = df["icd10_diag_sorted"].apply(retain_pdcs_only)
    df["icd10_proc"] = df["icd10_proc_sorted"].apply(lambda codes: [])
    df["target"] = df["icd10_diag"]

    is_test = df["split"] == "test"
    df_test = df[is_test].copy()
    df_test["icd10_diag"] = df_test["icd10_diag_sorted"].apply(first_code)
    df_test["target"] = df_test["icd10_diag"]

    df = pd.concat([df[~is_test], df_test], axis=0)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def prepare_plm_pdc(df):
    """Train and evaluate on the PDC only"""
    df["target"] = df["icd10_diag_sorted"].apply(first_code)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["icd10_diag"] = df["target"]
    df["icd10_proc"] = df["target"].apply(lambda codes: [])
    return df


def main():
    with step("Loading and formatting entity text"):
        df = read_entity_text(ENTITY_TEXT_DIR)
        codes = pd.read_feather(
            MIMICIV_PATH,
            columns=[
                "_id",
                "split",
                "icd10_diag_sorted",
                "icd10_proc_sorted",
                "target",
            ],
        )
        df = format_entity_text(
            pd.merge(df, codes, how="left", on="_id"), SECTIONS, TYPES
        )

    with step("Preparing and exporting datasets"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for name, prepare in [
            ("plm_icd", prepare_plm_icd),
            ("plm_pdc", prepare_plm_pdc),
        ]:
            dataset = prepare(df.copy()).drop(
                columns=["icd10_diag_sorted", "icd10_proc_sorted"]
            )
            dataset["num_targets"] = dataset["target"].apply(len)
            dataset.to_feather(
                f"{OUTPUT_DIR}/mimiciv_icd10_prime_automation_{name}.feather"
            )
    print(
        f"Prime automation datasets are exported to {OUTPUT_DIR}/mimiciv_icd10_prime_automation_{{plm_icd,plm_pdc}}.feather!"
    )


if __name__ == "__main__":
    main()
