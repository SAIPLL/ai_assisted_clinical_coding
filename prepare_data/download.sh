#!/usr/bin/env bash
# Download the raw MIMIC files from PhysioNet and RoBERTa-base-PM-M3-Voc into resources/
# Usage: bash prepare_data/download.sh <physionet-username>

wget -c --user "$1" --ask-password -P resources/mdace \
    https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz

wget -c --user "$1" --ask-password -P resources/mimiciv \
    https://physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz \
    https://physionet.org/files/mimiciv/2.2/hosp/procedures_icd.csv.gz \
    https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz

# RoBERTa-base-PM-M3-Voc, used by several approaches
mkdir -p resources/models
wget -O - https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz | tar -xz --strip-components=1 -C resources/models
