#!/usr/bin/env bash
# Parcellate HCP CIFTI movie time series and export subject-level CSV files.
#
# This script intentionally contains no cluster account names, email addresses,
# or project-specific absolute paths. Supply all paths at runtime.
#
# Usage:
#   bash parcellate_hcp_movie.sh \
#     /path/to/input_dtseries \
#     /path/to/atlas.dlabel.nii \
#     /path/to/parcellated_output \
#     /path/to/csv_output \
#     '*_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii'
#
# Optional environment variable:
#   WB_COMMAND=/path/to/wb_command
#
# The original study used Connectome Workbench mean aggregation within parcels.

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 INPUT_DIR LABEL_FILE PARC_DIR CSV_DIR [FILE_GLOB]" >&2
    exit 2
fi

INPUT_DIR="$1"
LABEL_FILE="$2"
PARC_DIR="$3"
CSV_DIR="$4"
FILE_GLOB="${5:-*_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii}"
WB_COMMAND="${WB_COMMAND:-wb_command}"

mkdir -p "$PARC_DIR" "$CSV_DIR"

if ! command -v "$WB_COMMAND" >/dev/null 2>&1; then
    echo "Could not find Connectome Workbench command: $WB_COMMAND" >&2
    exit 1
fi

"$WB_COMMAND" -version

shopt -s nullglob
files=("$INPUT_DIR"/$FILE_GLOB)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No input files matched: $INPUT_DIR/$FILE_GLOB" >&2
    exit 1
fi

for dtseries in "${files[@]}"; do
    fname="$(basename "$dtseries")"
    subject="${fname%%_*}"

    parcellated="$PARC_DIR/${subject}_parcellated.ptseries.nii"
    csv_out="$CSV_DIR/${subject}_parcellated.csv"

    echo "Processing subject $subject"

    "$WB_COMMAND" -cifti-parcellate \
        "$dtseries" \
        "$LABEL_FILE" \
        COLUMN \
        "$parcellated" \
        -method MEAN

    "$WB_COMMAND" -cifti-convert -to-text \
        "$parcellated" \
        "$csv_out" \
        -col-delim ","
done

echo "Parcellation complete."
echo "Run merge_parcellated_data.py separately to remove initial TRs and merge subjects."
