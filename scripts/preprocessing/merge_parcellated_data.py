#!/usr/bin/env python3
"""Merge subject-level parcellated CSV files into one analysis table.

The script accepts CSV files produced by Connectome Workbench after CIFTI
parcellation. Each subject file may be stored as either ``time x parcels`` or
``parcels x time``. The script standardizes the orientation, removes the first
N TRs, and concatenates subjects into a long table with columns::

    subject, time, 1, 2, ..., N_PARCELS

No participant data or project-specific paths are stored in this repository.

Example
-------
python merge_parcellated_data.py \
    --csv-dir /path/to/parcellated_csv \
    --output-file /path/to/movie4.csv \
    --n-parcels 379 \
    --remove-trs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing subject-level *_parcellated.csv files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Destination CSV for the merged subject-by-time table.",
    )
    parser.add_argument(
        "--n-parcels",
        type=int,
        default=379,
        help="Expected number of parcels (default: 379).",
    )
    parser.add_argument(
        "--remove-trs",
        type=int,
        default=10,
        help="Number of initial TRs to remove from every subject (default: 10).",
    )
    parser.add_argument(
        "--pattern",
        default="*_parcellated.csv",
        help="Glob pattern used to locate subject files.",
    )
    return parser.parse_args()


def load_subject_csv(path: Path, n_parcels: int, remove_trs: int) -> pd.DataFrame:
    """Load one Workbench CSV and return a time-by-parcel DataFrame."""
    data = pd.read_csv(path, header=None).to_numpy()

    if data.shape[0] == n_parcels:
        data = data.T
    elif data.shape[1] != n_parcels:
        raise ValueError(
            f"{path.name} has shape {data.shape}; expected "
            f"{n_parcels} x T or T x {n_parcels}."
        )

    if remove_trs < 0:
        raise ValueError("--remove-trs must be >= 0.")
    if data.shape[0] <= remove_trs:
        raise ValueError(
            f"{path.name} has only {data.shape[0]} time points; "
            f"cannot remove {remove_trs}."
        )

    data = data[remove_trs:, :]
    subject_id = path.name.split("_")[0]

    frame = pd.DataFrame(data, columns=[str(i + 1) for i in range(n_parcels)])
    frame.insert(0, "time", range(len(frame)))
    frame.insert(0, "subject", subject_id)

    print(f"{subject_id}: retained shape {data.shape}")
    return frame


def main() -> None:
    args = parse_args()
    csv_files = sorted(args.csv_dir.glob(args.pattern))
    csv_files = [p for p in csv_files if p.resolve() != args.output_file.resolve()]

    if not csv_files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} were found in {args.csv_dir}."
        )

    print(f"Found {len(csv_files)} subject files.")
    frames = [
        load_subject_csv(path, args.n_parcels, args.remove_trs)
        for path in csv_files
    ]

    merged = pd.concat(frames, axis=0, ignore_index=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_file, index=False)

    print(f"Saved merged table: {args.output_file}")
    print(f"Final shape: {merged.shape}")


if __name__ == "__main__":
    main()
