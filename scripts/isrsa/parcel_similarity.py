#!/usr/bin/env python3
"""Create a subject-by-subject similarity matrix for a user-specified parcel.

The script does not encode any study result or selected parcel identity. The parcel
number must be supplied at runtime. It saves the original-order similarity matrix
and an optional BMI-sorted version for visualization.

The brain array is expected to have shape ``subjects x TRs x parcels`` and the
phenotype rows must preserve the same subject order as the first array dimension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--parcel", type=int, required=True, help="One-based parcel number.")
    parser.add_argument("--parcel-name", default=None, help="Optional display label.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--bmi-column", default="BMI")
    parser.add_argument("--n-cortical-parcels", type=int, default=360)
    parser.add_argument("--vmin", type=float, default=-0.25)
    parser.add_argument("--vmax", type=float, default=0.25)
    parser.add_argument(
        "--skip-bmi-sorted",
        action="store_true",
        help="Do not create the BMI-sorted matrix/plot.",
    )
    return parser.parse_args()


def save_heatmap(matrix: np.ndarray, path: Path, title: str, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        matrix,
        ax=ax,
        square=True,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson correlation between parcel time series"},
    )
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.brain)
    if data.ndim != 3:
        raise ValueError("Expected subjects x TRs x parcels brain array.")
    if data.shape[2] < args.n_cortical_parcels:
        raise ValueError("Brain array contains fewer parcels than requested.")
    if not 1 <= args.parcel <= args.n_cortical_parcels:
        raise ValueError("--parcel must be within the retained cortical parcel range.")
    data = data[:, :, : args.n_cortical_parcels]

    phenotype = pd.read_csv(args.phenotype)
    if args.subject_column not in phenotype.columns:
        raise ValueError(f"Missing phenotype column: {args.subject_column}")
    if len(phenotype) != data.shape[0]:
        raise ValueError("Phenotype rows must match brain-array subject order.")

    subject_labels = phenotype[args.subject_column].astype(str).str.strip().to_numpy()
    parcel_index = args.parcel - 1
    parcel_label = args.parcel_name or f"parcel_{args.parcel}"
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in parcel_label)

    parcel_data = data[:, :, parcel_index]
    if not np.isfinite(parcel_data).all():
        raise ValueError("Selected parcel contains non-finite values.")
    if np.any(np.isclose(np.std(parcel_data, axis=1), 0.0)):
        raise ValueError("Selected parcel contains a constant subject time series.")

    similarity = 1.0 - pairwise_distances(parcel_data, metric="correlation")
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)

    stem = f"parcel_{args.parcel}_{safe_label}"
    np.save(args.output_dir / f"{stem}_similarity.npy", similarity)
    pd.DataFrame(similarity, index=subject_labels, columns=subject_labels).to_csv(
        args.output_dir / f"{stem}_similarity.csv"
    )
    save_heatmap(
        similarity,
        args.output_dir / f"{stem}_similarity.png",
        f"{parcel_label} (parcel {args.parcel}): original subject order",
        args.vmin,
        args.vmax,
    )

    if not args.skip_bmi_sorted:
        if args.bmi_column not in phenotype.columns:
            raise ValueError(
                f"BMI-sorted output requested but phenotype lacks {args.bmi_column!r}."
            )
        bmi = pd.to_numeric(phenotype[args.bmi_column], errors="coerce").to_numpy(float)
        if not np.isfinite(bmi).all():
            raise ValueError("BMI contains missing/non-finite values.")
        order = np.argsort(bmi, kind="stable")
        sorted_similarity = similarity[np.ix_(order, order)]
        sorted_subjects = subject_labels[order]
        sorted_bmi = bmi[order]

        np.save(args.output_dir / f"{stem}_similarity_sorted_by_BMI.npy", sorted_similarity)
        pd.DataFrame(sorted_similarity, index=sorted_subjects, columns=sorted_subjects).to_csv(
            args.output_dir / f"{stem}_similarity_sorted_by_BMI.csv"
        )
        save_heatmap(
            sorted_similarity,
            args.output_dir / f"{stem}_similarity_sorted_by_BMI.png",
            f"{parcel_label} (parcel {args.parcel}): subjects sorted by BMI",
            args.vmin,
            args.vmax,
        )

    print("Saved parcel-similarity outputs to", args.output_dir)


if __name__ == "__main__":
    main()
