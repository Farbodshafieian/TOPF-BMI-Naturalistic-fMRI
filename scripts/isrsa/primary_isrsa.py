#!/usr/bin/env python3
"""Run the primary parcel-wise BMI intersubject RSA (IS-RSA).

For each cortical parcel, the script computes a subject-by-subject neural
similarity matrix using Pearson correlation between parcel time series. BMI is
ranked and converted to a nearest-neighbor similarity matrix. Neural similarity
is then related to BMI similarity with Spearman correlation using nltools,
including subject-label permutation testing and Benjamini-Hochberg FDR across
parcels.

The study used 360 HCP-MMP cortical parcels and 5,000 permutations.

Important
---------
Rows in the phenotype CSV must be in the same subject order as the first axis
of the brain ``.npy`` array.

Example
-------
python primary_isrsa.py \
    --brain df_X_bridgeville_subjects_TRs_parcels.npy \
    --phenotype Subjects_Info.csv \
    --output-dir outputs/bridgeville_primary \
    --n-permutations 5000
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
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import fdrcorrection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, required=True, help="subjects x TRs x parcels .npy file")
    parser.add_argument("--phenotype", type=Path, required=True, help="CSV containing BMI")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bmi-column", default="BMI")
    parser.add_argument("--n-parcels", type=int, default=360)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def sort_square_matrix(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    return matrix[np.ix_(order, order)]


def main() -> None:
    args = parse_args()
    from nltools.data import Adjacency
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.brain)
    if data.ndim != 3:
        raise ValueError(f"Expected subjects x TRs x parcels; found {data.shape}.")
    if data.shape[2] < args.n_parcels:
        raise ValueError(
            f"Requested {args.n_parcels} parcels but array contains {data.shape[2]}."
        )
    data = data[:, :, : args.n_parcels]

    phenotype = pd.read_csv(args.phenotype)
    if args.bmi_column not in phenotype.columns:
        raise ValueError(f"Missing BMI column: {args.bmi_column}")
    bmi = pd.to_numeric(phenotype[args.bmi_column], errors="coerce").to_numpy(float)

    if len(bmi) != data.shape[0]:
        raise ValueError(
            "Phenotype row count must match the subject dimension of the brain array."
        )
    if not np.isfinite(bmi).all():
        raise ValueError("BMI contains missing or non-finite values.")

    print("Brain data shape:", data.shape)

    brain_adjacencies = []
    for parcel in range(args.n_parcels):
        if parcel % 25 == 0:
            print(f"Building parcel {parcel + 1}/{args.n_parcels}")
        brain_similarity = 1.0 - pairwise_distances(
            data[:, :, parcel], metric="correlation"
        )
        brain_adjacencies.append(
            Adjacency(brain_similarity, matrix_type="similarity")
        )
    brain_adjacencies = Adjacency(brain_adjacencies)

    bmi_rank = rankdata(bmi, method="average")
    bmi_similarity = Adjacency(
        pairwise_distances(bmi_rank.reshape(-1, 1), metric="euclidean"),
        matrix_type="distance",
    ).distance_to_similarity()

    # Sanity-check figures used during the study analysis.
    brain_adjacencies[0].plot(vmin=-0.25, vmax=0.25, cmap="RdBu_r")
    plt.savefig(args.output_dir / "parcel1_similarity.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bmi_similarity.plot(axes=axes[0])
    axes[0].set_title("BMI nearest-neighbor similarity")
    sns.heatmap(
        sort_square_matrix(bmi_similarity.squareform(), bmi),
        ax=axes[1],
        square=True,
    )
    axes[1].set_title("BMI similarity sorted by BMI")
    fig.tight_layout()
    fig.savefig(args.output_dir / "bmi_similarity_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for parcel in range(args.n_parcels):
        print(f"IS-RSA parcel {parcel + 1}/{args.n_parcels}")
        stats = brain_adjacencies[parcel].similarity(
            bmi_similarity,
            metric="spearman",
            n_permute=args.n_permutations,
            n_jobs=args.n_jobs,
        )
        rows.append(
            {
                "parcel": parcel + 1,
                "bmi_nn_r": stats["correlation"],
                "bmi_nn_perm_p": stats["p"],
            }
        )

    results = pd.DataFrame(rows)
    significant, q_values = fdrcorrection(results["bmi_nn_perm_p"], alpha=0.05)
    results["bmi_nn_qval_fdr"] = q_values
    results["bmi_nn_significant_fdr"] = significant
    results["n_subjects"] = data.shape[0]
    results["n_trs"] = data.shape[1]
    results["n_permutations"] = args.n_permutations
    results = results.sort_values("bmi_nn_r", ascending=False)

    output = args.output_dir / "bmi_nn_isrsa_results.csv"
    results.to_csv(output, index=False)
    print(f"Saved: {output}")
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
