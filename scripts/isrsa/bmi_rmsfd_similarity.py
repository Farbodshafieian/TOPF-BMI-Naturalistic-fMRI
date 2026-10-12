#!/usr/bin/env python3
"""Test whether participants with similar BMI also have similar RMSFD.

This script reproduces the study's BMI-similarity versus RMSFD-similarity
analysis for a user-specified naturalistic-viewing condition.

For BMI and RMSFD separately, the script:
1. average-ranks participant values;
2. computes pairwise Euclidean distances between ranks; and
3. converts rank distance to similarity as s_ij = 1 - d_ij.

It then relates the upper triangles of the two subject-by-subject similarity
matrices using Spearman correlation and a two-sided subject-label permutation
test. In each permutation, the same random permutation is applied to the rows
and columns of the RMSFD similarity matrix while the BMI similarity matrix stays
fixed. The empirical p-value uses a +1 correction in the numerator and
denominator.

This analysis tests whether BMI similarity is associated with RMSFD similarity.
It does not test whether RMSFD explains neural similarity.

Example
-------
python bmi_rmsfd_similarity.py \
    --phenotype /work/bridgeville/Subjects_Info.csv \
    --motion /work/bridgeville/RMSFD_subjects.csv \
    --output-dir /work/isrsa/bridgeville_bmi_rmsfd \
    --stimulus Bridgeville \
    --n-permutations 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import pairwise_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phenotype",
        type=Path,
        required=True,
        help="CSV containing participant IDs and BMI.",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        required=True,
        help="CSV containing participant IDs and HCP-provided RMSFD.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stimulus",
        default="unspecified",
        help="Label stored in the output table, e.g. Bridgeville or Movie4.",
    )
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--bmi-column", default="BMI")
    parser.add_argument("--motion-column", default="RMSFD")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for reproducible subject-label permutations.",
    )
    return parser.parse_args()


def clean_subject_id(series: pd.Series) -> pd.Series:
    """Standardize subject IDs before merging tables."""
    return (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    """Raise an informative error when required columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def rank_based_similarity(values: np.ndarray) -> np.ndarray:
    """Create a rank-based subject-by-subject similarity matrix.

    Average ranks are assigned first. Pairwise Euclidean distance on these
    one-dimensional ranks equals absolute rank difference. Similarity is then
    defined as s_ij = 1 - d_ij, matching the primary IS-RSA convention used in
    the study repository.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Similarity input contains missing or non-finite values.")

    ranked = rankdata(values, method="average")
    distance = pairwise_distances(
        ranked.reshape(-1, 1),
        metric="euclidean",
    )
    return 1.0 - distance


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the unique off-diagonal entries of a symmetric matrix."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def permutation_test(
    bmi_similarity: np.ndarray,
    rmsfd_similarity: np.ndarray,
    n_permutations: int,
    random_seed: int,
) -> tuple[float, float]:
    """Run a two-sided subject-label permutation test.

    RMSFD participant labels are shuffled by applying the same permutation to
    matrix rows and columns. BMI similarity remains fixed. Each permutation
    recomputes Spearman's rho from the upper triangles.
    """
    bmi_vector = upper_triangle(bmi_similarity)
    rmsfd_vector = upper_triangle(rmsfd_similarity)

    observed = float(spearmanr(rmsfd_vector, bmi_vector).statistic)
    if not np.isfinite(observed):
        raise ValueError("Observed Spearman correlation is non-finite.")

    rng = np.random.default_rng(random_seed)
    permuted_rhos = np.empty(n_permutations, dtype=float)

    n_subjects = bmi_similarity.shape[0]
    for index in range(n_permutations):
        order = rng.permutation(n_subjects)
        permuted_rmsfd = rmsfd_similarity[np.ix_(order, order)]
        permuted_vector = upper_triangle(permuted_rmsfd)
        permuted_rhos[index] = spearmanr(
            permuted_vector,
            bmi_vector,
        ).statistic

    valid = np.isfinite(permuted_rhos)
    if not valid.any():
        raise ValueError("All permutation correlations are non-finite.")

    permutation_p = (
        np.sum(np.abs(permuted_rhos[valid]) >= abs(observed)) + 1
    ) / (valid.sum() + 1)

    return observed, float(permutation_p)


def main() -> None:
    args = parse_args()

    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    phenotype = pd.read_csv(args.phenotype)
    motion = pd.read_csv(args.motion)

    require_columns(
        phenotype,
        [args.subject_column, args.bmi_column],
        "Phenotype file",
    )
    require_columns(
        motion,
        [args.subject_column, args.motion_column],
        "Motion file",
    )

    phenotype = phenotype[
        [args.subject_column, args.bmi_column]
    ].copy()
    motion = motion[
        [args.subject_column, args.motion_column]
    ].copy()

    phenotype["_subject"] = clean_subject_id(phenotype[args.subject_column])
    motion["_subject"] = clean_subject_id(motion[args.subject_column])

    if phenotype["_subject"].duplicated().any():
        duplicates = phenotype.loc[
            phenotype["_subject"].duplicated(keep=False), "_subject"
        ].unique()
        raise ValueError(
            "Duplicate subject IDs in phenotype file: "
            + ", ".join(map(str, duplicates[:10]))
        )

    if motion["_subject"].duplicated().any():
        duplicates = motion.loc[
            motion["_subject"].duplicated(keep=False), "_subject"
        ].unique()
        raise ValueError(
            "Duplicate subject IDs in motion file: "
            + ", ".join(map(str, duplicates[:10]))
        )

    merged = phenotype.merge(
        motion[["_subject", args.motion_column]],
        on="_subject",
        how="inner",
        validate="one_to_one",
    )

    merged[args.bmi_column] = pd.to_numeric(
        merged[args.bmi_column],
        errors="coerce",
    )
    merged[args.motion_column] = pd.to_numeric(
        merged[args.motion_column],
        errors="coerce",
    )

    before_complete_case = len(merged)
    merged = merged.dropna(
        subset=[args.bmi_column, args.motion_column]
    ).copy()
    n_incomplete = before_complete_case - len(merged)

    if len(merged) < 5:
        raise ValueError(
            "Too few participants remain after matching and complete-case "
            f"filtering: {len(merged)}"
        )

    bmi = merged[args.bmi_column].to_numpy(dtype=float)
    rmsfd = merged[args.motion_column].to_numpy(dtype=float)

    if np.unique(bmi).size < 2:
        raise ValueError("BMI has no between-participant variation.")
    if np.unique(rmsfd).size < 2:
        raise ValueError("RMSFD has no between-participant variation.")

    bmi_similarity = rank_based_similarity(bmi)
    rmsfd_similarity = rank_based_similarity(rmsfd)

    spearman_rho, permutation_p = permutation_test(
        bmi_similarity=bmi_similarity,
        rmsfd_similarity=rmsfd_similarity,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
    )

    results = pd.DataFrame(
        {
            "stimulus": [args.stimulus],
            "analysis": ["BMI_similarity_vs_RMSFD_similarity"],
            "spearman_rho": [spearman_rho],
            "permutation_p": [permutation_p],
            "n_subjects": [len(merged)],
            "n_permutations": [args.n_permutations],
            "random_seed": [args.random_seed],
            "bmi_similarity": ["average-rank Euclidean distance; s=1-d"],
            "rmsfd_similarity": ["average-rank Euclidean distance; s=1-d"],
            "permuted_matrix": ["RMSFD similarity"],
            "test_tail": ["two-sided"],
        }
    )

    safe_stimulus = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in args.stimulus.strip()
    ).strip("_") or "unspecified"

    results_file = (
        args.output_dir
        / f"bmi_rmsfd_similarity_{safe_stimulus.lower()}.csv"
    )
    subjects_file = (
        args.output_dir
        / f"subjects_included_bmi_rmsfd_{safe_stimulus.lower()}.csv"
    )

    results.to_csv(results_file, index=False)
    merged[
        ["_subject", args.bmi_column, args.motion_column]
    ].rename(columns={"_subject": "Subject"}).to_csv(
        subjects_file,
        index=False,
    )

    print(f"Stimulus: {args.stimulus}")
    print(f"Participants included: {len(merged)}")
    print(f"Incomplete matched participants excluded: {n_incomplete}")
    print(f"Spearman rho: {spearman_rho:.6f}")
    print(f"Permutation p: {permutation_p:.6f}")
    print(f"Saved results: {results_file}")
    print(f"Saved included-subject table: {subjects_file}")


if __name__ == "__main__":
    main()
