#!/usr/bin/env python3
"""Run covariate-controlled BMI IS-RSA with age, sex, and RMSFD.

This script reproduces the study's covariate-controlled Bridgeville analysis.
For each parcel, the upper-triangle vectors of the neural similarity matrix and
rank-based BMI nearest-neighbor similarity matrix are ranked and separately
residualized with respect to covariate dissimilarity vectors. Their residuals
are then correlated to obtain a partial Spearman association.

Covariates in the reported analysis
-----------------------------------
- RMSFD: continuous pairwise Euclidean distance
- Age: continuous pairwise Euclidean distance
- Sex: categorical dissimilarity (0=same, 1=different)

Race and ethnicity are deliberately not included.

Permutation testing permutes BMI subject labels (rows and columns together)
while leaving covariate matrices fixed. Two-sided empirical p-values are FDR
corrected across cortical parcels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import fdrcorrection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--bmi-column", default="BMI")
    parser.add_argument("--age-column", default="Age")
    parser.add_argument("--sex-column", default="Sex")
    parser.add_argument("--motion-column", default="RMSFD")
    parser.add_argument("--n-parcels", type=int, default=360)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def clean_subject_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def rank_similarity_matrix(values: np.ndarray) -> np.ndarray:
    """Create a pairwise similarity matrix from ranked values."""
    
    ranked = rankdata(
        np.asarray(values, dtype=float),
        method="average"
    )

    distance = pairwise_distances(
        ranked.reshape(-1, 1),
        metric="euclidean"
    )

    if np.nanmax(distance) == 0:
        raise ValueError("Values have no between-subject variation.")

    similarity = 1.0 - distance
    np.fill_diagonal(similarity, 1.0)

    return similarity


def continuous_distance(values: np.ndarray) -> np.ndarray:
    return pairwise_distances(np.asarray(values, float).reshape(-1, 1), metric="euclidean")


def categorical_distance(values: np.ndarray) -> np.ndarray:
    values = pd.Series(values).astype(str).str.strip().str.upper().to_numpy()
    return (values[:, None] != values[None, :]).astype(float)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    cov_rank = np.column_stack(
        [rankdata(covariates[:, i], method="average") for i in range(covariates.shape[1])]
    )
    valid = np.isfinite(x_rank) & np.isfinite(y_rank) & np.all(np.isfinite(cov_rank), axis=1)
    x_resid = residualize(x_rank[valid], cov_rank[valid])
    y_resid = residualize(y_rank[valid], cov_rank[valid])
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.brain)
    if data.ndim != 3:
        raise ValueError(f"Expected subjects x TRs x parcels; found {data.shape}.")
    if data.shape[2] < args.n_parcels:
        raise ValueError("Brain array contains fewer parcels than requested.")
    data = data[:, :, : args.n_parcels]

    pheno = pd.read_csv(args.phenotype)
    motion = pd.read_csv(args.motion)

    required_pheno = [args.subject_column, args.bmi_column, args.age_column, args.sex_column]
    missing = [c for c in required_pheno if c not in pheno.columns]
    if missing:
        raise ValueError(f"Phenotype file is missing columns: {missing}")
    if args.subject_column not in motion.columns or args.motion_column not in motion.columns:
        raise ValueError("Motion file must contain subject and RMSFD columns.")

    if len(pheno) != data.shape[0]:
        raise ValueError(
            "Phenotype rows must match the subject dimension and preserve brain-array order."
        )

    pheno = pheno.copy()
    pheno["_subject"] = clean_subject_id(pheno[args.subject_column])
    motion = motion.copy()
    motion["_subject"] = clean_subject_id(motion[args.subject_column])
    motion = motion.drop_duplicates("_subject").set_index("_subject")

    aligned_motion = motion.reindex(pheno["_subject"])[args.motion_column]
    if aligned_motion.isna().any():
        missing_ids = pheno.loc[aligned_motion.isna().to_numpy(), "_subject"].tolist()
        raise ValueError(f"Missing RMSFD for subjects: {missing_ids}")

    bmi = pd.to_numeric(pheno[args.bmi_column], errors="coerce").to_numpy(float)
    age = pd.to_numeric(pheno[args.age_column], errors="coerce").to_numpy(float)
    sex = pheno[args.sex_column].astype(str).to_numpy()
    rmsfd = pd.to_numeric(aligned_motion, errors="coerce").to_numpy(float)

    valid = np.isfinite(bmi) & np.isfinite(age) & np.isfinite(rmsfd) & pd.notna(sex)
    if not valid.all():
        print(f"Excluding {(~valid).sum()} incomplete subjects.")
        data = data[valid]
        pheno = pheno.loc[valid].reset_index(drop=True)
        bmi, age, sex, rmsfd = bmi[valid], age[valid], sex[valid], rmsfd[valid]

    bmi_similarity = rank_similarity_matrix(bmi)
    rmsfd_similarity = rank_similarity_matrix(rmsfd)

    covariate_vectors = np.column_stack(
        [
            upper_triangle(continuous_distance(rmsfd)),
            upper_triangle(continuous_distance(age)),
            upper_triangle(categorical_distance(sex)),
        ]
    )

    rng = np.random.default_rng(args.random_seed)
    permutations = [rng.permutation(len(bmi)) for _ in range(args.n_permutations)]

    def analyze_parcel(parcel_index: int) -> tuple[int, float, float]:
        brain_similarity = 1.0 - pairwise_distances(
            data[:, :, parcel_index], metric="correlation"
        )
        brain_vector = upper_triangle(brain_similarity)
        observed = partial_spearman(brain_vector, bmi_vector, covariate_vectors)

        if not np.isfinite(observed):
            return parcel_index + 1, np.nan, np.nan

        permuted = np.empty(args.n_permutations, dtype=float)
        for i, order in enumerate(permutations):
            bmi_perm = bmi_similarity[np.ix_(order, order)]
            permuted[i] = partial_spearman(
                brain_vector, upper_triangle(bmi_perm), covariate_vectors
            )
        valid_perm = np.isfinite(permuted)
        p_value = (
            np.sum(np.abs(permuted[valid_perm]) >= abs(observed)) + 1
        ) / (valid_perm.sum() + 1)
        return parcel_index + 1, observed, float(p_value)

    rows = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(analyze_parcel)(i) for i in range(args.n_parcels)
    )
    results = pd.DataFrame(rows, columns=["parcel", "bmi_nn_partial_r", "bmi_nn_partial_perm_p"])

    results["bmi_nn_partial_qval_fdr"] = np.nan
    results["bmi_nn_partial_significant_fdr"] = False
    valid_p = results["bmi_nn_partial_perm_p"].notna()
    if valid_p.any():
        significant, q_values = fdrcorrection(
            results.loc[valid_p, "bmi_nn_partial_perm_p"], alpha=0.05
        )
        results.loc[valid_p, "bmi_nn_partial_qval_fdr"] = q_values
        results.loc[valid_p, "bmi_nn_partial_significant_fdr"] = significant

    results["covariates_controlled"] = "RMSFD, Age, Sex"
    results["n_subjects"] = len(bmi)
    results["n_permutations"] = args.n_permutations
    results = results.sort_values("bmi_nn_partial_r", ascending=False, na_position="last")

    results_file = args.output_dir / "bmi_nn_isrsa_covariate_controlled.csv"
    results.to_csv(results_file, index=False)

    included = pd.DataFrame(
        {
            "Subject": pheno["_subject"],
            "BMI": bmi,
            "Age": age,
            "Sex": sex,
            "RMSFD": rmsfd,
        }
    )
    included.to_csv(args.output_dir / "subjects_included.csv", index=False)

    print(f"Saved: {results_file}")
    print("Minimum q:", results["bmi_nn_partial_qval_fdr"].min())
    print("FDR-significant parcels:", int(results["bmi_nn_partial_significant_fdr"].sum()))


if __name__ == "__main__":
    main()
