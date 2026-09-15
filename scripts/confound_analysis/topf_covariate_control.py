#!/usr/bin/env python3
"""Evaluate TOPF BMI predictions after post-prediction covariate residualization.

Observed BMI and cross-validated predicted BMI are residualized separately with
respect to selected covariates, then prediction metrics are recomputed. This
matches the study's post-prediction covariate analysis; covariates are not
entered into the ridge model itself.

The reported manuscript analyses used age, sex, and HCP-provided RMSFD. Race
and ethnicity were not included.

Example
-------
python topf_covariate_control.py \
    --predictions test_ridge_movie4_seed1.csv \
    --phenotype Subjects_Info.csv \
    --motion RMSFD_subjects.csv \
    --output residualized_movie4.csv \
    --covariates age sex rmsfd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional CSV for pooled and fold-wise performance metrics.",
    )
    parser.add_argument(
        "--covariates",
        nargs="+",
        choices=["age", "sex", "rmsfd"],
        default=["age", "sex", "rmsfd"],
    )
    return parser.parse_args()


def encode_sex(series: pd.Series) -> pd.Series:
    """Encode common HCP sex labels as F=0 and M=1."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"f": 0.0, "female": 0.0, "m": 1.0, "male": 1.0}
    encoded = normalized.map(mapping)

    if encoded.isna().any():
        bad = sorted(series[encoded.isna()].astype(str).unique())
        raise ValueError(f"Unexpected sex labels: {bad}")
    return encoded.astype(float)


def residualize(y: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Return residuals from linear regression of y on the confounds."""
    model = LinearRegression().fit(confounds, np.asarray(y).reshape(-1, 1))
    return (np.asarray(y).reshape(-1, 1) - model.predict(confounds)).ravel()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the regression metrics used in the project."""
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()

    pred = pd.read_csv(args.predictions)
    pheno = pd.read_csv(args.phenotype)
    motion = pd.read_csv(args.motion)

    for name, frame, required in [
        ("predictions", pred, ["Subject", "true", "pred"]),
        ("phenotype", pheno, ["Subject", "Age", "Sex"]),
        ("motion", motion, ["Subject", "RMSFD"]),
    ]:
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"{name} file is missing columns: {missing}")

    pheno = pheno[["Subject", "Age", "Sex"]].copy()
    motion = motion[["Subject", "RMSFD"]].copy()
    pheno["Sex"] = encode_sex(pheno["Sex"])

    merged = pred.merge(pheno, on="Subject", how="inner", validate="many_to_one")
    merged = merged.merge(motion, on="Subject", how="inner", validate="many_to_one")
    merged = merged.dropna(subset=["true", "pred", "Age", "Sex", "RMSFD"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError("No complete subjects remained after merging input tables.")

    covariate_columns = {
        "age": "Age",
        "sex": "Sex",
        "rmsfd": "RMSFD",
    }
    selected = [covariate_columns[name] for name in args.covariates]
    confounds = merged[selected].astype(float).to_numpy()

    merged["true_resid"] = residualize(merged["true"].to_numpy(), confounds)
    merged["pred_resid"] = residualize(merged["pred"].to_numpy(), confounds)

    raw = compute_metrics(merged["true"].to_numpy(), merged["pred"].to_numpy())
    adjusted = compute_metrics(merged["true_resid"].to_numpy(), merged["pred_resid"].to_numpy())

    rows = [
        {"scope": "pooled_raw", "fold": np.nan, **raw},
        {"scope": "pooled_residualized", "fold": np.nan, **adjusted},
    ]

    if "fold" in merged.columns:
        for fold, frame in merged.groupby("fold", sort=True):
            if len(frame) < 3:
                continue
            metrics = compute_metrics(frame["true_resid"], frame["pred_resid"])
            rows.append({"scope": "fold_residualized", "fold": fold, **metrics})

    metrics_df = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    metrics_output = args.metrics_output
    if metrics_output is None:
        metrics_output = args.output.with_name(args.output.stem + "_metrics.csv")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_output, index=False)

    print("Covariates:", ", ".join(args.covariates))
    print("Subjects:", len(merged))
    print("\nRaw pooled metrics")
    print(pd.Series(raw).to_string())
    print("\nResidualized pooled metrics")
    print(pd.Series(adjusted).to_string())
    print(f"\nSaved residualized table: {args.output}")
    print(f"Saved metrics: {metrics_output}")


if __name__ == "__main__":
    main()
