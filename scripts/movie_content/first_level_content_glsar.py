#!/usr/bin/env python3
"""Estimate subject-specific Bridgeville movie-content response betas.

The script fits a first-level GLSAR(1) model for each subject and each parcel
selected from a local IS-RSA result table supplied at runtime. BOLD time series
are z-scored across time within subject and parcel. HRF-convolved movie
regressors are standardized across TRs, and a standardized linear drift term
plus intercept are included in the design.

Only beta coefficients for the six primary movie-content regressors are saved;
first-level p-values are not used for BMI inference.

No result-derived parcel identities are stored in this source file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PRIMARY_FEATURES = [
    "human_presence", "communication", "face_attention", "active_movement",
    "music_performance", "road_vehicle_presence",
]
NUISANCE_FEATURES = ["outdoor_scene", "on_screen_text"]
MOVIE_FEATURES = PRIMARY_FEATURES + NUISANCE_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, required=True, help="subjects x TRs x parcels .npy file")
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--hrf-features", type=Path, required=True)
    parser.add_argument("--parcel-results", type=Path, required=True, help="Local primary IS-RSA result CSV.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--parcel-column", default="parcel")
    parser.add_argument("--significance-column", default="bmi_nn_significant_fdr")
    parser.add_argument("--n-trs", type=int, default=222)
    parser.add_argument("--n-cortical-parcels", type=int, default=360)
    parser.add_argument("--maxiter", type=int, default=20)
    return parser.parse_args()


def load_selected_parcels(args: argparse.Namespace) -> list[int]:
    table = pd.read_csv(args.parcel_results)
    for column in (args.parcel_column, args.significance_column):
        if column not in table.columns:
            raise ValueError(f"Parcel-result table is missing column: {column}")

    flag = table[args.significance_column]
    if flag.dtype != bool:
        flag = flag.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    parcels = (
        pd.to_numeric(table.loc[flag, args.parcel_column], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    if not parcels:
        raise ValueError("No parcels were selected by the supplied significance column.")
    invalid = [p for p in parcels if p < 1 or p > args.n_cortical_parcels]
    if invalid:
        raise ValueError(f"Selected parcel numbers are outside the retained range: {invalid}")
    return parcels


def main() -> None:
    args = parse_args()
    parcels = load_selected_parcels(args)

    data = np.load(args.brain)
    if data.ndim != 3:
        raise ValueError("Expected brain array: subjects x TRs x parcels.")
    if data.shape[1] < args.n_trs or data.shape[2] < args.n_cortical_parcels:
        raise ValueError("Brain array is smaller than the requested TR/parcel dimensions.")
    data = data[:, : args.n_trs, : args.n_cortical_parcels]

    phenotype = pd.read_csv(args.phenotype)
    if len(phenotype) != data.shape[0]:
        raise ValueError("Phenotype rows must match the brain-array subject dimension.")
    if args.subject_column in phenotype.columns:
        subject_ids = phenotype[args.subject_column].astype(str).str.strip().to_numpy()
    else:
        subject_ids = np.arange(data.shape[0]).astype(str)

    features = pd.read_csv(args.hrf_features)
    if len(features) != args.n_trs:
        raise ValueError(f"Expected {args.n_trs} HRF rows; found {len(features)}.")
    missing = [col for col in MOVIE_FEATURES if col not in features.columns]
    if missing:
        raise ValueError(f"Missing HRF regressors: {missing}")

    x_movie = features[MOVIE_FEATURES].to_numpy(float)
    movie_sd = x_movie.std(axis=0, ddof=1, keepdims=True)
    if np.any(~np.isfinite(movie_sd) | (movie_sd == 0)):
        raise ValueError("One or more movie regressors has zero/non-finite SD.")
    x_movie_z = (x_movie - x_movie.mean(axis=0, keepdims=True)) / movie_sd

    drift = np.linspace(-1.0, 1.0, args.n_trs)
    drift = (drift - drift.mean()) / drift.std(ddof=1)
    design = np.column_stack([x_movie_z, drift])
    design_names = MOVIE_FEATURES + ["linear_drift"]
    design = sm.add_constant(design, has_constant="add")
    design_names = ["intercept"] + design_names
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("First-level design matrix is not full rank.")

    bold_mean = np.nanmean(data, axis=1, keepdims=True)
    bold_sd = np.nanstd(data, axis=1, ddof=1, keepdims=True)
    bold_sd[~np.isfinite(bold_sd) | (bold_sd == 0)] = np.nan
    data_z = (data - bold_mean) / bold_sd

    rows = []
    for subject_index in range(data.shape[0]):
        if subject_index % 25 == 0 or subject_index == data.shape[0] - 1:
            print(f"Subject {subject_index + 1}/{data.shape[0]}")
        for parcel in parcels:
            y = data_z[subject_index, :, parcel - 1]
            valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
            if valid.sum() <= design.shape[1] + 5:
                continue
            try:
                model = sm.GLSAR(y[valid], design[valid], rho=1)
                fit = model.iterative_fit(maxiter=args.maxiter)
                params = np.asarray(fit.params, float)
                try:
                    rho = float(np.atleast_1d(model.rho)[0])
                except Exception:
                    rho = np.nan
            except Exception as exc:
                print(f"WARNING: GLSAR failed for subject row {subject_index}, parcel {parcel}: {exc}")
                continue

            for feature in PRIMARY_FEATURES:
                rows.append(
                    {
                        "subject": subject_ids[subject_index],
                        "subject_row": subject_index,
                        "parcel": parcel,
                        "feature": feature,
                        "beta": params[design_names.index(feature)],
                        "rho_AR1": rho,
                        "n_TRs_used": int(valid.sum()),
                    }
                )

    betas = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    betas.to_csv(args.output, index=False)
    expected = data.shape[0] * len(parcels) * len(PRIMARY_FEATURES)
    print("Selected parcels:", len(parcels))
    print("Beta rows:", len(betas), "Expected if every fit succeeded:", expected)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
