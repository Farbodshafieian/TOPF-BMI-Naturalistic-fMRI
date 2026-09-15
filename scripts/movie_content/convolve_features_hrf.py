#!/usr/bin/env python3
"""Convolve Bridgeville content features with a canonical double-gamma HRF.

The study used TR=1 s and a 32 s HRF kernel. Six primary content regressors and
two nuisance regressors (outdoor scene and on-screen text) are exported. The
script also prints regressor correlations, VIFs, matrix rank, and condition
number as design-matrix quality checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma
from statsmodels.stats.outliers_influence import variance_inflation_factor

PRIMARY_FEATURES = [
    "human_presence",
    "communication",
    "face_attention",
    "active_movement",
    "music_performance",
    "road_vehicle_presence",
]
NUISANCE_FEATURES = ["outdoor_scene", "on_screen_text"]
ALL_FEATURES = PRIMARY_FEATURES + NUISANCE_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--hrf-duration", type=float, default=32.0)
    parser.add_argument("--expected-trs", type=int, default=222)
    return parser.parse_args()


def canonical_hrf(tr: float, duration: float) -> np.ndarray:
    time = np.arange(0, duration, tr)
    hrf = gamma.pdf(time, 6) - gamma.pdf(time, 16) / 6.0
    return hrf / hrf.sum()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features)
    if len(frame) != args.expected_trs:
        raise ValueError(f"Expected {args.expected_trs} rows; found {len(frame)}.")
    missing = [col for col in ALL_FEATURES if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    hrf = canonical_hrf(args.tr, args.hrf_duration)
    output = frame[["brain_TR", "annotation_ID"]].copy()
    for feature in ALL_FEATURES:
        raw = frame[feature].to_numpy(float)
        output[feature] = np.convolve(raw, hrf, mode="full")[: len(raw)]

    correlation = output[ALL_FEATURES].corr()
    print("Post-HRF correlations")
    print(correlation.round(2).to_string())

    x = output[ALL_FEATURES].to_numpy(float)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    if np.any(~np.isfinite(sd) | (sd == 0)):
        raise ValueError("At least one HRF regressor has zero/non-finite variance.")
    xz = x / sd

    vif = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "VIF": [variance_inflation_factor(xz, i) for i in range(xz.shape[1])],
        }
    )
    print("\nVIF")
    print(vif.sort_values("VIF", ascending=False).to_string(index=False))

    design = np.column_stack([np.ones(len(xz)), xz])
    print("\nDesign rank:", np.linalg.matrix_rank(design), "/", design.shape[1])
    print("Condition number:", round(float(np.linalg.cond(design)), 2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
