#!/usr/bin/env python3
"""Test continuous BMI associations with Bridgeville movie-content betas.

Each parcel-by-feature model controls for age, sex, and HCP-provided RMSFD and
uses OLS with HC3 robust standard errors and a finite-sample t reference. All
BMI-slope p-values generated from the supplied beta table are corrected together
with the Benjamini-Hochberg FDR procedure.

For every parcel/feature model, adjusted responses are estimated at BMI mean - 1
SD, mean BMI, and BMI mean + 1 SD with age and RMSFD held at their sample means
and sex held at the sample-average coding. No result-derived parcel identities
are embedded in this script; parcels are read from the beta table at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

PRIMARY_FEATURES = [
    "human_presence",
    "communication",
    "face_attention",
    "active_movement",
    "music_performance",
    "road_vehicle_presence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", type=Path, required=True)
    parser.add_argument("--bmi", type=Path, required=True, help="CSV with Subject,BMI")
    parser.add_argument("--demographics", type=Path, required=True, help="CSV with Subject,Age,Sex")
    parser.add_argument("--motion", type=Path, required=True, help="CSV with Subject,RMSFD")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def clean_subject_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(".0", "", regex=False)


def check_unique_subjects(frame: pd.DataFrame, label: str) -> None:
    duplicate = frame["Subject"].duplicated(keep=False)
    if duplicate.any():
        ids = sorted(frame.loc[duplicate, "Subject"].unique().tolist())
        raise ValueError(f"Duplicate Subject IDs in {label}: {ids}")


def scalar(value) -> float:
    return float(np.asarray(value).squeeze())


def contrast_summary(fit, vector: list[float]) -> dict[str, float]:
    test = fit.t_test(np.asarray(vector, dtype=float))
    ci = np.asarray(test.conf_int(), dtype=float).reshape(-1)
    return {
        "estimate": scalar(test.effect),
        "se": scalar(test.sd),
        "t": scalar(test.tvalue),
        "p": scalar(test.pvalue),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def response_direction(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "positive"
    if ci_high < 0:
        return "negative"
    return "uncertain"


def fit_hc3(frame: pd.DataFrame, include_motion: bool = True):
    predictors = ["BMI_centered", "Age_z", "Sex_M_centered"]
    if include_motion:
        predictors.append("RMSFD_z")

    y = pd.to_numeric(frame["beta"], errors="coerce")
    x = frame[predictors].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(y.to_numpy()) & np.all(np.isfinite(x.to_numpy()), axis=1)
    y = y.loc[valid]
    x = sm.add_constant(x.loc[valid], has_constant="add")

    if len(y) <= x.shape[1] + 2 or np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        return None
    return sm.OLS(y, x).fit(cov_type="HC3", use_t=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    betas = pd.read_csv(args.betas)
    require_columns(betas, ["subject", "parcel", "feature", "beta"], "Beta file")
    betas["subject"] = clean_subject_id(betas["subject"])
    betas["parcel"] = pd.to_numeric(betas["parcel"], errors="raise").astype(int)
    betas["beta"] = pd.to_numeric(betas["beta"], errors="coerce")
    betas = betas.loc[betas["feature"].isin(PRIMARY_FEATURES)].copy()

    if betas.duplicated(["subject", "parcel", "feature"], keep=False).any():
        raise ValueError("Duplicate subject x parcel x feature beta rows detected.")
    if set(betas["feature"].unique()) != set(PRIMARY_FEATURES):
        raise ValueError("Beta file does not contain exactly the six primary features.")
    parcels = sorted(betas["parcel"].dropna().unique().astype(int).tolist())
    if not parcels:
        raise ValueError("Beta file contains no parcels after filtering primary features.")

    bmi = pd.read_csv(args.bmi)[["Subject", "BMI"]].copy()
    demographics = pd.read_csv(args.demographics)[["Subject", "Sex", "Age"]].copy()
    motion = pd.read_csv(args.motion)[["Subject", "RMSFD"]].copy()

    for frame in (bmi, demographics, motion):
        frame["Subject"] = clean_subject_id(frame["Subject"])
    for frame, label in ((bmi, "BMI"), (demographics, "demographics"), (motion, "RMSFD")):
        check_unique_subjects(frame, label)

    bmi["BMI"] = pd.to_numeric(bmi["BMI"], errors="coerce")
    demographics["Age"] = pd.to_numeric(demographics["Age"], errors="coerce")
    motion["RMSFD"] = pd.to_numeric(motion["RMSFD"], errors="coerce")
    demographics["Sex"] = demographics["Sex"].astype(str).str.strip().str.upper()

    subjects = (
        bmi.merge(demographics, on="Subject", validate="one_to_one")
        .merge(motion, on="Subject", validate="one_to_one")
        .dropna(subset=["BMI", "Age", "Sex", "RMSFD"])
    )
    subjects = subjects.loc[subjects["Sex"].isin(["F", "M"])].copy()
    subjects = subjects.loc[subjects["Subject"].isin(betas["subject"])].copy()
    if subjects.empty:
        raise ValueError("No complete subjects remain after merging inputs.")

    bmi_mean = float(subjects["BMI"].mean())
    bmi_sd = float(subjects["BMI"].std(ddof=1))
    age_mean, age_sd = float(subjects["Age"].mean()), float(subjects["Age"].std(ddof=1))
    rmsfd_mean, rmsfd_sd = float(subjects["RMSFD"].mean()), float(subjects["RMSFD"].std(ddof=1))
    for name, value in (("BMI SD", bmi_sd), ("Age SD", age_sd), ("RMSFD SD", rmsfd_sd)):
        if not np.isfinite(value) or value == 0:
            raise ValueError(f"{name} is zero or non-finite.")

    subjects["BMI_centered"] = subjects["BMI"] - bmi_mean
    subjects["Age_z"] = (subjects["Age"] - age_mean) / age_sd
    subjects["RMSFD_z"] = (subjects["RMSFD"] - rmsfd_mean) / rmsfd_sd
    subjects["Sex_M"] = (subjects["Sex"] == "M").astype(float)
    subjects["Sex_M_centered"] = subjects["Sex_M"] - subjects["Sex_M"].mean()

    low_bmi = bmi_mean - bmi_sd
    high_bmi = bmi_mean + bmi_sd
    vif_x = sm.add_constant(
        subjects[["BMI_centered", "Age_z", "Sex_M_centered", "RMSFD_z"]],
        has_constant="add",
    )
    vif = pd.DataFrame(
        {
            "predictor": vif_x.columns,
            "VIF": [variance_inflation_factor(vif_x.to_numpy(), i) for i in range(vif_x.shape[1])],
        }
    )
    vif.to_csv(args.output_dir / "second_level_VIF.csv", index=False)

    model_cols = [
        "Subject", "BMI", "BMI_centered", "Age", "Age_z", "Sex", "Sex_M",
        "Sex_M_centered", "RMSFD", "RMSFD_z",
    ]
    merged = betas.merge(
        subjects[model_cols], left_on="subject", right_on="Subject",
        how="inner", validate="many_to_one",
    )
    expected_rows = len(subjects) * len(parcels) * len(PRIMARY_FEATURES)
    if len(merged) != expected_rows:
        raise ValueError(f"Incomplete beta coverage: {len(merged)} rows; expected {expected_rows}.")

    rows = []
    model_frames: dict[tuple[int, str], pd.DataFrame] = {}
    for feature in PRIMARY_FEATURES:
        print("Processing", feature)
        for parcel in parcels:
            temp = merged.loc[(merged["feature"] == feature) & (merged["parcel"] == parcel)].copy()
            fit = fit_hc3(temp, include_motion=True)
            sensitivity = fit_hc3(temp, include_motion=False)
            if fit is None:
                raise ValueError(f"Primary model failed: parcel={parcel}, feature={feature}")

            # Parameter order: const, BMI_centered, Age_z, Sex_M_centered, RMSFD_z
            slope = contrast_summary(fit, [0, 1, 0, 0, 0])
            low = contrast_summary(fit, [1, -bmi_sd, 0, 0, 0])
            mean = contrast_summary(fit, [1, 0, 0, 0, 0])
            high = contrast_summary(fit, [1, bmi_sd, 0, 0, 0])
            difference = contrast_summary(fit, [0, 2 * bmi_sd, 0, 0, 0])

            if sensitivity is not None:
                slope_no_motion = contrast_summary(sensitivity, [0, 1, 0, 0])
            else:
                slope_no_motion = {k: np.nan for k in ["estimate", "p", "ci_low", "ci_high"]}

            rows.append(
                {
                    "parcel": parcel,
                    "feature": feature,
                    "n_subjects": int(fit.nobs),
                    "BMI_mean": bmi_mean,
                    "BMI_SD": bmi_sd,
                    "low_BMI_value": low_bmi,
                    "high_BMI_value": high_bmi,
                    "BMI_slope": slope["estimate"],
                    "BMI_SE_HC3": slope["se"],
                    "BMI_t_HC3": slope["t"],
                    "BMI_p": slope["p"],
                    "BMI_CI95_low": slope["ci_low"],
                    "BMI_CI95_high": slope["ci_high"],
                    "low_activity": low["estimate"],
                    "low_activity_CI95_low": low["ci_low"],
                    "low_activity_CI95_high": low["ci_high"],
                    "low_activity_direction": response_direction(low["ci_low"], low["ci_high"]),
                    "mean_activity": mean["estimate"],
                    "mean_activity_CI95_low": mean["ci_low"],
                    "mean_activity_CI95_high": mean["ci_high"],
                    "mean_activity_direction": response_direction(mean["ci_low"], mean["ci_high"]),
                    "high_activity": high["estimate"],
                    "high_activity_CI95_low": high["ci_low"],
                    "high_activity_CI95_high": high["ci_high"],
                    "high_activity_direction": response_direction(high["ci_low"], high["ci_high"]),
                    "high_minus_low": difference["estimate"],
                    "high_minus_low_CI95_low": difference["ci_low"],
                    "high_minus_low_CI95_high": difference["ci_high"],
                    "model_R2": float(fit.rsquared),
                    "model_adj_R2": float(fit.rsquared_adj),
                    "BMI_slope_no_motion": slope_no_motion["estimate"],
                    "BMI_p_no_motion": slope_no_motion["p"],
                    "BMI_CI95_low_no_motion": slope_no_motion["ci_low"],
                    "BMI_CI95_high_no_motion": slope_no_motion["ci_high"],
                }
            )
            model_frames[(parcel, feature)] = temp

    results = pd.DataFrame(rows)
    reject, q_values, _, _ = multipletests(
        results["BMI_p"].to_numpy(), alpha=args.fdr_alpha, method="fdr_bh"
    )
    results["BMI_q_FDR"] = q_values
    results["FDR_significant"] = reject
    results = results.sort_values(["BMI_q_FDR", "BMI_p"]).reset_index(drop=True)
    results.to_csv(args.output_dir / "continuous_BMI_direction_results.csv", index=False)

    significant = results.loc[results["FDR_significant"]].copy()
    significant.to_csv(args.output_dir / "continuous_BMI_FDR_significant.csv", index=False)

    # Create one adjusted-response figure per FDR-significant slope.
    for _, row in significant.iterrows():
        parcel = int(row["parcel"])
        feature = str(row["feature"])
        temp = model_frames[(parcel, feature)]
        fit = fit_hc3(temp, include_motion=True)
        bmi_grid = np.linspace(subjects["BMI"].min(), subjects["BMI"].max(), 100)
        x_grid = pd.DataFrame(
            {
                "const": 1.0,
                "BMI_centered": bmi_grid - bmi_mean,
                "Age_z": 0.0,
                "Sex_M_centered": 0.0,
                "RMSFD_z": 0.0,
            }
        )
        prediction = fit.get_prediction(x_grid).summary_frame(alpha=0.05)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(temp["BMI"], temp["beta"], alpha=0.45, s=22)
        ax.plot(bmi_grid, prediction["mean"].to_numpy())
        ax.fill_between(
            bmi_grid,
            prediction["mean_ci_lower"].to_numpy(),
            prediction["mean_ci_upper"].to_numpy(),
            alpha=0.2,
        )
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("BMI (kg/m²)")
        ax.set_ylabel("Adjusted feature-response beta")
        ax.set_title(f"Parcel {parcel}: {feature}")
        fig.tight_layout()
        fig.savefig(figure_dir / f"parcel_{parcel}_{feature}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"N complete subjects: {len(subjects)}")
    print(f"BMI mean (SD): {bmi_mean:.3f} ({bmi_sd:.3f})")
    print(f"Models: {len(results)}; FDR-significant: {len(significant)}")
    print("Saved outputs to", args.output_dir)


if __name__ == "__main__":
    main()
