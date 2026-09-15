#!/usr/bin/env python3
"""Run TOPF ridge-regression prediction for a naturalistic-viewing condition.

This is a path-independent wrapper around the TOPF implementation used in the
study. It preserves the study's 10-fold outer / 5-fold inner cross-validation
settings and allows all data locations to be supplied at runtime.

Expected input layout
---------------------
``data_dir`` must contain:

- ``df_X_<condition>.csv``: concatenated fMRI table with columns ``subject``,
  ``time``, and parcel columns ``1..n_roi``.
- ``Subjects_Info.csv``: HCP phenotype table containing ``Subject`` and the
  target phenotype (BMI in this study).
- ``famid.txt`` and ``FamilyIDinfo.csv``: family-structure helper files used by
  the study TOPF implementation.

Example
-------
python run_topf_bmi.py \
    --data-dir /path/to/data/dataframe \
    --subject-list /path/to/subjects.txt \
    --results-dir /path/to/results \
    --condition movie4 \
    --settings ridge_settings.txt \
    --phenotype BMI \
    --threshold 0.20 \
    --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subject-list", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        required=True,
        help="Condition name matching df_X_<condition>.csv (e.g., movie4, bridgeville).",
    )
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--phenotype", default="BMI")
    parser.add_argument("--n-roi", type=int, default=379)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-tr", type=int, default=0, help="0 uses the full time series.")
    parser.add_argument("--feature-type", choices=["singlePC", "combinedPC"], default="singlePC")
    parser.add_argument("--pc-index", type=int, default=1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="PC variance-explained feature-selection threshold (reported analysis: 0.20).",
    )
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from julearn.utils import configure_logging
    import TOPF

    configure_logging(level="INFO")
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    fmri_file = data_dir / f"df_X_{args.condition}.csv"
    pheno_file = data_dir / "Subjects_Info.csv"
    famid_file = data_dir / "famid.txt"
    family_members_file = data_dir / "FamilyIDinfo.csv"

    for path in [fmri_file, pheno_file, famid_file, family_members_file, args.subject_list, args.settings]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    df_fmri = pd.read_csv(fmri_file)
    df_pheno = pd.read_csv(pheno_file)
    famid = np.loadtxt(famid_file)
    family_df = pd.read_csv(family_members_file)

    family_cols = [
        "FamilyMembers_1",
        "FamilyMembers_2",
        "FamilyMembers_3",
        "FamilyMembers_4",
    ]
    missing_family_cols = [c for c in family_cols if c not in family_df.columns]
    if missing_family_cols:
        raise ValueError(f"FamilyIDinfo.csv is missing columns: {missing_family_cols}")
    family_members = family_df[family_cols].values

    subject_list = np.loadtxt(args.subject_list).astype(int)
    print("Number of subjects:", len(subject_list))

    missing, _ = TOPF.check_missing_data_phenotype(
        df_pheno, subject_list, args.phenotype
    )
    if len(missing):
        raise ValueError("The subject list contains missing phenotype values.")

    n_tr = None if args.n_tr <= 0 else args.n_tr

    result_folder = (
        f"/{args.phenotype}/cut_{n_tr}_t{args.threshold}_"
        f"{args.feature_type}_{args.pc_index}_ko{args.outer_folds}_"
        f"ki{args.inner_folds}_n{args.n_roi}"
    )
    result_dir = str(results_dir) + result_folder
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    result_struct = TOPF.init_result_dir(result_dir)
    feature_dir = str(results_dir) + (
        f"/features/cut_{n_tr}_ko{args.outer_folds}_"
        f"ki{args.inner_folds}_n{args.n_roi}/"
    )
    Path(feature_dir).mkdir(parents=True, exist_ok=True)

    model = TOPF.create_ML_julearn_model(args.model, str(args.settings))

    TOPF.main_TOPF_kfold(
        df_fmri,
        args.condition,
        args.model,
        args.n_roi,
        args.seed,
        subject_list,
        df_pheno,
        famid,
        family_members,
        args.phenotype,
        model,
        result_struct,
        feature_dir,
        args.feature_type,
        args.pc_index,
        n_tr,
        args.threshold,
        k_inner=args.inner_folds,
        k_outer=args.outer_folds,
        flip=None,
        clean=None,
        norm=1,
    )

    save_path = eval(result_struct.scorefname_test)
    _, _, fold_metrics, _ = TOPF.main_compute_prediction_scores(
        args.condition,
        args.model,
        model.probtype,
        result_struct,
        model.metric_list,
        [args.seed],
        args.outer_folds,
        save_path,
        foldwise=1,
        test=True,
    )

    print("Condition:", args.condition)
    print("Fold-wise scores:")
    print(fold_metrics)


if __name__ == "__main__":
    main()
