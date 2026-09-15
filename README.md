# TOPF-BMI-Naturalistic-fMRI

Analysis code for a study of body mass index (BMI), individual-specific brain topography, and intersubject neural similarity during naturalistic movie viewing in the Human Connectome Project (HCP) Young Adult 7T dataset.

The repository accompanies the manuscript in preparation, **A Topography-Based BMI Predictive Model**.

## Analysis overview

The repository contains code for the following analysis sequence:

1. Parcellate HCP naturalistic-viewing CIFTI time series into 379 regions and remove the first 10 TRs.
2. Extract individual-specific topographic features using the Topography-Based Predictive Framework (TOPF).
3. Predict BMI with ridge regression using nested cross-validation and a variance-explained feature-selection threshold of 0.20.
4. Evaluate the effect of age, sex, and HCP-provided RMSFD on TOPF prediction performance.
5. Run parcel-wise intersubject representational similarity analysis (IS-RSA) across 360 HCP-MMP cortical parcels.
6. Repeat the Bridgeville IS-RSA while controlling for age, sex, and RMSFD similarity/dissimilarity matrices.
7. Convert HCP WordNet annotations for Bridgeville into movie-content regressors and convolve them with a canonical double-gamma HRF.
8. Estimate subject-specific movie-content response coefficients in parcels selected from the primary IS-RSA output.
9. Test continuous BMI associations with movie-content response estimates while controlling for age, sex, and RMSFD.
10. Optionally visualize a user-specified parcel-level subject-similarity matrix.

## Repository structure

```text
scripts/
├── preprocessing/
│   ├── parcellate_hcp_movie.sh
│   └── merge_parcellated_data.py
├── topf/
│   ├── README.md
│   ├── run_topf_bmi.py
│   └── ridge_settings.txt
├── confound_analysis/
│   └── topf_covariate_control.py
├── isrsa/
│   ├── primary_isrsa.py
│   ├── covariate_controlled_isrsa.py
│   └── parcel_similarity.py
└── movie_content/
    ├── create_bridgeville_features.py
    ├── convolve_features_hrf.py
    ├── first_level_content_glsar.py
    └── continuous_bmi_analysis.py

docs/
├── analysis_workflow.md
└── data_inputs.md

results/
└── README.md
```

## Data and results policy

This repository contains **code and documentation only**. It does not include participant-level HCP data, subject identifiers, phenotype tables, motion tables, CIFTI/NIfTI files, NumPy arrays, HCP WordNet annotation files, derived result tables, statistical output, or generated figures.

Researchers with appropriate HCP access should obtain required inputs directly from the Human Connectome Project and keep all participant-level inputs and generated outputs outside the tracked repository. All scripts accept input and output paths as command-line arguments; no user-specific filesystem paths are hard-coded.

See [`docs/data_inputs.md`](docs/data_inputs.md) for expected input schemas.

## Main analysis settings

- TOPF parcellation: **379 regions** (360 cortical + 19 subcortical)
- IS-RSA: **360 cortical HCP-MMP parcels**
- TOPF component: **PC1**
- TOPF feature-selection threshold used by the analysis scripts: **0.20 variance explained**
- Outer cross-validation: **10 folds**
- Inner cross-validation: **5 folds**
- Ridge alpha grid: **500, 750, 1000, 1250, 1500, 1750, 2000**
- IS-RSA permutations: **5,000**
- Multiple-comparison correction: **Benjamini-Hochberg FDR, q < 0.05**
- Covariates: **age, sex, and RMSFD**

These are analysis settings, not observed study results.

## Environment

The analyses were run in Python 3.11. See [`requirements.txt`](requirements.txt) for the package versions recorded for the project.

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Connectome Workbench (`wb_command`) is also required for the CIFTI parcellation step.

## Running the analyses

Each script documents its required inputs and provides command-line help. For example:

```bash
python scripts/isrsa/primary_isrsa.py --help
python scripts/movie_content/create_bridgeville_features.py --help
python scripts/confound_analysis/topf_covariate_control.py --help
```

A step-by-step workflow is provided in [`docs/analysis_workflow.md`](docs/analysis_workflow.md).

## TOPF attribution

The TOPF runner is adapted to the BMI analysis from the public TOPF implementation by Xuan Li and colleagues. The upstream `TOPF.py` file is not redistributed here; see `scripts/topf/README.md` for setup instructions.

- Upstream repository: <https://github.com/xuanli-ac/TOPF>
- Li et al. (2023), Topography-Based Predictive Framework (TOPF)

See [`NOTICE.md`](NOTICE.md) for provenance information.

## Reproducibility notes

- Two Men, Bridgeville, and Pockets were separated from the HCP Movie 1 fMRI run for clip-level analyses.
- HCP-provided RMSFD was available separately for Movie 1 through Movie 4. Movie 1 RMSFD was used for Movie 1 clip-level analyses.
- Race and ethnicity are not included as covariates in the analysis scripts.
- Parcel selection for the movie-content follow-up is read from a local IS-RSA result file at runtime; the selected parcel identities are not stored in this repository.

## Contact

For questions about this analysis repository, please use the GitHub issue tracker.
