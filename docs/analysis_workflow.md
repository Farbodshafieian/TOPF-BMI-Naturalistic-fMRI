# Analysis workflow

This document summarizes the analysis order and gives example commands. Paths are examples only. Participant-level data and generated results should be stored outside the tracked repository.

## 1. Parcellate HCP movie data

```bash
bash scripts/preprocessing/parcellate_hcp_movie.sh   /data/HCP_MOVIE4   /data/parcellation_labels.dlabel.nii   /work/parcellated   /work/parcellated_csv
```

Merge subject files and remove the first 10 TRs:

```bash
python scripts/preprocessing/merge_parcellated_data.py   --csv-dir /work/parcellated_csv   --output-file /work/data/dataframe/df_X_movie4.csv   --n-parcels 379   --remove-trs 10
```

Two Men, Bridgeville, and Pockets were separated from the Movie 1 fMRI run before clip-level analyses.

## 2. Run TOPF BMI prediction

```bash
python scripts/topf/run_topf_bmi.py   --data-dir /work/data/dataframe   --subject-list /work/subjects.txt   --results-dir /work/results   --condition movie4   --settings scripts/topf/ridge_settings.txt   --phenotype BMI   --n-roi 379   --pc-index 1   --threshold 0.20   --seed 1
```

The runner uses 10 outer folds and 5 inner folds by default. The ridge alpha grid is stored in `ridge_settings.txt`.

## 3. Evaluate age, sex, and motion effects on TOPF prediction

```bash
python scripts/confound_analysis/topf_covariate_control.py   --predictions /work/results/test_ridge_movie4_seed1.csv   --phenotype /work/Subjects_Info.csv   --motion /work/RMSFD_subjects.csv   --output /work/covariates/movie4_age_sex_rmsfd.csv   --covariates age sex rmsfd
```

Other covariate combinations can be evaluated by changing `--covariates`.

## 4. Run primary BMI IS-RSA

```bash
python scripts/isrsa/primary_isrsa.py   --brain /work/df_X_bridgeville_subjects_TRs_parcels.npy   --phenotype /work/bridgeville/Subjects_Info.csv   --output-dir /work/isrsa/bridgeville_primary   --n-permutations 5000
```

The analysis uses 360 cortical parcels and Benjamini-Hochberg FDR across parcels.

## 5. Run covariate-controlled Bridgeville IS-RSA

```bash
python scripts/isrsa/covariate_controlled_isrsa.py   --brain /work/df_X_bridgeville_subjects_TRs_parcels.npy   --phenotype /work/bridgeville/Subjects_Info.csv   --motion /work/bridgeville/RMSFD_subjects.csv   --output-dir /work/isrsa/bridgeville_covariate_controlled   --n-permutations 5000
```

The covariate matrices are age, sex, and RMSFD only.

## 6. Build Bridgeville movie-content regressors

```bash
python scripts/movie_content/create_bridgeville_features.py   --annotations /data/HCP/WordNet_MOVIE1_CC1.txt   --output /work/movie_content/bridgeville_content_features.csv
```

```bash
python scripts/movie_content/convolve_features_hrf.py   --features /work/movie_content/bridgeville_content_features.csv   --output /work/movie_content/bridgeville_content_features_HRF.csv
```

## 7. Estimate subject-specific movie-content response betas

The parcel set is selected at runtime from the local primary IS-RSA output rather than being embedded in source code.

```bash
python scripts/movie_content/first_level_content_glsar.py   --brain /work/df_X_bridgeville_subjects_TRs_parcels.npy   --phenotype /work/bridgeville/Subjects_Info.csv   --hrf-features /work/movie_content/bridgeville_content_features_HRF.csv   --parcel-results /work/isrsa/bridgeville_primary/bmi_nn_isrsa_results.csv   --output /work/movie_content/bridgeville_subject_parcel_content_betas.csv
```

The first-level model uses GLSAR(1), standardized HRF regressors, a linear drift term, and temporally z-scored BOLD.

## 8. Run the continuous-BMI movie-content analysis

```bash
python scripts/movie_content/continuous_bmi_analysis.py   --betas /work/movie_content/bridgeville_subject_parcel_content_betas.csv   --bmi /work/bridgeville/Subjects_Info.csv   --demographics /work/bridgeville/Covariates/Subjects_Info.csv   --motion /work/bridgeville/Covariates/RMSFD_subjects.csv   --output-dir /work/movie_content/continuous_bmi
```

The model for each parcel x feature combination is:

```text
feature beta ~ BMI + age + sex + RMSFD
```

Inference uses HC3 robust standard errors. All parcel-by-feature BMI tests present in the input beta table are corrected together with Benjamini-Hochberg FDR.

## 9. Optional parcel-similarity visualization

A user-specified parcel can be visualized without storing any result-derived parcel identity in the repository:

```bash
python scripts/isrsa/parcel_similarity.py   --brain /work/df_X_bridgeville_subjects_TRs_parcels.npy   --phenotype /work/bridgeville/Subjects_Info.csv   --parcel <parcel-number>   --parcel-name <optional-label>   --output-dir /work/isrsa/parcel_similarity
```
