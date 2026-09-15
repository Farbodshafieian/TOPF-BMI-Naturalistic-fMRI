# Expected data inputs

This repository does not distribute HCP participant-level or restricted data. The descriptions below define the local inputs expected by the scripts so that authorized users can reproduce the analyses.

## Parcellation inputs

`parcellate_hcp_movie.sh` expects HCP CIFTI dense time-series files (`*.dtseries.nii`) and the parcellation label file used for the study. The script applies mean aggregation within parcels using Connectome Workbench and writes one parcellated CIFTI and one CSV per subject.

`merge_parcellated_data.py` expects subject-level `*_parcellated.csv` files. It accepts either `379 x T` or `T x 379` orientation, removes the first 10 TRs by default, and writes a table with columns:

```text
subject,time,1,2,...,379
```

## TOPF inputs

The TOPF runner expects a local data directory containing:

```text
df_X_<condition>.csv
Subjects_Info.csv
famid.txt
FamilyIDinfo.csv
```

The fMRI table should contain `subject`, `time`, and parcel columns. `Subjects_Info.csv` must include `Subject` and the phenotype column used as the target. Family helper files are required by the study version of the TOPF cross-validation code.

The subject list supplied with `--subject-list` should contain one HCP subject ID per line. Subject lists are not distributed in this repository.

## TOPF covariate-control inputs

`topf_covariate_control.py` expects:

- prediction CSV with `Subject`, `true`, and `pred` (plus optional `fold` and `seed`)
- phenotype CSV with `Subject`, `Age`, and `Sex`
- motion CSV with `Subject` and `RMSFD`

RMSFD was provided by HCP for each movie run. For clips separated from Movie 1, the Movie 1 RMSFD value is used.

## IS-RSA inputs

Brain arrays are local NumPy files with shape:

```text
subjects x TRs x parcels
```

The primary and covariate-controlled IS-RSA retain the first 360 cortical parcels. Phenotype rows must preserve the same subject order as the first dimension of the brain array.

Primary IS-RSA phenotype file:

```text
Subject,BMI,...
```

Covariate-controlled IS-RSA additionally requires age and sex in the phenotype file and a motion file containing `Subject,RMSFD`.

Race and ethnicity are not used in the covariate-controlled analysis.

## HCP WordNet annotation input

`create_bridgeville_features.py` expects the HCP Movie 1 WordNet annotation text file used in the study. The file itself is not redistributed.

The Bridgeville alignment used by the script is part of the analysis specification and does not contain participant data.

## Parcel-selection input for movie-content models

`first_level_content_glsar.py` reads the local output of `primary_isrsa.py` through `--parcel-results`. By default it selects rows where `bmi_nn_significant_fdr` is true and uses the corresponding values in the `parcel` column. This keeps result-derived parcel identities out of the source code.

## First-level movie-content output

`first_level_content_glsar.py` writes one row per subject x selected parcel x primary feature with columns such as:

```text
subject,subject_row,parcel,feature,beta,rho_AR1,n_TRs_used
```

This generated table contains derived participant-level information and should remain outside version control.

## Continuous-BMI movie-content inputs

`continuous_bmi_analysis.py` expects the local first-level beta table plus:

- BMI file: `Subject,BMI`
- demographic file: `Subject,Sex,Age`
- motion file: `Subject,RMSFD`

The script uses all parcels present in the supplied beta table and applies FDR across all parcel-by-feature BMI tests generated from that table.
