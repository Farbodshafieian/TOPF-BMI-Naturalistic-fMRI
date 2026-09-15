# TOPF dependency

The study used the public Topography-Based Predictive Framework (TOPF) implementation from:

<https://github.com/xuanli-ac/TOPF>

To avoid redistributing third-party code whose upstream repository does not currently specify a repository-level license, `TOPF.py` is not copied into this repository.

For reproducibility, obtain the upstream implementation and place `TOPF.py` in this directory next to `run_topf_bmi.py` before running the wrapper. The upstream repository's current `main` branch was inspected at commit:

```text
ddc9f4eb2657969823464ba68dba2f0b846a0c82
```

The local study wrapper documents the BMI-specific inputs, threshold, cross-validation settings, and ridge parameter grid used here.
