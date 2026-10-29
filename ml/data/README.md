# ML Data Storage Guide

This directory stores datasets across different stages of the data collection and modeling pipeline.

## Directory Layout

- `raw/`: Unmodified sensor recordings collected from the system (`data/runs/*.csv`) or raw external benchmarks. Never edit raw data files directly.
- `processed/`: Cleaned, validated, and feature-engineered datasets ready for model training and cross-validation.
- `external/`: Public benchmark datasets (e.g. UCI gas sensor datasets).

## Strict Data Integrity Policies

1. **No Synthetic Fakes**: Never fabricate sensor data or label distributions.
2. **Commit Policy**: Large datasets (`> 1 MB`) must be excluded from Git via `.gitignore`.
3. **Session Demarcation**: All recorded sensor files must include `run_id` and timestamps `ts` to allow session-aware evaluation and prevent time-series data leakage.
