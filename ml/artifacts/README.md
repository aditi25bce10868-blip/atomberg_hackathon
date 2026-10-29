# Model Artifacts Directory

This directory stores trained model bundles (`.joblib`), feature contracts, and model version metadata.

## Zero-Fabrication Rule
- **No Placeholder Models**: Do not commit mock or empty binary files and represent them as trained models.
- If no real model bundle exists, the control runtime automatically falls back to the transparent `rules-placeholder` classifier.
- When a model is legitimately trained via `python -m ml.training.train --data <dataset.csv>`, the bundle will be stored here with complete cross-validation metadata.
