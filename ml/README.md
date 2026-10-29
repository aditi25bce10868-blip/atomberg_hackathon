# Efficio Apex Machine Learning Lifecycle Architecture (`ml/`)

This package houses the complete machine learning architecture for the **Atomberg Efficio Apex** smart ventilation system (AtomQUEST 2026).

---

## 1. Package Architecture

```
ml/
├── __init__.py               # Package metadata and exports
├── README.md                 # Complete ML documentation (this file)
├── config.py                 # Central contract: feature lists (29 canonical, 23 legacy), classes, paths
│
├── datasets/                 # Dataset schema inspection, quality validation, and adapters
│   ├── __init__.py
│   ├── adapters.py           # Adapters for legacy prototypes & external benchmarks
│   ├── inspect_dataset.py    # Statistical profiling, sampling rates, missingness, sessions
│   └── validate_dataset.py   # Schema validation, label vocabulary, benchmark differentiation
│
├── features/                 # Deterministic, leakage-free feature engineering
│   ├── __init__.py
│   └── build_features.py     # FeatureExtractor (canonical/legacy) & build_features(history)
│
├── training/                 # Model training and session-aware evaluation
│   ├── __init__.py
│   ├── train.py              # Cross-validation with GroupKFold session isolation or TimeSeriesSplit
│   └── evaluate.py           # Independent model evaluation and report generation
│
├── inference/                # Real-time inference independent of hardware/fan control
│   ├── __init__.py
│   └── predictor.py          # AirQualityPredictor with input validation, status tracking, and fallback
│
├── data/                     # Data storage with strict source control rules
│   ├── raw/                  # Unmodified recordings from data/runs/*.csv
│   ├── processed/            # Verified and engineered feature matrices
│   └── external/             # Public benchmark datasets (e.g. UCI gas array)
│
├── artifacts/                # Trained model bundles (.joblib) and metadata
└── tests/                    # Unit tests specific to ML components
```

---

## 2. Sensor Schema & Hardware Architecture

The intended physical hardware architecture consists of:
- **MQ-3** (CH0 on MCP3008 ADC): Alcohol, benzene, hexane, ethanol (analog voltage 0–3.3V after divider)
- **MQ-4** (CH1 on MCP3008 ADC): Methane, natural gas, CNG, smoke
- **MQ-135** (CH2 on MCP3008 ADC): General air quality: NH3, NOx, benzene, CO2, smoke
- **Optional Context Channels**: Relative humidity (%RH), ambient temperature (°C), PIR presence (0 or 1)

### Canonical vs Legacy Schemas

| Channel | Canonical Physical Hardware | Legacy Prototype / Early Simulation |
|---|---|---|
| Gas Channel 0 | `mq3` (MQ-3) | N/A |
| Gas Channel 1 | `mq4` (MQ-4) | `mq2` (MQ-2 general combustible) |
| Gas Channel 2 | `mq135` (MQ-135) | `mq135` (MQ-135 air quality) |
| Optional Context | `humidity`, `temp_c`, `pir` | `humidity`, `temp_c`, `pir` |
| Feature Vector Count | **32 features** | **26 features** |

> [!IMPORTANT]
> **No Silent Coercion**: MQ-2 measurements from early prototypes are **never** silently renamed or coerced into MQ-3 or MQ-4 data. The `LegacyPrototypeAdapter` enforces strict separation and raises actionable errors if an attempt is made to treat legacy data as canonical physical hardware recordings.

---

## 3. Authoritative Feature Contracts

Feature lists and counts are derived dynamically from authoritative tuples in [`ml/config.py`](file:///c:/Users/mshas/OneDrive/Documents/GitHub/atomberg_hackathon/ml/config.py):

### Canonical Hardware Contract (32 Features)
1. **Raw Sensor Measurements (6)**: `raw_mq3`, `raw_mq4`, `raw_mq135`, `raw_humidity`, `raw_temp_c`, `raw_pir`
2. **Baseline Differences ($\Delta x = x - x_{\text{base}}$) (5)**: `delta_mq3`, `delta_mq4`, `delta_mq135`, `delta_humidity`, `delta_temp_c`
3. **Rolling Means (last 10 ticks) (5)**: `mean_mq3`, `mean_mq4`, `mean_mq135`, `mean_humidity`, `mean_temp_c`
4. **Rolling Standard Deviations (5)**: `std_mq3`, `std_mq4`, `std_mq135`, `std_humidity`, `std_temp_c`
5. **Rate of Change / Slope ($\frac{\Delta x}{\Delta t}$) (5)**: `slope_mq3`, `slope_mq4`, `slope_mq135`, `slope_humidity`, `slope_temp_c`
6. **Domain Ratios (3)**:
   - `ratio_mq4_mq135`: Methane/combustible to VOC ratio
   - `ratio_mq3_mq135`: Alcohol/solvent to VOC ratio
   - `ratio_humidity_temp`: Moisture-to-thermal index
7. **Explicit Sensor Availability Indicators (3)**:
   - `has_humidity`: 1.0 if humidity sensor is connected and providing valid measurements; 0.0 if missing/null.
   - `has_temp_c`: 1.0 if temperature sensor is connected and providing valid measurements; 0.0 if missing/null.
   - `has_pir`: 1.0 if PIR presence detector is connected; 0.0 if missing/null.

### Legacy Prototype Contract (26 Features)
For backwards compatibility with early bench recordings, the 26-feature contract uses `mq2` and `mq135` across raw, baseline, rolling mean, rolling std, slope, domain ratios (`ratio_mq2_mq135`, `ratio_humidity_temp`), and the 3 explicit availability indicators (`has_humidity`, `has_temp_c`, `has_pir`).

### Leakage-Free Rolling Window & Warm-Up Rules
- **Past-Only**: Rolling windows use strictly observations where $t \le t_{\text{current}}$, never looking forward.
- **Warm-Up Behavior**: During system startup or session transitions when window length is 1, slopes and standard deviations evaluate to exactly `0.0`. All values remain finite (no `NaN` or `inf`).
- **Session Isolation**: In batch extraction (`extract_from_dataframe`), rolling windows reset on every change of `run_id`, preventing cross-session boundary leakage.

### Optional Sensor Missing-Value Policy & Distinguishability
- **Missingness vs Genuine Measurements**: The feature vector strictly distinguishes genuine measurements from missing sensors:
  - A genuinely measured humidity of 50.0% produces `raw_humidity = 50.0, has_humidity = 1.0`. A missing humidity sensor produces `raw_humidity = 50.0, has_humidity = 0.0`.
  - A genuinely measured temperature of 25.0°C produces `raw_temp_c = 25.0, has_temp_c = 1.0`. A missing temperature sensor produces `raw_temp_c = 25.0, has_temp_c = 0.0`.
  - A genuinely measured unoccupied PIR reading produces `raw_pir = 0.0, has_pir = 1.0`. A missing/disconnected PIR sensor produces `raw_pir = 0.0, has_pir = 0.0`.
- **Domain Ratio Protection**: `ratio_humidity_temp` uses denominator protection `max(1.0, abs(raw_temp_c))` to prevent division by zero or denominator instability near freezing.

---

## 4. Evaluation Leakage Prevention & Training Pipeline

Sensor time-series exhibit high autocorrelation between adjacent observations. Standard random `KFold(shuffle=True)` creates severe temporal leakage, resulting in artificially inflated validation scores.

### Evaluation Strategies
The training pipeline (`ml.training.train`) enforces explicit, leak-free evaluation strategies:

1. **`group_kfold` (Default)**:
   - Groups observations strictly by `run_id` (recording session).
   - Entire recording runs are held out together.
   - **Zero Silent Fallback**: If fewer distinct `run_id` groups exist than `n_splits`, training aborts immediately with an actionable error. It strictly forbids falling back to random shuffled splits.
2. **`time_series_split`**:
   - For continuous recordings, cross-validation respects strict chronological ordering ($t_{\text{train}} < t_{\text{val}}$).
3. **`session_holdout`**:
   - Isolates the complete last session as an independent holdout test set (zero test leakage).
   - The saved model artifact is fitted strictly on training sessions without the holdout session.
   - Refuses execution if only a single session is present.

### Learned Preprocessing Isolation
All scalers and feature transformers are fitted exclusively on training folds and applied to validation/test sets without refitting.

---

## 5. Model Inference Statuses & Safety Interlocks

Inference returns a structured `PredictionResult` exposing explicit provenance and status:

| Status Code | Description | Fan Actuation Policy |
|---|---|---|
| `VALID_MODEL` | Valid prediction from verified `.joblib` bundle matching feature contract. | Allows state machine threshold scaling when `confidence >= min_confidence`. |
| `RULES_FALLBACK` | Heuristic fallback when no trained model is loaded. | **Inhibited**: Does NOT lower trigger thresholds or authorize actuation unless `allow_rules_fallback_trigger: true`. `confidence` is explicitly `None`. |
| `MODEL_UNAVAILABLE` | Model file does not exist. | Safe state: Automation relies strictly on raw rule thresholds. |
| `MODEL_ERROR` | Model file corrupted or pickle load failed. | Safe state: Logs error diagnostic; automation relies on raw rule thresholds. |
| `INPUT_INVALID` | Feature length mismatch, >50% NaN/inf values. | Safe state: Rejects inference vector. |
| `SENSOR_FAULT` | Gas sensors disconnected or returning nulls. | State machine immediately trips into `FAULT` state and lights fault LED. |

---

## 6. Target Environmental Classes

| Code | Label | Physical Description |
|---|---|---|
| `0` | `clean` | Ambient baseline air quality |
| `1` | `steam` | Elevated relative humidity with mild thermal rise and low VOC/combustible gas |
| `2` | `cooking` | Elevated temperature, elevated methane/smoke (`mq4`/`mq2`), moderate VOC |
| `3` | `foul` | Elevated VOCs / NH3 / CO2 (`mq135` / `mq3`), normal humidity and temperature |

*Note on Odor Discrimination*: Low-cost metal-oxide semiconductor (MOS) sensors exhibit broad cross-sensitivity. Environmental condition classification identifies high-level atmospheric patterns rather than specific chemical odor signatures.

---

## 7. Command-Line Workflow

### 7.1 Inspect a Dataset
Profile row counts, sampling intervals, missing values, class distribution, and session groups:
```bash
python -m ml.datasets.inspect_dataset data/runs/run_01_clean.csv
# JSON format:
python -m ml.datasets.inspect_dataset data/runs/run_01_clean.csv --json
```

### 7.2 Validate a Dataset Against Contracts
```bash
# Validate against general schema:
python -m ml.datasets.validate_dataset data/runs/run_01_clean.csv

# Enforce strict physical hardware schema (MQ-3, MQ-4, MQ-135):
python -m ml.datasets.validate_dataset data/runs/run_01_clean.csv --strict-canonical
```

### 7.3 Train a Model Bundle
```bash
# Train using session-based GroupKFold cross-validation:
python -m ml.training.train --data data/training_dataset.csv --eval-strategy group_kfold --n-splits 3

# Train using chronological TimeSeriesSplit:
python -m ml.training.train --data data/training_dataset.csv --eval-strategy time_series_split --n-splits 3
```

### 7.4 Evaluate an Artifact
```bash
python -m ml.training.evaluate --model ml/artifacts/model.joblib --data data/test_dataset.csv
```

### 7.5 Run Pre-Training System Preflight Audit
```bash
python -m ml.diagnostics.preflight
# Strict mode (enforces dataset presence):
python -m ml.diagnostics.preflight --strict-dataset
```

---

## 8. Real-Data Arrival & Production Preparation Workflow (10 Steps)

When genuine sensor recordings become available in approximately two days, follow this verified, zero-leakage workflow:

1. **Record Real Sensor Sessions**:
   Use the dashboard Collect tab or start the control loop in data-collection mode. Ensure sensors have completed their 5-minute preheat warmup cycle.

2. **Assign Valid Ground-Truth Labels**:
   Tag each session with an approved label from `common.schema.LABELS`:
   - `clean` (ambient air)
   - `steam` (water vapor, boiling)
   - `cooking` (frying, hot oil, food preparation)
   - `foul` (organic decay, VOCs, cleaning agents)

3. **Preserve Timestamps and `run_id`**:
   Never concatenate sessions without preserving `run_id`. Session identifiers are strictly required for temporal isolation and leakage prevention. Keep raw session CSVs in `data/runs/`.

4. **Validate Dataset Schema**:
   Verify column names, data types, and sensor channels:
   ```bash
   python -m ml.datasets.validate_dataset data/dataset.csv --strict-canonical
   ```
   Must pass with status `PASSED [VALID]` and type `CANONICAL_HARDWARE`.

5. **Inspect Class Balance and Session Distribution**:
   Check representation across classes and verify at least 2 distinct `run_id` sessions per class:
   ```bash
   python -m ml.datasets.inspect_dataset data/dataset.csv
   ```

6. **Execute Preflight Diagnostics**:
   Ensure all 17 preflight software checks pass before training:
   ```bash
   python -m ml.diagnostics.preflight --strict-dataset
   ```

7. **Execute Session-Aware Training**:
   Train the classifier using group k-fold cross-validation (zero temporal leakage):
   ```bash
   python -m ml.training.train --data data/dataset.csv --eval-strategy group_kfold --n-splits 3 --model-type rf
   ```

8. **Evaluate the Serialized Artifact**:
   Inspect generated metrics, confusion matrix, and feature importances:
   ```bash
   python -m ml.training.evaluate --model data/models/latest.joblib --data data/dataset.csv
   ```
   Inspect report artifacts in `ml/reports/latest/`.

9. **Load and Verify Artifact in Dashboard**:
   Trigger model reload from the dashboard Classifier tab or via API:
   ```bash
   curl -X POST http://localhost:8000/api/ml/reload
   ```
   Verify that `/api/ml/status` transitions to `VALID_MODEL` and contract version displays `CANONICAL_32`.

10. **Verify Inference in Simulation Before Physical Actuation**:
    Keep `hardware_enabled: false` and `fan_control_enabled: false`. Verify live predictions on simulated and replay sensor streams before connecting the relay and exhaust fan.

---

## 9. Model Artifact Lifecycle & Safe Reload Mechanism

The system enforces strict safeguards when deploying or replacing trained models:

- **Atomic Validation**: When `POST /api/ml/reload` is invoked, the replacement artifact is loaded and verified in memory before mutating the running predictor.
- **Rollback on Error**: If a candidate model bundle is corrupted, unreadable, or exhibits a feature-contract mismatch (e.g., wrong feature count or missing indicators), the reload fails with HTTP 400 and preserves the currently active model.
- **Contract Verification**: Only artifacts satisfying the 32-feature canonical contract or 26-feature legacy contract can be activated.
- **Provenance Transparency**: Every prediction records the exact model file name, feature contract version, and prediction timestamp.
- **Safety Defaults**: Loading a model never automatically enables physical GPIO or fan control. Hardware and relay actuation remain disabled until verified by an operator.

