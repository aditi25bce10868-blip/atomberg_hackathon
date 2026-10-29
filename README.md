# Efficio Apex: AI-Powered Smart Ventilation System

[![Python Tests](https://img.shields.io/badge/tests-146%20passed-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey.svg)]()

**Atomberg Efficio Apex** is an AI-powered environmental monitoring and automatic exhaust ventilation system developed for **AtomQUEST 2026**.

The system is designed for deployment on a **Raspberry Pi 5** interfaced with three distinct analog metal-oxide gas sensors (**MQ-3, MQ-4, MQ-135**) via an **MCP3008 10-bit SPI ADC**, operating an exhaust fan through an optoisolated relay subject to a deterministic control state machine and safety interlocks.

---

> [!CAUTION]
> **SAFETY & PROTOTYPE NOTICE**
> - Efficio Apex is a prototype smart ventilation system, **not a certified gas detector or life-safety alarm**.
> - Hardware access (`hardware_enabled`) and fan relay actuation (`fan_control_enabled`) are **disabled by default** in `config.yaml`.
> - **NEVER switch mains AC voltage (110V/230V)** during development or automated tests. Use low-voltage DC fans or bench LEDs.
> - An ML model or rules-based placeholder **never directly triggers the fan relay**. The deterministic state machine remains the sole actuation authority.
> - Rules-based fallback inference (`RULES_FALLBACK`) does **not** fabricate calibrated confidence and cannot lower trigger thresholds unless `allow_rules_fallback_trigger: true` is explicitly configured.
> - Final physical deployment requires voltage verification, optical relay isolation, an external 5V sensor heater power supply, and inspection by a qualified electrician.

---

## 1. System Architecture & Data Path

The system processes data across a strictly decoupled 7-stage pipeline:

```
[ Sensor Input ]  (MCP3008 SPI / Simulated)
       ↓
[ Input Validation ]  (Physical range & numeric checks in common/schema.py)
       ↓
[ Feature Extraction ]  (29 canonical features with zero-lookahead in ml/features/)
       ↓
[ ML Inference ]  (AirQualityPredictor in ml/inference/ with status tagging)
       ↓
[ Prediction Validation ]  (Status verification: VALID_MODEL vs RULES_FALLBACK)
       ↓
[ Control Decision ]  (Deterministic state machine: debounce, cooldown, hard caps)
       ↓
[ Fan Actuation ]  (Optoisolated relay driven only if fan_control_enabled=true)
```

### Directory Structure

```
atomberg_hackathon/
├── common/             # Shared contracts, configuration validation, SQLite DB, logging
│   ├── config.py       # Configuration loading, validation, safe defaults
│   ├── db.py           # Thread-safe SQLite event, row, and command store
│   ├── logger.py       # Application logging
│   └── schema.py       # SensorReading, PredictionResult, InferenceStatus, physical validation
│
├── control/            # Application orchestration & deterministic state machine
│   ├── main.py         # 1-Hz control loop (source -> ML/rules -> state machine -> DB)
│   ├── ml_runtime.py   # Runtime classifier wrapper with fail-safe fallback
│   ├── sources.py      # Data source adapters (Fake, Replay, Serial, Live)
│   └── state_machine.py# Pure-logic state machine (debounce, run-on, hard cap, cooldown)
│
├── hardware/           # Physical device communication & simulated hardware adapter
│   ├── interface.py    # BaseSensorHardware and BaseActuatorHardware ABCs
│   ├── mcp3008.py      # 10-bit SPI ADC driver with voltage conditioning guards
│   ├── README.md       # Hardware pinouts, wiring diagrams, and electrical guide
│   ├── sensors.py      # Physical Raspberry Pi sensor (MQ-3 CH0, MQ-4 CH1, MQ-135 CH2)
│   └── simulated.py    # Deterministic simulated sensor adapter for tests/bench
│
├── ml/                 # Machine learning package lifecycle
│   ├── artifacts/      # Storage for trained model bundles (.joblib) & metadata
│   ├── config.py       # Central contract: 29 canonical features, 23 legacy features
│   ├── data/           # Structured storage: raw/, processed/, external/
│   ├── datasets/       # inspect_dataset.py, validate_dataset.py, adapters.py
│   ├── features/       # FeatureExtractor (canonical/legacy) & build_features(history)
│   ├── inference/      # AirQualityPredictor with schema validation & fallback
│   ├── README.md       # Comprehensive ML lifecycle & evaluation documentation
│   ├── tests/          # ML-specific unit test suite (24 tests)
│   └── training/       # train.py (GroupKFold session CV) & evaluate.py
│
├── server/             # Dashboard API & WebSocket server
│   ├── app.py          # FastAPI application serving /ws/live and /api/*
│   └── static/         # Zero-dependency dark-mode UI (Bootstrap 5, Chart.js)
│
├── tests/              # Full repository test suite (69 tests)
├── config.yaml         # Central system configuration
├── main.py             # Primary application launcher
└── requirements.txt    # Project dependencies
```

---

## 2. Sensor Schema & Feature Contracts

### Canonical Hardware Channels (MCP3008 SPI)
- **CH0**: `mq3` — MQ-3 (Alcohol / VOCs / Ethanol)
- **CH1**: `mq4` — MQ-4 (Methane / Natural Gas / Smoke)
- **CH2**: `mq135` — MQ-135 (Air Quality / NH3 / CO2 / Smoke)
- **Context**: `humidity` (%RH), `temp_c` (°C), `pir` (Occupancy / Motion)

### Canonical 29-Feature Contract
1. **Raw Sensor Channels (6)**: `raw_mq3`, `raw_mq4`, `raw_mq135`, `raw_humidity`, `raw_temp_c`, `raw_pir`
2. **Baseline Differences (5)**: `delta_mq3`, `delta_mq4`, `delta_mq135`, `delta_humidity`, `delta_temp_c`
3. **Rolling Means (5)**: `mean_mq3`, `mean_mq4`, `mean_mq135`, `mean_humidity`, `mean_temp_c`
4. **Rolling Standard Deviations (5)**: `std_mq3`, `std_mq4`, `std_mq135`, `std_humidity`, `std_temp_c`
5. **Rate of Change / Slopes (5)**: `slope_mq3`, `slope_mq4`, `slope_mq135`, `slope_humidity`, `slope_temp_c`
6. **Domain Ratios (3)**: `ratio_mq4_mq135`, `ratio_mq3_mq135`, `ratio_humidity_temp`

### Legacy 23-Feature Contract
Maintained for backwards compatibility with early bench recordings using `mq2` and `mq135`. The `LegacyPrototypeAdapter` strictly refuses to silently coerce MQ-2 into MQ-3 or MQ-4 without genuine hardware calibration.

---

## 3. Quick Start (Simulation Mode)

No physical Raspberry Pi or sensor hardware is required to run and verify the entire system.

### 3.1 Setup Environment

```bash
git clone https://github.com/aditi25bce10868-blip/atomberg_hackathon.git
cd atomberg_hackathon

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3.2 Run the Automated Test Suite

```bash
# Run all tests across the repository:
pytest

# Run tests with short summary:
pytest -q
```
*Total 113 tests passing: 89 in `tests/` and 24 in `ml/tests/` (100% pass rate).*

### 3.3 Launch the Application (Simulation Mode)

```bash
# Launch both control loop and web server:
python main.py
```
Open **`http://localhost:8000`** in your browser to access the real-time dark-mode dashboard.

---

## 4. Machine Learning Commands

The `ml` package CLI tools enforce strict contracts, leakage prevention, and clear diagnostics:

```bash
# 1. Inspect a dataset schema, session groups, and class balance
python -m ml.datasets.inspect_dataset data/runs/run_01_clean.csv

# 2. Validate against schema contracts (enforcing canonical MQ-3/MQ-4/MQ-135 channels)
python -m ml.datasets.validate_dataset data/runs/run_01_clean.csv --strict-canonical

# 3. Train using session-based GroupKFold cross-validation (zero temporal leakage)
python -m ml.training.train --data data/training_dataset.csv --eval-strategy group_kfold --n-splits 3

# 4. Train using chronological TimeSeriesSplit
python -m ml.training.train --data data/training_dataset.csv --eval-strategy time_series_split --n-splits 3

# 5. Train using Session Holdout validation (complete holdout of last session)
python -m ml.training.train --data data/training_dataset.csv --eval-strategy session_holdout

# 6. Evaluate an artifact against a benchmark set
python -m ml.training.evaluate --model ml/artifacts/model.joblib --data data/test_dataset.csv

# 7. Run ML system pre-training preflight diagnostic (17 checks)
python -m ml.diagnostics.preflight
```

---

## 5. Fail-Safe Inference & Control Safety

Inference explicitly returns status tracking:
- `VALID_MODEL`: Genuine model prediction from a verified artifact matching canonical 32-feature schema. Threshold scaling applied only if `confidence >= min_confidence` and label is in `("steam", "cooking", "foul")`.
- `RULES_FALLBACK`: Transparent heuristic rules. `confidence` is `None` (never fabricated). Does not trigger automation or scale thresholds unless `allow_rules_fallback_trigger: true` is explicitly configured. Critical sensor faults are never hidden or converted to clean air.
- `MODEL_UNAVAILABLE` / `MODEL_ERROR`: Safe state; system operates solely on baseline delta thresholds.
- `INPUT_INVALID`: Empty, corrupted, or incompatible feature vector rejected.
- `SENSOR_FAULT`: Critical gas sensors disconnected or returning null/non-finite; immediate trip to `FAULT` state, automation is locked out, software fault LED is flagged in telemetry, and manual wall switch retains override priority.

### Optional Sensor Policy & Distinguishability
Optional context sensors (`humidity`, `temp_c`, `pir`) are handled deterministically with explicit availability indicators:
- Missing humidity defaults to neutral `50.0%` RH (`has_humidity = 0.0`). Measured 50.0% retains `has_humidity = 1.0`.
- Missing temperature defaults to neutral `25.0°C` (`has_temp_c = 0.0`). Measured 25.0°C retains `has_temp_c = 1.0`.
- Missing PIR defaults to `0.0` (`has_pir = 0.0`). Measured unoccupied PIR retains `has_pir = 1.0`.
- Genuine zero readings (`0.0%` humidity, `0.0°C` freezing) are preserved as `0.0` and are strictly distinguishable from missing sensors.
- Clamped denominator in `ratio_humidity_temp` (`max(1.0, abs(temp_c))`) prevents division by zero or NaN propagation.

### Implementation Status
- **Simulation**: 100% implemented and tested (146 automated tests across full-path runtime, ML preflight, and dashboard scenarios).
- **Physical Hardware**: Disabled by default (`hardware_enabled: false`, `fan_control_enabled: false`). Physical hardware assembly, MCP3008 SPI reference voltage calibration, sensor module burn-in (24-48h heater warm-up), physical LED wiring, and optocoupler relay isolation remain pending bench assembly.
