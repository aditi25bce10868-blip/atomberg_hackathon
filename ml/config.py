"""Machine learning configuration, canonical feature contracts, and storage paths."""
import pathlib
from typing import Dict, List, Tuple
from common.schema import (
    LABELS,
    LABEL_NAMES,
    LABEL_CODES,
    CANONICAL_GAS_SENSORS,
    CANONICAL_OPTIONAL_SENSORS,
    CANONICAL_SENSORS,
    LEGACY_PROTOTYPE_SENSORS,
)

# Package paths
ML_ROOT = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent

# Storage locations
DATA_DIR = ML_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
ARTIFACTS_DIR = ML_ROOT / "artifacts"
REPORTS_DIR = ML_ROOT / "reports" / "latest"

# Target classification classes
CLASS_NAMES: List[str] = list(LABEL_NAMES)
CLASS_CODES: Dict[str, int] = dict(LABEL_CODES)
CODE_TO_CLASS: Dict[int, str] = dict(LABELS)

# Supported evaluation strategies for sensor time-series (NO random shuffled KFold)
VALID_EVAL_STRATEGIES = ("group_kfold", "time_series_split", "session_holdout")
DEFAULT_EVAL_STRATEGY = "group_kfold"

# Training hyperparameter defaults
DEFAULT_MODEL_NAME = "model.joblib"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_FOR_TRAINING = 60  # Minimum samples strictly required before training is permitted
MIN_CLASSES_FOR_TRAINING = 2

# ==============================================================================
# CANONICAL FEATURE CONTRACT (Intended Hardware: MQ-3, MQ-4, MQ-135)
# ==============================================================================
# All feature transformations are strictly past/current-looking (t <= t_now).
# No future information or cross-session leakage is permitted.
CANONICAL_FEATURE_NAMES: List[str] = [
    # 1. Raw measurements (6 features)
    "raw_mq3",
    "raw_mq4",
    "raw_mq135",
    "raw_humidity",
    "raw_temp_c",
    "raw_pir",
    # 2. Baseline deltas: x - x_baseline (5 features)
    "delta_mq3",
    "delta_mq4",
    "delta_mq135",
    "delta_humidity",
    "delta_temp_c",
    # 3. Rolling window means over past W observations (5 features)
    "mean_mq3",
    "mean_mq4",
    "mean_mq135",
    "mean_humidity",
    "mean_temp_c",
    # 4. Rolling window standard deviations (5 features)
    "std_mq3",
    "std_mq4",
    "std_mq135",
    "std_humidity",
    "std_temp_c",
    # 5. Rate of change / slope: dx/dt over past observations (5 features)
    "slope_mq3",
    "slope_mq4",
    "slope_mq135",
    "slope_humidity",
    "slope_temp_c",
    # 6. Domain interaction ratios (3 features)
    "ratio_mq4_mq135",      # Methane/smoke to general air quality ratio
    "ratio_mq3_mq135",      # Alcohol/VOC to general air quality ratio
    "ratio_humidity_temp",  # Moisture-thermal interaction index
    # 7. Optional sensor availability indicators (3 features)
    # Strictly distinguishes genuine 0/neutral readings from missing/disconnected sensors
    "has_humidity",
    "has_temp_c",
    "has_pir",
]
CANONICAL_FEATURE_COUNT: int = len(CANONICAL_FEATURE_NAMES)  # Exactly 32 features

# ==============================================================================
# LEGACY PROTOTYPE FEATURE CONTRACT (For backward-compatible test fixtures & replay)
# ==============================================================================
LEGACY_FEATURE_NAMES: List[str] = [
    # 1. Raw measurements (5 features)
    "raw_mq135",
    "raw_mq2",
    "raw_humidity",
    "raw_temp_c",
    "raw_pir",
    # 2. Baseline deltas (4 features)
    "delta_mq135",
    "delta_mq2",
    "delta_humidity",
    "delta_temp_c",
    # 3. Rolling window means (4 features)
    "mean_mq135",
    "mean_mq2",
    "mean_humidity",
    "mean_temp_c",
    # 4. Rolling window standard deviations (4 features)
    "std_mq135",
    "std_mq2",
    "std_humidity",
    "std_temp_c",
    # 5. Slopes (4 features)
    "slope_mq135",
    "slope_mq2",
    "slope_humidity",
    "slope_temp_c",
    # 6. Domain interaction ratios (2 features)
    "ratio_mq2_mq135",
    "ratio_humidity_temp",
    # 7. Optional sensor availability indicators (3 features)
    "has_humidity",
    "has_temp_c",
    "has_pir",
]
LEGACY_FEATURE_COUNT: int = len(LEGACY_FEATURE_NAMES)  # Exactly 26 features

# Authoritative active feature contract aliases
FEATURE_NAMES: List[str] = CANONICAL_FEATURE_NAMES
FEATURE_COUNT: int = len(FEATURE_NAMES)
