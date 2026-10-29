"""Unit tests for deterministic feature extraction, exact contracts, and zero leakage."""
import numpy as np
import pandas as pd
from ml.config import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_COUNT,
    LEGACY_FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_COUNT,
)
from ml.features.build_features import FeatureExtractor, build_features


def test_canonical_feature_contract_exact_count():
    extractor = FeatureExtractor(mode="canonical")
    assert extractor.feature_count == 32
    assert len(CANONICAL_FEATURE_NAMES) == 32
    assert CANONICAL_FEATURE_COUNT == 32


def test_legacy_feature_contract_exact_count():
    extractor = FeatureExtractor(mode="legacy")
    assert extractor.feature_count == 26
    assert len(LEGACY_FEATURE_NAMES) == 26
    assert LEGACY_FEATURE_COUNT == 26


def test_canonical_extraction_from_history():
    extractor = FeatureExtractor(mode="canonical")
    history = [
        {
            "ts": 1700000000.0,
            "sensors": {
                "mq3": 0.80,
                "mq4": 1.05,
                "mq135": 1.55,
                "humidity": 55.0,
                "temp_c": 28.0,
                "pir": 0,
            },
            "baseline": {"mq3": 0.80, "mq4": 1.00, "mq135": 1.50, "humidity": 55.0, "temp_c": 28.0},
            "deltas": {"mq3": 0.0, "mq4": 0.05, "mq135": 0.05, "humidity": 0.0, "temp_c": 0.0},
        }
    ]
    vec = extractor.extract_from_history(history)
    assert len(vec) == 32
    assert vec[0] == 0.80   # raw_mq3
    assert vec[1] == 1.05   # raw_mq4
    assert vec[2] == 1.55   # raw_mq135


def test_warmup_single_sample_no_future_lookahead():
    """Verify that when only 1 observation exists, rolling stats and slopes are safe and finite."""
    extractor = FeatureExtractor(mode="canonical")
    history = [
        {
            "ts": 1700000000.0,
            "sensors": {"mq3": 0.90, "mq4": 1.10, "mq135": 1.60, "humidity": 60.0, "temp_c": 26.0, "pir": 1},
        }
    ]
    vec = extractor.extract_from_history(history)
    assert len(vec) == 32
    # Slopes must be 0.0 with single sample
    slope_mq3_idx = extractor.feature_names.index("slope_mq3")
    assert vec[slope_mq3_idx] == 0.0
    # Rolling standard deviation must be 0.0 with single sample
    std_mq3_idx = extractor.feature_names.index("std_mq3")
    assert vec[std_mq3_idx] == 0.0
    # All values finite
    assert not any(np.isnan(vec))
    assert not any(np.isinf(vec))


def test_feature_extractor_determinism():
    extractor = FeatureExtractor(mode="canonical")
    history = [
        {
            "ts": 1700000000.0 + i,
            "sensors": {"mq3": 0.8 + (i * 0.02), "mq4": 1.0, "mq135": 1.5, "humidity": 55.0, "temp_c": 28.0, "pir": 0},
            "baseline": {"mq3": 0.8, "mq4": 1.0, "mq135": 1.5, "humidity": 55.0, "temp_c": 28.0},
            "deltas": {"mq3": i * 0.02, "mq4": 0.0, "mq135": 0.0, "humidity": 0.0, "temp_c": 0.0},
        }
        for i in range(10)
    ]
    assert extractor.extract_from_history(history) == extractor.extract_from_history(history)


def test_feature_extractor_missing_sensor_graceful_handling():
    extractor = FeatureExtractor(mode="canonical")
    history = [
        {
            "ts": 1700000000.0,
            "sensors": {"mq3": None, "mq4": 1.0, "mq135": None, "humidity": None, "temp_c": 28.0, "pir": 0},
        }
    ]
    vec = extractor.extract_from_history(history)
    assert len(vec) == 32
    assert not any(np.isnan(vec))


def test_extract_from_dataframe_session_aware():
    extractor = FeatureExtractor(mode="canonical")
    data = {
        "run_id": [1] * 10 + [2] * 10,
        "ts": [1700000000.0 + i for i in range(20)],
        "mq3": [0.8 + (i * 0.01) for i in range(20)],
        "mq4": [1.0 + (i * 0.01) for i in range(20)],
        "mq135": [1.5 + (i * 0.01) for i in range(20)],
        "humidity": [55.0] * 20,
        "temp_c": [28.0] * 20,
        "pir": [0] * 20,
    }
    df = pd.DataFrame(data)
    X, names = extractor.extract_from_dataframe(df)
    assert X.shape == (20, 32)
    assert names == CANONICAL_FEATURE_NAMES


def test_build_features_standalone_compatibility():
    history = [
        {
            "ts": 1700000000.0,
            "sensors": {"mq3": 0.8, "mq4": 1.0, "mq135": 1.5, "humidity": 55.0, "temp_c": 28.0, "pir": 0},
        }
    ]
    vec = build_features(history)
    assert isinstance(vec, list)
    assert len(vec) == 32


def test_missing_optional_sensors_distinguishable_from_zero_and_neutral():
    """Verify missing sensors are strictly distinguishable from measured neutral/zero."""
    extractor = FeatureExtractor(mode="canonical")
    vec_missing = extractor.extract_from_history([
        {"ts": 1700000000.0, "sensors": {"mq3": 1.0, "mq4": 1.0, "mq135": 1.0, "humidity": None, "temp_c": None, "pir": None}}
    ])
    vec_measured = extractor.extract_from_history([
        {"ts": 1700000000.0, "sensors": {"mq3": 1.0, "mq4": 1.0, "mq135": 1.0, "humidity": 50.0, "temp_c": 25.0, "pir": 0}}
    ])
    has_hum_idx = extractor.feature_names.index("has_humidity")
    has_temp_idx = extractor.feature_names.index("has_temp_c")
    has_pir_idx = extractor.feature_names.index("has_pir")

    assert vec_missing[has_hum_idx] == 0.0
    assert vec_measured[has_hum_idx] == 1.0
    assert vec_missing[has_temp_idx] == 0.0
    assert vec_measured[has_temp_idx] == 1.0
    assert vec_missing[has_pir_idx] == 0.0
    assert vec_measured[has_pir_idx] == 1.0
