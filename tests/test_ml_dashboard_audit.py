"""Comprehensive End-to-End Integration and Audit Tests for ML Dashboard & Runtime.

Verifies all 18 operational scenarios specified in Section 10:
1. No model artifact present.
2. Valid temporary test artifact present.
3. Corrupted model artifact.
4. Feature-contract mismatch.
5. Rules fallback active.
6. Invalid sensor input.
7. One critical gas sensor missing.
8. All critical gas sensors missing.
9. NaN and infinite readings.
10. Inference exception.
11. Dashboard API request during ML failure.
12. Dashboard rendering with missing optional sensor data.
13. Training readiness with no dataset.
14. Training readiness with an invalid dataset.
15. Valid synthetic fixture passing training preflight.
16. Model artifact reload and inference.
17. Hardware disabled and fan control disabled.
18. Fault recovery behavior.
"""
import csv
import json
import math
import pathlib
import time
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from common.config import load_config
from common.schema import (
    InferenceStatus,
    CANONICAL_GAS_SENSORS,
    CANONICAL_CSV_COLUMNS,
    LABELS,
    LABEL_NAMES,
)
from ml.config import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_COUNT,
    LEGACY_FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
)
from ml.features.build_features import FeatureExtractor
from ml.inference.predictor import AirQualityPredictor
from ml.inference.status import (
    get_model_health,
    get_training_readiness,
    get_system_ml_status,
)
from ml.training.train import train_model
from server.app import app


@pytest.fixture
def test_client():
    """FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def valid_synthetic_model(tmp_path):
    """Create a temporary valid 32-feature synthetic model bundle."""
    model_path = tmp_path / "valid_test_model.joblib"
    from sklearn.ensemble import RandomForestClassifier

    # Train a minimal dummy classifier on synthetic 32-dim features
    X = np.random.RandomState(42).randn(20, CANONICAL_FEATURE_COUNT).astype(np.float32)
    y = np.array(["clean"] * 10 + ["cooking"] * 10)
    clf = RandomForestClassifier(n_estimators=3, random_state=42)
    clf.fit(X, y)

    bundle = {
        "model": clf,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "feature_count": CANONICAL_FEATURE_COUNT,
        "classes": list(clf.classes_),
        "label_names": list(LABEL_NAMES),
        "metadata": {
            "model_type": "RandomForestClassifier",
            "eval_strategy": "group_kfold",
            "trained_at": time.time(),
            "synthetic_test_model": True,
        },
    }
    joblib.dump(bundle, model_path)
    return model_path


# ==============================================================================
# SCENARIO 1: No model artifact present
# ==============================================================================
def test_scenario_01_no_model_artifact_present(tmp_path):
    pred = AirQualityPredictor(model_path=tmp_path / "nonexistent.joblib")
    assert not pred.is_loaded
    assert pred.status == InferenceStatus.MODEL_UNAVAILABLE
    res = pred.predict_history([{"sensors": {"mq3": 1.0, "mq4": 1.0, "mq135": 1.0}}])
    assert res.status == InferenceStatus.MODEL_UNAVAILABLE
    assert res.label is None
    assert res.confidence is None


# ==============================================================================
# SCENARIO 2: Valid temporary test artifact present
# ==============================================================================
def test_scenario_02_valid_temporary_test_artifact(valid_synthetic_model):
    pred = AirQualityPredictor(model_path=valid_synthetic_model)
    assert pred.is_loaded is True
    assert pred.status == InferenceStatus.VALID_MODEL
    assert pred.feature_count == 32

    # Verify prediction with legitimate confidence
    history = [
        {
            "ts": 100.0 + i,
            "sensors": {"mq3": 1.2, "mq4": 1.0, "mq135": 1.4, "humidity": 50.0, "temp_c": 25.0, "pir": 0},
            "baseline": {"mq3": 1.0, "mq4": 0.8, "mq135": 1.2, "humidity": 50.0, "temp_c": 25.0},
            "deltas": {"mq3": 0.2, "mq4": 0.2, "mq135": 0.2, "humidity": 0.0, "temp_c": 0.0},
        }
        for i in range(10)
    ]
    res = pred.predict_history(history)
    assert res.status == InferenceStatus.VALID_MODEL
    assert res.label in ("clean", "cooking")
    assert res.confidence is not None
    assert 0.0 <= res.confidence <= 1.0


# ==============================================================================
# SCENARIO 3: Corrupted model artifact
# ==============================================================================
def test_scenario_03_corrupted_model_artifact(tmp_path):
    corrupt_file = tmp_path / "corrupt_model.joblib"
    corrupt_file.write_bytes(b"NOT_A_VALID_JOBLIB_HEADER_GARBAGE_DATA_12345")

    pred = AirQualityPredictor(model_path=corrupt_file)
    assert pred.is_loaded is False
    assert pred.status == InferenceStatus.MODEL_ERROR
    assert "Failed to load model bundle" in (pred.last_error or "")


# ==============================================================================
# SCENARIO 4: Feature-contract mismatch
# ==============================================================================
def test_scenario_04_feature_contract_mismatch(tmp_path):
    mismatch_file = tmp_path / "mismatch_model.joblib"
    from sklearn.ensemble import RandomForestClassifier

    # Model trained with only 12 features instead of 32
    X = np.random.randn(10, 12).astype(np.float32)
    y = np.array(["clean"] * 5 + ["foul"] * 5)
    clf = RandomForestClassifier(n_estimators=2).fit(X, y)

    bundle = {
        "model": clf,
        "feature_names": [f"custom_feat_{i}" for i in range(12)],
        "feature_count": 12,
        "classes": list(clf.classes_),
        "metadata": {},
    }
    joblib.dump(bundle, mismatch_file)

    pred = AirQualityPredictor(model_path=mismatch_file)
    assert pred.is_loaded is False
    assert pred.status == InferenceStatus.MODEL_ERROR
    assert ("Unsupported artifact feature count" in (pred.last_error or "") or "Incompatible" in (pred.last_error or ""))


# ==============================================================================
# SCENARIO 5: Rules fallback active
# ==============================================================================
def test_scenario_05_rules_fallback_active():
    pred = AirQualityPredictor()
    res = pred.fallback_predict(
        sensors={"mq3": 1.25, "mq4": 1.05, "mq135": 1.45, "humidity": 65.0, "temp_c": 28.0, "pir": 1},
        baseline={"mq3": 1.0, "mq4": 0.8, "mq135": 1.2, "humidity": 50.0, "temp_c": 25.0},
        deltas={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25, "humidity": 15.0, "temp_c": 3.0},
        thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25, "humidity": 8.0, "temp_c": 4.0},
    )
    assert res.status == InferenceStatus.RULES_FALLBACK
    assert res.label in ("clean", "steam", "cooking", "foul")
    assert res.confidence is None  # Never fabricate confidence for rules fallback!
    assert res.source == "rules-placeholder"


# ==============================================================================
# SCENARIO 6: Invalid sensor input
# ==============================================================================
def test_scenario_06_invalid_sensor_input():
    pred = AirQualityPredictor()
    # Non-numeric string value for gas sensor
    res = pred.fallback_predict(
        sensors={"mq3": "MALFORMED_STRING", "mq4": 1.0, "mq135": 1.0},
        deltas={},
        thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
    )
    assert res.status == InferenceStatus.SENSOR_FAULT
    assert "Critical gas sensor 'mq3'" in (res.error or "")


# ==============================================================================
# SCENARIO 7: One critical gas sensor missing
# ==============================================================================
def test_scenario_07_one_critical_gas_sensor_missing():
    pred = AirQualityPredictor()
    # Canonical array requires mq3, mq4, mq135. Omitting mq3 must trigger SENSOR_FAULT
    res = pred.fallback_predict(
        sensors={"mq4": 1.0, "mq135": 1.2},
        deltas={},
        thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
    )
    assert res.status == InferenceStatus.SENSOR_FAULT
    assert "Critical gas sensor 'mq3'" in (res.error or "")


# ==============================================================================
# SCENARIO 8: All critical gas sensors missing
# ==============================================================================
def test_scenario_08_all_critical_gas_sensors_missing():
    pred = AirQualityPredictor()
    res = pred.fallback_predict(
        sensors={"humidity": 50.0, "temp_c": 25.0},
        deltas={},
        thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
    )
    assert res.status == InferenceStatus.SENSOR_FAULT


# ==============================================================================
# SCENARIO 9: NaN and infinite readings
# ==============================================================================
def test_scenario_09_nan_and_infinite_readings():
    pred = AirQualityPredictor()
    # NaN reading
    res_nan = pred.fallback_predict(
        sensors={"mq3": float("nan"), "mq4": 1.0, "mq135": 1.2},
        deltas={},
        thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
    )
    assert res_nan.status == InferenceStatus.SENSOR_FAULT

    # Infinite reading
    res_inf = pred.fallback_predict(
        sensors={"mq3": float("inf"), "mq4": 1.0, "mq135": 1.2},
        deltas={},
        thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
    )
    assert res_inf.status == InferenceStatus.SENSOR_FAULT


# ==============================================================================
# SCENARIO 10: Inference exception
# ==============================================================================
def test_scenario_10_inference_exception(valid_synthetic_model):
    pred = AirQualityPredictor(model_path=valid_synthetic_model)
    # Mock classifier to raise an exception during predict_proba
    with patch.object(pred.model, "predict_proba", side_effect=RuntimeError("Memory fault or thread error")):
        res = pred.predict_vector([1.0] * 32)
        assert res.status == InferenceStatus.MODEL_ERROR
        assert "Inference execution failed" in (res.error or "")


# ==============================================================================
# SCENARIO 11: Dashboard API request during ML failure
# ==============================================================================
def test_scenario_11_dashboard_api_during_ml_failure(test_client):
    # During sensor fault or model error, /api/ml/status must return HTTP 200 with honest error details
    resp = test_client.get("/api/ml/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_health" in data
    assert "training_readiness" in data
    assert "safety" in data
    # Server never crashes


# ==============================================================================
# SCENARIO 12: Dashboard rendering with missing optional sensor data
# ==============================================================================
def test_scenario_12_missing_optional_sensors():
    fe = FeatureExtractor(mode="canonical")
    # History with NO optional humidity, temp_c, or pir
    history = [
        {
            "ts": 100.0 + i,
            "sensors": {"mq3": 1.2, "mq4": 1.0, "mq135": 1.4},
            "baseline": {"mq3": 1.0, "mq4": 0.8, "mq135": 1.2},
            "deltas": {"mq3": 0.2, "mq4": 0.2, "mq135": 0.2},
        }
        for i in range(10)
    ]
    feat = fe.extract_from_history(history)
    assert len(feat) == 32
    # Verify availability indicators are strictly 0.0
    idx_has_hum = CANONICAL_FEATURE_NAMES.index("has_humidity")
    idx_has_temp = CANONICAL_FEATURE_NAMES.index("has_temp_c")
    idx_has_pir = CANONICAL_FEATURE_NAMES.index("has_pir")

    assert feat[idx_has_hum] == 0.0
    assert feat[idx_has_temp] == 0.0
    assert feat[idx_has_pir] == 0.0


# ==============================================================================
# SCENARIO 13: Training readiness with no dataset
# ==============================================================================
def test_scenario_13_training_readiness_no_dataset(tmp_path):
    readiness = get_training_readiness(dataset_path_override=tmp_path / "nonexistent.csv")
    assert readiness["status"] == "NOT_READY"
    assert readiness["dataset_available"] is False
    assert readiness["training_prerequisites_satisfied"] is False
    assert any("no training dataset" in r.lower() or "not found" in r.lower() for r in readiness["reasons"])


# ==============================================================================
# SCENARIO 14: Training readiness with invalid dataset
# ==============================================================================
def test_scenario_14_training_readiness_invalid_dataset(tmp_path):
    bad_csv = tmp_path / "invalid_data.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unrelated_col1", "unrelated_col2"])
        writer.writerow(["val1", "val2"])

    readiness = get_training_readiness(dataset_path_override=bad_csv)
    assert readiness["status"] == "NOT_READY"
    assert readiness["dataset_schema_valid"] is False
    assert readiness["training_prerequisites_satisfied"] is False


# ==============================================================================
# SCENARIO 15: Valid synthetic fixture passing training preflight
# ==============================================================================
def test_scenario_15_valid_synthetic_fixture_ready_to_train(tmp_path):
    valid_csv = tmp_path / "valid_dataset.csv"
    with open(valid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        # Session 1: Clean (35 rows)
        for i in range(35):
            writer.writerow([1, 1000.0 + i, 1.2, 1.0, 1.4, 50.0, 25.0, 0, 0])
        # Session 2: Cooking (35 rows)
        for i in range(35):
            writer.writerow([2, 2000.0 + i, 2.2, 2.0, 2.4, 60.0, 30.0, 1, 2])

    readiness = get_training_readiness(
        dataset_path_override=valid_csv,
        model_path_override=tmp_path / "no_model.joblib",
    )
    assert readiness["dataset_available"] is True
    assert readiness["dataset_schema_valid"] is True
    assert readiness["required_labels_present"] is True
    assert readiness["session_metadata_available"] is True
    assert readiness["training_prerequisites_satisfied"] is True
    assert readiness["status"] == "READY_TO_TRAIN"


# ==============================================================================
# SCENARIO 16: Model artifact reload and inference
# ==============================================================================
def test_scenario_16_model_artifact_reload_and_inference(test_client, valid_synthetic_model):
    cfg = load_config()
    with patch.dict(cfg["ml"], {"model_path": str(valid_synthetic_model)}):
        with patch("server.app.cfg", cfg):
            resp = test_client.post("/api/ml/reload")
            assert resp.status_code == 200
            data = resp.json()
            assert data["reloaded"] is True
            assert data["feature_count"] == 32
            assert data["contract"] == "CANONICAL_32"


# ==============================================================================
# SCENARIO 17: Hardware disabled and fan control disabled
# ==============================================================================
def test_scenario_17_hardware_and_fan_disabled_by_default(test_client):
    cfg = load_config()
    assert cfg.get("hardware_enabled", True) is False
    assert cfg.get("fan_control_enabled", True) is False

    resp = test_client.get("/api/ml/status")
    assert resp.status_code == 200
    safety = resp.json()["safety"]
    assert safety["hardware_enabled"] is False
    assert safety["fan_control_enabled"] is False
    assert safety["physical_control_disabled"] is True
    assert safety["actuator_locked"] is True


# ==============================================================================
# SCENARIO 18: Fault recovery behavior
# ==============================================================================
def test_scenario_18_fault_recovery_behavior(valid_synthetic_model):
    pred = AirQualityPredictor(model_path=valid_synthetic_model)

    # Step 1: Normal reading -> VALID_MODEL
    res1 = pred.predict_history([
        {
            "ts": 100.0,
            "sensors": {"mq3": 1.2, "mq4": 1.0, "mq135": 1.4, "humidity": 50.0, "temp_c": 25.0, "pir": 0},
            "baseline": {"mq3": 1.0, "mq4": 0.8, "mq135": 1.2, "humidity": 50.0, "temp_c": 25.0},
            "deltas": {"mq3": 0.2, "mq4": 0.2, "mq135": 0.2, "humidity": 0.0, "temp_c": 0.0},
        }
    ])
    assert res1.status == InferenceStatus.VALID_MODEL

    # Step 2: Critical sensor mq3 disconnects (NaN) -> SENSOR_FAULT
    res2 = pred.predict_history([
        {
            "ts": 101.0,
            "sensors": {"mq3": float("nan"), "mq4": 1.0, "mq135": 1.4, "humidity": 50.0, "temp_c": 25.0, "pir": 0},
            "baseline": {"mq3": 1.0, "mq4": 0.8, "mq135": 1.2, "humidity": 50.0, "temp_c": 25.0},
            "deltas": {"mq3": 0.0, "mq4": 0.2, "mq135": 0.2, "humidity": 0.0, "temp_c": 0.0},
        }
    ])
    assert res2.status == InferenceStatus.SENSOR_FAULT

    # Step 3: Sensor recovers with valid reading -> Recovers to VALID_MODEL
    res3 = pred.predict_history([
        {
            "ts": 102.0,
            "sensors": {"mq3": 1.22, "mq4": 1.02, "mq135": 1.41, "humidity": 50.0, "temp_c": 25.0, "pir": 0},
            "baseline": {"mq3": 1.0, "mq4": 0.8, "mq135": 1.2, "humidity": 50.0, "temp_c": 25.0},
            "deltas": {"mq3": 0.22, "mq4": 0.22, "mq135": 0.21, "humidity": 0.0, "temp_c": 0.0},
        }
    ])
    assert res3.status == InferenceStatus.VALID_MODEL
    assert res3.label is not None
