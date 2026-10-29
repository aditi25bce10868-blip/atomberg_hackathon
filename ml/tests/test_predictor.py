"""Unit tests for ML inference runtime, input validation, status tracking, and fallback mechanisms."""
import pytest
from common.schema import LABEL_NAMES, InferenceStatus
from ml.config import CANONICAL_FEATURE_COUNT, CANONICAL_FEATURE_NAMES
from ml.inference.predictor import AirQualityPredictor


def test_predictor_initialization_without_model():
    predictor = AirQualityPredictor("non_existent_model.joblib")
    assert predictor.is_loaded is False
    assert predictor.status == InferenceStatus.MODEL_UNAVAILABLE
    assert predictor.model_source == "none"


def test_predictor_fallback_predict():
    predictor = AirQualityPredictor()
    sensors = {"mq135": 1.5, "mq2": 1.0, "humidity": 75.0, "temp_c": 30.0, "pir": 0}
    deltas = {"mq135": 0.0, "mq2": 0.0, "humidity": 20.0, "temp_c": 2.0}
    thresholds = {"mq135": 0.25, "mq2": 0.25, "humidity": 8.0, "temp_c": 4.0}

    res = predictor.fallback_predict(sensors, deltas, thresholds)
    assert res.status == InferenceStatus.RULES_FALLBACK
    assert res.source == "rules-placeholder"
    assert res.label in LABEL_NAMES
    # Safety: Confidence must NOT be fabricated for heuristic rules fallback
    assert res.confidence is None
    assert set(res.probs.keys()) == set(LABEL_NAMES)
    assert abs(sum(res.probs.values()) - 1.0) < 0.01


def test_predictor_returns_unloaded_status_when_not_loaded():
    predictor = AirQualityPredictor()
    res = predictor.predict_vector([1.0] * CANONICAL_FEATURE_COUNT)
    assert res.status == InferenceStatus.MODEL_UNAVAILABLE
    assert res.label is None
    assert res.confidence is None


def test_predictor_corrupted_model_file(tmp_path):
    bad_model = tmp_path / "corrupted.joblib"
    bad_model.write_text("not a pickle file", encoding="utf-8")

    predictor = AirQualityPredictor(bad_model)
    assert predictor.is_loaded is False
    assert predictor.status == InferenceStatus.MODEL_ERROR
    assert predictor.model_source == "corrupted"


def test_predictor_feature_vector_dimension_mismatch(tmp_path):
    import joblib
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0.0] * CANONICAL_FEATURE_COUNT], [0])
    bundle = {
        "model": dummy,
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "class_names": LABEL_NAMES,
        "model_id": "dummy_test",
        "training_timestamp": 1700000000.0,
    }
    bundle_path = tmp_path / "dummy.joblib"
    joblib.dump(bundle, bundle_path)

    predictor = AirQualityPredictor(bundle_path)
    assert predictor.is_loaded is True

    res = predictor.predict_vector([1.0] * 20)
    assert res.status == InferenceStatus.INPUT_INVALID
    assert "Feature length mismatch" in res.error
    assert res.label is None


def test_predictor_sensor_fault_detection_in_history(tmp_path):
    import joblib
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0.0] * CANONICAL_FEATURE_COUNT], [0])
    bundle = {
        "model": dummy,
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "class_names": LABEL_NAMES,
        "model_id": "dummy_test",
    }
    bundle_path = tmp_path / "dummy.joblib"
    joblib.dump(bundle, bundle_path)

    predictor = AirQualityPredictor(bundle_path)
    faulty_history = [
        {
            "ts": 1700000000.0,
            "sensors": {"mq3": None, "mq4": None, "mq135": None, "humidity": 50.0},
            "baseline": {},
            "deltas": {},
        }
    ]
    res = predictor.predict_history(faulty_history)
    assert res.status == InferenceStatus.SENSOR_FAULT
    assert "faulty" in res.error.lower()


def test_predictor_rejects_incompatible_feature_names(tmp_path):
    import joblib
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0.0] * CANONICAL_FEATURE_COUNT], [0])
    bundle = {
        "model": dummy,
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_names": [f"f_{i}" for i in range(CANONICAL_FEATURE_COUNT)],
        "class_names": LABEL_NAMES,
        "model_id": "bad_names",
    }
    bundle_path = tmp_path / "bad_names.joblib"
    joblib.dump(bundle, bundle_path)

    predictor = AirQualityPredictor(bundle_path)
    assert predictor.is_loaded is False
    assert predictor.status == InferenceStatus.MODEL_ERROR
