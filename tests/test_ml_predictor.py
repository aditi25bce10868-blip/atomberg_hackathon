"""Unit tests for ML inference runtime, input validation, status tracking, and fallback mechanisms."""
import pytest
from common import config
from common.schema import LABEL_NAMES, InferenceStatus
from control.ml_runtime import Classifier
from ml.config import CANONICAL_FEATURE_COUNT, LEGACY_FEATURE_COUNT, CANONICAL_FEATURE_NAMES
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
    """Verify predictor rejects wrong feature length with INPUT_INVALID."""
    import joblib
    from sklearn.dummy import DummyClassifier

    # Train a dummy model
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

    # Pass 20 features instead of CANONICAL_FEATURE_COUNT
    res = predictor.predict_vector([1.0] * 20)
    assert res.status == InferenceStatus.INPUT_INVALID
    assert "Feature length mismatch" in res.error
    assert res.label is None


def test_predictor_sensor_fault_detection_in_history(tmp_path):
    """Verify predictor flags SENSOR_FAULT when all gas sensor values are null."""
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
    # History with broken gas sensors
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
    """Verify predictor rejects artifact whose feature names don't match canonical or legacy contracts."""
    import joblib
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0.0] * CANONICAL_FEATURE_COUNT], [0])
    bundle = {
        "model": dummy,
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_names": [f"arbitrary_{i}" for i in range(CANONICAL_FEATURE_COUNT)],
        "class_names": LABEL_NAMES,
        "model_id": "mismatched_test",
    }
    bundle_path = tmp_path / "mismatched.joblib"
    joblib.dump(bundle, bundle_path)

    predictor = AirQualityPredictor(bundle_path)
    assert predictor.is_loaded is False
    assert predictor.status == InferenceStatus.MODEL_ERROR
    assert "mismatch" in predictor.last_error.lower()


def test_predictor_rejects_unknown_classes(tmp_path):
    """Verify predictor rejects artifact containing unknown class names."""
    import joblib
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0.0] * CANONICAL_FEATURE_COUNT], [0])
    bundle = {
        "model": dummy,
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "class_names": ["unknown_hazard", "clean"],
        "model_id": "bad_classes",
    }
    bundle_path = tmp_path / "bad_classes.joblib"
    joblib.dump(bundle, bundle_path)

    predictor = AirQualityPredictor(bundle_path)
    assert predictor.is_loaded is False
    assert predictor.status == InferenceStatus.MODEL_ERROR
    assert "unknown class" in predictor.last_error.lower()


def test_control_classifier_wrapper_interface():
    """Verify Classifier in control/ml_runtime.py preserves exact contract for control loop."""
    cfg = config.load()
    clf = Classifier(cfg)
    assert hasattr(clf, "source")

    sensors = {"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 30.0, "pir": 0}
    deltas = {"mq135": 0.0, "mq2": 0.0, "humidity": 0.0, "temp_c": 0.0}
    history = [{"sensors": sensors, "baseline": sensors, "deltas": deltas, "ts": 1700000000.0}]

    pred = clf.predict(sensors, deltas, cfg["thresholds"], history)
    assert isinstance(pred, dict)
    assert "label" in pred
    assert "confidence" in pred
    assert "probs" in pred
    assert "source" in pred
    assert "status" in pred
    assert pred["label"] in LABEL_NAMES
