"""Unit tests for root application startup and dashboard server endpoints."""
import pytest
from starlette.testclient import TestClient
import main


@pytest.fixture
def client():
    return TestClient(main.app)


def test_main_module_exports_fastapi_app():
    assert hasattr(main, "app")
    assert main.app.title == "Efficio Apex Dashboard API"


def test_endpoint_root_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200


def test_endpoint_debug_view(client):
    response = client.get("/debug")
    assert response.status_code == 200
    assert "Efficio Apex" in response.text


def test_endpoint_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "schema" in data
    assert "db" in data
    assert data["hardware_enabled"] is False
    assert data["fan_control_enabled"] is False


def test_endpoint_api_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "source" in data
    assert "timers" in data
    assert "thresholds" in data


def test_endpoint_collect_start_validation(client):
    # Valid label
    res_valid = client.post("/api/collect/start", json={"label": "clean", "note": "test_note"})
    assert res_valid.status_code == 200
    assert res_valid.json()["queued"] is True

    # Invalid label returns 400
    res_invalid = client.post("/api/collect/start", json={"label": "invalid_label_123"})
    assert res_invalid.status_code == 400


def test_endpoint_api_ml_status(client):
    """GET /api/ml/status returns 200 with all telemetry sections and locked safety controls."""
    resp = client.get("/api/ml/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "inference_status" in data
    assert "model_health" in data
    assert "training_readiness" in data
    assert "safety" in data
    safety = data["safety"]
    assert safety["hardware_enabled"] is False
    assert safety["fan_control_enabled"] is False
    assert safety["physical_control_disabled"] is True
    assert safety["actuator_locked"] is True


def test_endpoint_api_ml_reload_no_model_artifact(client, tmp_path):
    """POST /api/ml/reload with no model artifact returns 404 with structured detail and preserves active model."""
    from unittest.mock import patch
    import server.app as srv
    missing_path = tmp_path / "nonexistent_model.joblib"
    original_predictor = srv._get_active_predictor()

    with patch.dict(srv.cfg["ml"], {"model_path": str(missing_path)}):
        resp = client.post("/api/ml/reload")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
        assert f"Model artifact not found at '{missing_path.name}'" in data["detail"]
        # Ensure active model was preserved
        assert srv._active_predictor is original_predictor


def test_endpoint_api_ml_reload_valid_artifact(client, tmp_path):
    """POST /api/ml/reload with valid artifact updates active model and returns 200."""
    from unittest.mock import patch
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    import server.app as srv
    from ml.config import CANONICAL_FEATURE_NAMES, CANONICAL_FEATURE_COUNT
    from common.schema import LABEL_NAMES

    model_path = tmp_path / "test_valid.joblib"
    X = np.random.RandomState(42).randn(10, CANONICAL_FEATURE_COUNT).astype(np.float32)
    y = np.array(["clean"] * 5 + ["cooking"] * 5)
    clf = RandomForestClassifier(n_estimators=2, random_state=42)
    clf.fit(X, y)
    bundle = {
        "model": clf,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "feature_count": CANONICAL_FEATURE_COUNT,
        "classes": list(clf.classes_),
        "label_names": list(LABEL_NAMES),
        "metadata": {"model_type": "RandomForestClassifier", "test_artifact": True},
    }
    joblib.dump(bundle, model_path)

    with patch.dict(srv.cfg["ml"], {"model_path": str(model_path)}):
        resp = client.post("/api/ml/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["reloaded"] is True
        assert data["feature_count"] == 32
        assert data["contract"] == "CANONICAL_32"
        assert srv._active_predictor.is_loaded is True
        assert srv._active_predictor.model_id == "RandomForestClassifier"


def test_endpoint_api_ml_reload_corrupted_artifact_preserves_active_model(client, tmp_path):
    """POST /api/ml/reload with corrupted artifact returns 400 and preserves active model."""
    from unittest.mock import patch
    from ml.inference.predictor import AirQualityPredictor
    import server.app as srv

    initial_predictor = AirQualityPredictor()
    srv._active_predictor = initial_predictor

    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"corrupted binary data not a valid joblib")

    with patch.dict(srv.cfg["ml"], {"model_path": str(corrupt_path)}):
        resp = client.post("/api/ml/reload")
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "Model artifact validation failed" in data["detail"]
        assert srv._active_predictor is initial_predictor


def test_endpoint_api_ml_reload_incompatible_artifact_preserves_active_model(client, tmp_path):
    """POST /api/ml/reload with incompatible feature count returns 400 and preserves active model."""
    from unittest.mock import patch
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from ml.inference.predictor import AirQualityPredictor
    import server.app as srv

    initial_predictor = AirQualityPredictor()
    srv._active_predictor = initial_predictor

    incompatible_path = tmp_path / "incompatible.joblib"
    X = np.random.RandomState(42).randn(10, 15).astype(np.float32)
    y = np.array(["clean"] * 5 + ["cooking"] * 5)
    clf = RandomForestClassifier(n_estimators=2, random_state=42)
    clf.fit(X, y)
    bundle = {
        "model": clf,
        "feature_names": [f"feat_{i}" for i in range(15)],
        "feature_count": 15,
        "classes": list(clf.classes_),
        "metadata": {"model_type": "IncompatibleModel"},
    }
    joblib.dump(bundle, incompatible_path)

    with patch.dict(srv.cfg["ml"], {"model_path": str(incompatible_path)}):
        resp = client.post("/api/ml/reload")
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "Model artifact validation failed" in data["detail"]
        assert "Unsupported artifact feature count: 15" in data["detail"]
        assert srv._active_predictor is initial_predictor

