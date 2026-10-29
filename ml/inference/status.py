"""Consolidated ML status, model health, and training readiness inspector.

Provides a unified, reliable interface between ML components, the FastAPI backend,
and the frontend dashboard. Strictly follows zero-fabrication and honest provenance rules.
"""
import math
import pathlib
import time
from typing import Dict, Any, Optional, List, Union

from common import config
from common.logger import get_logger
from common.schema import (
    InferenceStatus,
    LABEL_NAMES,
    LABELS,
    CANONICAL_GAS_SENSORS,
    CANONICAL_OPTIONAL_SENSORS,
)
from ml.config import (
    CANONICAL_FEATURE_COUNT,
    CANONICAL_FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
    LEGACY_FEATURE_NAMES,
    DEFAULT_EVAL_STRATEGY,
    ARTIFACTS_DIR,
)
from ml.datasets.validate_dataset import validate_dataset

logger = get_logger("ml.inference.status")


def get_model_health(
    cfg: Optional[Dict[str, Any]] = None,
    model_path_override: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Inspect and report the health and compatibility of the configured model artifact."""
    if cfg is None:
        cfg = config.load()

    ml_cfg = cfg.get("ml", {})
    if model_path_override is not None:
        model_path = pathlib.Path(model_path_override)
    else:
        mp = ml_cfg.get("model_path")
        model_path = config.path(cfg, str(mp)) if mp else (ARTIFACTS_DIR / "model.joblib")

    artifact_exists = model_path.exists()
    health: Dict[str, Any] = {
        "artifact_path": model_path.name if artifact_exists else (str(model_path.name) if model_path else "none"),
        "artifact_exists": artifact_exists,
        "is_loaded": False,
        "feature_count": None,
        "expected_feature_count": CANONICAL_FEATURE_COUNT,
        "contract_compatibility": "NONE",
        "expected_contract": "CANONICAL_32",
        "classes": [],
        "model_id": None,
        "model_type": None,
        "load_status": "Artifact missing on disk",
        "last_error": None,
    }

    if not artifact_exists:
        health["load_status"] = f"Artifact not found at '{model_path.name}'"
        return health

    # Test loading using AirQualityPredictor
    try:
        from ml.inference.predictor import AirQualityPredictor
        predictor = AirQualityPredictor(model_path)
        health["is_loaded"] = predictor.is_loaded
        health["model_id"] = predictor.model_id
        health["classes"] = list(predictor.class_names)
        health["feature_count"] = predictor.feature_count

        if predictor.is_loaded:
            if predictor.feature_count == CANONICAL_FEATURE_COUNT:
                health["contract_compatibility"] = "CANONICAL_32"
                health["load_status"] = f"Loaded ({predictor.feature_count} canonical features)"
            elif predictor.feature_count == LEGACY_FEATURE_COUNT:
                health["contract_compatibility"] = "LEGACY_26"
                health["load_status"] = f"Loaded ({predictor.feature_count} legacy prototype features)"
            else:
                health["contract_compatibility"] = "INCOMPATIBLE"
                health["load_status"] = f"Feature count mismatch: {predictor.feature_count} != {CANONICAL_FEATURE_COUNT}"
        else:
            health["load_status"] = predictor.last_error or "Model load rejected"
            health["last_error"] = predictor.last_error

    except Exception as e:
        logger.warning("Error inspecting model artifact at %s: %e", model_path, e)
        health["is_loaded"] = False
        health["load_status"] = f"Corrupted or incompatible artifact: {e}"
        health["last_error"] = str(e)

    return health


def get_training_readiness(
    cfg: Optional[Dict[str, Any]] = None,
    dataset_path_override: Optional[Union[str, pathlib.Path]] = None,
    model_path_override: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Inspect dataset availability, schema validity, and training prerequisites."""
    if cfg is None:
        cfg = config.load()

    eval_strategy = cfg.get("ml", {}).get("eval_strategy", DEFAULT_EVAL_STRATEGY)

    # Search candidate dataset paths
    candidate_paths: List[pathlib.Path] = []
    if dataset_path_override is not None:
        candidate_paths = [pathlib.Path(dataset_path_override)]
    else:
        cfg_data_path = cfg.get("ml", {}).get("dataset_path")
        if cfg_data_path:
            candidate_paths.append(config.path(cfg, str(cfg_data_path)))

        candidate_paths.append(config.path(cfg, "data/training_dataset.csv"))
        runs_dir = config.path(cfg, "data/runs")
        if runs_dir.exists():
            csv_runs = list(runs_dir.glob("*.csv"))
            if csv_runs:
                candidate_paths.extend(csv_runs)

    active_dataset: Optional[pathlib.Path] = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            active_dataset = p
            break

    readiness: Dict[str, Any] = {
        "status": "NOT_READY",
        "dataset_available": False,
        "dataset_path": None,
        "dataset_type": None,
        "dataset_rows": 0,
        "dataset_schema_valid": False,
        "required_labels_present": False,
        "session_metadata_available": False,
        "feature_extraction_valid": False,
        "eval_strategy_configured": eval_strategy,
        "training_prerequisites_satisfied": False,
        "model_trained": False,
        "model_artifact_validated": False,
        "reasons": [],
    }

    # Check model artifact state
    model_health = get_model_health(cfg, model_path_override=model_path_override)
    readiness["model_trained"] = model_health["artifact_exists"]
    readiness["model_artifact_validated"] = model_health["is_loaded"] and model_health["contract_compatibility"] in ("CANONICAL_32", "LEGACY_26")

    if active_dataset is None:
        readiness["reasons"].append(
            "No training dataset found in data/ or configured in config.yaml. "
            "Record labeled sensor runs or prepare 'data/training_dataset.csv'."
        )
        if readiness["model_artifact_validated"]:
            readiness["status"] = "TRAINED"
        return readiness

    readiness["dataset_available"] = True
    readiness["dataset_path"] = active_dataset.name

    # Validate dataset
    try:
        report = validate_dataset(active_dataset, strict_canonical=False)
        readiness["dataset_type"] = report.dataset_type
        readiness["dataset_rows"] = report.row_count
        readiness["dataset_schema_valid"] = report.is_valid

        if not report.is_valid:
            for err in report.errors:
                readiness["reasons"].append(f"Dataset error: {err}")

        # Check required labels
        import pandas as pd
        df = pd.read_csv(active_dataset)
        label_col = "label" if "label" in df.columns else ("target" if "target" in df.columns else None)
        if label_col:
            found_classes = set(df[label_col].dropna().unique())
            readiness["required_labels_present"] = len(found_classes) >= 2
            if len(found_classes) < 2:
                readiness["reasons"].append(f"Insufficient distinct classes ({len(found_classes)} found; minimum 2 required).")
        else:
            readiness["required_labels_present"] = False
            readiness["reasons"].append("No label column ('label' or 'target') in dataset.")

        # Check session metadata
        if "run_id" in df.columns:
            sessions = df["run_id"].dropna().unique()
            readiness["session_metadata_available"] = len(sessions) >= 2
            if len(sessions) < 2:
                readiness["reasons"].append(
                    f"Session metadata ('run_id') has only {len(sessions)} session(s). "
                    "GroupKFold and session_holdout require >= 2 distinct sessions."
                )
        else:
            readiness["session_metadata_available"] = False
            readiness["reasons"].append("Missing 'run_id' session column for leakage-free evaluation.")

        # Check feature extraction validity
        from ml.features.build_features import FeatureExtractor
        try:
            mode = "canonical" if report.dataset_type == "CANONICAL_HARDWARE" else "legacy"
            extractor = FeatureExtractor(mode=mode)
            X, feat_names = extractor.extract_from_dataframe(df.head(10))
            readiness["feature_extraction_valid"] = X.shape[0] > 0 and len(feat_names) in (CANONICAL_FEATURE_COUNT, LEGACY_FEATURE_COUNT)
        except Exception as e:
            readiness["feature_extraction_valid"] = False
            readiness["reasons"].append(f"Feature extraction check failed: {e}")

        # Synthesize prerequisite satisfaction
        readiness["training_prerequisites_satisfied"] = (
            readiness["dataset_available"]
            and readiness["dataset_schema_valid"]
            and readiness["required_labels_present"]
            and readiness["session_metadata_available"]
            and readiness["feature_extraction_valid"]
        )

        if readiness["model_artifact_validated"]:
            readiness["status"] = "TRAINED"
        elif readiness["training_prerequisites_satisfied"]:
            readiness["status"] = "READY_TO_TRAIN"
        else:
            readiness["status"] = "NOT_READY"

    except Exception as e:
        readiness["reasons"].append(f"Dataset inspection exception: {e}")
        readiness["status"] = "NOT_READY"

    return readiness


def get_system_ml_status(
    cfg: Optional[Dict[str, Any]] = None,
    latest_row: Optional[Dict[str, Any]] = None,
    model_path_override: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Compile comprehensive ML status including inference state, model health, and safety."""
    if cfg is None:
        cfg = config.load()

    model_health = get_model_health(cfg, model_path_override=model_path_override)
    training_readiness = get_training_readiness(cfg)

    ml_data = latest_row.get("ml") if latest_row else None
    sensors = latest_row.get("sensors") if latest_row else {}
    health = latest_row.get("health") if latest_row else {}
    state = latest_row.get("state") if latest_row else "WARMUP"
    mode = latest_row.get("mode") if latest_row else cfg.get("source", "fake").upper()
    ts = latest_row.get("ts") if latest_row else None

    # Derive inference status
    if ml_data and "status" in ml_data:
        status = ml_data["status"]
        label = ml_data.get("label")
        confidence = ml_data.get("confidence")
        source = ml_data.get("source") or model_health["load_status"]
        model_id = ml_data.get("model_id") or model_health["model_id"]
        probs = ml_data.get("probs")
        error = ml_data.get("error")
        pred_ts = ml_data.get("ts") or ts
    elif model_health["is_loaded"]:
        status = InferenceStatus.VALID_MODEL
        label = None
        confidence = None
        source = f"model:{model_health['artifact_path']}"
        model_id = model_health["model_id"]
        probs = None
        error = None
        pred_ts = None
    else:
        status = InferenceStatus.MODEL_UNAVAILABLE
        label = None
        confidence = None
        source = "rules-placeholder"
        model_id = None
        probs = None
        error = "No trained model loaded. ML predictions are unavailable or using the explicitly identified experimental fallback."
        pred_ts = None

    # Identify sensor faults
    faulty_sensors = [k for k, h in health.items() if h != "ok"] if health else []

    safety_dict = {
        "source_mode": mode,
        "is_simulated": mode == "SIMULATED",
        "hardware_mode": mode,
        "hardware_enabled": cfg.get("hardware_enabled", False),
        "fan_control_enabled": cfg.get("fan_control_enabled", False),
        "physical_control_disabled": not cfg.get("hardware_enabled", False),
        "controller_state": state,
        "sensor_faults": faulty_sensors,
        "ml_fault": status in (InferenceStatus.MODEL_ERROR, InferenceStatus.INPUT_INVALID, InferenceStatus.SENSOR_FAULT),
        "ml_faults": [error] if (status in (InferenceStatus.MODEL_ERROR, InferenceStatus.INPUT_INVALID, InferenceStatus.SENSOR_FAULT) and error) else [],
        "actuator_locked": not cfg.get("fan_control_enabled", False),
        "actuation_errors": [],
        "fault_led": latest_row.get("fault_led", False) if latest_row else False,
        "override": latest_row.get("override", False) if latest_row else False,
    }

    status_payload = {
        "status": status,
        "inference_status": status,
        "label": label if status in (InferenceStatus.VALID_MODEL, InferenceStatus.RULES_FALLBACK) else None,
        "predicted_class": label if status in (InferenceStatus.VALID_MODEL, InferenceStatus.RULES_FALLBACK) else None,
        "confidence": confidence if (status == InferenceStatus.VALID_MODEL and confidence is not None) else None,
        "probs": probs if probs else None,
        "source": source,
        "prediction_source": source,
        "model_id": model_id,
        "timestamp": pred_ts,
        "last_inference_ts": pred_ts,
        "error": error,
        "last_inference_error": error,
        "sensor_health": health,
        "faulty_sensors": faulty_sensors,
        "model_health": model_health,
        "training_readiness": training_readiness,
        "safety": safety_dict,
        "system_safety": safety_dict,
    }
    return status_payload
