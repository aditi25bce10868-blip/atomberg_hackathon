"""Fail-safe real-time model inference runtime with explicit provenance and status reporting."""
import math
import pathlib
import time
from typing import Dict, List, Optional, Any, Sequence, Union
import numpy as np

from common.logger import get_logger
from common.schema import PredictionResult, InferenceStatus, LABELS, LABEL_NAMES
from ml.config import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_COUNT,
    LEGACY_FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
)
from ml.features.build_features import FeatureExtractor

logger = get_logger("ml.inference.predictor")


class InferenceError(Exception):
    """Raised when model inference execution cannot proceed."""
    pass


class AirQualityPredictor:
    """Inference engine for environmental condition classification.
    
    Safety and Provenance Rules:
    - Never presents a placeholder as a genuine trained model prediction.
    - Rejects malformed artifacts and feature mismatches with explicit error states.
    - Rejects invalid, stale, or all-NaN inputs with INPUT_INVALID status.
    - Never invents calibrated confidence for placeholder or error results.
    """

    def __init__(self, model_path: Optional[Union[str, pathlib.Path]] = None):
        self.model = None
        self.feature_names = list(CANONICAL_FEATURE_NAMES)
        self.feature_count = CANONICAL_FEATURE_COUNT
        self.class_names = list(LABEL_NAMES)
        self.model_id: Optional[str] = None
        self.model_source = "rules-placeholder"
        self.is_loaded = False
        self.status = InferenceStatus.MODEL_UNAVAILABLE
        self.last_error: Optional[str] = None
        self._extractor = FeatureExtractor(mode="canonical")

        if model_path:
            self.load(model_path)

    def load(self, model_path: Union[str, pathlib.Path]) -> bool:
        """Load and validate a saved model bundle (.joblib).
        
        Returns:
            True if valid model bundle loaded, False otherwise.
        """
        p = pathlib.Path(model_path)
        if not p.exists():
            self.model = None
            self.is_loaded = False
            self.status = InferenceStatus.MODEL_UNAVAILABLE
            self.model_source = "none"
            self.last_error = f"Model artifact not found at '{p}'"
            logger.info("Model file '%s' not found. Inference set to MODEL_UNAVAILABLE.", p)
            return False

        try:
            import joblib
            bundle = joblib.load(p)

            if isinstance(bundle, dict) and "model" in bundle:
                loaded_model = bundle["model"]
                self.feature_names = bundle.get("feature_names", list(CANONICAL_FEATURE_NAMES))
                self.feature_count = len(self.feature_names)
                self.class_names = bundle.get("class_names") or [
                    LABELS.get(c, str(c)) for c in loaded_model.classes_
                ]
                self.model_id = bundle.get("metadata", {}).get("model_type", p.stem)
            else:
                loaded_model = bundle
                self.class_names = [
                    LABELS.get(c, str(c)) if isinstance(c, (int, np.integer)) else str(c)
                    for c in getattr(loaded_model, "classes_", LABEL_NAMES)
                ]
                self.model_id = p.stem

            if not hasattr(loaded_model, "predict_proba"):
                self.model = None
                self.is_loaded = False
                self.status = InferenceStatus.MODEL_ERROR
                self.last_error = "Model artifact missing 'predict_proba' method"
                logger.error("Loaded model from %s lacks 'predict_proba'.", p)
                return False

            # Strict feature contract validation: must match CANONICAL (29) or LEGACY (23) schema
            if self.feature_count == CANONICAL_FEATURE_COUNT:
                if self.feature_names != CANONICAL_FEATURE_NAMES:
                    self.model = None
                    self.is_loaded = False
                    self.status = InferenceStatus.MODEL_ERROR
                    self.last_error = "Artifact feature names mismatch for canonical 29-feature contract"
                    logger.error("Loaded model from %s: %s", p, self.last_error)
                    return False
                mode = "canonical"
            elif self.feature_count == LEGACY_FEATURE_COUNT:
                if self.feature_names != LEGACY_FEATURE_NAMES:
                    self.model = None
                    self.is_loaded = False
                    self.status = InferenceStatus.MODEL_ERROR
                    self.last_error = "Artifact feature names mismatch for legacy 23-feature contract"
                    logger.error("Loaded model from %s: %s", p, self.last_error)
                    return False
                mode = "legacy"
            else:
                self.model = None
                self.is_loaded = False
                self.status = InferenceStatus.MODEL_ERROR
                self.last_error = (
                    f"Unsupported artifact feature count: {self.feature_count}. "
                    f"Must match canonical ({CANONICAL_FEATURE_COUNT}) or legacy ({LEGACY_FEATURE_COUNT})."
                )
                logger.error("Loaded model from %s: %s", p, self.last_error)
                return False

            # Strict class label validation: all classes must be known in LABEL_NAMES
            unknown_classes = [c for c in self.class_names if c not in LABEL_NAMES]
            if unknown_classes:
                self.model = None
                self.is_loaded = False
                self.status = InferenceStatus.MODEL_ERROR
                self.last_error = f"Artifact contains unknown class labels: {unknown_classes}. Expected {LABEL_NAMES}."
                logger.error("Loaded model from %s: %s", p, self.last_error)
                return False

            # Configure feature extractor mode to match verified bundle
            self._extractor = FeatureExtractor(mode=mode)
            self.model = loaded_model
            self.model_source = f"model:{p.name}"
            self.is_loaded = True
            self.status = InferenceStatus.VALID_MODEL
            self.last_error = None
            logger.info("Loaded ML model bundle successfully from %s (features: %d, classes: %s)", p, self.feature_count, self.class_names)
            return True

        except Exception as e:
            self.model = None
            self.is_loaded = False
            self.status = InferenceStatus.MODEL_ERROR
            self.model_source = "corrupted"
            self.last_error = f"Failed to load model bundle: {e}"
            logger.error("Error loading model from %s: %s", p, e)
            return False

    def predict_vector(self, feature_vector: Sequence[float]) -> PredictionResult:
        """Execute prediction on a pre-extracted feature vector."""
        if not self.is_loaded or self.model is None:
            return PredictionResult(
                status=self.status,
                label=None,
                confidence=None,
                probs=None,
                source=self.model_source,
                model_id=self.model_id,
                error=self.last_error or "Model is not loaded",
                ts=time.time(),
            )

        if len(feature_vector) != self.feature_count:
            return PredictionResult(
                status=InferenceStatus.INPUT_INVALID,
                label=None,
                confidence=None,
                probs=None,
                source=self.model_source,
                model_id=self.model_id,
                error=f"Feature length mismatch: expected {self.feature_count}, got {len(feature_vector)}",
                ts=time.time(),
            )

        # Check for non-finite values and sanitize
        clean_vec: List[float] = []
        nan_count = 0
        for v in feature_vector:
            if v is None or math.isnan(v) or math.isinf(v):
                clean_vec.append(0.0)
                nan_count += 1
            else:
                clean_vec.append(float(v))

        if nan_count > len(feature_vector) // 2:
            return PredictionResult(
                status=InferenceStatus.INPUT_INVALID,
                label=None,
                confidence=None,
                probs=None,
                source=self.model_source,
                model_id=self.model_id,
                error=f"Input feature vector corrupted: {nan_count}/{len(feature_vector)} values are NaN/Inf",
                ts=time.time(),
            )

        try:
            X = np.array([clean_vec], dtype=np.float32)
            probabilities = self.model.predict_proba(X)[0]
            probs_dict = {
                cls_name: float(prob)
                for cls_name, prob in zip(self.class_names, probabilities)
            }
            best_label = max(probs_dict, key=probs_dict.get)
            confidence = probs_dict[best_label]

            # Reject unknown or corrupted label outputs
            if best_label not in LABEL_NAMES:
                return PredictionResult(
                    status=InferenceStatus.MODEL_ERROR,
                    label=None,
                    confidence=None,
                    probs=None,
                    source=self.model_source,
                    model_id=self.model_id,
                    error=f"Prediction returned unknown label '{best_label}'. Must be one of {LABEL_NAMES}",
                    ts=time.time(),
                )

            return PredictionResult(
                status=InferenceStatus.VALID_MODEL,
                label=best_label,
                confidence=round(confidence, 4),
                probs={k: round(v, 4) for k, v in probs_dict.items()},
                source=self.model_source,
                model_id=self.model_id,
                error=None,
                ts=time.time(),
            )
        except Exception as e:
            logger.error("Inference execution exception: %s", e)
            return PredictionResult(
                status=InferenceStatus.MODEL_ERROR,
                label=None,
                confidence=None,
                probs=None,
                source=self.model_source,
                model_id=self.model_id,
                error=f"Inference execution failed: {e}",
                ts=time.time(),
            )

    def _get_critical_gas_keys(self, sensors: Dict[str, Any]) -> Tuple[str, ...]:
        """Identify critical gas sensor keys required for valid inference."""
        if self.is_loaded and hasattr(self, "feature_names") and self.feature_names:
            if "raw_mq3" in self.feature_names:
                return ("mq3", "mq4", "mq135")
            elif "raw_mq2" in self.feature_names:
                return ("mq135", "mq2")
        # In fallback or unloaded mode, infer from provided sensors
        if "mq3" in sensors or "mq4" in sensors:
            return ("mq3", "mq4", "mq135")
        if "mq2" in sensors and "mq3" not in sensors:
            return ("mq135", "mq2")
        return ("mq3", "mq4", "mq135")

    def predict_history(self, history: Sequence[Dict[str, Any]]) -> PredictionResult:
        """Extract features from recent history and execute model inference."""
        if not self.is_loaded:
            return PredictionResult(
                status=self.status,
                label=None,
                confidence=None,
                probs=None,
                source=self.model_source,
                model_id=self.model_id,
                error=self.last_error or "Model is not loaded",
                ts=time.time(),
            )

        if not history:
            return PredictionResult(
                status=InferenceStatus.INPUT_INVALID,
                label=None,
                confidence=None,
                probs=None,
                source=self.model_source,
                model_id=self.model_id,
                error="History buffer is empty",
                ts=time.time(),
            )

        latest_sensors = history[-1].get("sensors", {}) or {}
        critical_keys = self._get_critical_gas_keys(latest_sensors)
        for k in critical_keys:
            val = latest_sensors.get(k)
            if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return PredictionResult(
                    status=InferenceStatus.SENSOR_FAULT,
                    label=None,
                    confidence=None,
                    probs=None,
                    source=self.model_source,
                    model_id=self.model_id,
                    error=f"Critical gas sensor '{k}' is missing, null, faulty, or non-finite ({val})",
                    ts=time.time(),
                )

        feat_vec = self._extractor.extract_from_history(history)
        return self.predict_vector(feat_vec)

    def fallback_predict(
        self,
        sensors: Dict[str, Any],
        deltas: Dict[str, float],
        thresholds: Dict[str, float],
        baseline: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> PredictionResult:
        """Deterministic heuristic rules-based fallback for provisional testing.
        
        SAFETY NOTE:
        This prediction explicitly returns InferenceStatus.RULES_FALLBACK.
        Confidence is NOT calibrated and MUST NOT independently authorize fan actuation
        or lower state machine trigger thresholds unless explicitly allowed by config.
        """
        # Strictly verify that all critical gas channels are present and finite
        critical_keys = self._get_critical_gas_keys(sensors)
        for k in critical_keys:
            val = sensors.get(k)
            if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return PredictionResult(
                    status=InferenceStatus.SENSOR_FAULT,
                    label=None,
                    confidence=None,
                    probs=None,
                    source="rules-placeholder",
                    model_id=None,
                    error=f"Critical gas sensor '{k}' is missing, null, faulty, or non-finite ({val})",
                    ts=time.time(),
                )

        # Gas delta normalization
        r = {
            k: max(0.0, deltas.get(k, 0.0) / thresholds[k]) if thresholds.get(k) else 0.0
            for k in thresholds
        }
        gas_smoke = max(r.get("mq4", 0.0), r.get("mq2", 0.0))
        gas_voc = max(r.get("mq3", 0.0), r.get("mq135", 0.0))

        steam = max(0.0, r.get("humidity", 0.0) - 0.5 * max(gas_smoke, r.get("temp_c", 0.0)))
        cook = (r.get("temp_c", 0.0) + gas_smoke) / 2.0
        foul = max(0.0, gas_voc - 0.5 * max(r.get("humidity", 0.0), r.get("temp_c", 0.0), gas_smoke))
        s = {
            "clean": max(0.2, 1.0 - max(steam, cook, foul)),
            "steam": steam,
            "cooking": cook,
            "foul": foul,
        }
        tot = sum(s.values()) or 1.0
        probs = {k: round(v / tot, 4) for k, v in s.items()}
        lab = max(probs, key=probs.get)

        return PredictionResult(
            status=InferenceStatus.RULES_FALLBACK,
            label=lab,
            confidence=None,  # NEVER fabricate calibrated confidence for rules fallback!
            probs=probs,
            source="rules-placeholder",
            model_id=None,
            error="Provisional rules-based fallback active. Model is not loaded.",
            ts=time.time(),
        )
