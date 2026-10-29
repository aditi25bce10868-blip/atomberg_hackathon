"""Classifier runtime wrapper bridging control orchestration with ml.inference."""
import pathlib
from typing import Dict, Any, Sequence, Optional
from common import config
from common.logger import get_logger
from common.schema import LABEL_NAMES, InferenceStatus
from ml.inference.predictor import AirQualityPredictor

logger = get_logger("control.ml_runtime")


class Classifier:
    """Wraps model inference for the live control loop with fail-safe fallback."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.names = list(LABEL_NAMES)
        ml_cfg = cfg.get("ml", {})
        model_path = ml_cfg.get("model_path")
        self.allow_fallback = ml_cfg.get("allow_rules_fallback", True)

        if model_path:
            p = pathlib.Path(model_path)
            mp = p if p.is_absolute() else config.path(cfg, str(model_path))
            self.predictor = AirQualityPredictor(mp)
        else:
            self.predictor = AirQualityPredictor()

        self.source = self.predictor.model_source
        self.status = self.predictor.status

    def predict(
        self,
        sensors: Dict[str, Any],
        deltas: Dict[str, float],
        thr: Dict[str, float],
        history: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform classification on recent readings with explicit status reporting."""
        # 1. If genuine model bundle is loaded, execute model inference
        if self.predictor.is_loaded:
            res = self.predictor.predict_history(history)
            if res.status == InferenceStatus.VALID_MODEL:
                return res.to_dict()
            # Do NOT conceal hardware sensor faults or invalid inputs behind rules fallback!
            if res.status in (InferenceStatus.SENSOR_FAULT, InferenceStatus.INPUT_INVALID):
                return res.to_dict()
            logger.warning("Model prediction returned non-valid status '%s': %s", res.status, res.error)
            if not self.allow_fallback:
                return res.to_dict()

        # 2. Rules-based provisional fallback if permitted
        if self.allow_fallback:
            res = self.predictor.fallback_predict(sensors, deltas, thr)
            return res.to_dict()

        # 3. Model unavailable / fallback disabled state
        return {
            "status": self.predictor.status,
            "label": None,
            "confidence": None,
            "probs": None,
            "source": self.predictor.model_source,
            "model_id": self.predictor.model_id,
            "error": self.predictor.last_error or "Model unavailable and fallback disabled",
            "ts": 0.0,
        }
