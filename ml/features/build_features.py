"""Deterministic, leakage-free feature engineering pipeline for gas sensor time-series."""
import math
from typing import Dict, List, Sequence, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from common.logger import get_logger
from ml.config import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_COUNT,
    LEGACY_FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_COUNT,
)

logger = get_logger("ml.features.build_features")


class FeatureExtractor:
    """Extracts ordered, deterministic numerical features from live history or batch data.
    
    Guarantees:
    - Strictly past/current temporal window (t <= t_now); zero future information.
    - No cross-session data leakage when processing batches.
    - Deterministic output matching the exact feature contract.
    - Transparent warm-up and missing value handling.
    """

    def __init__(self, mode: str = "canonical", window_size: int = 10, rolling_window: Optional[int] = None, **kwargs):
        """Initialize extractor.
        
        Args:
            mode: 'canonical' (MQ-3, MQ-4, MQ-135, 32 features) or
                  'legacy' (MQ-135, MQ-2, 26 features).
            window_size: Number of past seconds/ticks in rolling window.
            rolling_window: Alias for window_size.
        """
        self.mode = mode.lower()
        effective_window = rolling_window if rolling_window is not None else window_size
        self.window_size = effective_window
        self.rolling_window = effective_window

        if self.mode == "canonical":
            self.feature_names = list(CANONICAL_FEATURE_NAMES)
            self.feature_count = CANONICAL_FEATURE_COUNT
        elif self.mode == "legacy":
            self.feature_names = list(LEGACY_FEATURE_NAMES)
            self.feature_count = LEGACY_FEATURE_COUNT
        else:
            raise ValueError(f"Unknown FeatureExtractor mode: '{mode}'. Must be 'canonical' or 'legacy'.")

    def extract_from_history(self, history: Sequence[Dict[str, Any]]) -> List[float]:
        """Extract a 1D feature vector from recent operational history.
        
        Args:
            history: Sequence of system row dicts in chronological order.
                     The most recent observation is history[-1].
                     
        Returns:
            List of floats matching the exact order and length of self.feature_names.
        """
        if not history:
            return [0.0] * self.feature_count

        latest = history[-1]
        sensors = latest.get("sensors", {}) or {}
        baseline = latest.get("baseline", {}) or {}
        deltas = latest.get("deltas", {}) or {}

        # Slicing window up to current observation only (t <= t_now)
        window = list(history)[-self.window_size:]

        if self.mode == "canonical":
            return self._extract_canonical(latest, sensors, baseline, deltas, window)
        else:
            return self._extract_legacy(latest, sensors, baseline, deltas, window)

    def _extract_canonical(
        self,
        latest: Dict[str, Any],
        sensors: Dict[str, Any],
        baseline: Dict[str, Any],
        deltas: Dict[str, Any],
        window: List[Dict[str, Any]],
    ) -> List[float]:
        """Extract canonical 29-feature vector for MQ-3, MQ-4, MQ-135 array."""
        # 1. Raw measurements (default to neutral baselines if sensor read failed)
        def _get_val(k: str, default: float) -> float:
            v = sensors.get(k)
            if v is None or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                return default
            return float(v)

        raw_mq3 = _get_val("mq3", 0.80)
        raw_mq4 = _get_val("mq4", 1.00)
        raw_mq135 = _get_val("mq135", 1.50)
        raw_humidity = _get_val("humidity", 50.0)
        raw_temp_c = _get_val("temp_c", 25.0)

        # Explicit availability indicators: strictly distinguish genuine readings from missing sensors
        has_humidity = 1.0 if (sensors.get("humidity") is not None and isinstance(sensors["humidity"], (int, float)) and not (math.isnan(sensors["humidity"]) or math.isinf(sensors["humidity"]))) else 0.0
        has_temp_c = 1.0 if (sensors.get("temp_c") is not None and isinstance(sensors["temp_c"], (int, float)) and not (math.isnan(sensors["temp_c"]) or math.isinf(sensors["temp_c"]))) else 0.0
        v_pir = sensors.get("pir")
        has_pir = 1.0 if (v_pir is not None and not (isinstance(v_pir, float) and (math.isnan(v_pir) or math.isinf(v_pir)))) else 0.0
        raw_pir = 1.0 if (has_pir == 1.0 and v_pir) else 0.0

        # 2. Baseline deltas (x - x_baseline)
        def _get_delta(k: str, raw_val: float) -> float:
            if k in deltas and deltas[k] is not None:
                d = deltas[k]
                if isinstance(d, (int, float)) and not (math.isnan(d) or math.isinf(d)):
                    return float(d)
            base_val = baseline.get(k, raw_val)
            if base_val is None or not isinstance(base_val, (int, float)) or math.isnan(base_val) or math.isinf(base_val):
                base_val = raw_val
            return float(raw_val - base_val)

        d_mq3 = _get_delta("mq3", raw_mq3)
        d_mq4 = _get_delta("mq4", raw_mq4)
        d_mq135 = _get_delta("mq135", raw_mq135)
        d_humidity = _get_delta("humidity", raw_humidity)
        d_temp = _get_delta("temp_c", raw_temp_c)

        # 3. Rolling window collections (strictly past observations in window)
        win_mq3: List[float] = []
        win_mq4: List[float] = []
        win_mq135: List[float] = []
        win_hum: List[float] = []
        win_temp: List[float] = []
        win_ts: List[float] = []

        for row in window:
            s = row.get("sensors", {}) or {}
            if s.get("mq3") is not None and isinstance(s["mq3"], (int, float)):
                win_mq3.append(float(s["mq3"]))
            if s.get("mq4") is not None and isinstance(s["mq4"], (int, float)):
                win_mq4.append(float(s["mq4"]))
            if s.get("mq135") is not None and isinstance(s["mq135"], (int, float)):
                win_mq135.append(float(s["mq135"]))
            if s.get("humidity") is not None and isinstance(s["humidity"], (int, float)):
                win_hum.append(float(s["humidity"]))
            if s.get("temp_c") is not None and isinstance(s["temp_c"], (int, float)):
                win_temp.append(float(s["temp_c"]))
            if "ts" in row and row["ts"] is not None and isinstance(row["ts"], (int, float)):
                win_ts.append(float(row["ts"]))

        def _stats(arr: List[float], fallback: float) -> Tuple[float, float]:
            if not arr:
                return fallback, 0.0
            mean = sum(arr) / len(arr)
            var = sum((x - mean) ** 2 for x in arr) / len(arr) if len(arr) > 1 else 0.0
            return mean, math.sqrt(var)

        def _slope(arr: List[float], ts: List[float]) -> float:
            if len(arr) < 2 or len(ts) < 2:
                return 0.0
            dt = ts[-1] - ts[0]
            if dt <= 0:
                dt = float(len(arr) - 1)
            return (arr[-1] - arr[0]) / (dt or 1.0)

        mean_mq3, std_mq3 = _stats(win_mq3, raw_mq3)
        mean_mq4, std_mq4 = _stats(win_mq4, raw_mq4)
        mean_mq135, std_mq135 = _stats(win_mq135, raw_mq135)
        mean_hum, std_hum = _stats(win_hum, raw_humidity)
        mean_temp, std_temp = _stats(win_temp, raw_temp_c)

        slope_mq3 = _slope(win_mq3, win_ts)
        slope_mq4 = _slope(win_mq4, win_ts)
        slope_mq135 = _slope(win_mq135, win_ts)
        slope_hum = _slope(win_hum, win_ts)
        slope_temp = _slope(win_temp, win_ts)

        # 4. Domain interaction ratios
        ratio_mq4_mq135 = raw_mq4 / (raw_mq135 + 1e-4)
        ratio_mq3_mq135 = raw_mq3 / (raw_mq135 + 1e-4)
        ratio_humidity_temp = raw_humidity / (raw_temp_c + 1e-4)

        f_map = {
            "raw_mq3": raw_mq3,
            "raw_mq4": raw_mq4,
            "raw_mq135": raw_mq135,
            "raw_humidity": raw_humidity,
            "raw_temp_c": raw_temp_c,
            "raw_pir": raw_pir,
            "delta_mq3": d_mq3,
            "delta_mq4": d_mq4,
            "delta_mq135": d_mq135,
            "delta_humidity": d_humidity,
            "delta_temp_c": d_temp,
            "mean_mq3": mean_mq3,
            "mean_mq4": mean_mq4,
            "mean_mq135": mean_mq135,
            "mean_humidity": mean_hum,
            "mean_temp_c": mean_temp,
            "std_mq3": std_mq3,
            "std_mq4": std_mq4,
            "std_mq135": std_mq135,
            "std_humidity": std_hum,
            "std_temp_c": std_temp,
            "slope_mq3": slope_mq3,
            "slope_mq4": slope_mq4,
            "slope_mq135": slope_mq135,
            "slope_humidity": slope_hum,
            "slope_temp_c": slope_temp,
            "ratio_mq4_mq135": ratio_mq4_mq135,
            "ratio_mq3_mq135": ratio_mq3_mq135,
            "ratio_humidity_temp": ratio_humidity_temp,
            "has_humidity": has_humidity,
            "has_temp_c": has_temp_c,
            "has_pir": has_pir,
        }
        return [float(f_map[k]) for k in self.feature_names]

    def _extract_legacy(
        self,
        latest: Dict[str, Any],
        sensors: Dict[str, Any],
        baseline: Dict[str, Any],
        deltas: Dict[str, Any],
        window: List[Dict[str, Any]],
    ) -> List[float]:
        """Extract legacy 23-feature vector for MQ-135, MQ-2 prototype array."""
        def _get_val(k: str, default: float) -> float:
            v = sensors.get(k)
            if v is None or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                return default
            return float(v)

        raw_mq135 = _get_val("mq135", 1.50)
        raw_mq2 = _get_val("mq2", 1.00)
        raw_humidity = _get_val("humidity", 50.0)
        raw_temp_c = _get_val("temp_c", 25.0)

        # Explicit availability indicators: strictly distinguish genuine readings from missing sensors
        has_humidity = 1.0 if (sensors.get("humidity") is not None and isinstance(sensors["humidity"], (int, float)) and not (math.isnan(sensors["humidity"]) or math.isinf(sensors["humidity"]))) else 0.0
        has_temp_c = 1.0 if (sensors.get("temp_c") is not None and isinstance(sensors["temp_c"], (int, float)) and not (math.isnan(sensors["temp_c"]) or math.isinf(sensors["temp_c"]))) else 0.0
        v_pir = sensors.get("pir")
        has_pir = 1.0 if (v_pir is not None and not (isinstance(v_pir, float) and (math.isnan(v_pir) or math.isinf(v_pir)))) else 0.0
        raw_pir = 1.0 if (has_pir == 1.0 and v_pir) else 0.0

        def _get_delta(k: str, raw_val: float) -> float:
            if k in deltas and deltas[k] is not None:
                d = deltas[k]
                if isinstance(d, (int, float)) and not (math.isnan(d) or math.isinf(d)):
                    return float(d)
            base_val = baseline.get(k, raw_val)
            if base_val is None or not isinstance(base_val, (int, float)) or math.isnan(base_val) or math.isinf(base_val):
                base_val = raw_val
            return float(raw_val - base_val)

        d_mq135 = _get_delta("mq135", raw_mq135)
        d_mq2 = _get_delta("mq2", raw_mq2)
        d_humidity = _get_delta("humidity", raw_humidity)
        d_temp = _get_delta("temp_c", raw_temp_c)

        win_mq135, win_mq2, win_hum, win_temp, win_ts = [], [], [], [], []
        for row in window:
            s = row.get("sensors", {}) or {}
            if s.get("mq135") is not None: win_mq135.append(float(s["mq135"]))
            if s.get("mq2") is not None: win_mq2.append(float(s["mq2"]))
            if s.get("humidity") is not None: win_hum.append(float(s["humidity"]))
            if s.get("temp_c") is not None: win_temp.append(float(s["temp_c"]))
            if "ts" in row and row["ts"] is not None: win_ts.append(float(row["ts"]))

        def _stats(arr: List[float], fallback: float) -> Tuple[float, float]:
            if not arr: return fallback, 0.0
            mean = sum(arr) / len(arr)
            var = sum((x - mean) ** 2 for x in arr) / len(arr) if len(arr) > 1 else 0.0
            return mean, math.sqrt(var)

        def _slope(arr: List[float], ts: List[float]) -> float:
            if len(arr) < 2 or len(ts) < 2: return 0.0
            dt = ts[-1] - ts[0]
            if dt <= 0: dt = float(len(arr) - 1)
            return (arr[-1] - arr[0]) / (dt or 1.0)

        mean_mq135, std_mq135 = _stats(win_mq135, raw_mq135)
        mean_mq2, std_mq2 = _stats(win_mq2, raw_mq2)
        mean_hum, std_hum = _stats(win_hum, raw_humidity)
        mean_temp, std_temp = _stats(win_temp, raw_temp_c)

        slope_mq135 = _slope(win_mq135, win_ts)
        slope_mq2 = _slope(win_mq2, win_ts)
        slope_hum = _slope(win_hum, win_ts)
        slope_temp = _slope(win_temp, win_ts)

        ratio_mq2_mq135 = raw_mq2 / (raw_mq135 + 1e-4)
        ratio_humidity_temp = raw_humidity / (raw_temp_c + 1e-4)

        f_map = {
            "raw_mq135": raw_mq135,
            "raw_mq2": raw_mq2,
            "raw_humidity": raw_humidity,
            "raw_temp_c": raw_temp_c,
            "raw_pir": raw_pir,
            "delta_mq135": d_mq135,
            "delta_mq2": d_mq2,
            "delta_humidity": d_humidity,
            "delta_temp_c": d_temp,
            "mean_mq135": mean_mq135,
            "mean_mq2": mean_mq2,
            "mean_humidity": mean_hum,
            "mean_temp_c": mean_temp,
            "std_mq135": std_mq135,
            "std_mq2": std_mq2,
            "std_humidity": std_hum,
            "std_temp_c": std_temp,
            "slope_mq135": slope_mq135,
            "slope_mq2": slope_mq2,
            "slope_humidity": slope_hum,
            "slope_temp_c": slope_temp,
            "ratio_mq2_mq135": ratio_mq2_mq135,
            "ratio_humidity_temp": ratio_humidity_temp,
            "has_humidity": has_humidity,
            "has_temp_c": has_temp_c,
            "has_pir": has_pir,
        }
        return [float(f_map[k]) for k in self.feature_names]

    def extract_from_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a tabular session dataset DataFrame.
        
        Performs rolling computations grouped by 'run_id' (or contiguous sessions)
        to strictly prevent temporal data leakage across independent recording sessions.
        """
        if df.empty:
            return np.empty((0, self.feature_count), dtype=np.float32), list(self.feature_names)

        # Automatically detect schema if mode matches
        if "mq3" in df.columns and "mq4" in df.columns and self.mode != "canonical":
            logger.info("Auto-switching FeatureExtractor to 'canonical' mode based on dataset columns.")
            self.__init__(mode="canonical", window_size=self.window_size)
        elif "mq2" in df.columns and "mq3" not in df.columns and self.mode != "legacy":
            logger.info("Auto-switching FeatureExtractor to 'legacy' mode based on dataset columns.")
            self.__init__(mode="legacy", window_size=self.window_size)

        features_list = []
        has_run_id = "run_id" in df.columns

        grouped = df.groupby("run_id", sort=False) if has_run_id else [(None, df)]

        for _, session_df in grouped:
            if session_df.empty:
                continue
            rows_data: List[Dict[str, Any]] = []
            # Calculate stable session baseline from first valid row in session
            first_row = session_df.iloc[0]
            session_baseline: Dict[str, float] = {}
            for col in session_df.columns:
                val = first_row.get(col)
                if pd.notna(val) and isinstance(val, (int, float, np.number)):
                    session_baseline[col] = float(val)

            for _, r in session_df.iterrows():
                sensors_dict = {
                    "humidity": float(r["humidity"]) if "humidity" in r and pd.notna(r["humidity"]) else None,
                    "temp_c": float(r["temp_c"]) if "temp_c" in r and pd.notna(r["temp_c"]) else None,
                    "pir": int(r["pir"]) if "pir" in r and pd.notna(r["pir"]) else None,
                }
                if self.mode == "canonical":
                    sensors_dict["mq3"] = float(r["mq3"]) if "mq3" in r and pd.notna(r["mq3"]) else None
                    sensors_dict["mq4"] = float(r["mq4"]) if "mq4" in r and pd.notna(r["mq4"]) else None
                    sensors_dict["mq135"] = float(r["mq135"]) if "mq135" in r and pd.notna(r["mq135"]) else None
                else:
                    sensors_dict["mq135"] = float(r["mq135"]) if "mq135" in r and pd.notna(r["mq135"]) else None
                    sensors_dict["mq2"] = float(r["mq2"]) if "mq2" in r and pd.notna(r["mq2"]) else None

                row_dict = {
                    "ts": float(r["ts"]) if "ts" in r and pd.notna(r["ts"]) else 0.0,
                    "sensors": sensors_dict,
                    "baseline": session_baseline,
                    "deltas": {},
                }
                rows_data.append(row_dict)
                feat_vec = self.extract_from_history(rows_data)
                features_list.append(feat_vec)

        X = np.array(features_list, dtype=np.float32)
        return X, list(self.feature_names)

    def extract_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract DataFrame of features for batch training."""
        if df.empty:
            return pd.DataFrame(columns=self.feature_names)
        X, cols = self.extract_from_dataframe(df)
        return pd.DataFrame(X, columns=cols)

    def extract_inference_features(self, df_or_history: Union[pd.DataFrame, Sequence[Dict[str, Any]]]) -> pd.DataFrame:
        """Extract DataFrame containing feature vector for the most recent observation."""
        if isinstance(df_or_history, pd.DataFrame):
            if df_or_history.empty:
                return pd.DataFrame(columns=self.feature_names)
            X, cols = self.extract_from_dataframe(df_or_history)
            return pd.DataFrame([X[-1]], columns=cols)
        elif isinstance(df_or_history, (list, tuple)):
            if not df_or_history:
                return pd.DataFrame([[0.0] * self.feature_count], columns=self.feature_names)
            vec = self.extract_from_history(df_or_history)
            return pd.DataFrame([vec], columns=self.feature_names)
        return pd.DataFrame(columns=self.feature_names)


# Module-level instances
_default_canonical_extractor = FeatureExtractor(mode="canonical")
_default_legacy_extractor = FeatureExtractor(mode="legacy")


def build_features(history: Sequence[Dict[str, Any]], mode: str = "canonical") -> List[float]:
    """Compatibility entry point.
    
    Given a history sequence, extracts and returns the deterministic feature vector.
    """
    if mode == "canonical":
        # Check if latest history entry only has mq2 (legacy)
        if history and "mq2" in (history[-1].get("sensors") or {}) and "mq3" not in (history[-1].get("sensors") or {}):
            return _default_legacy_extractor.extract_from_history(history)
        return _default_canonical_extractor.extract_from_history(history)
    else:
        return _default_legacy_extractor.extract_from_history(history)
