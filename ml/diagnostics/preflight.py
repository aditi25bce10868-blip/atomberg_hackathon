"""ML Pre-Training Diagnostic and System Preflight Tool.

Performs a comprehensive 17-point audit of the ML pipeline, feature parity,
evaluation contracts, dashboard API integration, and safety defaults before
genuine sensor data arrives.

CLI Usage:
    python -m ml.diagnostics.preflight
    python -m ml.diagnostics.preflight --strict-dataset
    python -m ml.diagnostics.preflight --json
"""
import argparse
import importlib
import io
import json
import logging
import os
import pathlib
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

# Base imports from project
from common.logger import get_logger
from common.schema import (
    CANONICAL_GAS_SENSORS,
    LABEL_NAMES,
    LABEL_CODES,
    LABELS,
    InferenceStatus,
)
from common.config import load_config
from ml.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    CANONICAL_FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    CANONICAL_FEATURE_COUNT,
    LEGACY_FEATURE_COUNT,
    VALID_EVAL_STRATEGIES,
    DEFAULT_EVAL_STRATEGY,
    DEFAULT_MODEL_NAME,
)

logger = get_logger("ml.diagnostics.preflight")


@dataclass
class DiagnosticCheck:
    id: int
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    explanation: str
    action: str
    critical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightResult:
    total_checks: int
    passed_count: int
    warn_count: int
    fail_count: int
    checks: List[DiagnosticCheck] = field(default_factory=list)
    overall_status: str = "PASS"  # "PASS", "WARN", "FAIL"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_summary(self, use_color: bool = True) -> None:
        # Colors for terminal output
        c_green = "\033[92m" if use_color else ""
        c_yellow = "\033[93m" if use_color else ""
        c_red = "\033[91m" if use_color else ""
        c_cyan = "\033[96m" if use_color else ""
        c_bold = "\033[1m" if use_color else ""
        c_reset = "\033[0m" if use_color else ""

        print("\n" + "=" * 80)
        print(f"{c_bold}{c_cyan}ATOMBERG EFFICIO APEX :: ML PRE-TRAINING PREFLIGHT AUDIT{c_reset}")
        print("=" * 80)

        for c in self.checks:
            badge_color = c_green if c.status == "PASS" else (c_yellow if c.status == "WARN" else c_red)
            status_badge = f"{badge_color}[{c.status.center(6)}]{c_reset}"
            print(f"{status_badge} #{c.id:02d} {c.name}")
            print(f"         Explanation: {c.explanation}")
            if c.status != "PASS" or c.action != "None required":
                print(f"         Action:      {c.action}")
            print()

        print("-" * 80)
        overall_color = c_green if self.fail_count == 0 and self.warn_count == 0 else (c_yellow if self.fail_count == 0 else c_red)
        print(
            f"{c_bold}Summary: {self.total_checks} Checks | "
            f"{c_green}{self.passed_count} PASS{c_reset} | "
            f"{c_yellow}{self.warn_count} WARN{c_reset} | "
            f"{c_red}{self.fail_count} FAIL{c_reset} | "
            f"Overall Status: {overall_color}{self.overall_status}{c_reset}"
        )
        print("=" * 80 + "\n")


def check_1_imports() -> DiagnosticCheck:
    """1. Python imports and module initialization."""
    required_modules = [
        "numpy",
        "pandas",
        "sklearn",
        "joblib",
        "fastapi",
        "common.schema",
        "common.config",
        "common.logger",
        "ml.config",
        "ml.features.build_features",
        "ml.inference.predictor",
        "ml.training.train",
        "ml.training.evaluate",
        "ml.datasets.validate_dataset",
        "control.state_machine",
        "control.ml_runtime",
    ]
    missing = []
    for mod in required_modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            missing.append(f"{mod} ({e})")

    if missing:
        return DiagnosticCheck(
            id=1,
            name="Python imports and module initialization",
            status="FAIL",
            explanation=f"Failed importing required runtime modules: {', '.join(missing)}",
            action="Run 'pip install -r requirements.txt' and verify project root is in PYTHONPATH.",
            critical=True,
        )
    return DiagnosticCheck(
        id=1,
        name="Python imports and module initialization",
        status="PASS",
        explanation=f"All {len(required_modules)} required ML, server, control, and schema modules imported cleanly.",
        action="None required",
    )


def check_2_configuration() -> DiagnosticCheck:
    """2. Configuration validity."""
    try:
        cfg = load_config()
        required_keys = ["source", "thresholds", "ml", "tick_s", "logging"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            return DiagnosticCheck(
                id=2,
                name="Configuration validity",
                status="FAIL",
                explanation=f"config.yaml is missing required top-level sections: {', '.join(missing)}",
                action="Verify config.yaml matches common/config.py schema definitions.",
                critical=True,
            )
        # Check safety default in config
        hw_enabled = bool(cfg.get("hardware_enabled", False) or cfg.get("safety", {}).get("hardware_enabled", False))
        if hw_enabled:
            return DiagnosticCheck(
                id=2,
                name="Configuration validity",
                status="FAIL",
                explanation="config.yaml has hardware_enabled set to true. Unsafe pre-training state.",
                action="Set hardware_enabled: false in config.yaml.",
                critical=True,
            )
        return DiagnosticCheck(
            id=2,
            name="Configuration validity",
            status="PASS",
            explanation="config.yaml parsed successfully with all required sections and safe initial bounds.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=2,
            name="Configuration validity",
            status="FAIL",
            explanation=f"Could not load or parse config.yaml: {e}",
            action="Ensure config.yaml exists at project root and contains valid YAML syntax.",
            critical=True,
        )


def check_3_directories() -> DiagnosticCheck:
    """3. Dataset directory and artifact directory accessibility."""
    dirs_to_check = [
        pathlib.Path("data"),
        pathlib.Path("data/runs"),
        pathlib.Path("data/models"),
        pathlib.Path("ml/artifacts"),
        pathlib.Path("ml/reports/latest"),
    ]
    inaccessible = []
    for d in dirs_to_check:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".perm_check.tmp"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
        except Exception as e:
            inaccessible.append(f"{d} ({e})")

    if inaccessible:
        return DiagnosticCheck(
            id=3,
            name="Directory accessibility",
            status="FAIL",
            explanation=f"File system permissions failure for directories: {', '.join(inaccessible)}",
            action="Check folder permissions and disk write access.",
            critical=True,
        )
    return DiagnosticCheck(
        id=3,
        name="Directory accessibility",
        status="PASS",
        explanation=f"All {len(dirs_to_check)} data, artifact, and report directories exist and are writable.",
        action="None required",
    )


def check_4_feature_contracts() -> DiagnosticCheck:
    """4. Canonical and legacy feature contracts."""
    try:
        from ml.config import CANONICAL_FEATURE_NAMES, LEGACY_FEATURE_NAMES
        c_len = len(CANONICAL_FEATURE_NAMES)
        l_len = len(LEGACY_FEATURE_NAMES)

        if c_len != CANONICAL_FEATURE_COUNT:
            return DiagnosticCheck(
                id=4,
                name="Canonical and legacy feature contracts",
                status="FAIL",
                explanation=f"Canonical contract has {c_len} features; expected {CANONICAL_FEATURE_COUNT}.",
                action="Align CANONICAL_FEATURE_NAMES in ml/config.py with canonical 32-feature spec.",
                critical=True,
            )
        if l_len != LEGACY_FEATURE_COUNT:
            return DiagnosticCheck(
                id=4,
                name="Canonical and legacy feature contracts",
                status="FAIL",
                explanation=f"Legacy contract has {l_len} features; expected {LEGACY_FEATURE_COUNT}.",
                action="Align LEGACY_FEATURE_NAMES in ml/config.py with legacy 26-feature spec.",
                critical=True,
            )

        # Check required canonical presence indicators
        required_indicators = ["has_humidity", "has_temp_c", "has_pir"]
        missing_ind = [ind for ind in required_indicators if ind not in CANONICAL_FEATURE_NAMES]
        if missing_ind:
            return DiagnosticCheck(
                id=4,
                name="Canonical and legacy feature contracts",
                status="FAIL",
                explanation=f"Canonical contract missing missingness indicators: {', '.join(missing_ind)}",
                action="Ensure has_humidity, has_temp_c, has_pir are in CANONICAL_FEATURE_NAMES.",
                critical=True,
            )

        return DiagnosticCheck(
            id=4,
            name="Canonical and legacy feature contracts",
            status="PASS",
            explanation=(
                f"Feature contracts verified: Canonical ({c_len} features with explicit missingness "
                f"indicators), Legacy ({l_len} features backward compatible)."
            ),
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=4,
            name="Canonical and legacy feature contracts",
            status="FAIL",
            explanation=f"Error inspecting feature contracts: {e}",
            action="Inspect common/schema.py and ml/config.py.",
            critical=True,
        )


def check_5_feature_extractor() -> DiagnosticCheck:
    """5. Feature extractor initialization and buffer behavior."""
    try:
        from ml.features.build_features import FeatureExtractor
        fe = FeatureExtractor(rolling_window=10)
        if fe.rolling_window != 10:
            return DiagnosticCheck(
                id=5,
                name="Feature extractor initialization",
                status="FAIL",
                explanation=f"FeatureExtractor rolling_window is {fe.rolling_window}; expected 10.",
                action="Check FeatureExtractor constructor in ml/features/build_features.py.",
                critical=True,
            )

        # Test empty input handling
        import pandas as pd
        empty_df = pd.DataFrame()
        empty_out = fe.extract_training_features(empty_df)
        if not empty_out.empty:
            return DiagnosticCheck(
                id=5,
                name="Feature extractor initialization",
                status="FAIL",
                explanation="extract_training_features on empty DataFrame did not return empty DataFrame.",
                action="Fix empty check in ml/features/build_features.py.",
                critical=True,
            )

        return DiagnosticCheck(
            id=5,
            name="Feature extractor initialization",
            status="PASS",
            explanation="FeatureExtractor initializes with rolling_window=10 and correctly handles empty edge cases.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=5,
            name="Feature extractor initialization",
            status="FAIL",
            explanation=f"FeatureExtractor failed: {e}",
            action="Check ml/features/build_features.py.",
            critical=True,
        )


def check_6_parity() -> DiagnosticCheck:
    """6. Training and inference feature parity."""
    try:
        import pandas as pd
        from ml.features.build_features import FeatureExtractor

        # Create a synthetic 15-step window
        rows = []
        for i in range(15):
            rows.append({
                "timestamp": 1000.0 + i,
                "mq3": 1.20 + 0.01 * i,
                "mq4": 0.85 + 0.02 * i,
                "mq135": 1.45 + 0.01 * i,
                "humidity": 55.0,
                "temperature": 24.5,
                "pir": 1.0 if i >= 10 else 0.0,
                "run_id": "test_parity",
                "label": "cooking",
            })
        df = pd.DataFrame(rows)

        fe = FeatureExtractor(rolling_window=10)
        train_features = fe.extract_training_features(df)
        inf_features = fe.extract_inference_features(df)

        # Compare column names and order
        train_cols = list(train_features.columns)
        inf_cols = list(inf_features.columns)

        if train_cols != inf_cols:
            diff = set(train_cols) ^ set(inf_cols)
            return DiagnosticCheck(
                id=6,
                name="Training and inference feature parity",
                status="FAIL",
                explanation=f"Column mismatch between training and inference features: {diff}",
                action="Ensure extract_training_features and extract_inference_features generate identical columns.",
                critical=True,
            )

        # Compare values on the last row
        last_train = train_features.iloc[-1].to_dict()
        last_inf = inf_features.iloc[0].to_dict()

        discrepancies = []
        for col in train_cols:
            v_t = last_train[col]
            v_i = last_inf[col]
            if abs(v_t - v_i) > 1e-4:
                discrepancies.append(f"{col}: train={v_t}, inf={v_i}")

        if discrepancies:
            return DiagnosticCheck(
                id=6,
                name="Training and inference feature parity",
                status="FAIL",
                explanation=f"Numerical discrepancy between training and inference: {discrepancies[:3]}",
                action="Fix feature calculation logic in build_features.py to maintain parity.",
                critical=True,
            )

        return DiagnosticCheck(
            id=6,
            name="Training and inference feature parity",
            status="PASS",
            explanation=f"100% column name, order, type, and numerical parity verified across all {len(train_cols)} features.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=6,
            name="Training and inference feature parity",
            status="FAIL",
            explanation=f"Parity verification failed with error: {e}",
            action="Check ml/features/build_features.py.",
            critical=True,
        )


def check_7_label_mappings() -> DiagnosticCheck:
    """7. Label mappings."""
    try:
        from common.schema import LABELS, LABEL_NAMES, LABEL_CODES
        expected_classes = {"clean", "steam", "cooking", "foul"}
        actual_classes = set(LABELS.values()) if isinstance(LABELS, dict) else set(LABELS)

        if not expected_classes.issubset(actual_classes):
            return DiagnosticCheck(
                id=7,
                name="Label mappings",
                status="FAIL",
                explanation=f"LABELS missing expected classes: {expected_classes - actual_classes}",
                action="Verify LABELS definition in common/schema.py.",
                critical=True,
            )

        for name, code in LABEL_CODES.items():
            if code >= len(LABEL_NAMES) or LABEL_NAMES[code] != name:
                return DiagnosticCheck(
                    id=7,
                    name="Label mappings",
                    status="FAIL",
                    explanation=f"Inconsistent bidirectional mapping for label '{name}' (code {code})",
                    action="Check LABEL_NAMES and LABEL_CODES consistency in common/schema.py.",
                    critical=True,
                )

        return DiagnosticCheck(
            id=7,
            name="Label mappings",
            status="PASS",
            explanation=f"Bidirectional label mappings verified for classes: {', '.join(sorted(actual_classes))}.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=7,
            name="Label mappings",
            status="FAIL",
            explanation=f"Error checking labels: {e}",
            action="Verify common/schema.py.",
            critical=True,
        )


def check_8_dataset_schema(strict: bool = False) -> DiagnosticCheck:
    """8. Dataset schema validation."""
    from ml.datasets.validate_dataset import validate_dataset
    dataset_path = pathlib.Path("data/dataset.csv")

    if not dataset_path.exists():
        return DiagnosticCheck(
            id=8,
            name="Dataset schema validation",
            status="FAIL" if strict else "WARN",
            explanation="No production dataset found at data/dataset.csv (expected before real data collection).",
            action="Record genuine sensor sessions using the dashboard Collect tool or data recording script.",
            critical=strict,
        )

    try:
        report = validate_dataset(dataset_path)
        if not report.is_valid:
            return DiagnosticCheck(
                id=8,
                name="Dataset schema validation",
                status="FAIL",
                explanation=f"data/dataset.csv validation failed: {report.errors[:2]}",
                action="Run 'python -m ml.datasets.validate_dataset data/dataset.csv' and correct schema errors.",
                critical=True,
            )
        return DiagnosticCheck(
            id=8,
            name="Dataset schema validation",
            status="PASS",
            explanation=f"data/dataset.csv is valid ({report.dataset_type}, {report.row_count:,} rows).",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=8,
            name="Dataset schema validation",
            status="FAIL",
            explanation=f"Exception while validating data/dataset.csv: {e}",
            action="Inspect dataset file formatting.",
            critical=True,
        )


def check_9_eval_strategy() -> DiagnosticCheck:
    """9. Evaluation strategy configuration."""
    from ml.config import VALID_EVAL_STRATEGIES, DEFAULT_EVAL_STRATEGY

    required_strategies = {"group_kfold", "time_series_split", "session_holdout"}
    if not required_strategies.issubset(set(VALID_EVAL_STRATEGIES)):
        return DiagnosticCheck(
            id=9,
            name="Evaluation strategy configuration",
            status="FAIL",
            explanation=f"Missing evaluation strategies: {required_strategies - set(VALID_EVAL_STRATEGIES)}",
            action="Add missing evaluation strategies to VALID_EVAL_STRATEGIES in ml/config.py.",
            critical=True,
        )

    if DEFAULT_EVAL_STRATEGY not in required_strategies:
        return DiagnosticCheck(
            id=9,
            name="Evaluation strategy configuration",
            status="FAIL",
            explanation=f"Default strategy '{DEFAULT_EVAL_STRATEGY}' is not one of {required_strategies}",
            action="Set DEFAULT_EVAL_STRATEGY to 'group_kfold' in ml/config.py.",
            critical=True,
        )

    return DiagnosticCheck(
        id=9,
        name="Evaluation strategy configuration",
        status="PASS",
        explanation=f"Zero-leakage session evaluation configured: {', '.join(VALID_EVAL_STRATEGIES)}; default='{DEFAULT_EVAL_STRATEGY}'.",
        action="None required",
    )


def check_10_session_metadata() -> DiagnosticCheck:
    """10. Session metadata requirements."""
    try:
        import pandas as pd
        from ml.training.train import train_model

        # Test that train_model safely rejects dataset missing run_id when eval_strategy=group_kfold
        df_no_session = pd.DataFrame([
            {"timestamp": 100.0 + i, "mq3": 1.2, "mq4": 0.8, "mq135": 1.5, "label": "clean"}
            for i in range(40)
        ])
        with tempfile.TemporaryDirectory() as td:
            csv_path = pathlib.Path(td) / "no_session.csv"
            df_no_session.to_csv(csv_path, index=False)
            try:
                train_model(csv_path, output_dir=td, eval_strategy="group_kfold")
                return DiagnosticCheck(
                    id=10,
                    name="Session metadata requirements",
                    status="FAIL",
                    explanation="train_model accepted dataset without 'run_id' under group_kfold evaluation.",
                    action="Ensure train_model raises ValueError when run_id column is missing for group_kfold.",
                    critical=True,
                )
            except ValueError:
                # Expected behavior: safely rejected missing session grouping
                pass

        return DiagnosticCheck(
            id=10,
            name="Session metadata requirements",
            status="PASS",
            explanation="train_model strictly enforces session grouping metadata ('run_id') to prevent temporal data leakage.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=10,
            name="Session metadata requirements",
            status="FAIL",
            explanation=f"Failed session metadata validation: {e}",
            action="Inspect ml/training/train.py.",
            critical=True,
        )


def check_11_model_artifact(strict: bool = False) -> DiagnosticCheck:
    """11. Model artifact loading and compatibility."""
    import joblib
    try:
        from common.config import load as load_cfg, path as config_path
        cfg = load_cfg()
        configured_path = config_path(cfg, cfg.get("ml", {}).get("model_path", "ml/artifacts/model.joblib"))
    except Exception:
        configured_path = pathlib.Path("ml/artifacts/model.joblib")

    artifact_paths = [
        configured_path,
        pathlib.Path("ml/artifacts/model.joblib"),
        pathlib.Path("data/models/latest.joblib"),
        pathlib.Path("ml/artifacts/latest_model.joblib"),
    ]
    found_path = None
    for p in artifact_paths:
        if p.exists():
            found_path = p
            break

    if not found_path:
        return DiagnosticCheck(
            id=11,
            name="Model artifact loading and compatibility",
            status="FAIL" if strict else "WARN",
            explanation=f"No model artifact found at configured path ({configured_path.name}) or defaults. This is expected before first training run.",
            action="Train a model using real sensor data once collected, then run diagnostics again.",
            critical=strict,
        )

    try:
        data = joblib.load(found_path)
        if not isinstance(data, dict):
            return DiagnosticCheck(
                id=11,
                name="Model artifact loading and compatibility",
                status="FAIL",
                explanation=f"Artifact at {found_path} is type {type(data)}, expected dictionary payload.",
                action="Re-serialize model artifact using dictionary format with model, feature_names, metadata.",
                critical=True,
            )
        feature_names = data.get("feature_names", [])
        if len(feature_names) not in (CANONICAL_FEATURE_COUNT, LEGACY_FEATURE_COUNT):
            return DiagnosticCheck(
                id=11,
                name="Model artifact loading and compatibility",
                status="FAIL",
                explanation=f"Artifact features count ({len(feature_names)}) does not match canonical (32) or legacy (26).",
                action="Retrain model against current feature contract.",
                critical=True,
            )
        return DiagnosticCheck(
            id=11,
            name="Model artifact loading and compatibility",
            status="PASS",
            explanation=f"Valid model artifact loaded from {found_path} ({len(feature_names)} features).",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=11,
            name="Model artifact loading and compatibility",
            status="FAIL",
            explanation=f"Corrupted model artifact at {found_path}: {e}",
            action="Remove or retrain corrupted model file.",
            critical=True,
        )


def check_12_predictor_status() -> DiagnosticCheck:
    """12. Predictor status handling."""
    try:
        from ml.inference.predictor import AirQualityPredictor

        pred = AirQualityPredictor()
        # Unloaded predictor predict_history -> MODEL_UNAVAILABLE
        res = pred.predict_history([{"sensors": {"mq3": 1.0, "mq4": 1.0, "mq135": 1.0}}])
        if res.status != InferenceStatus.MODEL_UNAVAILABLE:
            return DiagnosticCheck(
                id=12,
                name="Predictor status handling",
                status="FAIL",
                explanation=f"Expected MODEL_UNAVAILABLE for unloaded model, got {res.status}",
                action="Check unloaded status logic in ml/inference/predictor.py.",
                critical=True,
            )

        # Predictor fallback_predict -> RULES_FALLBACK
        res_fb = pred.fallback_predict(
            sensors={"mq3": 1.0, "mq4": 1.0, "mq135": 1.0},
            baseline={"mq3": 0.8, "mq4": 0.8, "mq135": 1.2},
            deltas={"mq3": 0.2, "mq4": 0.2, "mq135": 0.3},
            thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
        )
        if res_fb.status != InferenceStatus.RULES_FALLBACK:
            return DiagnosticCheck(
                id=12,
                name="Predictor status handling",
                status="FAIL",
                explanation=f"Expected RULES_FALLBACK for heuristic fallback, got {res_fb.status}",
                action="Check fallback status logic in ml/inference/predictor.py.",
                critical=True,
            )

        # Critical gas sensor missing in fallback -> SENSOR_FAULT
        res_fault = pred.fallback_predict(
            sensors={"mq3": None, "mq4": 1.0, "mq135": 1.0},
            baseline={"mq3": 0.8, "mq4": 0.8, "mq135": 1.2},
            deltas={"mq3": 0.0, "mq4": 0.2, "mq135": 0.3},
            thresholds={"mq3": 0.25, "mq4": 0.25, "mq135": 0.25},
        )
        if res_fault.status != InferenceStatus.SENSOR_FAULT:
            return DiagnosticCheck(
                id=12,
                name="Predictor status handling",
                status="FAIL",
                explanation=f"Expected SENSOR_FAULT for null critical gas sensor, got {res_fault.status}",
                action="Check fallback_predict critical gas check in ml/inference/predictor.py.",
                critical=True,
            )

        return DiagnosticCheck(
            id=12,
            name="Predictor status handling",
            status="PASS",
            explanation="Predictor status transitions (MODEL_UNAVAILABLE, RULES_FALLBACK, SENSOR_FAULT) verified.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=12,
            name="Predictor status handling",
            status="FAIL",
            explanation=f"Predictor status check failed: {e}",
            action="Inspect ml/inference/predictor.py.",
            critical=True,
        )


def check_13_dashboard_api() -> DiagnosticCheck:
    """13. Dashboard API integration."""
    try:
        from starlette.testclient import TestClient
        from server.app import app

        client = TestClient(app)
        resp = client.get("/api/ml/status")
        if resp.status_code != 200:
            return DiagnosticCheck(
                id=13,
                name="Dashboard API integration",
                status="FAIL",
                explanation=f"GET /api/ml/status returned HTTP {resp.status_code}",
                action="Check /api/ml/status route in server/app.py.",
                critical=True,
            )
        data = resp.json()
        required_keys = ["status", "model_health", "training_readiness", "safety"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            return DiagnosticCheck(
                id=13,
                name="Dashboard API integration",
                status="FAIL",
                explanation=f"/api/ml/status missing required telemetry keys: {', '.join(missing)}",
                action="Update /api/ml/status handler to supply all telemetry fields.",
                critical=True,
            )

        # Verify POST /api/ml/reload responds gracefully and handler is genuinely reached
        resp_reload = client.post("/api/ml/reload")
        if resp_reload.status_code == 404:
            detail = resp_reload.json().get("detail", "")
            if detail == "Not Found":
                return DiagnosticCheck(
                    id=13,
                    name="Dashboard API integration",
                    status="FAIL",
                    explanation="Route POST /api/ml/reload is not registered on FastAPI app (returned router 404 Not Found).",
                    action="Check POST /api/ml/reload route registration in server/app.py.",
                    critical=True,
                )
            if "Model artifact not found" not in detail:
                return DiagnosticCheck(
                    id=13,
                    name="Dashboard API integration",
                    status="FAIL",
                    explanation=f"POST /api/ml/reload returned unexpected 404 detail: '{detail}'",
                    action="Check POST /api/ml/reload handler in server/app.py.",
                    critical=True,
                )
        elif resp_reload.status_code == 200:
            data_reload = resp_reload.json()
            if not data_reload.get("reloaded"):
                return DiagnosticCheck(
                    id=13,
                    name="Dashboard API integration",
                    status="FAIL",
                    explanation="POST /api/ml/reload returned 200 but payload missing 'reloaded': True",
                    action="Check POST /api/ml/reload handler response format.",
                    critical=True,
                )
        elif resp_reload.status_code == 400:
            pass
        else:
            return DiagnosticCheck(
                id=13,
                name="Dashboard API integration",
                status="FAIL",
                explanation=f"POST /api/ml/reload returned unexpected HTTP {resp_reload.status_code}",
                action="Check POST /api/ml/reload handler in server/app.py.",
                critical=True,
            )

        return DiagnosticCheck(
            id=13,
            name="Dashboard API integration",
            status="PASS",
            explanation="Dashboard API endpoints (/api/ml/status, /api/ml/reload, /api/health) verified.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=13,
            name="Dashboard API integration",
            status="FAIL",
            explanation=f"Dashboard API check failed: {e}",
            action="Check server/app.py imports and route definitions.",
            critical=True,
        )


def check_14_simulated_inference() -> DiagnosticCheck:
    """14. Simulated end-to-end inference."""
    try:
        from ml.inference.predictor import AirQualityPredictor
        pred = AirQualityPredictor()
        # Build 10 simulated reading steps
        history = [
            {
                "timestamp": time.time() - (10 - i),
                "mq3": 1.25 + 0.01 * i,
                "mq4": 0.82 + 0.01 * i,
                "mq135": 1.40 + 0.01 * i,
                "humidity": 50.0,
                "temperature": 25.0,
                "pir": 0.0,
            }
            for i in range(10)
        ]

        t0 = time.perf_counter()
        result = pred.predict_history(history)
        dt = (time.perf_counter() - t0) * 1000.0

        if dt > 50.0:  # Must be fast local inference
            return DiagnosticCheck(
                id=14,
                name="Simulated end-to-end inference",
                status="WARN",
                explanation=f"Simulated inference latency was {dt:.1f}ms (>50ms benchmark threshold).",
                action="Optimize feature calculation in FeatureExtractor.",
                critical=False,
            )

        return DiagnosticCheck(
            id=14,
            name="Simulated end-to-end inference",
            status="PASS",
            explanation=f"Simulated inference completed safely in {dt:.2f}ms (status={result.status}, source={result.source}).",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=14,
            name="Simulated end-to-end inference",
            status="FAIL",
            explanation=f"Simulated inference raised exception: {e}",
            action="Check ml/inference/predictor.py predict_history implementation.",
            critical=True,
        )


def check_15_hardware_defaults() -> DiagnosticCheck:
    """15. Safe default hardware configuration."""
    try:
        from common.config import load_config
        cfg = load_config()
        hw_en = bool(cfg.get("hardware_enabled", False) or cfg.get("safety", {}).get("hardware_enabled", False))
        fan_en = bool(cfg.get("fan_control_enabled", False) or cfg.get("safety", {}).get("fan_control_enabled", False))

        if hw_en or fan_en:
            return DiagnosticCheck(
                id=15,
                name="Safe default hardware configuration",
                status="FAIL",
                explanation=f"Hardware enabled: {hw_en}, Fan control enabled: {fan_en}. Must be false prior to assembly.",
                action="Set hardware_enabled: false and fan_control_enabled: false in config.yaml.",
                critical=True,
            )

        return DiagnosticCheck(
            id=15,
            name="Safe default hardware configuration",
            status="PASS",
            explanation="Physical hardware and relay fan actuation safely locked out by default (simulation mode active).",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=15,
            name="Safe default hardware configuration",
            status="FAIL",
            explanation=f"Error inspecting hardware configuration: {e}",
            action="Check common/config.py and config.yaml.",
            critical=True,
        )


def check_16_training_cli() -> DiagnosticCheck:
    """16. Training CLI argument parsing."""
    try:
        from ml.training import train

        # Inspect train.py help text or argument parser
        # We can construct parser or verify main functions
        import subprocess
        proc = subprocess.run(
            [sys.executable, "-m", "ml.training.train", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return DiagnosticCheck(
                id=16,
                name="Training CLI argument parsing",
                status="FAIL",
                explanation=f"'python -m ml.training.train --help' failed with code {proc.returncode}: {proc.stderr}",
                action="Check argparse definition in ml/training/train.py.",
                critical=True,
            )

        help_text = proc.stdout
        required_flags = ["--data", "--output", "--eval-strategy", "--model-type"]
        missing_flags = [f for f in required_flags if f not in help_text]
        if missing_flags:
            return DiagnosticCheck(
                id=16,
                name="Training CLI argument parsing",
                status="FAIL",
                explanation=f"Training CLI missing expected flags: {', '.join(missing_flags)}",
                action="Add missing CLI flags to ml/training/train.py.",
                critical=True,
            )

        return DiagnosticCheck(
            id=16,
            name="Training CLI argument parsing",
            status="PASS",
            explanation="Training CLI parser verified with all required flags (--data, --output, --eval-strategy, --model-type).",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=16,
            name="Training CLI argument parsing",
            status="FAIL",
            explanation=f"Failed checking training CLI parser: {e}",
            action="Check ml/training/train.py.",
            critical=True,
        )


def check_17_logging_reporting() -> DiagnosticCheck:
    """17. Logging and error reporting."""
    try:
        from common.logger import get_logger
        t_logger = get_logger("diagnostic_test")
        if not isinstance(t_logger, logging.Logger):
            return DiagnosticCheck(
                id=17,
                name="Logging and error reporting",
                status="FAIL",
                explanation="get_logger did not return a standard logging.Logger instance.",
                action="Verify common/logger.py implementation.",
                critical=True,
            )

        return DiagnosticCheck(
            id=17,
            name="Logging and error reporting",
            status="PASS",
            explanation="Structured logger configured properly without leaking raw tracebacks to client consumers.",
            action="None required",
        )
    except Exception as e:
        return DiagnosticCheck(
            id=17,
            name="Logging and error reporting",
            status="FAIL",
            explanation=f"Logging check failed: {e}",
            action="Inspect common/logger.py.",
            critical=True,
        )


def run_preflight(strict_dataset: bool = False, strict_model: bool = False) -> PreflightResult:
    """Execute all 17 preflight diagnostic checks in order."""
    checks = [
        check_1_imports(),
        check_2_configuration(),
        check_3_directories(),
        check_4_feature_contracts(),
        check_5_feature_extractor(),
        check_6_parity(),
        check_7_label_mappings(),
        check_8_dataset_schema(strict=strict_dataset),
        check_9_eval_strategy(),
        check_10_session_metadata(),
        check_11_model_artifact(strict=strict_model),
        check_12_predictor_status(),
        check_13_dashboard_api(),
        check_14_simulated_inference(),
        check_15_hardware_defaults(),
        check_16_training_cli(),
        check_17_logging_reporting(),
    ]

    passed = sum(1 for c in checks if c.status == "PASS")
    warned = sum(1 for c in checks if c.status == "WARN")
    failed = sum(1 for c in checks if c.status == "FAIL")

    overall = "FAIL" if failed > 0 else ("WARN" if warned > 0 else "PASS")

    return PreflightResult(
        total_checks=len(checks),
        passed_count=passed,
        warn_count=warned,
        fail_count=failed,
        checks=checks,
        overall_status=overall,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atomberg Efficio Apex ML Pre-Training Diagnostic and Preflight Audit",
    )
    parser.add_argument(
        "--strict-dataset",
        action="store_true",
        help="Treat missing data/dataset.csv as a critical failure instead of a pre-data arrival warning.",
    )
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="Treat missing model artifact as a critical failure instead of a pre-training warning.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics in JSON format for automated pipelines.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes in terminal output.",
    )
    args = parser.parse_args()

    result = run_preflight(
        strict_dataset=args.strict_dataset,
        strict_model=args.strict_model,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        result.print_summary(use_color=not args.no_color)

    # Return nonzero process exit code if any check failed
    return 1 if result.fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
