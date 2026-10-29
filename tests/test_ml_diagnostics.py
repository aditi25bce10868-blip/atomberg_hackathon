"""Automated tests for ML preflight diagnostic tool (python -m ml.diagnostics.preflight)."""
import subprocess
import sys
import pytest

from ml.diagnostics.preflight import (
    run_preflight,
    PreflightResult,
    check_1_imports,
    check_2_configuration,
    check_3_directories,
    check_4_feature_contracts,
    check_5_feature_extractor,
    check_6_parity,
    check_7_label_mappings,
    check_8_dataset_schema,
    check_9_eval_strategy,
    check_10_session_metadata,
    check_11_model_artifact,
    check_12_predictor_status,
    check_13_dashboard_api,
    check_14_simulated_inference,
    check_15_hardware_defaults,
    check_16_training_cli,
    check_17_logging_reporting,
)


def test_preflight_runs_all_17_checks():
    """Verify that run_preflight executes all 17 distinct diagnostic checks."""
    res = run_preflight(strict_dataset=False, strict_model=False)
    assert isinstance(res, PreflightResult)
    assert res.total_checks == 17
    assert len(res.checks) == 17
    assert res.fail_count == 0
    assert res.overall_status in ("PASS", "WARN")


def test_preflight_strict_dataset_fails_when_no_data():
    """Verify that --strict-dataset turns missing dataset into an explicit failure."""
    res = run_preflight(strict_dataset=True, strict_model=False)
    assert res.fail_count >= 1
    assert res.overall_status == "FAIL"
    chk8 = [c for c in res.checks if c.id == 8][0]
    assert chk8.status == "FAIL"


def test_preflight_individual_checks():
    """Verify core software checks pass in isolation without real hardware or dataset."""
    assert check_1_imports().status == "PASS"
    assert check_2_configuration().status == "PASS"
    assert check_3_directories().status == "PASS"
    assert check_4_feature_contracts().status == "PASS"
    assert check_5_feature_extractor().status == "PASS"
    assert check_6_parity().status == "PASS"
    assert check_7_label_mappings().status == "PASS"
    assert check_9_eval_strategy().status == "PASS"
    assert check_10_session_metadata().status == "PASS"
    assert check_12_predictor_status().status == "PASS"
    assert check_13_dashboard_api().status == "PASS"
    assert check_14_simulated_inference().status == "PASS"
    assert check_15_hardware_defaults().status == "PASS"
    assert check_16_training_cli().status == "PASS"
    assert check_17_logging_reporting().status == "PASS"


def test_preflight_cli_execution():
    """Verify python -m ml.diagnostics.preflight CLI returns exit code 0 under standard preflight."""
    proc = subprocess.run(
        [sys.executable, "-m", "ml.diagnostics.preflight", "--no-color"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0
    assert "Summary: 17 Checks" in proc.stdout
    assert "0 FAIL" in proc.stdout


def test_preflight_cli_json_output():
    """Verify CLI --json outputs structured valid JSON."""
    import json
    proc = subprocess.run(
        [sys.executable, "-m", "ml.diagnostics.preflight", "--json"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["total_checks"] == 17
    assert len(data["checks"]) == 17
    assert data["fail_count"] == 0
