"""Unit tests for ML dataset inspection, validation, and adapters."""
import csv
import pytest
from common.schema import CANONICAL_CSV_COLUMNS, LEGACY_CSV_COLUMNS
from ml.datasets.inspect_dataset import inspect_dataset
from ml.datasets.validate_dataset import validate_dataset
from ml.datasets.adapters import LegacyPrototypeAdapter, ExternalBenchmarkAdapter, DatasetAdapterError


@pytest.fixture
def canonical_csv(tmp_path):
    csv_file = tmp_path / "canonical_session.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(20):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.4, 1.5 + (i * 0.01), 55.0, 28.0, 0, 0])
        for i in range(15):
            writer.writerow([2, 1700000100.0 + i, 1.3, 1.5, 1.8, 75.0, 31.0, 0, 1])
    return csv_file


@pytest.fixture
def legacy_csv(tmp_path):
    csv_file = tmp_path / "legacy_session.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LEGACY_CSV_COLUMNS)
        for i in range(20):
            writer.writerow([1, 1700000000.0 + i, 1.5, 1.2, 55.0, 25.0, 0, "clean"])
        for i in range(20):
            writer.writerow([2, 1700000100.0 + i, 1.8, 1.4, 75.0, 30.0, 0, "steam"])
    return csv_file


def test_inspect_canonical_dataset(canonical_csv):
    summary = inspect_dataset(canonical_csv)
    assert summary.row_count == 35
    assert set(summary.column_names) == set(CANONICAL_CSV_COLUMNS)
    assert summary.session_count == 2
    assert "clean" in summary.label_distribution or "0" in summary.label_distribution
    assert "mq3" in summary.sensor_statistics
    assert "mq4" in summary.sensor_statistics
    assert "mq135" in summary.sensor_statistics
    assert summary.sensor_statistics["mq3"]["count"] == 35


def test_validate_dataset_canonical_valid(canonical_csv):
    report = validate_dataset(canonical_csv, min_rows=30)
    assert report.is_valid is True
    assert report.dataset_type == "CANONICAL_HARDWARE"
    assert report.errors == []


def test_validate_dataset_legacy_detected(legacy_csv):
    report = validate_dataset(legacy_csv, strict_canonical=False, min_rows=30)
    assert report.is_valid is True
    assert report.dataset_type == "LEGACY_PROTOTYPE"
    assert any("legacy prototype schema" in w for w in report.warnings)

    strict_report = validate_dataset(legacy_csv, strict_canonical=True, min_rows=30)
    assert strict_report.is_valid is False
    assert any("legacy prototype schema" in e for e in strict_report.errors)


def test_validate_dataset_missing_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "ts", "mq3", "humidity", "label"])
        for i in range(35):
            writer.writerow([1, 1700000000.0 + i, 1.5, 55.0, 0])

    report = validate_dataset(bad_csv, strict_canonical=True)
    assert report.is_valid is False
    assert any("Required column" in e for e in report.errors)


def test_validate_dataset_invalid_labels(tmp_path):
    bad_labels_csv = tmp_path / "bad_labels.csv"
    with open(bad_labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(35):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.4, 1.5, 55.0, 28.0, 0, "unsupported_condition_xyz"])

    report = validate_dataset(bad_labels_csv)
    assert report.is_valid is False
    assert any("Unsupported class labels found" in e for e in report.errors)


def test_legacy_prototype_adapter_refuses_silent_canonical_coercion(legacy_csv):
    import pandas as pd
    adapter = LegacyPrototypeAdapter()
    df = pd.read_csv(legacy_csv)
    assert "mq2" in df.columns
    with pytest.raises(DatasetAdapterError, match="Refusing silent coercion"):
        adapter.adapt(df, target_schema="canonical")


def test_external_benchmark_adapter_preserves_provenance(tmp_path):
    import pandas as pd
    external_csv = tmp_path / "uci_gas.csv"
    with open(external_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Gas_Ethanol", "Gas_Methane", "Air_Quality", "Target_Class"])
        for i in range(40):
            writer.writerow([i * 1.0, 2.5, 1.8, 0.9, 1 if i >= 20 else 0])

    adapter = ExternalBenchmarkAdapter(
        dataset_name="UCI_Gas_Benchmark_2016",
        column_mapping={
            "ts": "Time",
            "mq3": "Gas_Ethanol",
            "mq4": "Gas_Methane",
            "mq135": "Air_Quality",
            "label": "Target_Class",
        },
        label_mapping={0: "clean", 1: "cooking"},
    )
    df_raw = pd.read_csv(external_csv)
    df_can = adapter.adapt(df_raw)
    assert "mq3" in df_can.columns
    assert "mq4" in df_can.columns
    assert "mq135" in df_can.columns
    assert "label" in df_can.columns
    assert "_provenance" in df_can.columns
