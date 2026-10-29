"""Unit tests for ML training pipeline, refusal rules, and leakage prevention."""
import csv
import pytest
from common.schema import CANONICAL_CSV_COLUMNS, LEGACY_CSV_COLUMNS
from ml.inference.predictor import AirQualityPredictor
from ml.training.train import train_model
from ml.training.evaluate import evaluate_model


@pytest.fixture
def canonical_training_dataset(tmp_path):
    data_file = tmp_path / "canonical_runs.csv"
    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(30):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])
        for i in range(30):
            writer.writerow([2, 1700000100.0 + i, 1.3, 1.2, 1.6, 80.0, 30.0, 0, 1])
        for i in range(30):
            writer.writerow([3, 1700000200.0 + i, 2.5, 2.8, 2.2, 60.0, 35.0, 0, 2])
    return data_file


def test_train_model_refuses_small_dataset(tmp_path):
    small_file = tmp_path / "small.csv"
    with open(small_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(10):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])

    with pytest.raises(ValueError, match="Insufficient rows"):
        train_model(small_file)


def test_train_model_refuses_insufficient_groups(tmp_path):
    two_groups_file = tmp_path / "two_groups.csv"
    with open(two_groups_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(35):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])
        for i in range(35):
            writer.writerow([2, 1700000100.0 + i, 1.3, 1.2, 1.6, 80.0, 30.0, 0, 1])

    with pytest.raises(ValueError, match="Cannot use 'group_kfold'"):
        train_model(two_groups_file, eval_strategy="group_kfold", n_splits=3)


def test_train_canonical_dataset_32_features(canonical_training_dataset, tmp_path):
    out_dir = tmp_path / "artifacts"
    res = train_model(
        data_path=canonical_training_dataset,
        output_dir=out_dir,
        model_name="canonical_model.joblib",
        model_type="rf",
        eval_strategy="group_kfold",
        n_splits=3,
    )
    assert res["status"] == "success"
    assert res["feature_count"] == 32
    assert res["eval_strategy"] == "group_kfold"
