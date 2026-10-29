"""Unit tests for ML training pipeline, refusal rules, leakage prevention, and model bundle evaluation."""
import csv
import joblib
import numpy as np
import pytest
from common.schema import CANONICAL_CSV_COLUMNS, LEGACY_CSV_COLUMNS, InferenceStatus
from ml.inference.predictor import AirQualityPredictor
from ml.training.train import train_model
from ml.training.evaluate import evaluate_model


@pytest.fixture
def canonical_training_dataset(tmp_path):
    data_file = tmp_path / "canonical_runs.csv"
    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        # Session 1: Clean (30 samples)
        for i in range(30):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])
        # Session 2: Steam (30 samples)
        for i in range(30):
            writer.writerow([2, 1700000100.0 + i, 1.3, 1.2, 1.6, 80.0, 30.0, 0, 1])
        # Session 3: Cooking (30 samples)
        for i in range(30):
            writer.writerow([3, 1700000200.0 + i, 2.5, 2.8, 2.2, 60.0, 35.0, 0, 2])
    return data_file


@pytest.fixture
def legacy_training_dataset(tmp_path):
    data_file = tmp_path / "legacy_runs.csv"
    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LEGACY_CSV_COLUMNS)
        # Session 1: Clean (30 samples)
        for i in range(30):
            writer.writerow([1, 1700000000.0 + i, 1.5, 1.0, 55.0, 25.0, 0, 0])
        # Session 2: Steam (30 samples)
        for i in range(30):
            writer.writerow([2, 1700000100.0 + i, 1.6, 1.1, 80.0, 30.0, 0, 1])
        # Session 3: Cooking (30 samples)
        for i in range(30):
            writer.writerow([3, 1700000200.0 + i, 2.2, 2.5, 60.0, 35.0, 0, 2])
    return data_file


def test_train_model_refuses_nonexistent_dataset():
    with pytest.raises(ValueError, match="Cannot train model"):
        train_model("non_existent_dataset.csv")


def test_train_model_refuses_small_dataset(tmp_path):
    small_file = tmp_path / "small.csv"
    with open(small_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(10):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])

    with pytest.raises(ValueError, match="Insufficient rows"):
        train_model(small_file)


def test_train_model_refuses_invalid_eval_strategy(canonical_training_dataset):
    with pytest.raises(ValueError, match="Invalid evaluation strategy"):
        train_model(canonical_training_dataset, eval_strategy="random_kfold")


def test_train_model_refuses_insufficient_groups_without_silent_fallback(tmp_path):
    """Leakage prevention: GroupKFold with 2 groups and n_splits=3 must FAIL, not silently fall back."""
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


def test_train_model_time_series_split(canonical_training_dataset, tmp_path):
    """Verify chronological split runs without error and records methodology."""
    out_dir = tmp_path / "artifacts"
    res = train_model(
        data_path=canonical_training_dataset,
        output_dir=out_dir,
        model_name="ts_model.joblib",
        eval_strategy="time_series_split",
        n_splits=3,
    )
    assert res["status"] == "success"
    assert res["eval_strategy"] == "time_series_split"
    assert 0.0 <= res["cv_accuracy"] <= 1.0


def test_train_model_session_holdout(canonical_training_dataset, tmp_path):
    """Verify session_holdout isolates the entire last session as an independent test set."""
    out_dir = tmp_path / "artifacts"
    res = train_model(
        data_path=canonical_training_dataset,
        output_dir=out_dir,
        model_name="holdout_model.joblib",
        eval_strategy="session_holdout",
    )
    assert res["status"] == "success"
    assert res["eval_strategy"] == "session_holdout"
    assert 0.0 <= res["cv_accuracy"] <= 1.0

    # Verify bundle metadata explicitly identifies training and holdout sessions
    bundle = joblib.load(res["model_path"])
    assert "holdout_session" in bundle["metadata"]
    assert bundle["metadata"]["holdout_session"] == 3
    assert bundle["metadata"]["train_sessions"] == [1, 2]
    assert bundle["feature_count"] == 32


def test_train_model_session_holdout_refuses_single_session(tmp_path):
    one_session_file = tmp_path / "single_session.csv"
    with open(one_session_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for i in range(35):
            writer.writerow([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])
        for i in range(35):
            writer.writerow([1, 1700000100.0 + i, 1.3, 1.2, 1.6, 80.0, 30.0, 0, 1])

    with pytest.raises(ValueError, match="requires at least 2 distinct 'run_id' sessions"):
        train_model(one_session_file, eval_strategy="session_holdout")


def test_train_canonical_dataset_32_features_pipeline(canonical_training_dataset, tmp_path):
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
    assert res["cv_balanced_accuracy"] is not None

    # Verify evaluate_model runs on canonical artifact
    eval_res = evaluate_model(res["model_path"], canonical_training_dataset)
    assert eval_res["samples_evaluated"] == 90
    assert 0.0 <= eval_res["accuracy"] <= 1.0
    assert "clean" in eval_res["classes"]

    # Verify AirQualityPredictor loads and verifies 32 features
    predictor = AirQualityPredictor(res["model_path"])
    assert predictor.is_loaded is True
    assert predictor.status == InferenceStatus.VALID_MODEL
    assert predictor.feature_count == 32

    reading_history = [
        {
            "ts": 1700000500.0,
            "sensors": {"mq3": 1.2, "mq4": 1.1, "mq135": 1.5, "humidity": 55.0, "temp_c": 25.0, "pir": 0},
            "baseline": {"mq3": 1.2, "mq4": 1.1, "mq135": 1.5, "humidity": 55.0, "temp_c": 25.0},
            "deltas": {"mq3": 0.0, "mq4": 0.0, "mq135": 0.0, "humidity": 0.0, "temp_c": 0.0},
        }
    ]
    pred_out = predictor.predict_history(reading_history)
    assert pred_out.status == InferenceStatus.VALID_MODEL
    assert pred_out.label in ("clean", "steam", "cooking")
    assert 0.0 <= pred_out.confidence <= 1.0


def test_train_legacy_dataset_26_features_pipeline(legacy_training_dataset, tmp_path):
    out_dir = tmp_path / "artifacts"
    res = train_model(
        data_path=legacy_training_dataset,
        output_dir=out_dir,
        model_name="legacy_model.joblib",
        model_type="rf",
        eval_strategy="group_kfold",
        n_splits=3,
    )
    assert res["status"] == "success"
    assert res["feature_count"] == 26

    predictor = AirQualityPredictor(res["model_path"])
    assert predictor.is_loaded is True
    assert predictor.feature_count == 26


def test_session_holdout_complete_isolation_different_distribution(tmp_path):
    """Regression test proving session holdout is 100% leakage-free.
    
    Creates two datasets with identical training sessions (1 & 2):
    - Dataset A: Holdout session 3 has normal benign readings (1.2, 1.1, 1.5).
    - Dataset B: Holdout session 3 has extreme outlier readings (999.0, 888.0, 777.0).
    
    Verifies that:
    1. Both models trained under session_holdout are 100% mathematically identical in predictions
       and probability outputs on training data (proving zero leakage from session 3 into
       feature extraction, rolling window calculations, baseline derivation, scaling, or tree splits).
    2. Holdout session 3 is never used for fitting or threshold selection.
    3. Metadata accurately captures train_sessions=[1, 2] and holdout_session=3 in both models.
    """
    normal_training_data = []
    # Session 1: Clean (35 samples)
    for i in range(35):
        normal_training_data.append([1, 1700000000.0 + i, 1.2, 1.1, 1.5, 55.0, 25.0, 0, 0])
    # Session 2: Cooking (35 samples)
    for i in range(35):
        normal_training_data.append([2, 1700000100.0 + i, 2.5, 2.8, 2.2, 60.0, 35.0, 0, 2])

    # Dataset A: Benign holdout (Session 3)
    benign_holdout = []
    for i in range(35):
        benign_holdout.append([3, 1700000200.0 + i, 1.3, 1.2, 1.6, 56.0, 26.0, 0, 0])

    # Dataset B: Extreme outlier holdout (Session 3)
    outlier_holdout = []
    for i in range(35):
        outlier_holdout.append([3, 1700000200.0 + i, 999.0, 888.0, 777.0, 99.0, 80.0, 1, 2])

    file_a = tmp_path / "dataset_a_benign_holdout.csv"
    with open(file_a, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for row in normal_training_data + benign_holdout:
            writer.writerow(row)

    file_b = tmp_path / "dataset_b_outlier_holdout.csv"
    with open(file_b, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANONICAL_CSV_COLUMNS)
        for row in normal_training_data + outlier_holdout:
            writer.writerow(row)

    out_a = tmp_path / "run_a"
    res_a = train_model(
        data_path=file_a,
        output_dir=out_a,
        model_name="model_a.joblib",
        eval_strategy="session_holdout",
        random_state=42,
    )

    out_b = tmp_path / "run_b"
    res_b = train_model(
        data_path=file_b,
        output_dir=out_b,
        model_name="model_b.joblib",
        eval_strategy="session_holdout",
        random_state=42,
    )

    bundle_a = joblib.load(res_a["model_path"])
    bundle_b = joblib.load(res_b["model_path"])

    # 1. Verify train and holdout metadata
    assert bundle_a["metadata"]["holdout_session"] == 3
    assert bundle_a["metadata"]["train_sessions"] == [1, 2]
    assert bundle_b["metadata"]["holdout_session"] == 3
    assert bundle_b["metadata"]["train_sessions"] == [1, 2]

    # 2. Verify model fitted under session_holdout with benign holdout matches outlier holdout identically
    from ml.features.build_features import FeatureExtractor
    import pandas as pd
    df_eval = pd.DataFrame(normal_training_data, columns=CANONICAL_CSV_COLUMNS)
    X_eval, _ = FeatureExtractor(mode="canonical").extract_from_dataframe(df_eval)

    preds_a = bundle_a["model"].predict(X_eval)
    preds_b = bundle_b["model"].predict(X_eval)
    np.testing.assert_array_equal(preds_a, preds_b)

    proba_a = bundle_a["model"].predict_proba(X_eval)
    proba_b = bundle_b["model"].predict_proba(X_eval)
    np.testing.assert_allclose(proba_a, proba_b, atol=1e-5)
