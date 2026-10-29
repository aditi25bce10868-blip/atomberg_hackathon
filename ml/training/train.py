"""Model training pipeline with strict session isolation and zero-leakage evaluation."""
import argparse
import json
import pathlib
import sys
import time
from typing import Dict, Any, Optional, Union, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from common.logger import get_logger
from common.schema import LABEL_NAMES, LABEL_CODES, LABELS
from ml.config import (
    ARTIFACTS_DIR,
    REPORTS_DIR,
    DEFAULT_MODEL_NAME,
    MIN_SAMPLES_FOR_TRAINING,
    MIN_CLASSES_FOR_TRAINING,
    RANDOM_STATE,
    VALID_EVAL_STRATEGIES,
    DEFAULT_EVAL_STRATEGY,
)
from ml.datasets.validate_dataset import validate_dataset
from ml.features.build_features import FeatureExtractor

logger = get_logger("ml.training.train")


def compute_balanced_accuracy(y_true: Union[np.ndarray, List[int]], y_pred: Union[np.ndarray, List[int]]) -> float:
    """Compute balanced accuracy honestly across ground-truth classes.
    
    When a validation fold contains only a single class, balanced accuracy is
    mathematically identical to recall on that class (i.e. accuracy). For multiple
    classes, computes the macro-average recall over classes present in y_true,
    properly penalizing false predictions without triggering zero-division
    warnings or crashing on classes missing from y_true.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes_present = np.unique(y_t)
    if len(classes_present) == 0:
        return 0.0
    if len(classes_present) == 1:
        c = classes_present[0]
        return float(np.sum((y_t == c) & (y_p == c)) / len(y_t))
    
    recalls = [
        float(np.sum((y_t == c) & (y_p == c)) / np.sum(y_t == c))
        for c in classes_present
    ]
    return float(np.mean(recalls))


def train_model(
    data_path: Union[str, pathlib.Path],
    output_dir: Optional[Union[str, pathlib.Path]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_type: str = "rf",
    eval_strategy: str = DEFAULT_EVAL_STRATEGY,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Train environmental condition classifier with strict temporal isolation.
    
    CRITICAL EVALUATION LEAKAGE RULES:
    1. NEVER silently fall back to random/shuffled K-Fold for sensor time-series.
    2. GroupKFold keeps all observations from a recording session together.
    3. If grouping metadata is insufficient, explicitly require TimeSeriesSplit or fail.
    4. Fit preprocessing and models strictly on training folds.
    5. Refuse to claim real-world hardware accuracy when trained on benchmark or limited data.
    """
    if eval_strategy not in VALID_EVAL_STRATEGIES:
        raise ValueError(
            f"Invalid evaluation strategy: '{eval_strategy}'. "
            f"Must be one of: {', '.join(VALID_EVAL_STRATEGIES)}. "
            "Random shuffled splitting is strictly forbidden to prevent temporal leakage."
        )

    data_p = pathlib.Path(data_path)
    if not data_p.exists():
        raise ValueError(f"Cannot train model: dataset file does not exist at '{data_p}'.")

    # 1. Dataset schema and quality validation
    validation = validate_dataset(data_p, min_rows=MIN_SAMPLES_FOR_TRAINING)
    if not validation.is_valid:
        raise ValueError(
            f"Cannot train model on invalid dataset ({data_p.name}):\n"
            + "\n".join(f"  - {e}" for e in validation.errors)
        )

    df = pd.read_csv(data_p)
    row_count = len(df)
    if row_count < MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Insufficient samples: dataset has {row_count} rows, below the minimum threshold "
            f"({MIN_SAMPLES_FOR_TRAINING}). Refusing to train an unvalidated model."
        )

    # 2. Extract features deterministically per session
    extractor = FeatureExtractor()
    X, feature_names = extractor.extract_from_dataframe(df)
    feature_count = len(feature_names)

    # 3. Target label processing
    label_col = "label" if "label" in df.columns else "target"
    raw_y = df[label_col].values
    y: List[int] = []
    for val in raw_y:
        if isinstance(val, (int, np.integer)) or (isinstance(val, str) and val.isdigit()):
            code = int(val)
        else:
            code = LABEL_CODES.get(str(val).strip(), -1)
        if code == -1:
            raise ValueError(f"Unrecognized class label '{val}' in dataset.")
        y.append(code)

    y_arr = np.array(y, dtype=np.int32)
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < MIN_CLASSES_FOR_TRAINING:
        raise ValueError(
            f"Dataset contains only {len(unique_classes)} class ({unique_classes}). "
            f"Multi-class classification requires at least {MIN_CLASSES_FOR_TRAINING} classes."
        )

    # 4. Configure evaluation split strictly without random shuffled fallback
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    groups = None
    if eval_strategy == "group_kfold":
        if "run_id" not in df.columns:
            raise ValueError(
                "Cannot use 'group_kfold': dataset lacks 'run_id' column for session grouping. "
                "Shuffled K-Fold fallback is strictly forbidden due to time-series data leakage. "
                "Add session identifiers or specify '--eval-strategy time_series_split'."
            )
        unique_runs = df["run_id"].unique()
        if len(unique_runs) < n_splits:
            raise ValueError(
                f"Cannot use 'group_kfold' with {n_splits} folds: dataset contains only {len(unique_runs)} "
                f"distinct session(s) {list(unique_runs)}. "
                "Shuffled K-Fold fallback is strictly forbidden to prevent temporal leakage. "
                f"Either reduce --n-splits to <= {len(unique_runs)} or use '--eval-strategy time_series_split'."
            )
        groups = df["run_id"].values
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y_arr, groups=groups))
        logger.info("Using session-aware GroupKFold validation (%d folds across %d sessions)", n_splits, len(unique_runs))

    elif eval_strategy == "time_series_split":
        # Strict chronological order without lookahead
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X, y_arr))
        logger.info("Using chronological TimeSeriesSplit validation (%d folds)", n_splits)

    elif eval_strategy == "session_holdout":
        # Reserve last session as independent holdout
        if "run_id" not in df.columns:
            raise ValueError("Cannot use 'session_holdout': dataset lacks 'run_id' column for session grouping.")
        runs = list(df["run_id"].unique())
        if len(runs) < 2:
            raise ValueError(
                f"Cannot use 'session_holdout': requires at least 2 distinct 'run_id' sessions, but found only {len(runs)}."
            )
        holdout_run = runs[-1]
        train_idx = np.where(df["run_id"].values != holdout_run)[0]
        val_idx = np.where(df["run_id"].values == holdout_run)[0]
        splits = [(train_idx, val_idx)]
        logger.info(
            "Using session_holdout: training on runs %s (%d samples), holdout validation on session %s (%d samples)",
            runs[:-1],
            len(train_idx),
            holdout_run,
            len(val_idx),
        )

    # 5. Classifier Model Construction
    if model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=random_state)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)

    # 6. Cross-Validation Execution (with strict training-fold fitting)
    fold_accuracies: List[float] = []
    fold_balanced_accs: List[float] = []
    fold_macro_f1s: List[float] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # Ensure validation classes are represented
        if len(np.unique(y_train)) < len(unique_classes):
            logger.warning("Fold %d: training split does not contain all classes.", fold_idx)

        # Fit exclusively on training fold
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc = float(accuracy_score(y_val, preds))
        bacc = compute_balanced_accuracy(y_val, preds)
        f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        fold_accuracies.append(acc)
        fold_balanced_accs.append(bacc)
        fold_macro_f1s.append(f1)
        all_y_true.extend(y_val)
        all_y_pred.extend(preds)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_bacc = float(np.mean(fold_balanced_accs))
    mean_f1 = float(np.mean(fold_macro_f1s))

    # 7. Final Model Packaging
    if eval_strategy == "session_holdout":
        # Crucial: in session_holdout, the model saved in the artifact must NOT be fitted
        # on the holdout session (preserving the independent test set integrity)
        final_model = clf
    else:
        # Re-fit final deployable estimator on full dataset after cross-validation assessment
        from sklearn.base import clone
        final_model = clone(clf)
        final_model.fit(X, y_arr)

    # 8. Assemble Model Bundle with strict metadata
    class_names_in_model = [LABELS.get(c, str(c)) for c in final_model.classes_]
    bundle = {
        "model": final_model,
        "feature_names": feature_names,
        "feature_count": feature_count,
        "classes": clf.classes_.tolist(),
        "class_names": class_names_in_model,
        "metadata": {
            "trained_at": time.time(),
            "model_type": model_type,
            "eval_strategy": eval_strategy,
            "dataset_file": data_p.name,
            "dataset_type": validation.dataset_type,
            "dataset_rows": row_count,
            "feature_count": feature_count,
            "random_state": random_state,
            "train_sessions": [int(r) if isinstance(r, (int, np.integer)) else str(r) for r in runs[:-1]] if eval_strategy == "session_holdout" else None,
            "holdout_session": (int(runs[-1]) if isinstance(runs[-1], (int, np.integer)) else str(runs[-1])) if eval_strategy == "session_holdout" else None,
            "cv_metrics": {
                "macro_f1": round(mean_f1, 4),
                "balanced_accuracy": round(mean_bacc, 4),
                "accuracy": round(mean_acc, 4),
                "accuracy_std": round(std_acc, 4),
                "folds_evaluated": len(fold_accuracies),
            },
            "disclaimer": "Cross-validation and benchmark metrics are not proof of real-world physical hardware performance.",
        },
    }

    # 9. Save bundle to artifacts directory
    target_out_dir = pathlib.Path(output_dir or ARTIFACTS_DIR)
    target_out_dir.mkdir(parents=True, exist_ok=True)
    saved_model_path = target_out_dir / model_name
    joblib.dump(bundle, saved_model_path)
    logger.info("Saved trained model bundle to %s (features: %d, classes: %s)", saved_model_path, feature_count, class_names_in_model)

    # 10. Generate dashboard reporting artifacts in ml/reports/latest/
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "accuracy": round(mean_acc, 4),
        "balanced_accuracy": round(mean_bacc, 4),
        "accuracy_std": round(std_acc, 4),
        "macro_f1": round(mean_f1, 4),
        "samples": row_count,
        "eval_strategy": eval_strategy,
        "train_sessions": [int(r) if isinstance(r, (int, np.integer)) else str(r) for r in runs[:-1]] if eval_strategy == "session_holdout" else None,
        "holdout_session": (int(runs[-1]) if isinstance(runs[-1], (int, np.integer)) else str(runs[-1])) if eval_strategy == "session_holdout" else None,
        "classes": class_names_in_model,
        "trained_at": time.time(),
        "disclaimer": "Cross-validation metrics are not proof of real-world hardware performance.",
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

    eval_labels = sorted(list(set(all_y_true).union(set(all_y_pred)).union(set(clf.classes_))))
    if len(eval_labels) < 2:
        eval_labels = sorted(list(LABELS.keys()))
    eval_class_names = [LABELS.get(c, str(c)) for c in eval_labels]
    cm = confusion_matrix(all_y_true, all_y_pred, labels=eval_labels)
    cm_data = {
        "labels": eval_class_names,
        "matrix": cm.tolist(),
    }
    (reports_dir / "confusion_matrix.json").write_text(json.dumps(cm_data, indent=2))

    if hasattr(clf, "feature_importances_"):
        fi = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(
                zip(feature_names, clf.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        }
        (reports_dir / "feature_importance.json").write_text(json.dumps(fi, indent=2))

    dataset_summary_data = {
        "dataset_name": data_p.name,
        "dataset_type": validation.dataset_type,
        "rows": row_count,
        "class_counts": {
            LABELS.get(int(k), str(k)): int(v)
            for k, v in df[label_col].value_counts().to_dict().items()
        },
    }
    (reports_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary_data, indent=2))

    model_info_data = {
        "model_type": model_type,
        "artifact_path": str(saved_model_path),
        "features_count": feature_count,
        "trained_timestamp": time.time(),
        "eval_strategy": eval_strategy,
    }
    (reports_dir / "model_info.json").write_text(json.dumps(model_info_data, indent=2))

    return {
        "status": "success",
        "model_path": str(saved_model_path),
        "eval_strategy": eval_strategy,
        "cv_accuracy": mean_acc,
        "cv_balanced_accuracy": mean_bacc,
        "cv_macro_f1": mean_f1,
        "metrics": metrics_data,
        "feature_count": feature_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Train an environmental classification model.")
    parser.add_argument("--data", required=True, help="Path to training dataset CSV")
    parser.add_argument("--output", help="Directory to save the trained model artifact")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model artifact filename")
    parser.add_argument("--model-type", choices=["rf", "hgb"], default="rf", help="Classifier algorithm")
    parser.add_argument(
        "--eval-strategy",
        choices=VALID_EVAL_STRATEGIES,
        default=DEFAULT_EVAL_STRATEGY,
        help="Evaluation strategy (group_kfold | time_series_split | session_holdout)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    try:
        res = train_model(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.model_name,
            model_type=args.model_type,
            eval_strategy=args.eval_strategy,
            n_splits=args.n_splits,
        )
        print("Training completed successfully.")
        print(f"Artifact:          {res['model_path']}")
        print(f"Feature Count:     {res['feature_count']}")
        print(f"CV Accuracy:       {res['cv_accuracy']:.2%}")
        print(f"CV Balanced Acc:   {res['cv_balanced_accuracy']:.2%}")
        print(f"CV Macro F1:       {res['cv_macro_f1']:.4f}")
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
