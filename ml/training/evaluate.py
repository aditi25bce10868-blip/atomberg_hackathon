"""Model evaluation and comparison entry point with strict contract verification."""
import argparse
import json
import pathlib
import sys
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from common.logger import get_logger
from common.schema import LABEL_CODES, LABELS
from ml.features.build_features import FeatureExtractor

logger = get_logger("ml.training.evaluate")


def evaluate_model(
    model_path: Union[str, pathlib.Path],
    data_path: Union[str, pathlib.Path],
) -> Dict[str, Any]:
    """Evaluate a trained model bundle against an independent test dataset.
    
    Verifies feature count, feature ordering, and class mappings before prediction.
    """
    m_p = pathlib.Path(model_path)
    if not m_p.exists():
        raise FileNotFoundError(f"Model bundle not found: {m_p}")

    d_p = pathlib.Path(data_path)
    if not d_p.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {d_p}")

    bundle = joblib.load(m_p)
    if not isinstance(bundle, dict) or "model" not in bundle:
        model = bundle
        feature_names = None
        class_names = [LABELS.get(c, str(c)) for c in getattr(model, "classes_", [])]
        bundle_metadata = {}
    else:
        model = bundle["model"]
        feature_names = bundle.get("feature_names")
        class_names = bundle.get("class_names") or [LABELS.get(c, str(c)) for c in model.classes_]
        bundle_metadata = bundle.get("metadata", {})

    df = pd.read_csv(d_p)
    if len(df) == 0:
        raise ValueError(f"Evaluation dataset '{d_p.name}' is empty.")

    # Match extractor mode to bundle feature names if available
    mode = "legacy" if feature_names and "raw_mq2" in feature_names else "canonical"
    extractor = FeatureExtractor(mode=mode)
    X, extracted_features = extractor.extract_from_dataframe(df)

    # Feature contract check
    if feature_names is not None and extracted_features != feature_names:
        raise ValueError(
            f"Feature contract mismatch between model artifact and evaluation data extractor.\n"
            f"Model expected ({len(feature_names)} features): {feature_names[:3]}...\n"
            f"Extractor generated ({len(extracted_features)} features): {extracted_features[:3]}..."
        )

    label_col = "label" if "label" in df.columns else "target"
    if label_col not in df.columns:
        raise ValueError(f"Dataset missing '{label_col}' column.")

    raw_y = df[label_col].values
    y_true: list[int] = []
    for val in raw_y:
        if isinstance(val, (int, np.integer)) or (isinstance(val, str) and val.isdigit()):
            code = int(val)
        else:
            code = LABEL_CODES.get(str(val).strip(), -1)
        if code == -1:
            raise ValueError(f"Unrecognized class label '{val}' in evaluation dataset.")
        y_true.append(code)

    y_arr = np.array(y_true, dtype=np.int32)
    y_pred = model.predict(X)

    from ml.training.train import compute_balanced_accuracy

    acc = float(accuracy_score(y_arr, y_pred))
    bacc = compute_balanced_accuracy(y_arr, y_pred)
    macro_f1 = float(f1_score(y_arr, y_pred, average="macro", zero_division=0))
    eval_labels = sorted(list(set(model.classes_).union(set(y_arr)).union(set(y_pred))))
    if len(eval_labels) < 2:
        eval_labels = sorted(list(LABELS.keys()))
    eval_class_names = [LABELS.get(c, str(c)) for c in eval_labels]
    cm = confusion_matrix(y_arr, y_pred, labels=eval_labels)
    report_dict = classification_report(
        y_arr,
        y_pred,
        labels=eval_labels,
        target_names=eval_class_names,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model_file": m_p.name,
        "dataset_file": d_p.name,
        "samples_evaluated": len(y_arr),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
        "classification_report": report_dict,
        "training_metadata": bundle_metadata,
        "disclaimer": "Independent evaluation metrics are specific to the test dataset and do not prove real-world hardware performance.",
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model bundle against a dataset.")
    parser.add_argument("--model", required=True, help="Path to model bundle (.joblib)")
    parser.add_argument("--data", required=True, help="Path to evaluation dataset CSV")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    try:
        metrics = evaluate_model(args.model, args.data)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print("=" * 70)
            print(f"EVALUATION RESULTS: {metrics['model_file']} on {metrics['dataset_file']}")
            print("=" * 70)
            print(f"Samples:           {metrics['samples_evaluated']}")
            print(f"Accuracy:          {metrics['accuracy']:.2%}")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
            print(f"Macro F1:          {metrics['macro_f1']:.4f}")
            print("\nClassification Report:")
            for cls_name in metrics["classes"]:
                cls_m = metrics["classification_report"].get(cls_name, {})
                p = cls_m.get("precision", 0.0)
                r = cls_m.get("recall", 0.0)
                f = cls_m.get("f1-score", 0.0)
                print(f"  {cls_name:<12}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")
            print("=" * 70)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
