"""Dataset validation tool verifying schema contracts, sensor channels, and quality."""
import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union

from common.schema import (
    CANONICAL_CSV_COLUMNS,
    CANONICAL_GAS_SENSORS,
    LEGACY_CSV_COLUMNS,
    LABELS,
    LABEL_NAMES,
    LABEL_CODES,
)


@dataclass
class ValidationReport:
    file_path: str
    is_valid: bool
    dataset_type: str  # 'CANONICAL_HARDWARE', 'LEGACY_PROTOTYPE', 'EXTERNAL_BENCHMARK', 'MALFORMED'
    row_count: int
    missing_required_columns: List[str] = field(default_factory=list)
    invalid_labels: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        status_str = "PASSED [VALID]" if self.is_valid else "FAILED [INVALID]"
        lines = [
            "=" * 70,
            f"DATASET VALIDATION REPORT: {pathlib.Path(self.file_path).name}",
            "=" * 70,
            f"Status:          {status_str}",
            f"Dataset Type:    {self.dataset_type}",
            f"Row Count:       {self.row_count:,}",
        ]
        if self.missing_required_columns:
            lines.append(f"Missing Columns: {', '.join(self.missing_required_columns)}")
        if self.invalid_labels:
            lines.append(f"Invalid Labels:  {', '.join(self.invalid_labels[:10])}")

        if self.errors:
            lines.append("")
            lines.append("--- Fatal Errors ---")
            for e in self.errors:
                lines.append(f"  [x] {e}")

        if self.warnings:
            lines.append("")
            lines.append("--- Quality Warnings ---")
            for w in self.warnings:
                lines.append(f"  [!] {w}")

        lines.append("=" * 70)
        return "\n".join(lines)


def validate_dataset(
    file_path: Union[str, pathlib.Path],
    strict_canonical: bool = False,
    min_rows: int = 30,
) -> ValidationReport:
    """Validate a sensor dataset against schema rules and data quality standards.
    
    Args:
        file_path: Path to dataset CSV.
        strict_canonical: If True, requires physical hardware schema (MQ-3, MQ-4, MQ-135).
        min_rows: Minimum rows required for training viability.
    """
    p = pathlib.Path(file_path)
    if not p.exists():
        return ValidationReport(
            file_path=str(p),
            is_valid=False,
            dataset_type="MALFORMED",
            row_count=0,
            errors=[f"File not found: {p}"],
        )

    errors: List[str] = []
    warnings: List[str] = []
    missing_cols: List[str] = []
    invalid_labels: List[str] = []

    try:
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                return ValidationReport(
                    file_path=str(p),
                    is_valid=False,
                    dataset_type="MALFORMED",
                    row_count=0,
                    errors=["Dataset is empty or has no header row."],
                )
            rows = list(reader)
    except Exception as e:
        return ValidationReport(
            file_path=str(p),
            is_valid=False,
            dataset_type="MALFORMED",
            row_count=0,
            errors=[f"Failed to read CSV: {e}"],
        )

    row_count = len(rows)
    if row_count == 0:
        return ValidationReport(
            file_path=str(p),
            is_valid=False,
            dataset_type="MALFORMED",
            row_count=0,
            errors=["Dataset has zero data rows."],
        )

    # 1. Dataset Type Classification
    is_canonical = all(c in fieldnames for c in ("mq3", "mq4", "mq135"))
    is_legacy = "mq2" in fieldnames and "mq135" in fieldnames and not is_canonical

    if is_canonical:
        dataset_type = "CANONICAL_HARDWARE"
        required_cols = CANONICAL_CSV_COLUMNS
    elif is_legacy:
        dataset_type = "LEGACY_PROTOTYPE"
        required_cols = LEGACY_CSV_COLUMNS
        if strict_canonical:
            errors.append(
                "Dataset uses legacy prototype schema (contains 'mq2'). "
                "Canonical hardware pipeline requires dedicated 'mq3' and 'mq4' channels."
            )
        else:
            warnings.append(
                "Dataset uses legacy prototype schema (MQ-2/MQ-135). "
                "Valid for legacy testing only; not representative of final MQ-3/MQ-4 hardware."
            )
    else:
        if strict_canonical:
            dataset_type = "MALFORMED"
            required_cols = CANONICAL_CSV_COLUMNS
            errors.append("Dataset does not match canonical hardware schema (missing mq3, mq4, or mq135).")
        else:
            has_sensors = any("gas" in c.lower() or "sensor" in c.lower() or "mq" in c.lower() for c in fieldnames)
            has_label = "label" in fieldnames or "target" in fieldnames
            if has_sensors and has_label:
                dataset_type = "EXTERNAL_BENCHMARK"
                required_cols = []
                warnings.append("Identified as external benchmark dataset. Use ExternalBenchmarkAdapter before training.")
            else:
                dataset_type = "MALFORMED"
                required_cols = CANONICAL_CSV_COLUMNS

    # 2. Check required columns if canonical, legacy, or malformed
    if dataset_type in ("CANONICAL_HARDWARE", "LEGACY_PROTOTYPE", "MALFORMED"):
        for col in required_cols:
            if col not in fieldnames:
                missing_cols.append(col)
                errors.append(f"Required column '{col}' is missing.")

    # 3. Minimum row count check
    if row_count < min_rows:
        errors.append(f"Insufficient rows: {row_count} < minimum required {min_rows}.")

    # 4. Label Validation
    label_col = "label" if "label" in fieldnames else ("target" if "target" in fieldnames else None)
    if not label_col:
        errors.append("No label column found ('label' or 'target').")
    else:
        valid_label_names = set(LABEL_NAMES)
        valid_label_codes = {str(k) for k in LABELS.keys()}
        found_labels = set()

        for idx, r in enumerate(rows):
            lbl = r.get(label_col)
            if lbl is None or lbl.strip() == "":
                errors.append(f"Row {idx + 1} has missing/empty label.")
                break
            clean_lbl = lbl.strip()
            found_labels.add(clean_lbl)
            if clean_lbl not in valid_label_names and clean_lbl not in valid_label_codes:
                if clean_lbl not in invalid_labels:
                    invalid_labels.append(clean_lbl)

        if invalid_labels:
            errors.append(
                f"Unsupported class labels found: {invalid_labels[:5]}. "
                f"Expected label codes {list(LABELS.keys())} or names {LABEL_NAMES}."
            )

        if len(found_labels) < 2:
            errors.append(
                f"Dataset contains only {len(found_labels)} unique class: {found_labels}. "
                "Multi-class classification requires at least 2 distinct classes."
            )

    # 5. Timestamp Monotonicity and Session Integrity
    if "ts" in fieldnames and "run_id" in fieldnames:
        last_ts = None
        last_run = None
        seen_timestamps = set()
        duplicate_ts_count = 0

        for idx, r in enumerate(rows):
            run_val = r.get("run_id")
            ts_str = r.get("ts")
            if ts_str is not None and ts_str.strip():
                try:
                    ts = float(ts_str)
                    if (run_val, ts) in seen_timestamps:
                        duplicate_ts_count += 1
                    seen_timestamps.add((run_val, ts))

                    if last_run == run_val and last_ts is not None and ts < last_ts:
                        warnings.append(
                            f"Timestamp non-monotonic in run {run_val} at row {idx + 1}: "
                            f"{ts} < {last_ts}."
                        )
                    last_ts = ts
                    last_run = run_val
                except ValueError:
                    errors.append(f"Row {idx + 1} contains non-numeric timestamp: '{ts_str}'.")

        if duplicate_ts_count > 0:
            warnings.append(f"Found {duplicate_ts_count} duplicate (run_id, ts) timestamp records.")

    # 6. Physical Numerical Bounds & Sensor Integrity
    critical_to_check = ("mq3", "mq4", "mq135") if is_canonical else ("mq135", "mq2")
    for s in critical_to_check:
        if s in fieldnames:
            non_numeric = 0
            nan_or_inf = 0
            negative_count = 0
            values = []

            for idx, r in enumerate(rows):
                v_str = r.get(s)
                if v_str is not None and v_str.strip() not in ("", "null", "none"):
                    try:
                        v = float(v_str)
                        if math.isnan(v) or math.isinf(v):
                            nan_or_inf += 1
                        elif v < 0.0:
                            negative_count += 1
                        else:
                            values.append(v)
                    except ValueError:
                        non_numeric += 1

            if non_numeric > 0:
                errors.append(f"Sensor '{s}' has {non_numeric} non-numeric values.")
            if nan_or_inf > 0:
                errors.append(f"Sensor '{s}' has {nan_or_inf} NaN or Infinite values.")
            if negative_count > 0:
                warnings.append(f"Sensor '{s}' has {negative_count} negative readings.")
            if len(values) > 1 and max(values) == min(values):
                warnings.append(f"Sensor '{s}' is completely constant ({values[0]}) across all rows.")

    is_valid = len(errors) == 0
    return ValidationReport(
        file_path=str(p.resolve()),
        is_valid=is_valid,
        dataset_type=dataset_type,
        row_count=row_count,
        missing_required_columns=missing_cols,
        invalid_labels=invalid_labels,
        errors=errors,
        warnings=warnings,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate a sensor dataset against schema contracts.")
    parser.add_argument("file_path", help="Path to the dataset CSV file")
    parser.add_argument("--strict-canonical", action="store_true", help="Enforce physical MQ-3, MQ-4, MQ-135 hardware schema")
    parser.add_argument("--min-rows", type=int, default=30, help="Minimum required data rows (default: 30)")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")
    args = parser.parse_args()

    report = validate_dataset(args.file_path, strict_canonical=args.strict_canonical, min_rows=args.min_rows)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_text())

    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
