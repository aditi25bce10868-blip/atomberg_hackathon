"""Dataset schema and statistical inspection tool."""
import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union

from common.schema import SENSORS, LABELS, LABEL_NAMES, LABEL_CODES


@dataclass
class SensorStats:
    count: int = 0
    missing: int = 0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


@dataclass
class DatasetSummary:
    file_path: str
    file_size_bytes: int
    row_count: int
    column_names: List[str]
    column_types: Dict[str, str]
    missing_counts: Dict[str, int]
    label_distribution: Dict[str, int]
    session_count: int
    session_ids: List[Any]
    time_span_s: float
    estimated_sampling_hz: float
    sensor_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        lines = [
            "=" * 68,
            f"DATASET INSPECTION REPORT: {pathlib.Path(self.file_path).name}",
            "=" * 68,
            f"Path:            {self.file_path}",
            f"Size:            {self.file_size_bytes:,} bytes",
            f"Rows:            {self.row_count:,}",
            f"Columns ({len(self.column_names)}):     {', '.join(self.column_names)}",
            f"Sessions:        {self.session_count} unique run(s)",
            f"Time Span:       {self.time_span_s:.1f} seconds (~{self.time_span_s / 60.0:.1f} min)",
            f"Sampling Rate:   ~{self.estimated_sampling_hz:.2f} Hz",
            "",
            "--- Label Distribution ---",
        ]
        if self.label_distribution:
            for lbl, cnt in sorted(self.label_distribution.items()):
                pct = (cnt / self.row_count * 100.0) if self.row_count > 0 else 0.0
                lines.append(f"  {lbl:<12}: {cnt:>6} ({pct:>5.1f}%)")
        else:
            lines.append("  (No labels found)")

        lines.append("")
        lines.append("--- Sensor Statistics ---")
        lines.append(f"{'Sensor':<12} {'Count':>7} {'Missing':>8} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std':>8}")
        lines.append("-" * 68)
        for s, stats in self.sensor_statistics.items():
            lines.append(
                f"{s:<12} {stats.get('count', 0):>7} {stats.get('missing', 0):>8} "
                f"{stats.get('min', 0.0):>8.2f} {stats.get('mean', 0.0):>8.2f} "
                f"{stats.get('max', 0.0):>8.2f} {stats.get('std', 0.0):>8.2f}"
            )

        if self.warnings:
            lines.append("")
            lines.append("--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  [!] {w}")

        lines.append("=" * 68)
        return "\n".join(lines)


def inspect_dataset(file_path: Union[str, pathlib.Path]) -> DatasetSummary:
    """Inspect a CSV dataset and return comprehensive schema and statistical summary."""
    p = pathlib.Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {p}")

    size_bytes = p.stat().st_size
    warnings: List[str] = []

    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Dataset is empty or has no header row: {p}")

        columns = list(reader.fieldnames)
        rows = list(reader)

    row_count = len(rows)
    if row_count == 0:
        return DatasetSummary(
            file_path=str(p),
            file_size_bytes=size_bytes,
            row_count=0,
            column_names=columns,
            column_types={},
            missing_counts={c: 0 for c in columns},
            label_distribution={},
            session_count=0,
            session_ids=[],
            time_span_s=0.0,
            estimated_sampling_hz=0.0,
            sensor_statistics={},
            warnings=["Dataset has headers but zero data rows."],
        )

    # 1. Missing counts and types
    missing_counts = {c: 0 for c in columns}
    column_types: Dict[str, str] = {}
    timestamps: List[float] = []
    labels: List[str] = []
    session_ids: set = set()
    potential_sensors = ("mq3", "mq4", "mq135", "mq2", "humidity", "temp_c", "pir")
    sensor_vals: Dict[str, List[float]] = {s: [] for s in potential_sensors if s in columns}

    for idx, r in enumerate(rows):
        for c in columns:
            v = r.get(c)
            if v is None or v.strip() == "" or v.strip().lower() in ("nan", "null", "none"):
                missing_counts[c] += 1

        # Sessions
        if "run_id" in r and r["run_id"]:
            session_ids.add(r["run_id"])

        # Timestamps
        if "ts" in r and r["ts"]:
            try:
                timestamps.append(float(r["ts"]))
            except ValueError:
                pass

        # Labels
        if "label" in r and r["label"] is not None:
            raw_lbl = r["label"].strip()
            # Try numeric label code
            try:
                code = int(raw_lbl)
                lbl_name = LABELS.get(code, f"unknown_{code}")
            except ValueError:
                lbl_name = raw_lbl
            labels.append(lbl_name)

        # Numerical sensor columns
        for s in sensor_vals:
            val_str = r.get(s)
            if val_str is not None and val_str.strip() not in ("", "nan", "null", "none"):
                try:
                    sensor_vals[s].append(float(val_str))
                except ValueError:
                    pass

    # Type detection
    for c in columns:
        if c in ("ts", "run_id") or c in SENSORS:
            column_types[c] = "numeric"
        elif c == "label":
            column_types[c] = "categorical"
        else:
            column_types[c] = "text"

    # Label distribution
    label_dist: Dict[str, int] = {}
    for lbl in labels:
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    # Time span & sampling rate
    time_span = 0.0
    sampling_hz = 0.0
    if len(timestamps) >= 2:
        timestamps.sort()
        time_span = max(0.0, timestamps[-1] - timestamps[0])
        if time_span > 0:
            sampling_hz = (len(timestamps) - 1) / time_span

    # Sensor stats
    sensor_stats: Dict[str, Dict[str, Any]] = {}
    for s, vals in sensor_vals.items():
        missing = missing_counts.get(s, 0)
        if vals:
            cnt = len(vals)
            mn = min(vals)
            mx = max(vals)
            mean = sum(vals) / cnt
            variance = sum((x - mean) ** 2 for x in vals) / cnt if cnt > 1 else 0.0
            std = math.sqrt(variance)
            sensor_stats[s] = {
                "count": cnt,
                "missing": missing,
                "min": round(mn, 3),
                "max": round(mx, 3),
                "mean": round(mean, 3),
                "std": round(std, 3),
            }
        else:
            sensor_stats[s] = {"count": 0, "missing": missing, "min": None, "max": None, "mean": None, "std": None}

    # Warnings checks
    if row_count < 60:
        warnings.append(f"Small dataset size ({row_count} rows). Insufficient for robust ML training.")
    if len(label_dist) < 2:
        warnings.append(f"Only {len(label_dist)} class label(s) present. Multi-class classification requires at least 2 distinct classes.")
    for s, stats in sensor_stats.items():
        if stats.get("missing", 0) > 0.1 * row_count:
            warnings.append(f"Sensor '{s}' has high missingness ({stats['missing']}/{row_count} rows).")
        if stats.get("std", 0.0) == 0.0 and stats.get("count", 0) > 1:
            warnings.append(f"Sensor '{s}' is completely constant across all rows.")

    return DatasetSummary(
        file_path=str(p.resolve()),
        file_size_bytes=size_bytes,
        row_count=row_count,
        column_names=columns,
        column_types=column_types,
        missing_counts=missing_counts,
        label_distribution=label_dist,
        session_count=len(session_ids) if session_ids else 1,
        session_ids=sorted(list(session_ids)),
        time_span_s=round(time_span, 2),
        estimated_sampling_hz=round(sampling_hz, 2),
        sensor_statistics=sensor_stats,
        warnings=warnings,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect a sensor dataset schema and properties.")
    parser.add_argument("file_path", help="Path to the dataset CSV file")
    parser.add_argument("--json", action="store_true", help="Output summary in JSON format")
    args = parser.parse_args()

    try:
        summary = inspect_dataset(args.file_path)
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(summary.format_text())
    except Exception as e:
        print(f"Error inspecting dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
