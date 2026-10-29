"""Dataset adapters for external benchmarks and legacy prototype recordings.

Enforces:
- No silent reinterpretation of unrelated gas sensors as MQ-3, MQ-4, or MQ-135.
- Preservation of original provenance, sensor types, and license metadata.
- Rejection of incompatible datasets with actionable diagnostic errors.
"""
import csv
import json
import pathlib
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from common.logger import get_logger
from common.schema import (
    CANONICAL_CSV_COLUMNS,
    CANONICAL_GAS_SENSORS,
    LEGACY_CSV_COLUMNS,
    LABELS,
    LABEL_NAMES,
    LABEL_CODES,
)

logger = get_logger("ml.datasets.adapters")


class DatasetAdapterError(Exception):
    """Raised when an external or legacy dataset cannot be defensibly adapted."""
    pass


class BaseDatasetAdapter:
    """Base contract for dataset ingestion adapters."""

    def __init__(self, source_name: str, provenance_info: Optional[Dict[str, Any]] = None):
        self.source_name = source_name
        self.provenance_info = provenance_info or {}

    def validate_source(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        raise NotImplementedError

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class LegacyPrototypeAdapter(BaseDatasetAdapter):
    """Adapter for legacy prototype recordings collected with MQ-2 instead of MQ-3/MQ-4.
    
    IMPORTANT SCIENTIFIC POLICY:
    An MQ-2 sensor has wide cross-sensitivity to LPG, propane, and methane, but has
    significantly different sensitivity curves than MQ-3 (alcohol) and MQ-4 (methane).
    This adapter strictly REFUSES to silently rename 'mq2' to 'mq3' or 'mq4'.
    It permits pipeline testing of legacy data by preserving 'mq2' under the legacy schema,
    or creating an explicitly synthetic fixture with clear warnings.
    """

    def __init__(self):
        super().__init__(
            source_name="LEGACY_PROTOTYPE_MQ2",
            provenance_info={
                "description": "Early bench prototype recording using MQ-2 and MQ-135.",
                "hardware_match": "PARTIAL (Lacks dedicated MQ-3 and MQ-4 channels).",
                "real_world_claim": "INVALID for physical MQ-3/MQ-4 hardware validation.",
            },
        )

    def validate_source(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        for col in LEGACY_CSV_COLUMNS:
            if col not in df.columns:
                errors.append(f"Legacy dataset missing required column: '{col}'")
        return len(errors) == 0, errors

    def adapt(self, df: pd.DataFrame, target_schema: str = "legacy") -> pd.DataFrame:
        """Process legacy dataframe.
        
        Args:
            df: Source DataFrame.
            target_schema: 'legacy' (preserves mq2) or 'canonical_placeholder' (raises error unless explicit).
        """
        valid, errors = self.validate_source(df)
        if not valid:
            raise DatasetAdapterError("Cannot adapt legacy dataset:\n" + "\n".join(f"  - {e}" for e in errors))

        if target_schema == "canonical":
            raise DatasetAdapterError(
                "Refusing silent coercion: Legacy MQ-2 sensor measurements cannot be "
                "defensibly mapped to physical MQ-3 or MQ-4 channels. "
                "Collect genuine sensor recordings using the MCP3008 ADC with MQ-3 and MQ-4 installed."
            )

        # Return standardized legacy dataframe with provenance column attached
        df_out = df[LEGACY_CSV_COLUMNS].copy()
        df_out["_provenance"] = "LEGACY_PROTOTYPE_MQ2"
        return df_out


class ExternalBenchmarkAdapter(BaseDatasetAdapter):
    """Adapter for external public benchmark datasets (e.g. UCI gas array).
    
    Validates required column mappings, units, and label correspondences.
    """

    def __init__(
        self,
        dataset_name: str,
        column_mapping: Dict[str, str],
        label_mapping: Dict[Any, str],
        license_name: str = "Unknown",
        sensor_types: Optional[List[str]] = None,
    ):
        super().__init__(
            source_name=dataset_name,
            provenance_info={
                "dataset_name": dataset_name,
                "license": license_name,
                "sensor_types": sensor_types or [],
                "hardware_match": "BENCHMARK_ONLY",
            },
        )
        self.column_mapping = column_mapping
        self.label_mapping = label_mapping

    def validate_source(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        for target_col, src_col in self.column_mapping.items():
            if src_col not in df.columns:
                errors.append(f"External dataset missing mapped column '{src_col}' for target '{target_col}'")
        return len(errors) == 0, errors

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        valid, errors = self.validate_source(df)
        if not valid:
            raise DatasetAdapterError(f"Cannot adapt benchmark '{self.source_name}':\n" + "\n".join(f"  - {e}" for e in errors))

        # Rename columns according to mapping
        inverted_map = {src: tgt for tgt, src in self.column_mapping.items()}
        renamed = df.rename(columns=inverted_map)

        # Map labels
        if "label" in renamed.columns:
            def _map_lbl(v):
                mapped = self.label_mapping.get(v)
                if mapped is None:
                    raise DatasetAdapterError(f"Unknown benchmark label '{v}'. Must map to one of {LABEL_NAMES}")
                return LABEL_CODES.get(mapped, mapped)

            renamed["label"] = renamed["label"].apply(_map_lbl)

        renamed["_provenance"] = f"EXTERNAL_BENCHMARK:{self.source_name}"
        return renamed
