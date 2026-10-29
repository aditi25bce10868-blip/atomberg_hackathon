"""Single source of truth for system data contracts, sensor schemas, and inference states."""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import math
import time

SCHEMA_VERSION = 2

# ==============================================================================
# CANONICAL HARDWARE SENSOR DEFINITIONS (Raspberry Pi 5 + MCP3008 ADC)
# ==============================================================================
# The canonical physical hardware uses three distinct metal oxide gas sensors:
# - MQ-3:   Alcohol, benzene, hexane, ethanol (CH0 on MCP3008)
# - MQ-4:   Methane, natural gas, CNG, smoke (CH1 on MCP3008)
# - MQ-135: Air quality: NH3, NOx, benzene, CO2, smoke (CH2 on MCP3008)
# All gas sensors output analog voltages (conditioned to 0.0-3.3V range via divider).
# Raw ADC counts (0-1023) MUST NOT be represented as calibrated gas concentrations.
CANONICAL_GAS_SENSORS = ("mq3", "mq4", "mq135")

# Optional environmental context and presence sensors:
# - humidity: Relative humidity (%RH, 0-100)
# - temp_c:   Ambient temperature (°C)
# - pir:      Passive infrared motion/presence (0 or 1)
CANONICAL_OPTIONAL_SENSORS = ("humidity", "temp_c", "pir")

# Full canonical sensor array
CANONICAL_SENSORS = ["mq3", "mq4", "mq135", "humidity", "temp_c", "pir"]
CANONICAL_CRITICAL = ("mq3", "mq4", "mq135")  # Physical automation requires all 3 gas sensors

# Standard CSV columns for canonical recorded datasets
CANONICAL_CSV_COLUMNS = ["run_id", "ts", "mq3", "mq4", "mq135", "humidity", "temp_c", "pir", "label"]

# ==============================================================================
# LEGACY PROTOTYPE DEFINITIONS (Used in initial bench simulations & replays)
# ==============================================================================
# The early prototype used an MQ-2 general combustible gas sensor in place of MQ-3/MQ-4.
# Legacy recordings and tests must be handled distinctly without silent column coercion.
LEGACY_PROTOTYPE_SENSORS = ["mq135", "mq2", "humidity", "temp_c", "pir"]
LEGACY_CRITICAL = ("mq135", "mq2", "humidity", "temp_c")
LEGACY_CSV_COLUMNS = ["run_id", "ts", "mq135", "mq2", "humidity", "temp_c", "pir", "label"]

# Active defaults: SENSORS and CRITICAL alias canonical, while supporting legacy detection
SENSORS = list(CANONICAL_SENSORS)
CRITICAL = tuple(CANONICAL_CRITICAL)
CSV_COLUMNS = list(CANONICAL_CSV_COLUMNS)

# Physical ADC channel mapping on MCP3008 (SPI 0.0)
TARGET_ADC_CHANNELS = {
    0: "mq3",     # CH0: MQ-3 (Alcohol / VOC)
    1: "mq4",     # CH1: MQ-4 (Methane / Natural Gas)
    2: "mq135",   # CH2: MQ-135 (Air Quality / NH3 / Smoke / CO2)
}

# ==============================================================================
# ENVIRONMENTAL CLASSIFICATION LABELS
# ==============================================================================
LABELS = {0: "clean", 1: "steam", 2: "cooking", 3: "foul"}
LABEL_NAMES = list(LABELS.values())
LABEL_CODES = {v: k for k, v in LABELS.items()}

# State machine states
STATES = ["WARMUP", "IDLE", "RUNNING", "RUN-ON", "FAULT", "MANUAL"]


# ==============================================================================
# INFERENCE STATUS CONTRACT
# ==============================================================================
class InferenceStatus:
    """Explicit lifecycle status for model inference and prediction provenance."""
    VALID_MODEL = "VALID_MODEL"              # Genuine prediction from validated trained model
    RULES_FALLBACK = "RULES_FALLBACK"        # Heuristic rules fallback (provisional/testing only)
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"  # Model bundle file missing on disk
    MODEL_ERROR = "MODEL_ERROR"              # Corrupted artifact, unpickling failure, or schema mismatch
    INPUT_INVALID = "INPUT_INVALID"          # Missing, stale, NaN, or out-of-range sensor readings
    SENSOR_FAULT = "SENSOR_FAULT"            # Physical ADC/sensor read failure


# ==============================================================================
# DATA CONTRACTS
# ==============================================================================
@dataclass
class SensorReading:
    """Structured representation of a sensor acquisition event.
    
    Attributes:
        ts: Unix epoch timestamp in seconds.
        sensors: Mapping of sensor name to conditioned value (volts or unit scale).
        health: Mapping of sensor name to health state ('ok', 'fault', 'stale').
        override: Whether the manual wall-switch override is active.
        session_id: Optional session or run identifier.
        raw_adc: Optional raw 10-bit integer ADC counts (0-1023). Raw ADC counts
                 must NEVER be represented directly as calibrated gas concentrations.
        source: Provenance indicator ('SIMULATED', 'REPLAY', 'LIVE', etc.).
    """
    ts: float = field(default_factory=time.time)
    sensors: Dict[str, Optional[float]] = field(default_factory=dict)
    health: Dict[str, str] = field(default_factory=dict)
    override: bool = False
    session_id: Optional[str] = None
    raw_adc: Optional[Dict[str, int]] = None
    source: str = "SIMULATED"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ts": self.ts,
            "sensors": dict(self.sensors),
            "health": dict(self.health),
            "override": self.override,
            "source": self.source,
        }
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.raw_adc is not None:
            d["raw_adc"] = dict(self.raw_adc)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorReading":
        return cls(
            ts=float(data.get("ts", time.time())),
            sensors=data.get("sensors", {}),
            health=data.get("health", {}),
            override=bool(data.get("override", False)),
            session_id=data.get("session_id"),
            raw_adc=data.get("raw_adc"),
            source=str(data.get("source", "SIMULATED")),
        )


@dataclass
class PredictionResult:
    """Structured output from model inference, explicit fallback, or fault handler.
    
    Provenance and confidence rules:
    - If status == VALID_MODEL: confidence represents genuine model probability [0.0, 1.0].
    - If status != VALID_MODEL: confidence MUST BE None or clearly marked uncalibrated.
      A placeholder or error must NEVER fabricate calibrated confidence.
    """
    status: str = InferenceStatus.RULES_FALLBACK
    label: Optional[str] = None
    confidence: Optional[float] = None
    probs: Optional[Dict[str, float]] = None
    source: str = "rules-placeholder"
    model_id: Optional[str] = None
    error: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "label": self.label,
            "confidence": self.confidence,
            "source": self.source,
            "ts": self.ts,
        }
        if self.probs is not None:
            d["probs"] = dict(self.probs)
        if self.model_id is not None:
            d["model_id"] = self.model_id
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionResult":
        return cls(
            status=d.get("status", InferenceStatus.RULES_FALLBACK),
            label=d.get("label"),
            confidence=d.get("confidence"),
            probs=d.get("probs"),
            source=d.get("source", "rules-placeholder"),
            model_id=d.get("model_id"),
            error=d.get("error"),
            ts=float(d.get("ts", time.time())),
        )


@dataclass
class DiagnosticEvent:
    """Structured operational event / log."""
    ts: float
    level: str  # 'info', 'warn', 'error'
    kind: str   # 'start', 'fault', 'fan_on', etc.
    msg: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# VALIDATION UTILITIES
# ==============================================================================
def validate_sensor_reading(
    sensors: Dict[str, Any],
    required_sensors: Optional[Tuple[str, ...]] = None,
) -> Tuple[bool, List[str]]:
    """Validate a dictionary of sensor measurements against physical bounds.
    
    Checks:
    - Required sensors are present (auto-detects canonical vs legacy if not specified).
    - Values are numeric (int/float/None for failed reads).
    - Values are finite (no NaN or inf).
    - Non-negative voltage/concentration values.
    """
    errors: List[str] = []
    if required_sensors is not None:
        req = required_sensors
    elif "mq3" in sensors or "mq4" in sensors:
        req = CANONICAL_CRITICAL
    elif "mq2" in sensors:
        req = LEGACY_CRITICAL
    else:
        req = CANONICAL_CRITICAL

    for s in req:
        if s not in sensors:
            errors.append(f"Missing required sensor channel: '{s}'")
        else:
            v = sensors[s]
            if v is not None:
                if not isinstance(v, (int, float)):
                    errors.append(f"Invalid sensor value type for '{s}': expected numeric, got {type(v).__name__}")
                elif math.isnan(v) or math.isinf(v):
                    errors.append(f"Sensor '{s}' has non-finite value: {v}")
                elif v < 0.0:
                    errors.append(f"Sensor '{s}' has negative reading: {v}")

    # Check any other provided sensor channels for non-numeric or non-finite values
    for s, v in sensors.items():
        if s not in req and v is not None:
            if not isinstance(v, (int, float)):
                errors.append(f"Invalid sensor value type for '{s}': expected numeric, got {type(v).__name__}")
            elif math.isnan(v) or math.isinf(v):
                errors.append(f"Sensor '{s}' has non-finite value: {v}")

    # Check bounds on optional sensors if present
    hum = sensors.get("humidity")
    if hum is not None and isinstance(hum, (int, float)):
        if not (0.0 <= hum <= 100.0):
            errors.append(f"Humidity value out of physical bounds [0, 100]: {hum}")

    temp = sensors.get("temp_c")
    if temp is not None and isinstance(temp, (int, float)):
        if not (-20.0 <= temp <= 80.0):
            errors.append(f"Temperature value out of realistic bounds [-20, 80]: {temp}")

    return len(errors) == 0, errors
