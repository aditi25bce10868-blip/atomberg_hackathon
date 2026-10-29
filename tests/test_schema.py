"""Unit tests for data contracts, sensor schemas, and validation."""
import time
from common.schema import (
    SensorReading,
    PredictionResult,
    DiagnosticEvent,
    validate_sensor_reading,
    SENSORS,
    CRITICAL,
    TARGET_ADC_CHANNELS,
)


def test_sensor_reading_dataclass():
    reading = SensorReading(
        ts=1700000000.0,
        sensors={"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 28.0, "pir": 0},
        health={"mq135": "ok", "mq2": "ok", "humidity": "ok", "temp_c": "ok", "pir": "ok"},
        override=False,
        session_id="run_01",
        raw_adc={"mq135": 307, "mq2": 204},
        source="SIMULATED",
    )
    d = reading.to_dict()
    assert d["ts"] == 1700000000.0
    assert d["sensors"]["mq135"] == 1.5
    assert d["source"] == "SIMULATED"
    assert d["session_id"] == "run_01"
    assert d["raw_adc"]["mq135"] == 307

    # Round trip
    reconstructed = SensorReading.from_dict(d)
    assert reconstructed.ts == reading.ts
    assert reconstructed.sensors == reading.sensors
    assert reconstructed.session_id == "run_01"


def test_raw_adc_separation_from_measurements():
    """Ensure raw ADC counts are never confused with calibrated sensor values."""
    reading = SensorReading(
        sensors={"mq135": 1.52},
        raw_adc={"mq135": 311},
    )
    # Calibrated/normalized reading is a float in unit space
    assert reading.sensors["mq135"] == 1.52
    # Raw ADC count is integer count in [0, 1023]
    assert reading.raw_adc["mq135"] == 311
    assert reading.raw_adc["mq135"] != reading.sensors["mq135"]


def test_prediction_result_dataclass():
    res = PredictionResult(
        label="steam",
        confidence=0.88,
        probs={"clean": 0.05, "steam": 0.88, "cooking": 0.05, "foul": 0.02},
        source="model:model.joblib",
    )
    d = res.to_dict()
    assert d["label"] == "steam"
    assert d["confidence"] == 0.88
    assert d["probs"]["steam"] == 0.88
    assert d["source"] == "model:model.joblib"


def test_validate_sensor_reading_success():
    valid_sensors = {"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 28.0, "pir": 0}
    is_valid, errors = validate_sensor_reading(valid_sensors)
    assert is_valid is True
    assert errors == []


def test_validate_sensor_reading_missing_critical():
    incomplete = {"mq135": 1.5, "mq2": 1.0}
    is_valid, errors = validate_sensor_reading(incomplete)
    assert is_valid is False
    assert any("humidity" in e for e in errors)
    assert any("temp_c" in e for e in errors)


def test_validate_sensor_reading_invalid_type():
    bad_type = {"mq135": "corrupted_string", "mq2": 1.0, "humidity": 55.0, "temp_c": 28.0}
    is_valid, errors = validate_sensor_reading(bad_type)
    assert is_valid is False
    assert any("Invalid sensor value type" in e for e in errors)


def test_target_adc_channel_mapping():
    assert TARGET_ADC_CHANNELS[0] == "mq3"
    assert TARGET_ADC_CHANNELS[1] == "mq4"
    assert TARGET_ADC_CHANNELS[2] == "mq135"


def test_validate_sensor_reading_canonical_success():
    canonical_sensors = {"mq3": 1.2, "mq4": 1.1, "mq135": 1.5, "humidity": 55.0, "temp_c": 28.0, "pir": 0}
    is_valid, errors = validate_sensor_reading(canonical_sensors)
    assert is_valid is True
    assert errors == []

