"""Deterministic simulated sensor and relay adapter for development and testing."""
import math
import random
import time
from typing import Dict, Optional, Any
from common.schema import (
    SensorReading,
    CANONICAL_SENSORS,
    CANONICAL_GAS_SENSORS,
    TARGET_ADC_CHANNELS,
)
from hardware.interface import BaseSensorHardware, BaseActuatorHardware


class SimulatedHardwareAdapter(BaseSensorHardware, BaseActuatorHardware):
    """Simulates physical sensor acquisition and fan relay state deterministically.
    
    Clearly tagged as SIMULATED to prevent confusing simulated test runs with physical hardware.
    Generates signals for canonical gas sensors (MQ-3, MQ-4, MQ-135) and optional context sensors.
    """

    BASE_VALUES = {
        "mq3": 0.80,       # Alcohol / VOC (0-3.3V conditioned)
        "mq4": 1.00,       # Methane / Natural Gas (0-3.3V conditioned)
        "mq135": 1.50,     # Air Quality / NH3 / CO2 (0-3.3V conditioned)
        "mq2": 1.00,       # Legacy general combustible sensor
        "humidity": 55.0,  # %RH
        "temp_c": 28.0,    # °C
        "pir": 0,          # Presence
    }

    def __init__(self, seed: int = 42, tick_interval: float = 1.0):
        self.rng = random.Random(seed)
        self.tick_interval = tick_interval
        self._step_count = 0
        self._t0 = time.time()
        self._override = False
        self._fan_on = False
        self._injected_faults: Dict[str, str] = {}

    def is_available(self) -> bool:
        """Simulated hardware is always available."""
        return True

    def inject_fault(self, sensor_name: str, fault_type: str = "fault") -> None:
        """Inject a simulated sensor failure for testing fault-tolerance."""
        self._injected_faults[sensor_name] = fault_type

    def clear_faults(self) -> None:
        """Clear all injected faults."""
        self._injected_faults.clear()

    def set_override(self, active: bool) -> None:
        """Simulate physical wall-switch toggle."""
        self._override = active

    def read_override(self) -> bool:
        """Return simulated override state."""
        return self._override

    def read_sensors(self) -> SensorReading:
        """Return deterministic simulated sensor measurements."""
        self._step_count += 1
        current_time = self._t0 + (self._step_count * self.tick_interval)

        sensors: Dict[str, Optional[float]] = {}
        health: Dict[str, str] = {}
        raw_adc: Dict[str, int] = {}

        # Canonical sensors + legacy support
        all_channels = ("mq3", "mq4", "mq135", "mq2", "humidity", "temp_c", "pir")

        for sensor in all_channels:
            if sensor in self._injected_faults:
                sensors[sensor] = None
                health[sensor] = self._injected_faults[sensor]
                raw_adc[sensor] = 0
                continue

            base = self.BASE_VALUES.get(sensor, 1.0)
            if sensor == "pir":
                sensors["pir"] = 1 if (self._step_count % 30 == 0) else 0
                health["pir"] = "ok"
                raw_adc["pir"] = 1023 if sensors["pir"] == 1 else 0
            else:
                drift = 0.02 * base * math.sin(self._step_count / 50.0)
                noise = self.rng.gauss(0, 0.01 * base)
                val = round(base + drift + noise, 3)
                sensors[sensor] = max(0.0, val)
                health[sensor] = "ok"
                # Simulated 10-bit ADC count (0 to 1023 relative to 3.3V reference)
                raw_adc[sensor] = min(1023, max(0, int((val / 3.3) * 1023)))

        return SensorReading(
            ts=current_time,
            sensors=sensors,
            health=health,
            override=self._override,
            raw_adc=raw_adc,
            source="SIMULATED",
        )

    def check_health(self) -> Dict[str, str]:
        """Return current sensor health status."""
        health = {s: "ok" for s in ("mq3", "mq4", "mq135", "mq2", "humidity", "temp_c", "pir")}
        health.update(self._injected_faults)
        return health

    def set_fan(self, on: bool) -> None:
        """Track fan relay state in memory without driving physical pins."""
        self._fan_on = bool(on)

    def is_fan_on(self) -> bool:
        """Return commanded fan state."""
        return self._fan_on

    def close(self) -> None:
        """Turn off simulated fan."""
        self._fan_on = False
