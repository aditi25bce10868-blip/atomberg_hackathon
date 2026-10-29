"""Unit tests for hardware interfaces, simulated adapters, and safety gates."""
import pytest
from hardware.interface import (
    BaseSensorHardware,
    BaseActuatorHardware,
    HardwareDisabledError,
    HardwareNotAvailableError,
)
from hardware.simulated import SimulatedHardwareAdapter
from hardware.mcp3008 import MCP3008Reader
import hardware.sensors as hw_sensors


def test_simulated_adapter_reading_and_determinism():
    adapter1 = SimulatedHardwareAdapter(seed=42)
    adapter2 = SimulatedHardwareAdapter(seed=42)

    r1 = adapter1.read_sensors()
    r2 = adapter2.read_sensors()

    assert r1.source == "SIMULATED"
    assert r1.sensors["mq135"] == r2.sensors["mq135"]
    assert r1.sensors["humidity"] == r2.sensors["humidity"]
    assert r1.raw_adc is not None
    assert 0 <= r1.raw_adc["mq135"] <= 1023


def test_simulated_adapter_fault_injection():
    adapter = SimulatedHardwareAdapter()
    adapter.inject_fault("mq135", "fault")

    r = adapter.read_sensors()
    assert r.sensors["mq135"] is None
    assert r.health["mq135"] == "fault"

    adapter.clear_faults()
    r_cleared = adapter.read_sensors()
    assert r_cleared.sensors["mq135"] is not None
    assert r_cleared.health["mq135"] == "ok"


def test_simulated_adapter_actuator_state():
    adapter = SimulatedHardwareAdapter()
    assert adapter.is_fan_on() is False

    adapter.set_fan(True)
    assert adapter.is_fan_on() is True

    adapter.close()
    assert adapter.is_fan_on() is False


def test_mcp3008_safe_import_and_availability():
    """Verify MCP3008 driver initializes safely without hardware dependencies."""
    reader = MCP3008Reader(bus=0, device=0)
    # On Windows / non-Pi systems without spidev, it safely reports False
    if not reader.is_available():
        with pytest.raises(HardwareNotAvailableError):
            reader.read_raw(0)


def test_hardware_sensors_safety_disabled_by_default():
    """Verify physical hardware access is strictly blocked when hardware_enabled is False."""
    hw_sensors.configure_hardware(hardware_enabled=False, fan_control_enabled=False)
    status = hw_sensors.check_hardware_availability()
    assert status["hardware_enabled"] is False
    assert status["ready_for_live"] is False

    # Calling read_sensors must raise HardwareDisabledError
    with pytest.raises(HardwareDisabledError):
        hw_sensors.read_sensors()


def test_hardware_sensors_fan_control_disabled_by_default():
    """Verify fan relay actuation is suppressed when fan_control_enabled is False."""
    hw_sensors.configure_hardware(hardware_enabled=False, fan_control_enabled=False)
    # Commanding set_fan(True) must not energize physical relay
    hw_sensors.set_fan(True)
    assert hw_sensors._fan_state is False
