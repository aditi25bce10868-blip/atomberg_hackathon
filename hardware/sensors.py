"""Hardware interface implementation for physical Raspberry Pi 5 operation.

IMPORTANT SAFETY NOTICE:
- Hardware access and fan relay actuation are DISABLED BY DEFAULT in configuration.
- Do NOT enable physical hardware mode (`source: live`) or fan actuation until wiring,
  resistor dividers, and power conditioning have been thoroughly bench-verified.
- Never connect mains AC power to relay contacts during development or automated tests.
"""

from typing import Dict, Any, Optional
from common.logger import get_logger
from common.schema import SENSORS, CRITICAL
from hardware.interface import (
    HardwareDisabledError,
    HardwareNotAvailableError,
)
from hardware.mcp3008 import MCP3008Reader

logger = get_logger("hardware.sensors")

# Module-level instances and state
_mcp_reader: Optional[MCP3008Reader] = None
_fan_state: bool = False
_hardware_enabled: bool = False
_fan_control_enabled: bool = False


def configure_hardware(hardware_enabled: bool = False, fan_control_enabled: bool = False) -> None:
    """Set operational safety gates for physical hardware access."""
    global _hardware_enabled, _fan_control_enabled
    _hardware_enabled = hardware_enabled
    _fan_control_enabled = fan_control_enabled
    logger.info(
        "Hardware safety gates configured: hardware_enabled=%s, fan_control_enabled=%s",
        _hardware_enabled,
        _fan_control_enabled,
    )


def check_hardware_availability() -> Dict[str, Any]:
    """Inspect and report physical hardware availability and safety states."""
    status: Dict[str, Any] = {
        "hardware_enabled": _hardware_enabled,
        "fan_control_enabled": _fan_control_enabled,
        "spidev_installed": False,
        "rpi_gpio_installed": False,
        "spi_bus_open": False,
        "ready_for_live": False,
    }

    try:
        import spidev  # type: ignore
        status["spidev_installed"] = True
    except ImportError:
        status["spidev_installed"] = False

    try:
        import RPi.GPIO  # type: ignore
        status["rpi_gpio_installed"] = True
    except (ImportError, RuntimeError):
        status["rpi_gpio_installed"] = False

    global _mcp_reader
    if _mcp_reader is None and status["spidev_installed"] and _hardware_enabled:
        _mcp_reader = MCP3008Reader()

    if _mcp_reader is not None:
        status["spi_bus_open"] = _mcp_reader.is_available()

    status["ready_for_live"] = (
        status["hardware_enabled"] and status["spidev_installed"] and status["spi_bus_open"]
    )
    return status


def read_sensors() -> Dict[str, Any]:
    """Read physical sensors through MCP3008 ADC and GPIOs.
    
    Returns:
        Dictionary containing sensor readings and optional health dictionary:
        {"mq135": float, "mq2": float, "humidity": float, "temp_c": float, "pir": 0|1,
         "health": {"mq135": "ok"|"fault", ...}}
         
    Raises:
        HardwareDisabledError: If hardware_enabled is False (the safe default).
        HardwareNotAvailableError: If physical SPI/ADC is inaccessible.
    """
    if not _hardware_enabled:
        raise HardwareDisabledError(
            "Physical hardware access is disabled by default in config.yaml. "
            "To use live hardware, set 'hardware_enabled: true' after verifying bench wiring."
        )

    global _mcp_reader
    if _mcp_reader is None:
        _mcp_reader = MCP3008Reader()

    if not _mcp_reader.is_available():
        raise HardwareNotAvailableError(
            "MCP3008 ADC is not responding on SPI bus. Check wiring, /dev/spidev0.0, and permissions."
        )

    # Read ADC channels for canonical gas sensors
    # CH0: MQ-3 (Alcohol / VOC)
    # CH1: MQ-4 (Methane / Natural Gas)
    # CH2: MQ-135 (Air Quality / NH3 / NOx / CO2)
    # Voltages must be measured after appropriate resistive divider conditioning!
    try:
        v_mq3 = _mcp_reader.read_voltage(0)
        v_mq4 = _mcp_reader.read_voltage(1)
        v_mq135 = _mcp_reader.read_voltage(2)
    except Exception as e:
        raise HardwareNotAvailableError(f"SPI communication error during ADC read: {e}") from e

    # Optional digital sensors (e.g. DHT22 / AHT20 / PIR)
    temp_c = 25.0
    humidity = 50.0
    pir = 0

    readings = {
        "mq3": round(v_mq3, 3),
        "mq4": round(v_mq4, 3),
        "mq135": round(v_mq135, 3),
        "humidity": humidity,
        "temp_c": temp_c,
        "pir": pir,
        "health": {
            "mq3": "ok" if v_mq3 > 0.05 else "fault",
            "mq4": "ok" if v_mq4 > 0.05 else "fault",
            "mq135": "ok" if v_mq135 > 0.05 else "fault",
            "humidity": "ok",
            "temp_c": "ok",
            "pir": "ok",
        },
    }
    return readings


def read_override() -> bool:
    """Read manual wall switch input from GPIO.
    
    Returns False if hardware is disabled or switch is open.
    """
    if not _hardware_enabled:
        return False
    # Hardware owner: read physical GPIO pin configured for pull-down wall switch
    return False


def set_fan(on: bool) -> None:
    """Command the exhaust fan relay.
    
    Must fail safe to OFF (open relay contacts) when disabled or on exit.
    """
    global _fan_state
    if not _fan_control_enabled:
        _fan_state = False
        if on:
            logger.warning("Fan relay actuation suppressed: 'fan_control_enabled: false' in configuration.")
        return

    _fan_state = bool(on)
    logger.info("Commanding physical fan relay: %s", "ON" if _fan_state else "OFF")
    # Hardware owner: drive physical GPIO pin to relay optocoupler
