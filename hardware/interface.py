"""Hardware abstraction interfaces and exceptions."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from common.schema import SensorReading


class HardwareError(Exception):
    """Base exception for hardware communication errors."""
    pass


class HardwareNotAvailableError(HardwareError):
    """Raised when hardware access (SPI/GPIO) is attempted on an unsupported system."""
    pass


class HardwareDisabledError(HardwareError):
    """Raised when physical hardware operation is requested while disabled in configuration."""
    pass


class BaseSensorHardware(ABC):
    """Abstract interface for sensor acquisition hardware."""

    @abstractmethod
    def read_sensors(self) -> SensorReading:
        """Read all configured sensors and return a structured SensorReading."""
        raise NotImplementedError

    @abstractmethod
    def read_override(self) -> bool:
        """Check if physical manual override (wall switch) is active."""
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> Dict[str, str]:
        """Return diagnostic health status for each sensor ('ok', 'fault', 'stale')."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether physical hardware communication is initialized and functional."""
        raise NotImplementedError


class BaseActuatorHardware(ABC):
    """Abstract interface for ventilation fan control (relay)."""

    @abstractmethod
    def set_fan(self, on: bool) -> None:
        """Drive the fan relay. Must fail safely to OFF if unenergized or disabled."""
        raise NotImplementedError

    @abstractmethod
    def is_fan_on(self) -> bool:
        """Return current commanded relay state."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Safely release GPIO/relay resources, guaranteeing fan is left OFF."""
        raise NotImplementedError
