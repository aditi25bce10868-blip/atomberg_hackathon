"""MCP3008 10-bit 8-channel SPI ADC Interface.

CRITICAL HARDWARE SAFETY NOTICE:
1. Voltage Compatibility:
   - Raspberry Pi 5 GPIO pins are 3.3V logic ONLY. Exceeding 3.3V on any Pi GPIO will cause
     irreversible damage to the SoC.
   - If MCP3008 is powered at VDD = 3.3V (recommended):
     Maximum allowable analog input voltage on CH0-CH7 is 3.3V.
     MQ-series gas sensors (MQ-3, MQ-4, MQ-135) operate with 5V heaters and output analog
     voltages up to 5V under high gas concentrations.
     *MANDATORY*: You MUST install a resistive voltage divider (e.g. 10kΩ / 20kΩ) or an active
     voltage conditioning buffer between each MQ analog output and MCP3008 input channels!
   - If MCP3008 is powered at VDD = 5.0V:
     The DOUT (MISO) line from MCP3008 to Raspberry Pi GPIO will output 5V logic.
     *MANDATORY*: You MUST install a bidirectional logic level shifter or 2:1 resistive divider
     on the MISO line before connecting to Raspberry Pi Pin 21 (GPIO 9 / MISO).

2. Raw ADC Counts vs. Concentrations:
   Raw 10-bit integer counts (0-1023) MUST NEVER be represented directly as calibrated gas
   concentrations (ppm or mg/L). Raw counts represent normalized ADC voltage steps.
"""

from typing import Dict, List, Optional, Tuple
from common.logger import get_logger
from hardware.interface import HardwareNotAvailableError

logger = get_logger("hardware.mcp3008")


class MCP3008Reader:
    """Driver for MCP3008 8-channel 10-bit ADC over SPI."""

    def __init__(
        self,
        bus: int = 0,
        device: int = 0,
        vref: float = 3.3,
        max_speed_hz: int = 1350000,
    ):
        self.bus = bus
        self.device = device
        self.vref = vref
        self.max_speed_hz = max_speed_hz
        self._spi = None
        self._available = False

        self._init_spi()

    def _init_spi(self) -> None:
        """Attempt to initialize the SPI bus using spidev."""
        try:
            import spidev  # type: ignore
            self._spi = spidev.SpiDev()
            self._spi.open(self.bus, self.device)
            self._spi.max_speed_hz = self.max_speed_hz
            self._spi.mode = 0b00
            self._available = True
            logger.info("MCP3008 SPI bus opened successfully on /dev/spidev%d.%d", self.bus, self.device)
        except ImportError:
            logger.warning("spidev module not installed. MCP3008 physical SPI access is unavailable on this platform.")
            self._available = False
        except Exception as e:
            logger.warning("Failed to open SPI bus /dev/spidev%d.%d: %s", self.bus, self.device, e)
            self._available = False

    def is_available(self) -> bool:
        """Return True if the SPI connection to MCP3008 is active."""
        return self._available

    def read_raw(self, channel: int) -> int:
        """Read a single-ended raw 10-bit count (0-1023) from the specified channel (0-7).
        
        Raises:
            ValueError: If channel is not between 0 and 7.
            HardwareNotAvailableError: If physical SPI bus is not connected.
        """
        if not (0 <= channel <= 7):
            raise ValueError(f"MCP3008 channel must be in range 0-7, got {channel}")

        if not self._available or self._spi is None:
            raise HardwareNotAvailableError(
                f"MCP3008 SPI is not available on /dev/spidev{self.bus}.{self.device}. "
                "Ensure SPI is enabled via raspi-config and user has spi group permissions."
            )

        # MCP3008 protocol: Start bit (1), Single-ended mode (1) + 3-bit channel, dummy byte
        cmd = [1, (8 + channel) << 4, 0]
        try:
            resp = self._spi.xfer2(cmd)
        except Exception as e:
            raise HardwareNotAvailableError(f"SPI transfer failed on channel {channel}: {e}") from e
        # Extract 10 bits from the 3-byte response
        raw_count = ((resp[1] & 3) << 8) + resp[2]
        return raw_count

    def read_voltage(self, channel: int) -> float:
        """Read channel and return measured analog voltage (0.0 to vref)."""
        count = self.read_raw(channel)
        return (count * self.vref) / 1023.0

    def read_all_channels(self) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Read all 8 channels.
        
        Returns:
            (raw_counts_dict, voltages_dict)
        """
        counts = {}
        voltages = {}
        for ch in range(8):
            cnt = self.read_raw(ch)
            counts[f"ch{ch}"] = cnt
            voltages[f"ch{ch}"] = (cnt * self.vref) / 1023.0
        return counts, voltages

    def close(self) -> None:
        """Close SPI connection."""
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
            self._spi = None
        self._available = False
