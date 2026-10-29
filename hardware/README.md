# Hardware Integration & Wiring Guide: Efficio Apex

This document details the hardware architecture, pin mapping, electrical conditioning, and safety rules for deploying the Efficio Apex AI-powered smart ventilation system on physical hardware.

---

## 1. Safety and Reliability Mandates

> [!CAUTION]
> **PROTOTYPE SYSTEM - NOT A LIFE-SAFETY OR CERTIFIED GAS DETECTOR**
> - The software and hardware described here are an experimental prototype for ventilation automation. It is NOT a life-safety device and must never be relied upon as a certified toxic or explosive gas alarm.
> - Hardware access (`hardware_enabled`) and relay actuation (`fan_control_enabled`) are **disabled by default** in `config.yaml`.
> - **NEVER switch mains AC voltage (110V/230V) during testing or bench development.** Use a 5V/12V DC brushless fan or indicator LED.
> - Any eventual physical installation involving mains wiring must strictly comply with local electrical codes and be inspected by a qualified, licensed electrician.

---

## 2. Hardware Architecture & Components

The target hardware stack consists of:

1. **Host Compute**: Raspberry Pi 5 (running Raspberry Pi OS 64-bit).
2. **Analog-to-Digital Converter**: Microchip MCP3008 8-channel 10-bit SPI ADC.
3. **Sensor Array**:
   - **MQ-3**: Alcohol, benzene, and volatile organic compounds (VOCs).
   - **MQ-4**: Methane, natural gas, compressed natural gas (CNG), and smoke.
   - **MQ-135**: Air quality (ammonia, nitrogen oxides, alcohol, benzene, smoke, CO2).
   - **PIR Sensor**: Passive infrared motion/occupancy sensor (3.3V digital output).
   - **DHT22 / AHT20**: Digital relative humidity and temperature sensor.
4. **Actuation**: 5V Optocoupled Relay Module (driving fan motor).
5. **Power**: External regulated 5V DC power supply (minimum 2.5A) with shared ground.

---

## 3. Critical Electrical Conditioning & Voltage Verification

### 3.1 Voltage Limits and Level Shifting
- **Raspberry Pi 5 GPIOs operate strictly at 3.3V logic.** Exposing any GPIO pin to voltages above 3.3V will permanently destroy the Broadcom SoC.
- **MCP3008 Operating Voltage:**
  - **Scenario A (Recommended): MCP3008 powered at $V_{\text{DD}} = 3.3\text{V}$ and $V_{\text{REF}} = 3.3\text{V}$:**
    - SPI communication lines (MOSI, MISO, SCLK, CE0) are natively 3.3V compatible with the Pi.
    - **CRITICAL**: The maximum input voltage on any analog channel (CH0–CH7) cannot exceed $V_{\text{DD}} = 3.3\text{V}$.
    - Because MQ sensors are powered by 5V and can output analog signals up to 5V under heavy gas concentration, **each analog output line MUST pass through a voltage divider** before reaching the MCP3008.
    - *Recommended Divider*: $R_1 = 10\text{ k}\Omega$ (series), $R_2 = 20\text{ k}\Omega$ (to GND).  
      $V_{\text{ADC}} = V_{\text{sensor}} \times \frac{20}{10 + 20} = V_{\text{sensor}} \times 0.667$.  
      A 5.0V sensor peak becomes 3.33V at the ADC input.
  - **Scenario B: MCP3008 powered at $V_{\text{DD}} = 5.0\text{V}$:**
    - MCP3008 DOUT (MISO) will output 5V logic. You **MUST** insert a bidirectional logic level shifter or a 2:1 voltage divider on the MISO line (MCP3008 Pin 12 $\rightarrow$ Pi Pin 21) to prevent frying the Pi.

### 3.2 Sensor Heater Power Requirements
- MQ-series gas sensors contain an internal tin oxide ($\text{SnO}_2$) heating element requiring 5.0V DC.
- Each sensor heater draws approximately **150–180 mA**.
- Three MQ sensors operating concurrently draw over **500 mA** continuous current.
- **DO NOT power the MQ heaters from the Raspberry Pi 3.3V rail or 5V GPIO header pins.**  
  Doing so can cause brownouts, power resets, or damage to the Pi's power management IC (PMIC).
- **MANDATORY**: Power all MQ heaters from a dedicated external regulated 5V DC power supply (minimum 2.0A rating). Connect the ground of the external 5V power supply to the Raspberry Pi Ground to ensure a common reference potential.

### 3.3 Relay Isolation
- Use an optocoupled relay board featuring an isolation optical isolator.
- Remove the VCC/JD-VCC jumper to separate the relay coil power supply from the Raspberry Pi logic power.
- Connect relay coil (JD-VCC) to external 5V supply; connect VCC to Raspberry Pi 3.3V logic.
- Always use the **Normally Open (NO)** contact configuration so that if the control software crashes, power is lost, or the Pi reboots, the fan relay defaults to the safe **OFF** state.

---

## 4. Raspberry Pi 5 to MCP3008 Pinout Table

Enable SPI on the Raspberry Pi via `sudo raspi-config` $\rightarrow$ Interface Options $\rightarrow$ SPI $\rightarrow$ Enable.

| MCP3008 Pin | MCP3008 Function | Connection Target | Raspberry Pi 5 Physical Pin |
|---|---|---|---|
| Pin 1 | CH0 (Analog Input) | MQ-3 Analog Out (via voltage divider) | — |
| Pin 2 | CH1 (Analog Input) | MQ-4 Analog Out (via voltage divider) | — |
| Pin 3 | CH2 (Analog Input) | MQ-135 Analog Out (via voltage divider) | — |
| Pin 4–8 | CH3–CH7 | Reserved / Auxiliary Analog | — |
| Pin 9 | AGND (Analog Ground) | Common Ground (External 5V & Pi) | Physical Pin 6, 9, 14, 20, 25, 30, 34, or 39 |
| Pin 10 | $\overline{\text{CS}}$ (Chip Select) | SPI CE0 | Physical Pin 24 (GPIO 8) |
| Pin 11 | $\text{D}_{\text{IN}}$ (MOSI) | SPI MOSI | Physical Pin 19 (GPIO 10) |
| Pin 12 | $\text{D}_{\text{OUT}}$ (MISO) | SPI MISO | Physical Pin 21 (GPIO 9) |
| Pin 13 | CLK (Clock) | SPI SCLK | Physical Pin 23 (GPIO 11) |
| Pin 14 | DGND (Digital Ground)| Common Ground | Physical Pin 6 |
| Pin 15 | $V_{\text{REF}}$ | Regulated 3.3V reference | Physical Pin 1 (3.3V) |
| Pin 16 | $V_{\text{DD}}$ | Regulated 3.3V power | Physical Pin 1 or 17 (3.3V) |

---

## 5. Software Configuration Checklist for Hardware Deployment

1. **Verify SPI Device Node:**
   ```bash
   ls -l /dev/spidev0.0
   ```
2. **Install Hardware Packages on Raspberry Pi OS:**
   ```bash
   pip install spidev RPi.GPIO
   ```
3. **Bench Verification:**
   Before running `control.main`, run hardware verification:
   ```bash
   python -c "import hardware.sensors as s; print(s.check_hardware_availability())"
   ```
4. **Enable Hardware in `config.yaml`:**
   ```yaml
   source: live
   hardware_enabled: true
   fan_control_enabled: false   # Keep false until fan electrical isolation is certified
   ```
