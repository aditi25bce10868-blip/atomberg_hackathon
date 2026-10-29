"""HARDWARE OWNER: fill these three functions in (used when `source: live`).
Return raw sensor values in the SAME units used for the training data.

read_sensors() -> {"mq135": float, "mq2": float, "humidity": float, "temp_c": float, "pir": 0|1,
                   optional "health": {"mq135": "ok"|"fault", ...}}
   Use None for a sensor that fails to read.
read_override() -> bool      True while the wall switch / override button is ON
set_fan(on: bool)            drive the relay. MUST default to OFF at startup and on exit.
"""
def read_sensors():
    raise NotImplementedError("fill in hardware/sensors.py")

def read_override():
    return False

def set_fan(on):
    raise NotImplementedError("fill in hardware/sensors.py")
