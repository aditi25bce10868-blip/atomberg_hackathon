"""Data sources. Every source returns the same dict from read():
   {"ts", "sensors": {...}, "health": {...}, "override": bool, ["truth": label]}"""
import csv, math, random, threading, time
from common import config
from common.schema import CRITICAL, SENSORS, LABELS, LABEL_NAMES

class Source:
    mode = "SIMULATED"
    def read(self): raise NotImplementedError
    def set_fan(self, on): pass
    def close(self): pass

# --------------------------------------------------------------------- fake
class FakeSource(Source):
    """Simulated air. Rise/decay curves + noise + drift. Cycle: steam, cooking, foul, sensor fault."""
    mode = "SIMULATED"
    BASE = {"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 30.0}
    NOISE = {"mq135": 0.03, "mq2": 0.02, "humidity": 0.4, "temp_c": 0.1}
    AMP = {"steam":   {"humidity": 25.0, "temp_c": 3.0, "mq135": 0.10, "mq2": 0.05},
           "cooking": {"temp_c": 9.0, "mq2": 0.9, "mq135": 0.6, "humidity": 5.0},
           "foul":    {"mq135": 1.0, "mq2": 0.10}}
    CYCLE = [("steam", 30, 70), ("cooking", 200, 240), ("foul", 330, 370)]
    PERIOD, FAULT_WINDOW = 500, (440, 455)

    def __init__(self, cfg):
        c = cfg["fake"]
        self.rng, self.scn, self.n = random.Random(c.get("seed", 7)), c.get("scenario", "cycle"), 0
        self.t0 = time.time()
        self.dt = cfg["tick_s"]
        if self.scn in self.AMP:
            self.events, self.period = [(self.scn, 30, 70)], 150
        else:
            self.events, self.period = (self.CYCLE, self.PERIOD) if self.scn == "cycle" else ([], 10**9)

    def _bump(self, t):
        out, truth = {k: 0.0 for k in self.BASE}, "clean"
        for name, a, b in self.events:
            tr, td = 15.0, 15.0
            if t < a:
                continue
            k = (1 - math.exp(-(t - a) / tr)) if t <= b else (1 - math.exp(-(b - a) / tr)) * math.exp(-(t - b) / td)
            for s, amp in self.AMP[name].items():
                out[s] += k * amp
            if k > 0.5:
                truth = name
        return out, truth

    def read(self):
        t = (self.n * self.dt) % self.period
        self.n += 1
        bump, truth = self._bump(t)
        sens, health = {}, {}
        for k, b in self.BASE.items():
            drift = 0.01 * b * math.sin(self.n / 400.0)
            sens[k] = b + drift + bump[k] + self.rng.gauss(0, self.NOISE[k])
            health[k] = "ok"
        sens["pir"] = 0
        health["pir"] = "ok"
        fault = (self.scn == "fault" and t > 60) or (self.scn == "cycle" and self.FAULT_WINDOW[0] <= t < self.FAULT_WINDOW[1])
        if fault:
            sens["mq135"], health["mq135"] = None, "fault"
        return {"ts": self.t0 + self.n * self.dt, "sensors": sens, "health": health, "override": False, "truth": truth}

# ------------------------------------------------------------------- replay
class ReplaySource(Source):
    """Replays a recorded dataset CSV (column names mapped in config.yaml)."""
    mode = "REPLAY"
    def __init__(self, cfg):
        c = cfg["replay"]
        self.cols, self.loop, self.i = c["columns"], c.get("loop", True), 0
        with open(config.path(cfg, c["file"]), newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError("replay file is empty")
        self.t0, self.dt = time.time(), cfg["tick_s"]

    def _f(self, r, k):
        try:
            return float(r[self.cols[k]])
        except (KeyError, ValueError, TypeError):
            return None

    def read(self):
        if self.i >= len(self.rows):
            if not self.loop:
                raise StopIteration
            self.i = 0
        r = self.rows[self.i]; self.i += 1
        sens = {k: self._f(r, k) for k in SENSORS}
        if sens["pir"] is None:
            sens["pir"] = 0
        health = {k: ("ok" if sens[k] is not None else "fault") for k in SENSORS}
        truth = r.get(self.cols.get("label", "label"))
        try:
            truth = LABELS[int(truth)]
        except (TypeError, ValueError, KeyError):
            pass
        return {"ts": self.t0 + self.i * self.dt, "sensors": sens, "health": health, "override": False, "truth": truth}

# ------------------------------------------------------------------- serial
class SerialSource(Source):
    """ESP32 (or any board) streams one CSV line per second over USB serial:
         mq135,mq2,humidity,temp_c,pir,override      (use 'nan' for a failed sensor)
    The Pi receives it here. set_fan() sends 'F1' / 'F0' back to the board."""
    mode = "LIVE"
    def __init__(self, cfg):
        import serial
        c = cfg["serial"]
        self.port, self.baud, self.stale = c["port"], c["baud"], c["stale_s"]
        self.ser, self.latest, self.rx, self.lock = None, None, 0.0, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import serial
        while True:
            try:
                if self.ser is None:
                    self.ser = serial.Serial(self.port, self.baud, timeout=2)
                line = self.ser.readline().decode(errors="ignore").strip()
                p = line.split(",")
                if len(p) < 6:
                    continue
                v = [None if x.strip().lower() in ("nan", "") else float(x) for x in p[:6]]
                with self.lock:
                    self.latest, self.rx = v, time.time()
            except ValueError:
                continue
            except Exception as e:
                print("[serial] error:", e)
                try: self.ser.close()
                except Exception: pass
                self.ser = None
                time.sleep(2)

    def read(self):
        with self.lock:
            v, rx = self.latest, self.rx
        if v is None or time.time() - rx > self.stale:      # link down -> every sensor faults
            sens = {k: None for k in SENSORS}
            return {"ts": time.time(), "sensors": sens, "health": {k: "fault" for k in SENSORS}, "override": False}
        sens = dict(zip(["mq135", "mq2", "humidity", "temp_c", "pir"], v[:5]))
        health = {k: ("ok" if sens[k] is not None else "fault") for k in sens}
        return {"ts": time.time(), "sensors": sens, "health": health, "override": bool(v[5])}

    def set_fan(self, on):
        try: self.ser.write(b"F1\n" if on else b"F0\n")
        except Exception: pass

# --------------------------------------------------------------------- live
class LiveSource(Source):
    """Pi reads the sensors itself. The hardware owner fills in hardware/sensors.py."""
    mode = "LIVE"
    def __init__(self, cfg):
        from hardware import sensors
        self.hw = sensors

    def read(self):
        r = self.hw.read_sensors()
        health = r.get("health") or {k: ("ok" if r.get(k) is not None else "fault") for k in SENSORS}
        sens = {k: r.get(k) for k in SENSORS}
        if sens.get("pir") is None:
            sens["pir"] = 0
        return {"ts": time.time(), "sensors": sens, "health": health, "override": bool(self.hw.read_override())}

    def set_fan(self, on): self.hw.set_fan(on)
    def close(self): self.hw.set_fan(False)

def make_source(cfg):
    return {"fake": FakeSource, "replay": ReplaySource, "serial": SerialSource, "live": LiveSource}[cfg["source"]](cfg)
