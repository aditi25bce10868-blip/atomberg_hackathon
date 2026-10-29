"""Rule-based control: baseline -> trigger -> debounce -> RUNNING -> RUN-ON -> IDLE.
Time comes from the row timestamp, so it works with fake/replay/live alike."""
from common.schema import CRITICAL

class Controller:
    def __init__(self, cfg):
        self.cfg, self.T, self.thr = cfg, cfg["timers"], cfg["thresholds"]
        self.clear_ratio, self.mlc = cfg["clear_ratio"], cfg["ml"]
        self.alpha, self.alpha_fast = cfg["baseline"]["alpha"], cfg["baseline"]["alpha_fast"]
        self.base, self.state, self.fan = {}, "WARMUP", False
        self.t0 = self.last_ts = None
        self.streak = self.clear_streak = self.fault_ok = self.frozen_for = 0.0
        self.first_trig = self.start = self.air_clear = self.runon_end = self.trig_label = None
        self.cooldown_until = 0.0

    # --- helpers -------------------------------------------------------
    def deltas(self, sensors):
        return {k: (sensors[k] - self.base[k]) if sensors.get(k) is not None and k in self.base else 0.0
                for k in self.thr}

    def _update_base(self, sensors, alpha):
        for k in CRITICAL:
            x = sensors.get(k)
            if x is None:
                continue
            self.base[k] = x if k not in self.base else self.base[k] + alpha * (x - self.base[k])

    def _auto(self):
        return self.state in ("RUNNING", "RUN-ON")

    def _end(self, ts, reason):
        run = {"label": self.trig_label, "first_trigger_ts": self.first_trig, "start_ts": self.start, "end_ts": ts,
               "trigger_to_start_s": (self.start - self.first_trig) if self.first_trig is not None else None,
               "duration_s": ts - self.start,
               "run_on_s": (ts - self.air_clear) if self.air_clear is not None else None,
               "end_reason": reason}
        self.state, self.fan = "IDLE", False
        self.frozen_for = self.clear_streak = self.streak = 0.0
        self.air_clear = None
        return run

    def _out(self, ts, d, override, ev, run):
        s = self.state
        return {"state": s, "fan_on": self.fan, "relay_on": self.fan and self._auto(),
                "fault_led": s == "FAULT", "override": bool(override), "events": ev, "run": run,
                "baseline": dict(self.base), "deltas": d,
                "timers": {"warmup_left": max(0.0, self.T["warmup_s"] - (ts - self.t0)) if s == "WARMUP" else 0.0,
                           "run_on_left": max(0.0, self.runon_end - ts) if s == "RUN-ON" else 0.0,
                           "cap_left": max(0.0, self.T["cap_s"] - (ts - self.start)) if self._auto() else None}}

    # --- one tick --------------------------------------------------------
    def step(self, ts, sensors, health, override, ml_label=None, ml_conf=0.0):
        ev, run = [], None
        dt = 1.0 if self.last_ts is None else max(0.0, ts - self.last_ts)
        self.last_ts = ts
        if self.t0 is None:
            self.t0 = ts
        faults = [k for k in CRITICAL if health.get(k, "ok") != "ok" or sensors.get(k) is None]
        d = self.deltas(sensors)
        scale = 1.0
        if ml_label not in (None, "clean") and (ml_conf or 0) >= self.mlc["min_confidence"]:
            scale *= self.mlc["threshold_scale"]
        if sensors.get("pir"):
            scale *= self.cfg["pir_scale"]
        trig = any(d[k] >= self.thr[k] * scale for k in self.thr)
        clearing = all(d[k] < self.thr[k] * self.clear_ratio for k in self.thr)

        # 1) faults beat everything: automation off, wall switch keeps working
        if faults:
            self.fault_ok = 0.0
            if self.state != "FAULT":
                if self._auto():
                    run = self._end(ts, "fault")
                ev.append(("error", "fault", "Sensor fault: " + ", ".join(faults) + " - automation off"))
                self.state, self.fan, self.streak = "FAULT", False, 0.0
            return self._out(ts, d, override, ev, run)
        if self.state == "FAULT":
            self.fault_ok += dt
            if self.fault_ok < self.T["fault_clear_s"]:
                return self._out(ts, d, override, ev, run)
            self.state = "IDLE"
            ev.append(("info", "fault_clear", "Sensors healthy again"))

        # 2) wall switch (override) beats the sensors
        if override:
            if self.state != "MANUAL":
                if self._auto():
                    run = self._end(ts, "override")
                self.state, self.fan = "MANUAL", True
                ev.append(("info", "override", "Wall switch ON - manual control"))
            return self._out(ts, d, override, ev, run)
        if self.state == "MANUAL":
            self.state, self.fan = "IDLE", False
            ev.append(("info", "override_off", "Wall switch released - back to auto"))

        # 3) normal operation
        if self.state == "WARMUP":
            self._update_base(sensors, self.alpha_fast)
            if ts - self.t0 >= self.T["warmup_s"]:
                self.state = "IDLE"
                ev.append(("info", "ready", "Warm-up complete, baseline set"))
        elif self.state == "IDLE":
            if clearing or self.frozen_for > self.cfg["baseline"]["max_freeze_s"]:
                self._update_base(sensors, self.alpha if clearing else self.alpha_fast)
                self.frozen_for = 0.0
            else:
                self.frozen_for += dt
            if ts >= self.cooldown_until and trig:
                if self.streak == 0:
                    self.first_trig, self.trig_label = ts, ml_label
                self.streak += 1
                if self.streak >= self.T["debounce"]:
                    self.state, self.fan, self.start = "RUNNING", True, ts
                    self.clear_streak = self.streak = 0.0
                    self.air_clear = None
                    ev.append(("info", "fan_on", f"Trigger ({self.trig_label or 'rules'}) - fan ON"))
            else:
                self.streak = 0.0
        elif self._auto():
            if ts - self.start >= self.T["cap_s"]:
                ev.append(("warn", "cap", "Hard cap reached - fan OFF"))
                run = self._end(ts, "cap")
                self.cooldown_until = ts + self.T["cooldown_s"]
            elif self.state == "RUNNING":
                self.clear_streak = self.clear_streak + dt if clearing else 0.0
                if self.clear_streak >= self.T["clear_hold_s"]:
                    self.state, self.air_clear = "RUN-ON", ts
                    self.runon_end = ts + self.T["run_on_s"]
                    ev.append(("info", "air_clear", "Air clear - run-on started"))
            else:  # RUN-ON
                if trig:
                    self.state, self.clear_streak = "RUNNING", 0.0
                    ev.append(("info", "retrigger", "Air dirty again - back to RUNNING"))
                elif ts >= self.runon_end:
                    ev.append(("info", "fan_off", "Run-on finished - fan OFF"))
                    run = self._end(ts, "clear")
        return self._out(ts, d, override, ev, run)
