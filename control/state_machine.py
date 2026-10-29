"""Rule-based control: baseline -> trigger -> debounce -> RUNNING -> RUN-ON -> IDLE.
Time comes from the row timestamp, so it works with fake/replay/live alike."""
import math
from common.schema import CRITICAL, InferenceStatus

class Controller:
    def __init__(self, cfg):
        self.cfg, self.T, self.thr = cfg, cfg["timers"], cfg["thresholds"]
        self.clear_ratio, self.mlc = cfg["clear_ratio"], cfg.get("ml", {})
        self.alpha, self.alpha_fast = cfg["baseline"]["alpha"], cfg["baseline"]["alpha_fast"]
        # Critical sensors to monitor for faults: explicit config or auto-detected on first packet
        self.critical = tuple(cfg["critical_sensors"]) if "critical_sensors" in cfg else None
        self.base, self.state, self.fan = {}, "WARMUP", False
        self.t0 = self.last_ts = None
        self.streak = self.clear_streak = self.fault_ok = self.frozen_for = 0.0
        self.first_trig = self.start = self.air_clear = self.runon_end = self.trig_label = None
        self.cooldown_until = 0.0

    # --- helpers -------------------------------------------------------
    def deltas(self, sensors):
        return {k: (sensors[k] - self.base[k]) if sensors.get(k) is not None and k in self.base else 0.0
                for k in self.thr if k in sensors}

    def _update_base(self, sensors, alpha):
        for k in self.thr:
            x = sensors.get(k)
            if x is None or k == "pir":
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
    def step(self, ts, sensors, health, override, ml_label=None, ml_conf=0.0, ml_status=None):
        ev, run = [], None
        dt = 1.0 if self.last_ts is None else max(0.0, ts - self.last_ts)
        self.last_ts = ts
        if self.t0 is None:
            self.t0 = ts
        if self.critical is None:
            if "mq3" in sensors and "mq4" in sensors:
                self.critical = ("mq3", "mq4", "mq135")
            elif "mq2" in sensors:
                self.critical = ("mq135", "mq2")
            else:
                self.critical = tuple(k for k in self.thr if k in sensors and k != "pir")

        faults = [
            k for k in self.critical
            if health.get(k, "ok") != "ok"
            or sensors.get(k) is None
            or not isinstance(sensors.get(k), (int, float))
            or math.isnan(sensors[k])
            or math.isinf(sensors[k])
        ]
        d = self.deltas(sensors)
        scale = 1.0

        # SAFETY GATE:
        # A placeholder or unverified model must NOT independently authorize fan actuation
        # or lower trigger thresholds. Only genuine VALID_MODEL predictions with verified
        # confidence scale the thresholds unless allow_rules_fallback_trigger is explicitly True.
        # Note: ml_status is None is strictly prohibited from triggering threshold scaling.
        allow_fallback_trigger = bool(self.mlc.get("allow_rules_fallback_trigger", False))
        is_valid_model = (ml_status == InferenceStatus.VALID_MODEL)
        is_allowed_fallback = (ml_status == InferenceStatus.RULES_FALLBACK and allow_fallback_trigger)
        can_scale_thresholds = is_valid_model or is_allowed_fallback

        valid_contaminant_labels = ("steam", "cooking", "foul")
        is_valid_confidence = (
            isinstance(ml_conf, (int, float))
            and not (math.isnan(ml_conf) or math.isinf(ml_conf))
            and ml_conf >= self.mlc.get("min_confidence", 0.7)
        )

        if can_scale_thresholds and ml_label in valid_contaminant_labels and is_valid_confidence:
            scale *= self.mlc.get("threshold_scale", 0.7)

        if sensors.get("pir"):
            scale *= self.cfg.get("pir_scale", 1.0)
        trig = any(d[k] >= self.thr[k] * scale for k in d) if d else False
        clearing = all(d[k] < self.thr[k] * self.clear_ratio for k in d) if d else True

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
                    self.first_trig = ts
                    self.trig_label = ml_label if (is_valid_model and ml_label in valid_contaminant_labels) else "rules"
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
