import copy, pytest
from common import config
from common.schema import InferenceStatus
from control.state_machine import Controller

OK = {k: "ok" for k in ("mq135", "mq2", "humidity", "temp_c", "pir")}
CLEAN = {"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 30.0, "pir": 0}
DIRTY = dict(CLEAN, humidity=70.0)          # +15 %RH -> trigger

@pytest.fixture
def ctl():
    cfg = config.load()
    c = Controller(cfg)
    for t in range(25):                      # finish warm-up on clean air
        c.step(t, CLEAN, OK, False)
    assert c.state == "IDLE"
    c.t = 25
    return c

def run(c, n, sens=CLEAN, health=OK, override=False):
    out = None
    c.log = getattr(c, "log", [])
    for _ in range(n):
        out = c.step(c.t, sens, health, override); c.t += 1
        if out["run"]:
            c.log.append(out["run"])         # a run summary appears on the tick it ends
    return out

def test_debounce_needs_three_ticks(ctl):
    assert run(ctl, 2, DIRTY)["state"] == "IDLE"
    assert run(ctl, 1, DIRTY)["state"] == "RUNNING"

def test_single_spike_is_ignored(ctl):
    run(ctl, 1, DIRTY)
    assert run(ctl, 5, CLEAN)["state"] == "IDLE" and not ctl.fan

def test_full_cycle_and_runon(ctl):
    run(ctl, 3, DIRTY)
    assert ctl.fan
    run(ctl, ctl.T["clear_hold_s"] + 1, CLEAN)
    assert ctl.state == "RUN-ON" and ctl.fan
    out = run(ctl, ctl.T["run_on_s"] + 1, CLEAN)
    assert out["state"] == "IDLE" and not out["fan_on"] and ctl.log[-1]["end_reason"] == "clear"

def test_hard_cap_turns_fan_off(ctl):
    out = run(ctl, ctl.T["cap_s"] + 10, DIRTY)          # air never clears
    assert not out["fan_on"] and ctl.state == "IDLE"

def test_fault_turns_automation_off_and_lights_led(ctl):
    run(ctl, 3, DIRTY)
    bad = dict(OK, mq135="fault")
    out = run(ctl, 1, dict(DIRTY, mq135=None), bad)
    assert out["state"] == "FAULT" and out["fault_led"] and not out["fan_on"] and out["run"]["end_reason"] == "fault"

def test_fault_recovers_after_healthy_period(ctl):
    run(ctl, 1, dict(CLEAN, mq135=None), dict(OK, mq135="fault"))
    assert run(ctl, ctl.T["fault_clear_s"] + 2, CLEAN)["state"] == "IDLE"

def test_override_wins_and_releases(ctl):
    out = run(ctl, 1, CLEAN, override=True)
    assert out["state"] == "MANUAL" and out["fan_on"] and not out["relay_on"]
    assert run(ctl, 1, CLEAN)["state"] == "IDLE"

def test_override_during_auto_run_ends_the_run(ctl):
    run(ctl, 3, DIRTY)
    out = run(ctl, 1, DIRTY, override=True)
    assert out["state"] == "MANUAL" and out["run"]["end_reason"] == "override"


def test_ml_fallback_does_not_lower_thresholds_safely(ctl):
    """Verify that RULES_FALLBACK status does not lower thresholds or trigger fan."""
    from common.schema import InferenceStatus
    # Humidity delta = 6.0. Normal threshold = 8.0. Scaled threshold = 8.0 * 0.7 = 5.6
    sub_threshold_dirty = dict(CLEAN, humidity=61.0)  # delta = 6.0
    
    # 1. With RULES_FALLBACK, threshold remains 8.0 -> no trigger
    for _ in range(5):
        out = ctl.step(ctl.t, sub_threshold_dirty, OK, False, ml_label="steam", ml_conf=0.9, ml_status=InferenceStatus.RULES_FALLBACK)
        ctl.t += 1
    assert out["state"] == "IDLE"
    assert not out["fan_on"]

    # 2. With VALID_MODEL, threshold scales to 5.6 -> triggers after debounce
    for _ in range(4):
        out = ctl.step(ctl.t, sub_threshold_dirty, OK, False, ml_label="steam", ml_conf=0.9, ml_status=InferenceStatus.VALID_MODEL)
        ctl.t += 1
    assert out["state"] == "RUNNING"
    assert out["fan_on"]


def test_canonical_gas_sensor_fault_triggers_safety_lockout():
    """Verify missing canonical gas sensor (MQ-3, MQ-4, MQ-135) immediately trips FAULT state."""
    cfg = config.load()
    c = Controller(cfg)
    canonical_reading = {"mq3": 1.2, "mq4": 1.1, "mq135": 1.5, "humidity": 50.0, "temp_c": 25.0, "pir": 0}
    canonical_health = {k: "ok" for k in canonical_reading}
    
    # Initialize baseline
    for t in range(25):
        c.step(t, canonical_reading, canonical_health, False)
    assert c.state == "IDLE"
    assert c.critical == ("mq3", "mq4", "mq135")

    # MQ-3 fails
    faulty_reading = dict(canonical_reading, mq3=None)
    faulty_health = dict(canonical_health, mq3="disconnected")
    out = c.step(26, faulty_reading, faulty_health, False)
    assert out["state"] == "FAULT"
    assert out["fault_led"] is True
    assert not out["fan_on"]


def test_ml_status_none_never_lowers_thresholds_even_with_allow_fallback():
    """Verify that when ml_status is None, thresholds are NEVER lowered,
    even if allow_rules_fallback_trigger is True."""
    cfg = config.load()
    cfg["ml"]["allow_rules_fallback_trigger"] = True
    c = Controller(cfg)
    sensors = {"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 30.0, "pir": 0}
    health = {k: "ok" for k in sensors}

    # Warm up
    for t in range(25):
        c.step(t, sensors, health, False)

    # Sub-threshold dirty reading (baseline 1.5, raw 1.74 -> delta 0.24 < 0.25 threshold)
    sub_dirty = dict(sensors, mq135=1.74)
    # If scaled by 0.7, threshold would be 0.175 and trigger.
    # But ml_status=None must NOT scale!
    for t in range(25, 30):
        out = c.step(t, sub_dirty, health, False, ml_label="steam", ml_conf=0.95, ml_status=None)
    assert out["state"] == "IDLE"
    assert not out["fan_on"]


def test_unknown_ml_label_never_lowers_thresholds():
    """Verify that unknown ML labels cannot trigger threshold scaling."""
    cfg = config.load()
    c = Controller(cfg)
    sensors = {"mq135": 1.5, "mq2": 1.0, "humidity": 55.0, "temp_c": 30.0, "pir": 0}
    health = {k: "ok" for k in sensors}

    for t in range(25):
        c.step(t, sensors, health, False)

    sub_dirty = dict(sensors, mq135=1.74)
    for t in range(25, 30):
        out = c.step(t, sub_dirty, health, False, ml_label="unknown_hazard", ml_conf=0.99, ml_status=InferenceStatus.VALID_MODEL)
    assert out["state"] == "IDLE"
    assert not out["fan_on"]

