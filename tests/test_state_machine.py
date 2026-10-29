import copy, pytest
from common import config
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
