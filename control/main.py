"""Control process: source -> features/ML -> state machine -> relay, writing one row per second to SQLite.
Run:  python -m control.main            (stop with Ctrl+C; the relay is forced OFF on exit)"""
import collections, csv, pathlib, re, sys, time
from common import config
from common.db import DB
from common.schema import SCHEMA_VERSION, CSV_COLUMNS, LABEL_CODES, LABEL_NAMES
from control.sources import make_source
from control.state_machine import Controller
from control.ml_runtime import Classifier

class Collector:
    """Records labelled runs to CSV (training data) when the dashboard asks."""
    def __init__(self, runs_dir):
        self.dir, self.f, self.w, self.label, self.file, self.run_id = pathlib.Path(runs_dir), None, None, None, None, 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def start(self, label, note):
        self.stop()
        ids = [int(m.group(1)) for p in self.dir.glob("run_*.csv") if (m := re.match(r"run_(\d+)_", p.name))]
        self.run_id = max(ids, default=0) + 1
        note = re.sub(r"[^A-Za-z0-9_-]+", "_", note or "")[:40].strip("_")
        self.file = self.dir / f"run_{self.run_id:02d}_{label}{'_' + note if note else ''}.csv"
        self.f = open(self.file, "w", newline="")
        self.w = csv.writer(self.f); self.w.writerow(CSV_COLUMNS)
        self.label = label

    def write(self, row):
        if self.w:
            s = row["sensors"]
            self.w.writerow([self.run_id, row["ts"]] + [s.get(k) for k in CSV_COLUMNS[2:7]] + [LABEL_CODES[self.label]])
            self.f.flush()

    def stop(self):
        if self.f:
            self.f.close()
        self.f = self.w = None

    def status(self):
        return {"active": self.w is not None, "label": self.label if self.w else None,
                "file": self.file.name if self.w else None}

def main():
    cfg = config.load()
    db = DB(config.path(cfg, cfg["db_path"]))
    src, ctl, clf = make_source(cfg), Controller(cfg), Classifier(cfg)
    col = Collector(config.path(cfg, cfg["runs_dir"]))
    hist, last_relay, n = collections.deque(maxlen=120), False, 0
    fast = cfg["source"] in ("fake", "replay")
    sleep = cfg["tick_s"] / (cfg.get("speed", 1.0) if fast else 1.0)
    print(f"[control] source={cfg['source']} mode={src.mode} tick={cfg['tick_s']}s speed={cfg.get('speed',1.0) if fast else 1}")
    db.add_event(time.time(), "info", "start", f"Control started ({cfg['source']}, ML: {clf.source})")
    try:
        while True:
            t_loop = time.time()
            raw = src.read()
            ts, sens, health = raw["ts"], raw["sensors"], raw["health"]
            d = ctl.deltas(sens)
            ml = clf.predict(sens, d, cfg["thresholds"], hist)
            out = ctl.step(ts, sens, health, raw["override"], ml["label"], ml["confidence"])
            if out["relay_on"] != last_relay:                      # only the relay is ever driven
                src.set_fan(out["relay_on"]); last_relay = out["relay_on"]
            row = {"schema": SCHEMA_VERSION, "ts": ts, "mode": src.mode, "sensors": sens, "baseline": out["baseline"],
                   "deltas": out["deltas"], "health": health, "ml": ml, "state": out["state"], "fan_on": out["fan_on"],
                   "relay_on": out["relay_on"], "fault_led": out["fault_led"], "override": out["override"],
                   "timers": out["timers"], "demo_timers": cfg["demo_timers"], "collecting": col.status()}
            if "truth" in raw:
                row["truth"] = raw["truth"]
            hist.append(row)
            db.insert_row(row); col.write(row)
            for lvl, kind, msg in out["events"]:
                db.add_event(ts, lvl, kind, msg); print(f"[{lvl}] {kind}: {msg}")
            if out["run"]:
                db.add_run(out["run"])
            for cmd, p in db.poll_commands():
                if cmd == "collect_start" and p.get("label") in LABEL_NAMES:
                    col.start(p["label"], p.get("note", "")); db.add_event(ts, "info", "collect", f"Recording {col.file.name}")
                elif cmd == "collect_stop":
                    col.stop(); db.add_event(ts, "info", "collect", "Recording stopped")
            n += 1
            if n % 600 == 0:
                db.prune(cfg["retention_hours"])
            time.sleep(max(0.0, sleep - (time.time() - t_loop)))
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        col.stop()
        try: src.set_fan(False)                                    # relay default = OFF
        except Exception: pass
        src.close()
        print("[control] stopped, relay OFF")

if __name__ == "__main__":
    main()
