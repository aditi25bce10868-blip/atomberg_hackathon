import os, pathlib, yaml
ROOT = pathlib.Path(__file__).resolve().parent.parent

def load(path=None):
    p = pathlib.Path(path or os.environ.get("FAN_CONFIG") or ROOT / "config.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    cfg["timers"] = cfg["timers_demo"] if cfg.get("demo_timers", True) else cfg["timers_real"]
    return cfg

def path(cfg, rel):
    p = pathlib.Path(rel)
    return p if p.is_absolute() else pathlib.Path(cfg["_root"]) / p
