"""Configuration loading and validation with safety checks and defaults."""
import os
import pathlib
from typing import Any, Dict, List, Optional
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


class ConfigurationError(Exception):
    """Raised when application configuration is invalid or missing required fields."""
    pass


REQUIRED_TIMERS = (
    "warmup_s",
    "run_on_s",
    "cap_s",
    "cooldown_s",
    "clear_hold_s",
    "debounce",
    "fault_clear_s",
)


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate configuration values and return actionable error messages.
    
    Returns:
        List of validation error strings. If empty, the configuration is valid.
    """
    errors: List[str] = []

    # 1. Source validation
    source = cfg.get("source")
    valid_sources = ("fake", "replay", "serial", "live")
    if not source:
        errors.append("Missing required configuration key: 'source'")
    elif source not in valid_sources:
        errors.append(f"Invalid source '{source}'. Must be one of: {', '.join(valid_sources)}")

    # 2. Timing validation
    tick_s = cfg.get("tick_s")
    if tick_s is None:
        errors.append("Missing required configuration key: 'tick_s'")
    elif not isinstance(tick_s, (int, float)) or tick_s <= 0:
        errors.append(f"'tick_s' must be a positive number, got: {tick_s}")

    for timer_set in ("timers_demo", "timers_real"):
        t_dict = cfg.get(timer_set)
        if not isinstance(t_dict, dict):
            errors.append(f"Missing or invalid '{timer_set}' section in configuration")
        else:
            for k in REQUIRED_TIMERS:
                val = t_dict.get(k)
                if val is None or not isinstance(val, (int, float)) or val < 0:
                    errors.append(f"'{timer_set}.{k}' must be a non-negative number, got: {val}")

    # 3. Thresholds & trigger logic
    thr = cfg.get("thresholds")
    if not isinstance(thr, dict):
        errors.append("Missing 'thresholds' configuration map")
    else:
        for k, v in thr.items():
            if not isinstance(v, (int, float)) or v <= 0:
                errors.append(f"Threshold for '{k}' must be positive, got: {v}")

    clear_ratio = cfg.get("clear_ratio")
    if clear_ratio is None or not isinstance(clear_ratio, (int, float)) or not (0.0 < clear_ratio < 1.0):
        errors.append(f"'clear_ratio' must be a float between 0.0 and 1.0, got: {clear_ratio}")

    # 4. ML section
    ml_cfg = cfg.get("ml")
    if not isinstance(ml_cfg, dict):
        errors.append("Missing 'ml' configuration section")
    else:
        if "model_path" not in ml_cfg or not ml_cfg["model_path"]:
            errors.append("Missing 'ml.model_path' setting")
        min_conf = ml_cfg.get("min_confidence", 0.7)
        if not isinstance(min_conf, (int, float)) or not (0.0 <= min_conf <= 1.0):
            errors.append(f"'ml.min_confidence' must be between 0.0 and 1.0, got: {min_conf}")

    # 5. Security check: prohibit plain-text secrets in source-controlled config
    sensitive_keys = ("password", "secret", "token", "private_key", "api_key")
    for k in cfg.keys():
        if any(s in k.lower() for s in sensitive_keys):
            errors.append(f"Sensitive key '{k}' must not be stored in config.yaml. Use environment variables.")

    return errors


def apply_safe_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply secure defaults for hardware, actuation, and operational modes."""
    # Hardware and fan actuation are disabled by default for safety
    cfg["hardware_enabled"] = bool(cfg.get("hardware_enabled", False))
    cfg["fan_control_enabled"] = bool(cfg.get("fan_control_enabled", False))

    # Default runtime mode based on source if not explicitly given
    if "runtime_mode" not in cfg:
        cfg["runtime_mode"] = "simulation" if cfg.get("source") in ("fake", "replay") else "live"

    # Default logging config
    if "logging" not in cfg or not isinstance(cfg["logging"], dict):
        cfg["logging"] = {
            "level": "INFO",
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        }
    else:
        cfg["logging"].setdefault("level", "INFO")
        cfg["logging"].setdefault("format", "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

    # ML fallback safety defaults
    if "ml" not in cfg or not isinstance(cfg["ml"], dict):
        cfg["ml"] = {
            "min_confidence": 0.7,
            "threshold_scale": 0.7,
            "model_path": "ml/artifacts/model.joblib",
            "allow_rules_fallback": True,
            "allow_rules_fallback_trigger": False,
        }
    else:
        cfg["ml"].setdefault("allow_rules_fallback", True)
        cfg["ml"].setdefault("allow_rules_fallback_trigger", False)

    # Baseline defaults
    if "baseline" not in cfg or not isinstance(cfg["baseline"], dict):
        cfg["baseline"] = {"alpha": 0.02, "alpha_fast": 0.1, "max_freeze_s": 600}

    return cfg


def load(path: Optional[str] = None) -> Dict[str, Any]:
    """Load, validate, and return the application configuration.
    
    Raises:
        ConfigurationError: If configuration is invalid or missing required settings.
    """
    p = pathlib.Path(path or os.environ.get("FAN_CONFIG") or ROOT / "config.yaml")
    if not p.exists():
        raise ConfigurationError(f"Configuration file does not exist: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigurationError(f"Configuration in {p} must be a YAML mapping")

    errors = validate_config(cfg)
    if errors:
        raise ConfigurationError("Configuration errors in {}:\n  - {}".format(p, "\n  - ".join(errors)))

    apply_safe_defaults(cfg)
    cfg["_root"] = str(ROOT)
    cfg["timers"] = cfg["timers_demo"] if cfg.get("demo_timers", True) else cfg["timers_real"]
    return cfg


def path(cfg: Dict[str, Any], rel: str) -> pathlib.Path:
    """Resolve a relative path against the application root."""
    p = pathlib.Path(rel)
    return p if p.is_absolute() else pathlib.Path(cfg.get("_root", str(ROOT))) / p


# Convenience alias for config loader
load_config = load
