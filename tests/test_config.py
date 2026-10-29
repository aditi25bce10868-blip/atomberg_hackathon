"""Unit tests for configuration loading, validation, and safe defaults."""
import copy
import pytest
from common import config
from common.config import ConfigurationError, validate_config, apply_safe_defaults


def test_load_default_config():
    cfg = config.load()
    assert isinstance(cfg, dict)
    assert "_root" in cfg
    assert cfg["source"] in ("fake", "replay", "serial", "live")
    assert "timers" in cfg
    # Safety defaults
    assert cfg["hardware_enabled"] is False
    assert cfg["fan_control_enabled"] is False
    assert cfg["runtime_mode"] in ("simulation", "replay", "live")


def test_validate_valid_config():
    cfg = config.load()
    errors = validate_config(cfg)
    assert errors == []


def test_validate_missing_source():
    cfg = copy.deepcopy(config.load())
    cfg.pop("source")
    errors = validate_config(cfg)
    assert any("source" in e for e in errors)


def test_validate_invalid_source():
    cfg = copy.deepcopy(config.load())
    cfg["source"] = "magic_source"
    errors = validate_config(cfg)
    assert any("Invalid source" in e for e in errors)


def test_validate_invalid_tick_s():
    cfg = copy.deepcopy(config.load())
    cfg["tick_s"] = -1.0
    errors = validate_config(cfg)
    assert any("tick_s" in e for e in errors)


def test_validate_missing_timer_field():
    cfg = copy.deepcopy(config.load())
    del cfg["timers_demo"]["warmup_s"]
    errors = validate_config(cfg)
    assert any("timers_demo.warmup_s" in e for e in errors)


def test_validate_invalid_clear_ratio():
    cfg = copy.deepcopy(config.load())
    cfg["clear_ratio"] = 1.5
    errors = validate_config(cfg)
    assert any("clear_ratio" in e for e in errors)


def test_validate_prohibits_secrets_in_config():
    cfg = copy.deepcopy(config.load())
    cfg["api_secret_key"] = "super_secret_123"
    errors = validate_config(cfg)
    assert any("Sensitive key" in e for e in errors)


def test_load_nonexistent_file_raises_error():
    with pytest.raises(ConfigurationError):
        config.load("non_existent_config_file.yaml")
