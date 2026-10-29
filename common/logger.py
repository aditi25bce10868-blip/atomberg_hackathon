"""Centralized logging infrastructure with secret-safe output formatting."""
import logging
import re
from typing import Optional

_CONFIGURED = False
_SENSITIVE_PATTERNS = [
    re.compile(r"(password\s*[:=]\s*)[^\s,;]+", re.IGNORECASE),
    re.compile(r"(token\s*[:=]\s*)[^\s,;]+", re.IGNORECASE),
    re.compile(r"(key\s*[:=]\s*)[^\s,;]+", re.IGNORECASE),
    re.compile(r"(secret\s*[:=]\s*)[^\s,;]+", re.IGNORECASE),
]


class SecretSanitizingFilter(logging.Filter):
    """Filter that masks sensitive tokens, passwords, and API keys from log messages."""
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern in _SENSITIVE_PATTERNS:
                msg = pattern.sub(r"\1[REDACTED]", msg)
            record.msg = msg
        return True


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:
    """Initialize root application logging."""
    global _CONFIGURED
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = format_str or "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(SecretSanitizingFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if not _CONFIGURED:
        root_logger.handlers = [handler]
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with secret sanitization enabled."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
