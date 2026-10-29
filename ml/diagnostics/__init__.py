"""ML Diagnostics and Preflight Verification Package."""
from typing import Any

__all__ = ["run_preflight", "PreflightResult", "DiagnosticCheck"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import ml.diagnostics.preflight as pf
        return getattr(pf, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
