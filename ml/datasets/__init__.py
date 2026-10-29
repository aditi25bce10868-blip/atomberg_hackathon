"""Dataset tools for schema inspection, validation, and metadata management."""

def inspect_dataset(*args, **kwargs):
    from ml.datasets.inspect_dataset import inspect_dataset as _inspect
    return _inspect(*args, **kwargs)

def validate_dataset(*args, **kwargs):
    from ml.datasets.validate_dataset import validate_dataset as _validate
    return _validate(*args, **kwargs)

__all__ = ["inspect_dataset", "validate_dataset"]
