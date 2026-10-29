"""Model training and evaluation entry points."""

def train_model(*args, **kwargs):
    from ml.training.train import train_model as _train
    return _train(*args, **kwargs)

def evaluate_model(*args, **kwargs):
    from ml.training.evaluate import evaluate_model as _eval
    return _eval(*args, **kwargs)

__all__ = ["train_model", "evaluate_model"]
