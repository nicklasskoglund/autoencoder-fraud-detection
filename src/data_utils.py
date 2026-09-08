"""Utilities for loading prepared data splits."""
import numpy as np


def load_splits(path="../data/processed/splits.npz"):
    """Load the train/val/test splits saved by 02_data_preparation.ipynb.

    Parameters
    ----------
    path : str
        Path to the .npz file containing the saved arrays.

    Returns
    -------
    dict
        Dictionary with keys: X_train, X_val, X_test, y_test.
    """
    data = np.load(path)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
    }