"""Utilities for evaluating the autoencoder's reconstruction error."""
import numpy as np


def reconstruction_error(model, X):
    """Compute the per-row mean squared reconstruction error for X.

    Parameters
    ----------
    model : keras.Model
        A trained autoencoder.
    X : np.ndarray
        Input data of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Reconstruction error (MSE) per row, shape (n_samples,).
    """
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)


def classify_by_threshold(errors, threshold):
    """Flag rows with reconstruction error above threshold as anomalies.

    Parameters
    ----------
    errors : np.ndarray
        Per-row reconstruction error.
    threshold : float
        Reconstruction error threshold above which a row is flagged.

    Returns
    -------
    np.ndarray
        Binary array (1 = flagged as anomaly/fraud, 0 = normal).
    """
    return (errors > threshold).astype(int)