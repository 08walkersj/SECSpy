"""Metrics for comparing virtual magnetometer network configurations."""

from __future__ import annotations

import numpy as np


def root_mean_square_error(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return RMSE between observed and reconstructed perturbations."""
    observed = np.asarray(observed)
    reconstructed = np.asarray(reconstructed)
    if observed.shape != reconstructed.shape:
        raise ValueError(
            "observed and reconstructed arrays must have matching shapes: "
            f"{observed.shape} != {reconstructed.shape}"
        )
    return float(np.sqrt(np.mean((observed - reconstructed) ** 2)))

