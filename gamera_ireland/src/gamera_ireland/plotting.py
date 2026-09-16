"""Plotting helpers for Ireland ground perturbation diagnostics."""

from __future__ import annotations


def set_default_style() -> None:
    """Apply a compact Matplotlib style when Matplotlib is available."""
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

