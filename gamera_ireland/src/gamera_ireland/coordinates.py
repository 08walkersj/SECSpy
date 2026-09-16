"""Coordinate utilities for the Ireland analysis domain."""

from __future__ import annotations

IRELAND_BOUNDS = {
    "latitude_min": 51.0,
    "latitude_max": 56.0,
    "longitude_min": -11.0,
    "longitude_max": -5.0,
}


def delta_h(delta_bx: float, delta_by: float) -> float:
    """Return horizontal perturbation magnitude from north/east components."""
    return (delta_bx**2 + delta_by**2) ** 0.5

