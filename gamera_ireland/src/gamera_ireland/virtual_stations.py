"""Definitions for real and virtual magnetometer stations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    """Minimal station metadata."""

    name: str
    latitude: float
    longitude: float


REFERENCE_STATIONS = [
    Station("Dunsink", 53.386, -6.338),
    Station("Valentia", 51.938, -10.247),
    Station("Armagh", 54.353, -6.649),
    Station("Birr", 53.096, -7.914),
    Station("Florence Court", 54.258, -7.723),
]

