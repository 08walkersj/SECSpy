"""Input/output helpers for inspected MAGE/GAMERA-derived products.

Schema-specific HDF5 readers should be added only after representative real
MAGE, REMIX, RAIJU, or deltab files have been inspected.
"""

from __future__ import annotations

from pathlib import Path


def require_existing_file(path: str | Path) -> Path:
    """Return a resolved file path, raising a clear error if it is absent."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected an existing file: {resolved}")
    return resolved

