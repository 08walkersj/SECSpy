#!/usr/bin/env python3
"""Recursively inspect groups, datasets, shapes, dtypes, and attributes in HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py


def format_attr_value(value: Any) -> str:
    text = repr(value)
    if len(text) > 160:
        return text[:157] + "..."
    return text


def print_attrs(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
    for key, value in obj.attrs.items():
        print(f"{indent}@{key}: {format_attr_value(value)}")


def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
    depth = 0 if not name else name.count("/") + 1
    indent = "  " * depth
    display_name = "/" if not name else f"/{name}"

    if isinstance(obj, h5py.Dataset):
        print(
            f"{indent}DATASET {display_name} "
            f"shape={obj.shape} dtype={obj.dtype}"
        )
    elif isinstance(obj, h5py.Group):
        print(f"{indent}GROUP {display_name}")
    else:
        print(f"{indent}OBJECT {display_name} type={type(obj).__name__}")

    print_attrs(obj, indent + "  ")


def inspect_file(filename: Path) -> None:
    with h5py.File(filename, "r") as handle:
        visit("", handle)
        handle.visititems(visit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5 file recursively without assuming dataset names.",
    )
    parser.add_argument("filename", type=Path, help="HDF5 file to inspect")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.filename.exists():
        raise SystemExit(f"File not found: {args.filename}")
    if not args.filename.is_file():
        raise SystemExit(f"Not a file: {args.filename}")

    inspect_file(args.filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

