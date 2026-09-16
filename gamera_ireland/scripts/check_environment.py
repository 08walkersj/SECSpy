#!/usr/bin/env python3
"""Print diagnostics for the GAMERA Ireland Python and build environment."""

from __future__ import annotations

import importlib.util
import importlib.metadata as metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON_PACKAGES = [
    "kaipy",
    "numpy",
    "scipy",
    "h5py",
    "matplotlib",
    "pandas",
    "xarray",
    "tables",
    "mpi4py",
    "apexpy",
    "secsy",
    "cdflib",
    "cdasws",
    "astropy",
]

EXECUTABLES = [
    "mpicc",
    "mpicxx",
    "mpifort",
    "mpirun",
    "cmake",
    "gcc",
    "g++",
    "gfortran",
    "h5cc",
    "h5pcc",
    "h5fc",
    "h5pfc",
    "calcdb.x",
    "gamera.x",
    "mage.x",
    "remix.x",
    "raiju.x",
    "kaiju",
]


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def command_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 and not output:
        return None
    return output.splitlines()[0] if output else None


def package_location(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin:
        return spec.origin
    if spec.submodule_search_locations:
        return ", ".join(str(path) for path in spec.submodule_search_locations)
    return "unknown"


def print_python_info() -> None:
    print_section("Python")
    print(f"version: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"prefix: {sys.prefix}")
    print(f"base_prefix: {sys.base_prefix}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV')}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")


def print_package_info() -> None:
    print_section("Python Packages")
    for package in PYTHON_PACKAGES:
        location = package_location(package)
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            if location is None:
                print(f"{package}: MISSING")
            else:
                print(f"{package}: importable, no distribution metadata")
                print(f"  import: {location}")
            continue

        print(f"{package}: {dist.version}")
        print(f"  distribution: {dist.locate_file('')}")
        print(f"  import: {location or 'not importable'}")


def print_executable_info() -> None:
    print_section("Executables")
    for name in EXECUTABLES:
        path = shutil.which(name)
        if path is None:
            print(f"{name}: MISSING")
            continue

        version = command_output([path, "--version"])
        print(f"{name}: {path}")
        if version:
            print(f"  version: {version}")


def print_hdf5_info() -> None:
    print_section("HDF5")
    for wrapper in ("h5cc", "h5pcc", "h5fc", "h5pfc"):
        path = shutil.which(wrapper)
        print(f"{wrapper}: {path or 'MISSING'}")

    h5cc = shutil.which("h5cc")
    if not h5cc:
        return

    try:
        completed = subprocess.run(
            [h5cc, "-showconfig"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"h5cc -showconfig failed: {type(exc).__name__}: {exc}")
        return

    interesting_prefixes = (
        "HDF5 Version:",
        "Installation point:",
        "Parallel HDF5:",
        "C Compiler:",
        "Fortran:",
        "Fortran Compiler:",
        "C++:",
        "C++ Compiler:",
    )
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(interesting_prefixes):
            print(stripped)


def print_project_info() -> None:
    print_section("Project")
    print(f"cwd: {Path.cwd()}")
    print(f"script: {Path(__file__).resolve()}")


def main() -> int:
    print_project_info()
    print_python_info()
    print_package_info()
    print_executable_info()
    print_hdf5_info()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
