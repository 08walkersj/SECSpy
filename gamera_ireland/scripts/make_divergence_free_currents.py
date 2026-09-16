#!/usr/bin/env python3
"""Create a divergence-free SECS current test field over the Ireland grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

import h5py
import numpy as np
import pandas as pd

from SECSpy.SECS import SECS

from gamera_ireland.config import load_json_config, resolve_project_path


def finite_difference_divergence(
    east_current: np.ndarray,
    north_current: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Approximate horizontal divergence on the cubed-sphere local grid."""
    x_m = xi[0, :] * 6371.2e3
    y_m = eta[:, 0] * 6371.2e3
    d_east_dx = np.gradient(east_current, x_m, axis=1, edge_order=1)
    d_north_dy = np.gradient(north_current, y_m, axis=0, edge_order=1)
    return d_east_dx + d_north_dy


def make_current_product(
    config: dict,
    output: Path,
    amplitude: float,
    singularity_limit_km: float,
) -> None:
    """Generate and save a central-pole divergence-free current response."""
    paths = config["paths"]
    grid_csv = resolve_project_path(paths["grid_csv"], PROJECT_DIR)
    metadata_json = resolve_project_path(paths["grid_metadata"], PROJECT_DIR)

    grid = pd.read_csv(grid_csv)
    with metadata_json.open() as handle:
        metadata = json.load(handle)

    shape = tuple(metadata["shape"])
    lon = grid["glon_deg_signed"].to_numpy(dtype=float)
    lat = grid["glat_deg"].to_numpy(dtype=float)
    xi = grid["xi"].to_numpy(dtype=float).reshape(shape)
    eta = grid["eta"].to_numpy(dtype=float).reshape(shape)

    pole_lon = np.asarray([float(config["ireland_grid"]["lon_centre"]) - 360.0])
    pole_lat = np.asarray([float(config["ireland_grid"]["lat_centre"])])

    model = SECS(
        pole_lon,
        pole_lat,
        lon,
        lat,
        mode="default",
    )
    east_g, north_g = model.eval_G_Matrix_J(
        singularity_limit=singularity_limit_km * 1000.0,
        system="divergence-free",
    )

    east = (east_g @ np.asarray([amplitude])).reshape(shape)
    north = (north_g @ np.asarray([amplitude])).reshape(shape)
    divergence = finite_difference_divergence(east, north, xi, eta)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["description"] = (
            "Divergence-free SECS unit/current-response test field over Ireland. "
            "This is not a GAMERA-derived physical current system."
        )
        handle.attrs["event_time"] = config["event_time"]
        handle.attrs["source_grid"] = str(grid_csv)
        handle.attrs["secs_pole_longitude_deg"] = float(pole_lon[0])
        handle.attrs["secs_pole_latitude_deg"] = float(pole_lat[0])
        handle.attrs["secs_amplitude_A"] = float(amplitude)
        handle.attrs["singularity_limit_km"] = float(singularity_limit_km)
        handle.attrs["max_abs_divergence_A_per_m2"] = float(np.nanmax(np.abs(divergence)))
        handle.create_dataset("longitude_deg", data=lon.reshape(shape))
        handle.create_dataset("latitude_deg", data=lat.reshape(shape))
        handle.create_dataset("xi", data=xi)
        handle.create_dataset("eta", data=eta)
        handle.create_dataset("east_current_A_per_m", data=east)
        handle.create_dataset("north_current_A_per_m", data=north)
        handle.create_dataset("east_current_mA_per_m", data=east * 1000.0)
        handle.create_dataset("north_current_mA_per_m", data=north * 1000.0)
        handle.create_dataset("finite_difference_divergence_A_per_m2", data=divergence)

    print(f"Wrote {output}")
    print(f"central SECS pole: lon={pole_lon[0]:.4f}, lat={pole_lat[0]:.4f}")
    print(f"amplitude: {amplitude:.6g} A")
    print(f"max |finite-difference divergence|: {np.nanmax(np.abs(divergence)):.6e} A/m^2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json",
        help="Event configuration JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "gamera_ireland/output/optimisation/divergence_free_unit_current.h5",
        help="Output HDF5 current product.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0e6,
        help="SECS pole amplitude in amps.",
    )
    parser.add_argument(
        "--singularity-limit-km",
        type=float,
        default=10.0,
        help="Inner regularisation distance around the SECS pole.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    make_current_product(
        load_json_config(args.config),
        args.output,
        args.amplitude,
        args.singularity_limit_km,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
