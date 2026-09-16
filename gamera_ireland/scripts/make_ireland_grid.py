#!/usr/bin/env python3
"""Create the Ireland cubed-sphere virtual magnetometer grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

import numpy as np
import pandas as pd
from apexpy import Apex
from secsy import cubedsphere as cs

from gamera_ireland.config import load_json_config, resolve_project_path
from gamera_ireland.virtual_stations import REFERENCE_STATIONS


def signed_longitude(lon: np.ndarray) -> np.ndarray:
    """Convert 0-360 longitudes to -180..180."""
    return ((lon + 180.0) % 360.0) - 180.0


def build_grid(grid_config: dict) -> tuple[cs.CSgrid, float]:
    """Build the user-specified Apex-aligned cubed-sphere grid."""
    apex = Apex(pd.Timestamp(grid_config["apex_time"]))
    lon_centre = float(grid_config["lon_centre"])
    lat_centre = float(grid_config["lat_centre"])
    _, f2 = apex.basevectors_qd(lat_centre, lon_centre, 0, coords="geo")
    qd_north = f2 / np.linalg.norm(f2)
    east, north = qd_north[0], qd_north[1]
    orientation = float(np.rad2deg(np.arctan2(east, north)))
    projection = cs.CSprojection((lon_centre, lat_centre), orientation)
    grid = cs.CSgrid(
        projection,
        float(grid_config["L_km"]),
        float(grid_config["W_km"]),
        float(grid_config["Lres_km"]),
        float(grid_config["Wres_km"]),
    )
    return grid, orientation


def write_grid(config: dict) -> None:
    """Write virtual station and grid metadata files."""
    grid, orientation = build_grid(config["ireland_grid"])
    paths = config["paths"]
    grid_csv = resolve_project_path(paths["grid_csv"], PROJECT_DIR)
    metadata_json = resolve_project_path(paths["grid_metadata"], PROJECT_DIR)
    stations_csv = resolve_project_path(paths["reference_stations_csv"], PROJECT_DIR)

    grid_csv.parent.mkdir(parents=True, exist_ok=True)

    lon_signed = signed_longitude(grid.lon.flatten())
    grid_df = pd.DataFrame(
        {
            "station_id": [f"IEV{i:04d}" for i in range(grid.lon.size)],
            "glon_deg_east_0_360": grid.lon.flatten(),
            "glon_deg_signed": lon_signed,
            "glat_deg": grid.lat.flatten(),
            "xi": grid.xi.flatten(),
            "eta": grid.eta.flatten(),
        }
    )
    grid_df.to_csv(grid_csv, index=False)

    reference_df = pd.DataFrame(
        {
            "name": [station.name for station in REFERENCE_STATIONS],
            "glat_deg": [station.latitude for station in REFERENCE_STATIONS],
            "glon_deg_signed": [station.longitude for station in REFERENCE_STATIONS],
            "glon_deg_east_0_360": [
                station.longitude % 360.0 for station in REFERENCE_STATIONS
            ],
        }
    )
    reference_df.to_csv(stations_csv, index=False)

    metadata = {
        "apex_time": config["ireland_grid"]["apex_time"],
        "projection_centre": [
            config["ireland_grid"]["lon_centre"],
            config["ireland_grid"]["lat_centre"],
        ],
        "orientation_degrees": orientation,
        "shape": list(grid.lon.shape),
        "node_count": int(grid.lon.size),
        "longitude_signed_min": float(np.nanmin(lon_signed)),
        "longitude_signed_max": float(np.nanmax(lon_signed)),
        "latitude_min": float(np.nanmin(grid.lat)),
        "latitude_max": float(np.nanmax(grid.lat)),
        "L_km": config["ireland_grid"]["L_km"],
        "W_km": config["ireland_grid"]["W_km"],
        "Lres_km": config["ireland_grid"]["Lres_km"],
        "Wres_km": config["ireland_grid"]["Wres_km"],
    }
    with metadata_json.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {grid_csv}")
    print(f"Wrote {stations_csv}")
    print(f"Wrote {metadata_json}")
    print(
        "Grid summary: "
        f"{metadata['node_count']} nodes, shape={metadata['shape']}, "
        f"lon={metadata['longitude_signed_min']:.3f}.."
        f"{metadata['longitude_signed_max']:.3f}, "
        f"lat={metadata['latitude_min']:.3f}..{metadata['latitude_max']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json",
        help="Event configuration JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_grid(load_json_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

