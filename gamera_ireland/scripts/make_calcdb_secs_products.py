#!/usr/bin/env python3
"""Invert calcdb virtual magnetometers with divergence-free SECS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from apexpy import Apex
from scipy.interpolate import griddata
from secsy import cubedsphere as cs

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from SECSpy.constants import RE  # noqa: E402
from SECSpy.SECS import SECS  # noqa: E402
from gamera_ireland.config import load_json_config  # noqa: E402
from plot_mage_ireland_maps import load_calcdb_ground  # noqa: E402
from make_ireland_grid import build_grid  # noqa: E402


def make_grid(config_path: Path, length_km: float, width_km: float, res_km: float):
    config = load_json_config(config_path)
    grid_config = dict(config["ireland_grid"])
    grid_config["L_km"] = length_km
    grid_config["W_km"] = width_km
    grid_config["Lres_km"] = res_km
    grid_config["Wres_km"] = res_km
    return build_grid(grid_config)[0], config


def eval_grid_from_node_grid(node_grid, res_km: float):
    return cs.CSgrid(
        node_grid.projection,
        node_grid.L,
        node_grid.W,
        res_km,
        res_km,
        R=node_grid.R,
    )


def step_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        names = [name for name in handle if name.startswith("Step#")]
    return sorted(names, key=lambda name: int(name.split("#", 1)[1]))


def interpolate_ground_to_grid(deltab: Path, step: str, grid):
    ground = load_calcdb_ground(deltab, step)
    points = np.column_stack([ground["lon"].ravel(), ground["lat"].ravel()])
    target = (grid.lon, grid.lat)
    fields = {
        "South_ground": -griddata(points, ground["d_bn"].ravel(), target, method="linear") * 1e-9,
        "East_ground": griddata(points, ground["d_be"].ravel(), target, method="linear") * 1e-9,
        "Up_ground": griddata(points, ground["d_bu"].ravel(), target, method="linear") * 1e-9,
    }
    return ground, fields


def build_observation_table(deltab: Path, steps: list[str], station_grid, station_skip: int):
    rows = []
    data_vectors = []
    station_mask = np.zeros(station_grid.shape, dtype=bool)
    station_mask[::station_skip, ::station_skip] = True
    glon = station_grid.lon[station_mask]
    glat = station_grid.lat[station_mask]
    xi = station_grid.xi[station_mask]
    eta = station_grid.eta[station_mask]
    station_ids = np.array([f"V{index:04d}" for index in range(glon.size)])

    for step in steps:
        ground, fields = interpolate_ground_to_grid(deltab, step, station_grid)
        south = fields["South_ground"][station_mask]
        east = fields["East_ground"][station_mask]
        up = fields["Up_ground"][station_mask]
        if np.isnan(south).any() or np.isnan(east).any() or np.isnan(up).any():
            raise ValueError(f"NaN virtual magnetometer values for {step}; enlarge calcdb grid or reduce domain.")
        data_vectors.append(np.concatenate([south, east, up]))
        timestamp = np.datetime64(pd.Timestamp(ground["timestamp"]).to_datetime64(), "ns")
        for i, station_id in enumerate(station_ids):
            rows.append(
                {
                    "Date_UTC": pd.Timestamp(timestamp),
                    "iaga": station_id,
                    "glon": glon[i],
                    "glat": glat[i],
                    "xi": xi[i],
                    "eta": eta[i],
                    "South_ground": south[i],
                    "East_ground": east[i],
                    "Up_ground": up[i],
                }
            )

    table = pd.DataFrame(rows)
    table = table.pivot(index="Date_UTC", columns="iaga")
    table.columns.names = [None, "iaga"]
    return table, np.vstack(data_vectors), station_ids, glon, glat, xi, eta


def main() -> int:
    args = parse_args()
    steps = step_names(args.deltab)
    if args.max_steps is not None:
        steps = steps[: args.max_steps]
    if not steps:
        raise ValueError(f"No Step# groups found in {args.deltab}")

    node_grid, config = make_grid(args.config, args.length_km, args.width_km, args.pole_res_km)
    eval_grid = eval_grid_from_node_grid(node_grid, args.eval_res_km)
    station_grid = eval_grid_from_node_grid(node_grid, args.station_res_km)
    observations, data, station_ids, glon, glat, xi, eta = build_observation_table(
        args.deltab,
        steps,
        station_grid,
        args.station_skip,
    )

    event_time = pd.Timestamp(config["ireland_grid"]["apex_time"])
    apex = Apex(event_time)
    poles = SECS(
        node_grid.lon.flatten(),
        node_grid.lat.flatten(),
        eval_grid.lon.flatten(),
        eval_grid.lat.flatten(),
        mode="image",
        image_current_radius=RE - 500e3,
        current_radius=RE + 110e3,
    )
    poles.Fitting_Matrix_Regularisation(
        l1=args.l1,
        l2=args.l2,
        grid=node_grid,
        glon=glon,
        glat=glat,
        Apex=apex,
        block_rows=args.block_rows,
    )
    amps = poles.time_inversion(data)
    times = observations.index.values.astype("datetime64[ns]")
    amplitudes = poles.amps_to_xarray(
        amps,
        times,
        node_grid,
        mag_code=station_ids,
        meas_glon=glon,
        meas_glat=glat,
    )
    estimated = poles.estimated_to_xarray(
        times,
        eval_grid,
        node_grid,
        amps=amps,
        mag_code=station_ids,
        meas_glon=glon,
        meas_glat=glat,
        meas_xi=xi,
        meas_eta=eta,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    amplitudes_path = args.output_dir / "amplitudes.nc"
    estimated_path = args.output_dir / "estimated.nc"
    observations_path = args.output_dir / "virtual_magnetometers.csv"
    metadata_path = args.output_dir / "metadata.json"
    amplitudes.to_netcdf(amplitudes_path)
    estimated.to_netcdf(estimated_path)
    observations.to_csv(observations_path)
    metadata = {
        "deltab_h5": str(args.deltab),
        "steps": steps,
        "times": [str(pd.Timestamp(t)) for t in times],
        "grid": {
            "length_km": args.length_km,
            "width_km": args.width_km,
            "pole_res_km": args.pole_res_km,
            "eval_res_km": args.eval_res_km,
            "station_res_km": args.station_res_km,
            "station_skip": args.station_skip,
            "virtual_station_count": int(len(station_ids)),
            "node_shape": list(node_grid.shape),
            "eval_shape": list(eval_grid.shape),
        },
        "regularisation": {"l1": args.l1, "l2": args.l2},
        "component_convention": "calcdb dBn north, dBp east, dBr up; inversion uses South=-dBn, East=dBp, Up=dBr in tesla",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {amplitudes_path}")
    print(f"Wrote {estimated_path}")
    print(f"Wrote {observations_path}")
    print(f"Wrote {metadata_path}")
    return 0


def parse_args() -> argparse.Namespace:
    run_dir = PROJECT_DIR / "gamera_ireland/simulations/may2024_ireland"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltab", type=Path, default=run_dir / "may2024_storm1832.deltab.h5")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "gamera_ireland/output/optimisation/may2024_storm1832_secs")
    parser.add_argument("--length-km", type=float, default=900.0)
    parser.add_argument("--width-km", type=float, default=900.0)
    parser.add_argument("--pole-res-km", type=float, default=75.0)
    parser.add_argument("--eval-res-km", type=float, default=25.0)
    parser.add_argument("--station-res-km", type=float, default=25.0)
    parser.add_argument("--station-skip", type=int, default=2)
    parser.add_argument("--l1", type=float, default=3e3)
    parser.add_argument("--l2", type=float, default=6e3)
    parser.add_argument("--block-rows", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
