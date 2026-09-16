#!/usr/bin/env python3
"""Plot ReMIX equivalent currents and calcdb ground perturbations over Ireland."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "gamera_ireland/environment/matplotlib"))

import matplotlib.pyplot as plt

from gamera_ireland.coordinates import delta_h  # noqa: E402
from gamera_ireland.config import load_json_config  # noqa: E402
from make_ireland_grid import build_grid  # noqa: E402

try:
    from astropy.time import Time
except ImportError:  # pragma: no cover
    Time = None


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _mjd_to_iso(mjd: float) -> str:
    if Time is None:
        return f"MJD {mjd:.6f}"
    return Time(float(mjd), format="mjd").isot


def _lon_360_to_180(lon_deg: np.ndarray) -> np.ndarray:
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _cell_centres_from_vertices(values: np.ndarray) -> np.ndarray:
    """Return latitude/longitude cell centres for a latitude-vertex, longitude-periodic grid."""
    return 0.5 * (values[:-1, :] + values[1:, :])


def load_remix_current_proxy(mix_h5: Path, step: str) -> dict[str, np.ndarray | str]:
    """Load a ReMIX northern-hemisphere stream-function current proxy.

    The smoke run has zero Hall conductance at the selected step. For a
    divergence-free diagnostic map, use ionospheric potential as a stream
    function and plot zhat x grad(psi). This is a real ReMIX field-derived
    equivalent-current proxy, not a full SECS inversion.
    """
    earth_radius_m = 6378.1e3
    with h5py.File(mix_h5, "r") as handle:
        group = handle[step]
        lat_raw = np.rad2deg(group["Latitude (Geographic) NORTH"][...])
        # In this ReMIX output the northern grid coordinate is stored as
        # colatitude despite the dataset label, so convert to geographic latitude.
        lat_v = 90.0 - lat_raw
        lon_v = _lon_360_to_180(np.rad2deg(group["Longitude (Geographic) NORTH"][...]))
        psi = group["Potential NORTH"][...] * 1.0e3
        pedersen = group["Pedersen conductance NORTH"][...]
        hall = group["Hall conductance NORTH"][...]
        mjd = group.attrs.get("MJD")

    lat = _cell_centres_from_vertices(lat_v)
    lon = _cell_centres_from_vertices(lon_v)

    # ReMIX longitude is periodic. Unwrap for gradients, then wrap for plotting.
    lon_unwrapped_rad = np.unwrap(np.deg2rad(lon), axis=1)
    lat_rad = np.deg2rad(lat)

    dpsi_dlat = np.gradient(psi, axis=0) / np.gradient(lat_rad[:, 0])[:, None]
    dpsi_dlon = np.gradient(psi, axis=1) / np.gradient(lon_unwrapped_rad, axis=1)

    grad_north = dpsi_dlat / earth_radius_m
    grad_east = dpsi_dlon / (earth_radius_m * np.cos(lat_rad))

    # Divergence-free horizontal current proxy: J = zhat x grad(psi).
    j_east = -grad_north
    j_north = grad_east

    return {
        "lat": lat,
        "lon": lon,
        "j_east": j_east,
        "j_north": j_north,
        "j_mag": np.sqrt(j_east**2 + j_north**2),
        "pedersen": pedersen,
        "hall": hall,
        "timestamp": _mjd_to_iso(mjd) if mjd is not None else step,
    }


def load_calcdb_ground(deltab_h5: Path, step: str) -> dict[str, np.ndarray | str]:
    with h5py.File(deltab_h5, "r") as handle:
        group = handle[step]
        lat = group["smlat"][0, ...]
        lon = _lon_360_to_180(group["smlon"][0, ...])
        d_bn = group["dBn"][0, ...]
        d_be = group["dBp"][0, ...]
        d_bu = group["dBr"][0, ...]
        d_bt = group["dBt"][0, ...]
        mjd = group.attrs.get("MJD")

    return {
        "lat": lat,
        "lon": lon,
        "d_bn": d_bn,
        "d_be": d_be,
        "d_bu": d_bu,
        "d_bt": d_bt,
        "d_bh": delta_h(d_bn, d_be),
        "timestamp": _mjd_to_iso(mjd) if mjd is not None else step,
    }


def interpolate_to_points(lon, lat, target_lon, target_lat, fields):
    points = np.column_stack([lon.ravel(), lat.ravel()])
    out = {}
    for name, values in fields.items():
        out[name] = griddata(points, values.ravel(), (target_lon, target_lat), method="linear")
    return out


def geo_vectors_to_cube(node_grid, east, north):
    """Rotate geographic east/north vectors into the cubed-sphere local frame."""
    rotation = node_grid.projection.local2geo_enu_rotation(
        node_grid.lon.flatten(),
        node_grid.lat.flatten(),
    )
    vectors_geo = np.column_stack([east.flatten(), north.flatten()])
    vectors_cube = np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), vectors_geo)
    return vectors_cube[:, 0].reshape(east.shape), vectors_cube[:, 1].reshape(north.shape)


def add_reference_stations(ax: plt.Axes, path: Path, node_grid) -> None:
    if not path.exists():
        return
    stations = pd.read_csv(path)
    lon_candidates = ["longitude", "lon", "glon_deg_signed", "glon"]
    lat_candidates = ["latitude", "lat", "glat_deg", "glat"]
    lon_col = next((name for name in lon_candidates if name in stations.columns), None)
    lat_col = next((name for name in lat_candidates if name in stations.columns), None)
    if lon_col is None or lat_col is None:
        return
    name_col = "name" if "name" in stations.columns else stations.columns[0]
    station_lon = _lon_360_to_180(stations[lon_col].to_numpy(dtype=float))
    station_lat = stations[lat_col].to_numpy(dtype=float)
    station_xi, station_eta = node_grid.projection.geo2local(station_lon, station_lat)
    ax.scatter(station_xi, station_eta, s=28, c="white", edgecolor="black", zorder=5)
    for _, row in stations.iterrows():
        lon = float(_lon_360_to_180(np.asarray(row[lon_col], dtype=float)))
        xi, eta = node_grid.projection.geo2local(
            np.asarray([lon]),
            np.asarray([float(row[lat_col])]),
        )
        xi = float(xi[0])
        eta = float(eta[0])
        ax.text(xi + 0.0007, eta + 0.0005, str(row[name_col]), fontsize=7, zorder=6)


def add_coastlines(ax: plt.Axes, node_grid) -> None:
    try:
        node_grid.add_natural_earth_feature(ax, "coastline", resolution="10m", color="0.15", linewidth=0.8)
    except Exception as exc:
        print(f"Warning: could not add coastline: {exc}")


def make_figure(current, ground, output_png: Path, stations_csv: Path, node_grid) -> None:
    target_lon = _lon_360_to_180(node_grid.lon)
    target_lat = node_grid.lat
    cfields = interpolate_to_points(
        current["lon"],
        current["lat"],
        target_lon,
        target_lat,
        {
            "j_east": current["j_east"],
            "j_north": current["j_north"],
            "j_mag": current["j_mag"],
        },
    )
    bfields = interpolate_to_points(
        ground["lon"],
        ground["lat"],
        target_lon,
        target_lat,
        {
            "d_bn": ground["d_bn"],
            "d_be": ground["d_be"],
            "d_bh": ground["d_bh"],
            "d_bu": ground["d_bu"],
        },
    )
    j_xi, j_eta = geo_vectors_to_cube(node_grid, cfields["j_east"], cfields["j_north"])
    b_xi, b_eta = geo_vectors_to_cube(node_grid, bfields["d_be"], bfields["d_bn"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)

    ax = axes[0]
    mesh = ax.pcolormesh(node_grid.xi_mesh, node_grid.eta_mesh, cfields["j_mag"] * 1e3, shading="auto", cmap="viridis")
    qskip = (slice(None, None, 7), slice(None, None, 7))
    ax.quiver(
        node_grid.xi[qskip],
        node_grid.eta[qskip],
        j_xi[qskip],
        j_eta[qskip],
        color="black",
        scale=0.0015,
        width=0.003,
    )
    add_coastlines(ax, node_grid)
    add_reference_stations(ax, stations_csv, node_grid)
    fig.colorbar(mesh, ax=ax, label="|Jdf proxy| (mA/m)")
    ax.set_title("ReMIX divergence-free current proxy")

    ax = axes[1]
    vmax = float(np.nanmax(np.abs(bfields["d_bu"])))
    mesh = ax.pcolormesh(
        node_grid.xi_mesh,
        node_grid.eta_mesh,
        bfields["d_bu"],
        shading="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.contour(node_grid.xi, node_grid.eta, bfields["d_bh"], colors="0.15", linewidths=0.7, alpha=0.65)
    qskip = (slice(None, None, 8), slice(None, None, 8))
    ax.quiver(
        node_grid.xi[qskip],
        node_grid.eta[qskip],
        b_xi[qskip],
        b_eta[qskip],
        color="black",
        scale=22,
        width=0.003,
    )
    add_coastlines(ax, node_grid)
    add_reference_stations(ax, stations_csv, node_grid)
    fig.colorbar(mesh, ax=ax, label="Upward Delta B, dBr (nT)")
    ax.set_title("calcdb ground perturbation")

    for ax in axes:
        ax.set_xlim(node_grid.xi_min, node_grid.xi_max)
        ax.set_ylim(node_grid.eta_min, node_grid.eta_max)
        ax.set_xlabel("Cubed-sphere xi")
        ax.set_ylabel("Cubed-sphere eta")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    title = (
        f"Ireland May 2024 MAGE diagnostic map\n"
        f"ReMIX {current['timestamp']} UTC, calcdb {ground['timestamp']} UTC"
    )
    fig.suptitle(title, fontsize=13)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def write_summary(path: Path, current, ground, mix_h5: Path, deltab_h5: Path) -> None:
    summary = {
        "mix_h5": str(mix_h5),
        "deltab_h5": str(deltab_h5),
        "current_timestamp": current["timestamp"],
        "ground_timestamp": ground["timestamp"],
        "current_proxy_note": (
            "Divergence-free proxy J = zhat x grad(ReMIX Potential NORTH). "
            "This is not yet a fitted SECS inversion."
        ),
        "max_current_proxy_mA_per_m_ireland": float(np.nanmax(current["j_mag"]) * 1e3),
        "max_delta_h_nT_global": float(np.nanmax(ground["d_bh"])),
        "max_abs_delta_bz_nT_global": float(np.nanmax(np.abs(ground["d_bu"]))),
        "magnetic_colour_field": "calcdb dBr, radial/upward perturbation in nT, seismic colormap",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    run_dir = PROJECT_DIR / "gamera_ireland/simulations/may2024_ireland"
    parser.add_argument("--mix", type=Path, default=run_dir / "may2024_smoke.mix.h5")
    parser.add_argument("--deltab", type=Path, default=run_dir / "may2024_smoke.deltab.h5")
    parser.add_argument("--mix-step", default="Step#1")
    parser.add_argument("--deltab-step", default="Step#1")
    parser.add_argument(
        "--stations",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/config/ireland_reference_stations.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/output/figures/may2024_smoke_ireland_currents_groundB.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/output/figures/may2024_smoke_ireland_currents_groundB.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json_config(args.config)
    node_grid, _ = build_grid(config["ireland_grid"])
    current = load_remix_current_proxy(args.mix, args.mix_step)
    ground = load_calcdb_ground(args.deltab, args.deltab_step)
    make_figure(current, ground, args.output, args.stations, node_grid)
    write_summary(args.summary, current, ground, args.mix, args.deltab)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
