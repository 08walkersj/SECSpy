#!/usr/bin/env python3
"""Render SECS inversion frames using SECSpy's time_loop methodology."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "gamera_ireland/environment/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from apexpy import Apex

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from SECSpy.Plotting_Tools import save_time_loop_figure, time_loop  # noqa: E402
from gamera_ireland.config import load_json_config  # noqa: E402


def _values_t(values):
    data = np.asarray(values)
    units = str(getattr(values, "attrs", {}).get("units", "")).lower()
    if units == "nt":
        return data * 1e-9
    return data


def _values_a_per_m(values):
    data = np.asarray(values)
    units = str(getattr(values, "attrs", {}).get("units", "")).lower()
    if units == "ma/m":
        return data * 1e-3
    return data


def _remove_dynamic(ax, names):
    for name in names:
        artists = getattr(ax, name, None)
        if artists is None:
            continue
        if not isinstance(artists, (list, tuple)):
            artists = [artists]
        stack = list(artists)
        while stack:
            artist = stack.pop()
            if isinstance(artist, (list, tuple)):
                stack.extend(artist)
            elif hasattr(artist, "remove"):
                try:
                    artist.remove()
                except ValueError:
                    pass
            elif hasattr(artist, "collections"):
                for item in artist.collections:
                    try:
                        item.remove()
                    except ValueError:
                        pass
        setattr(ax, name, [] if name in {"clon", "clat"} else None)


def create_mage_secs_figure(node_grid, br_lim=10.0, current_lim=1.0, figsize=(13.5, 6.2)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    mag_ax, cur_ax = axes
    mag_ax.set_title("Ground Magnetic Field")
    cur_ax.set_title("Divergence-Free SECS Current")

    for ax in axes:
        ax.set_xlim(node_grid.xi_min, node_grid.xi_max)
        ax.set_ylim(node_grid.eta_min, node_grid.eta_max)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.coastlines = node_grid.add_natural_earth_feature(ax, "coastline", resolution="10m", color="0.15", linewidth=0.8)
        ax.Br = None
        ax.Q_B = None
        ax.Mag_scatter = None
        ax.Q_B2 = None
        ax.Q_J = None
        ax.A = None
        ax.clon = []
        ax.clat = []

    mag_cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-br_lim, vmax=br_lim), cmap="seismic"),
        ax=mag_ax,
        orientation="vertical",
        extend="both",
    )
    mag_cbar.set_label("Upward Delta B, dBr (nT)")
    cur_cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-current_lim, vmax=current_lim), cmap="seismic"),
        ax=cur_ax,
        orientation="vertical",
        extend="both",
    )
    cur_cbar.set_label("SECS amplitude / cell area")
    return fig


def mage_secs_frame(
    amplitudes,
    estimated,
    event_time,
    time,
    SM_data,
    Apex,
    fig=None,
    node_grid=None,
    br_lim=10.0,
    current_lim=1.0,
    plot_measured=False,
):
    if fig is None:
        fig = create_mage_secs_figure(node_grid, br_lim=br_lim, current_lim=current_lim)

    mag_ax, cur_ax = fig.axes[0], fig.axes[1]
    _remove_dynamic(mag_ax, ["Br", "Q_B", "Mag_scatter", "Q_B2", "clon", "clat"])
    _remove_dynamic(cur_ax, ["A", "Q_J", "clon", "clat"])

    absolute_time = pd.Timestamp(np.datetime64(pd.Timestamp(event_time).to_datetime64(), "ns") + time)
    frame = estimated.sel(time=absolute_time)
    amp_frame = amplitudes.sel(time=absolute_time)
    area = amplitudes.Area

    mag_ax.Br = mag_ax.pcolormesh(
        estimated.xi_mesh.values.T,
        estimated.eta_mesh.values.T,
        frame.Bu.values.T,
        cmap="seismic",
        vmin=-br_lim,
        vmax=br_lim,
        zorder=-1,
    )
    skip = 5
    be = frame.Be_cs.transpose("eta", "xi").isel(eta=slice(None, None, skip), xi=slice(None, None, skip))
    bn = frame.Bn_cs.transpose("eta", "xi").isel(eta=slice(None, None, skip), xi=slice(None, None, skip))
    mag_ax.Q_B = mag_ax.quiver(
        be.xi.values,
        be.eta.values,
        _values_t(be),
        _values_t(bn),
        scale=6e-6,
        zorder=100,
    )
    sm_frame = SM_data.loc[SM_data.index == absolute_time]
    if plot_measured and not sm_frame.empty:
        first = lambda column: sm_frame.loc[:, column].to_numpy().reshape(-1)
        mag_ax.Mag_scatter = mag_ax.scatter(
            first("xi"),
            first("eta"),
            c=first("Up_ground") * 1e9,
            cmap="seismic",
            vmin=-br_lim,
            vmax=br_lim,
            s=35,
            marker="o",
            edgecolors="black",
            zorder=120,
        )
        mag_ax.Q_B2 = mag_ax.quiver(
            first("xi"),
            first("eta"),
            first("Be_cs") if "Be_cs" in sm_frame.columns.get_level_values(0) else first("East_ground"),
            first("Bn_cs") if "Bn_cs" in sm_frame.columns.get_level_values(0) else -first("South_ground"),
            scale=6e-6,
            color="darkorange",
            zorder=150,
        )

    cur_ax.A = cur_ax.pcolormesh(
        amplitudes.xi_mesh.values.T,
        amplitudes.eta_mesh.values.T,
        (amp_frame.Amplitude / area).values.T,
        cmap="seismic",
        vmin=-current_lim,
        vmax=current_lim,
        zorder=-1,
    )
    je = frame.Je_cs.transpose("eta", "xi").isel(eta=slice(None, None, skip), xi=slice(None, None, skip))
    jn = frame.Jn_cs.transpose("eta", "xi").isel(eta=slice(None, None, skip), xi=slice(None, None, skip))
    cur_ax.Q_J = cur_ax.quiver(
        je.xi.values,
        je.eta.values,
        _values_a_per_m(je),
        _values_a_per_m(jn),
        scale=18,
        width=0.003,
        zorder=100,
    )

    lat = amplitudes.lat.values
    lon = amplitudes.lon.values
    mlat, mlt = Apex.convert(lat, lon, source="GEO", dest="mlt", datetime=absolute_time, height=80)
    mlt = np.asarray(mlt)
    mlt[np.asarray(mlat) > 80] = np.nan
    mlt[mlt > 12] -= 24
    for ax in [mag_ax, cur_ax]:
        ax.clon = ax.contour(amplitudes.xi.values, amplitudes.eta.values, mlt.T, levels=np.arange(-12, 13, 3), colors="black", alpha=0.35, linewidths=0.45)
        ax.clat = ax.contour(amplitudes.xi.values, amplitudes.eta.values, np.asarray(mlat).T, levels=np.arange(40, 90, 10), colors="black", alpha=0.35, linewidths=0.45)
    fig.suptitle(f"{absolute_time.strftime('%Y-%m-%d %H:%M:%S')} UTC", fontsize=16)
    return fig


def add_station_cube_vectors(sm_data, node_grid):
    east = sm_data.loc[:, "East_ground"].to_numpy()
    north = -sm_data.loc[:, "South_ground"].to_numpy()
    lon = sm_data.loc[:, "glon"].to_numpy()
    lat = sm_data.loc[:, "glat"].to_numpy()
    be_cs, bn_cs = node_grid.projection.vector_cube_projection(east, north, lon, lat, return_xi_eta=False)
    be_cs = np.asarray(be_cs).reshape(east.shape)
    bn_cs = np.asarray(bn_cs).reshape(north.shape)
    be = pd.DataFrame(be_cs, index=sm_data.index, columns=sm_data.loc[:, "East_ground"].columns)
    bn = pd.DataFrame(bn_cs, index=sm_data.index, columns=sm_data.loc[:, "South_ground"].columns)
    be.columns = pd.MultiIndex.from_product([["Be_cs"], be.columns], names=sm_data.columns.names)
    bn.columns = pd.MultiIndex.from_product([["Bn_cs"], bn.columns], names=sm_data.columns.names)
    return pd.concat([sm_data, be, bn], axis=1)


def encode_video(frames_root: Path, event_day: str, output: Path, framerate: float) -> None:
    frame_pattern = frames_root / event_day / "%d.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(framerate),
            "-i",
            str(frame_pattern),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    product_dir = PROJECT_DIR / "gamera_ireland/output/optimisation/may2024_storm1832_secs"
    out_dir = PROJECT_DIR / "gamera_ireland/output/figures/may2024_storm1832_secs_time_loop"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=product_dir)
    parser.add_argument("--frames-root", type=Path, default=out_dir / "frames")
    parser.add_argument("--output", type=Path, default=out_dir / "may2024_storm1832_secs_time_loop.mp4")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json")
    parser.add_argument("--framerate", type=float, default=1.0)
    parser.add_argument("--br-lim", type=float, default=10.0)
    parser.add_argument("--current-lim", type=float, default=0.15)
    parser.add_argument("--plot-measured", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    amplitudes = xr.open_dataset(args.products / "amplitudes.nc")
    estimated = xr.open_dataset(args.products / "estimated.nc")
    sm_data = pd.read_csv(args.products / "virtual_magnetometers.csv", header=[0, 1], index_col=0, parse_dates=True)

    from secsy.cubedsphere import CSgrid, CSprojection
    import json

    grid_dict = json.loads(amplitudes.attrs["cs_grid"])
    node_grid = CSgrid(projection=CSprojection(**grid_dict.pop("projection")), **grid_dict)
    sm_data = add_station_cube_vectors(sm_data, node_grid)
    config = load_json_config(args.config)
    apex = Apex(pd.Timestamp(config["ireland_grid"]["apex_time"]))
    event_time = pd.Timestamp(amplitudes.time.values[0])

    from SECSpy.utils import date2filename

    event_day = date2filename(np.datetime64(event_time.to_datetime64()), resolution="D")
    time_loop(
        amplitudes,
        estimated,
        functions=[mage_secs_frame, save_time_loop_figure],
        event_time=event_time,
        SM_data=sm_data,
        Apex=apex,
        node_grid=node_grid,
        save_root=str(args.frames_root),
        save_template="{frame_index}.png",
        savefig_kwargs={"dpi": 180, "bbox_inches": "tight"},
        br_lim=args.br_lim,
        current_lim=args.current_lim,
        plot_measured=args.plot_measured,
        progress=True,
    )
    encode_video(args.frames_root, event_day, args.output, args.framerate)
    print(f"Wrote frames under {args.frames_root / event_day}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
