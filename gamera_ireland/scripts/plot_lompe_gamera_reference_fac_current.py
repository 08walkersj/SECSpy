#!/usr/bin/env python3
"""Plot FAC and signed Hall electrojet from the LOMPE GAMERA reference file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "environment" / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def polar_xy(mlat: np.ndarray, mlt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = 90.0 - mlat
    angle = np.deg2rad((12.0 - mlt) * 15.0)
    return r * np.sin(angle), r * np.cos(angle)


def make_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.r_[first, mids, last]


def add_guides(ax: plt.Axes) -> None:
    phi = np.linspace(0, 2 * np.pi, 361)
    for mlat in [80, 70, 60, 50]:
        r = 90.0 - mlat
        ax.plot(r * np.sin(phi), r * np.cos(phi), ":", color="0.55", lw=0.6)
        ax.text(0, r, f"{mlat}", ha="left", va="bottom", fontsize=8, color="0.35")
    for label, angle in [(12, 0), (6, np.pi / 2), (0, np.pi), (18, 3 * np.pi / 2)]:
        ax.text(47 * np.sin(angle), 47 * np.cos(angle), f"{label:02d}", ha="center", va="center", fontsize=9)


def plot_reference(nc: Path, output: Path, summary: Path) -> None:
    ds = xr.open_dataset(nc)
    mlat = ds["mlat"].values
    mlt = ds["mlt"].values
    mlat_g, mlt_g = np.meshgrid(mlat, mlt, indexing="ij")
    x, y = polar_xy(mlat_g, mlt_g)

    mlat_edges = make_edges(mlat)
    r_edges = 90.0 - mlat_edges
    angle_edges = -2.0 * np.pi * np.arange(mlt.size + 1) / mlt.size
    rg, ag = np.meshgrid(r_edges, angle_edges, indexing="ij")
    xe = rg * np.sin(ag)
    ye = rg * np.cos(ag)

    fac = ds["FAC"].values
    sigma_h = ds["SigmaHall"].values
    e_theta = ds["Etheta"].values
    e_phi = ds["Ephi"].values
    theta = np.deg2rad(90.0 - mlat_g)
    nhcdip = -2.0 * np.cos(theta) / np.sqrt(1.0 + 3.0 * np.cos(theta) ** 2)
    hall_theta = -sigma_h * e_phi / nhcdip
    hall_phi = sigma_h * e_theta / nhcdip

    angle = np.deg2rad((12.0 - mlt_g) * 15.0)
    hall_x = hall_theta * np.sin(angle) - hall_phi * np.cos(angle)
    hall_y = hall_theta * np.cos(angle) + hall_phi * np.sin(angle)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), constrained_layout=True)
    fac_lim = max(0.25, float(np.nanpercentile(np.abs(fac), 99)))
    im = axes[0].pcolormesh(xe, ye, fac, cmap="seismic", vmin=-fac_lim, vmax=fac_lim, shading="flat")
    axes[0].contour(x, y, ds["potential"].values, levels=np.r_[-200:201:10], colors="black", linewidths=0.55)
    fig.colorbar(im, ax=axes[0], label="FAC [$\\mu$A m$^{-2}$]")
    axes[0].set_title("LOMPE GAMERA reference FAC")

    jphi = hall_phi * 1e3
    j_lim = max(20.0, float(np.nanpercentile(np.abs(jphi), 99)))
    im = axes[1].pcolormesh(xe, ye, jphi, cmap="seismic", vmin=-j_lim, vmax=j_lim, shading="flat")
    skip = (slice(None, None, 2), slice(None, None, 8))
    mag = np.hypot(hall_theta, hall_phi)
    mag95 = max(float(np.nanpercentile(mag, 95)), 1e-12)
    gain = 1.35 * np.clip(mag / mag95, 0.0, 1.6)
    axes[1].quiver(
        x[skip],
        y[skip],
        hall_x[skip] / np.maximum(mag[skip], 1e-12) * gain[skip],
        hall_y[skip] / np.maximum(mag[skip], 1e-12) * gain[skip],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0021,
    )
    axes[1].contour(x, y, ds["potential"].values, levels=np.r_[-200:201:10], colors="white", linewidths=0.45, alpha=0.65)
    fig.colorbar(im, ax=axes[1], label="Hall $J_\\phi$ [mA m$^{-1}$]")
    axes[1].set_title("LOMPE GAMERA signed Hall electrojet")

    for ax in axes:
        add_guides(ax)
        ax.set_aspect("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    report = {
        "source": str(nc),
        "time": str(ds["time"].values),
        "fac_min_max": [float(np.nanmin(fac)), float(np.nanmax(fac))],
        "hall_jphi_min_max_ma_m": [float(np.nanmin(jphi)), float(np.nanmax(jphi))],
        "orientation": "12 MLT top, 06 right, 00 bottom, 18 left, using explicit LOMPE mlt coordinate",
    }
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc", type=Path, default=Path("/home/simon/gits/lompe/examples/sample_dataset/gamera.nc"))
    parser.add_argument("--output", type=Path, default=Path("gamera_ireland/output/figures/lompe_reference_fac_signed_hall.png"))
    parser.add_argument("--summary", type=Path, default=Path("gamera_ireland/output/figures/lompe_reference_fac_signed_hall.json"))
    args = parser.parse_args()
    plot_reference(args.nc, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
