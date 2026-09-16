#!/usr/bin/env python3
"""Plot ReMIX FAC and Hall current on the native polar MLT grid.

This is a direct diagnostic of MAGE/ReMIX output. It does not run a SECS or
Lompe inversion, and it does not use the transformed geographic coordinate
arrays for the polar plot. ReMIX native phi has 12 MLT at phi=0.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "environment" / "matplotlib"),
)

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time


IONOSPHERE_RADIUS_M = 6500.0e3


def load_low_lat_boundary(xml_path: Path | None, fallback: float) -> float:
    if xml_path is None or not xml_path.exists():
        return fallback
    tree = ET.parse(xml_path)
    node = tree.find("./REMIX/grid")
    if node is None:
        return fallback
    return float(node.attrib.get("LowLatBoundary", fallback))


def load_remix(mix_h5: Path, step: str) -> dict:
    with h5py.File(mix_h5, "r") as handle:
        group = handle[step]
        return {
            "timestamp": Time(float(group.attrs["MJD"]), format="mjd").isot,
            "time_s": float(group.attrs["time"]),
            "nCPCP": float(group.attrs["nCPCP"]),
            "sCPCP": float(group.attrs["sCPCP"]),
            # Kaiju writes St%Vars(:,2:,v), and the HDF5 reader exposes this
            # transposed as (theta, phi). Potential is stored in kV.
            "potential_v": group["Potential NORTH"][...] * 1.0e3,
            "fac_ua_m2": group["Field-aligned current NORTH"][...],
            "hall_s": group["Hall conductance NORTH"][...],
        }


def native_grid(shape: tuple[int, int], low_lat_boundary_deg: float) -> dict:
    ntheta, nphi = shape
    dtheta = np.deg2rad(low_lat_boundary_deg / ntheta)
    theta = dtheta * (np.arange(ntheta) + 1.0)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    phig, thetag = np.meshgrid(phi, theta)

    theta_edges = np.empty(ntheta + 1)
    theta_edges[0] = 0.0
    theta_edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    theta_edges[-1] = theta[-1]
    phi_edges = 2.0 * np.pi * np.arange(nphi + 1) / nphi
    phi_edges_g, theta_edges_g = np.meshgrid(phi_edges, theta_edges)

    # Plot convention: 12 MLT at top, 06 MLT right, 00 MLT bottom, 18 MLT left.
    # Kaiju/ReMIX native phi=0 is noon and phi=pi/2 is dusk, so the display x
    # coordinate is -sin(phi), not +sin(phi).
    x = -np.rad2deg(thetag) * np.sin(phig)
    y = np.rad2deg(thetag) * np.cos(phig)
    xe = -np.rad2deg(theta_edges_g) * np.sin(phi_edges_g)
    ye = np.rad2deg(theta_edges_g) * np.cos(phi_edges_g)
    return {
        "theta": theta,
        "phi": phi,
        "thetag": thetag,
        "phig": phig,
        "x": x,
        "y": y,
        "xe": xe,
        "ye": ye,
        "dtheta_deg": np.rad2deg(dtheta),
    }


def derive_hall_current(remix: dict, grid: dict) -> dict:
    potential = np.asarray(remix["potential_v"], dtype=float)
    hall = np.asarray(remix["hall_s"], dtype=float)
    theta = grid["theta"]
    phi = grid["phi"]
    thetag = grid["thetag"]
    phig = grid["phig"]

    dpot_dtheta = np.gradient(potential, theta, axis=0, edge_order=2)
    dphi = phi[1] - phi[0]
    dpot_dphi = (np.roll(potential, -1, axis=1) - np.roll(potential, 1, axis=1)) / (2.0 * dphi)

    e_theta = -dpot_dtheta / IONOSPHERE_RADIUS_M
    e_phi = -dpot_dphi / (IONOSPHERE_RADIUS_M * np.maximum(np.sin(thetag), np.sin(np.deg2rad(1.0))))

    # Match Kaiju/calcdb's northern-hemisphere Hall-current convention,
    # including the dip-angle denominator used in calcdbutils.F90.
    nhcdip = -2.0 * np.cos(thetag) / np.sqrt(1.0 + 3.0 * np.cos(thetag) ** 2)
    hall_theta = -hall * e_phi / nhcdip
    hall_phi = hall * e_theta / nhcdip

    hall_x = -hall_theta * np.sin(phig) - hall_phi * np.cos(phig)
    hall_y = hall_theta * np.cos(phig) - hall_phi * np.sin(phig)
    return {
        "hall_theta": hall_theta,
        "hall_phi": hall_phi,
        "hall_x": hall_x,
        "hall_y": hall_y,
        "hall_mag": np.hypot(hall_theta, hall_phi),
    }


def add_native_guides(ax, low_lat_boundary_deg: float) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 361)
    for mlat in [80, 70, 60, 50]:
        r = 90.0 - mlat
        if r <= low_lat_boundary_deg:
            ax.plot(r * np.sin(phi), r * np.cos(phi), ":", color="0.55", lw=0.6)
            ax.text(0.0, r, f"{mlat}", ha="left", va="bottom", fontsize=8, color="0.35")

    for mlt, angle in [(12, 0.0), (18, np.pi / 2.0), (0, np.pi), (6, 3.0 * np.pi / 2.0)]:
        ax.plot(
            [0.0, -low_lat_boundary_deg * np.sin(angle)],
            [0.0, low_lat_boundary_deg * np.cos(angle)],
            ":",
            color="0.65",
            lw=0.6,
        )
        ax.text(
            -1.05 * low_lat_boundary_deg * np.sin(angle),
            1.05 * low_lat_boundary_deg * np.cos(angle),
            f"{mlt:02d}",
            ha="center",
            va="center",
            fontsize=9,
            color="0.25",
        )


def make_plot(
    mix_h5: Path,
    step: str,
    output: Path,
    summary_path: Path,
    low_lat_boundary_deg: float,
    theta_stride: int,
    phi_stride: int,
    quiver_length_deg: float,
) -> None:
    remix = load_remix(mix_h5, step)
    grid = native_grid(remix["fac_ua_m2"].shape, low_lat_boundary_deg)
    hall = derive_hall_current(remix, grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), constrained_layout=True)

    fac_limit = max(0.25, float(np.nanpercentile(np.abs(remix["fac_ua_m2"]), 99)))
    fac_mesh = axes[0].pcolormesh(
        grid["xe"],
        grid["ye"],
        remix["fac_ua_m2"],
        cmap="seismic",
        vmin=-fac_limit,
        vmax=fac_limit,
        shading="flat",
    )
    axes[0].contour(grid["x"], grid["y"], remix["potential_v"] * 1e-3, levels=np.r_[-200:201:10], colors="black", linewidths=0.55)
    fig.colorbar(fac_mesh, ax=axes[0], label="FAC [$\\mu$A m$^{-2}$]")
    axes[0].set_title("Field-aligned current")

    jphi_ma_m = hall["hall_phi"] * 1e3
    jphi_limit = max(20.0, float(np.nanpercentile(np.abs(jphi_ma_m), 99)))
    hall_mesh = axes[1].pcolormesh(
        grid["xe"],
        grid["ye"],
        jphi_ma_m,
        cmap="seismic",
        vmin=-jphi_limit,
        vmax=jphi_limit,
        shading="flat",
    )
    skip = (slice(None, None, theta_stride), slice(None, None, phi_stride))
    mag = hall["hall_mag"]
    mag95 = max(float(np.nanpercentile(mag, 95)), 1.0e-12)
    arrow_gain = quiver_length_deg * np.clip(mag / mag95, 0.0, 1.6)
    arrow_norm = np.maximum(mag, 1.0e-12)
    arrow_x = hall["hall_x"] / arrow_norm * arrow_gain
    arrow_y = hall["hall_y"] / arrow_norm * arrow_gain
    pole_vector_mask = grid["theta"][:, None] < np.deg2rad(2.5 * grid["dtheta_deg"])
    arrow_x = np.where(pole_vector_mask, np.nan, arrow_x)
    arrow_y = np.where(pole_vector_mask, np.nan, arrow_y)
    axes[1].quiver(
        grid["x"][skip],
        grid["y"][skip],
        arrow_x[skip],
        arrow_y[skip],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0021,
        headwidth=3.4,
        headlength=4.6,
        headaxislength=4.0,
        alpha=0.92,
    )
    axes[1].contour(grid["x"], grid["y"], remix["potential_v"] * 1e-3, levels=np.r_[-200:201:10], colors="white", linewidths=0.45, alpha=0.6)
    fig.colorbar(hall_mesh, ax=axes[1], label="Hall $J_\\phi$ [mA m$^{-1}$]")
    axes[1].set_title("Signed Hall electrojet")

    for ax in axes:
        add_native_guides(ax, low_lat_boundary_deg)
        ax.set_aspect("equal")
        ax.set_xlim(-low_lat_boundary_deg * 1.12, low_lat_boundary_deg * 1.12)
        ax.set_ylim(-low_lat_boundary_deg * 1.12, low_lat_boundary_deg * 1.12)
        ax.set_axis_off()

    fig.suptitle(
        f"{mix_h5.parent.name} {step}, {remix['timestamp']}  "
        f"CPCP(N/S)={remix['nCPCP']:.1f}/{remix['sCPCP']:.1f} kV"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "mix_h5": str(mix_h5),
        "step": step,
        "timestamp": remix["timestamp"],
        "time_s": remix["time_s"],
        "nCPCP_kV": remix["nCPCP"],
        "sCPCP_kV": remix["sCPCP"],
        "field_shape_theta_phi": list(remix["fac_ua_m2"].shape),
        "low_lat_boundary_colat_deg": low_lat_boundary_deg,
        "min_mlat_deg": 90.0 - low_lat_boundary_deg,
        "first_off_pole_colat_deg": grid["dtheta_deg"],
        "pole_handling": "The omitted ReMIX pole row is closed only for scalar colour cells; quivers are masked near the polar singularity.",
        "max_abs_fac_ua_m2": float(np.nanmax(np.abs(remix["fac_ua_m2"]))),
        "max_hall_current_ma_m": float(np.nanmax(hall["hall_mag"]) * 1e3),
        "p99_hall_current_ma_m": float(np.nanpercentile(hall["hall_mag"], 99) * 1e3),
        "p99_abs_hall_jphi_ma_m": float(np.nanpercentile(np.abs(jphi_ma_m), 99)),
        "orientation": "Native ReMIX polar grid: 12 MLT top, 00 MLT bottom, 06 right, 18 left; Kaiju phi=pi/2 is dusk and is plotted left",
        "method": "Hall current follows Kaiju/calcdb northern convention with dip-angle denominator: J_theta=-Sigma_H E_phi/nhcdip, J_phi=Sigma_H E_theta/nhcdip, E=-grad(Potential)",
        "not_plotted": "Divergence-free/equivalent current; this is signed physical Hall-current J_phi plus quivers, not a SECS decomposition.",
        "theta_stride": theta_stride,
        "phi_stride": phi_stride,
        "quiver_length_deg_at_p95": quiver_length_deg,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, default=Path("gamera_ireland/simulations/lompe_candidate_a/lompe_candidate_a.mix.h5"))
    parser.add_argument("--step", default="Step#5")
    parser.add_argument("--xml", type=Path, default=Path("gamera_ireland/simulations/lompe_candidate_a/lompe_candidate_a.xml"))
    parser.add_argument("--low-lat-boundary", type=float, default=45.0, help="ReMIX equatorward colatitude boundary in degrees.")
    parser.add_argument("--output", type=Path, default=Path("gamera_ireland/output/figures/lompe_candidate_a_fac_hall_native.png"))
    parser.add_argument("--summary", type=Path, default=Path("gamera_ireland/output/figures/lompe_candidate_a_fac_hall_native.json"))
    parser.add_argument("--theta-stride", type=int, default=2)
    parser.add_argument("--phi-stride", type=int, default=8)
    parser.add_argument("--quiver-length-deg", type=float, default=1.15, help="Display arrow length in plot degrees at the 95th percentile current magnitude.")
    args = parser.parse_args()
    low_lat = load_low_lat_boundary(args.xml, args.low_lat_boundary)
    make_plot(args.mix, args.step, args.output, args.summary, low_lat, args.theta_stride, args.phi_stride, args.quiver_length_deg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
