#!/usr/bin/env python3
"""Render the Figure 4 GAMERA-output column from a public GAMERA candidate run.

This intentionally plots only model outputs already present in ReMIX/calcdb
HDF5 files. It does not run or plot a LOMPE/SECS inversion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "environment" / "matplotlib"),
)

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from scipy.interpolate import griddata

import lompe


def cell_centres(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values[:-1, :] + values[1:, :])


def mlt_from_lon_rad(lon_rad: np.ndarray) -> np.ndarray:
    return (np.rad2deg(lon_rad) / 15.0) % 24.0


def build_lompe_grids():
    position = (0, 90)
    # With longitude = MLT * 15 deg, orientation=0 puts 12 MLT at +eta,
    # which is the top/north direction in the rendered panels.
    orientation = 0.0
    length, width, lres, wres = 15_000e3, 15_000e3, 200e3, 200e3

    grid = lompe.cs.CSgrid(
        lompe.cs.CSprojection(position, orientation),
        length,
        width,
        lres,
        wres,
        R=6380e3 + 120e3,
    )

    xi_edges = np.hstack((grid.xi_mesh[0], grid.xi_mesh[0, -1] + grid.dxi)) - grid.dxi / 2
    eta_edges = np.hstack((grid.eta_mesh[:, 0], grid.eta_mesh[-1, 0] + grid.deta)) - grid.deta / 2
    grid2 = lompe.cs.CSgrid(
        lompe.cs.CSprojection(grid.projection.position, grid.projection.orientation),
        grid.L + grid.Lres,
        grid.W + grid.Wres,
        grid.Lres,
        grid.Wres,
        edges=(xi_edges, eta_edges),
        R=grid.R,
    )
    return grid, grid2


def interp_to_grid(grid, lon_deg: np.ndarray, lat_deg: np.ndarray, values: np.ndarray) -> np.ndarray:
    lon_deg = np.asarray(lon_deg).flatten()
    lat_deg = np.asarray(lat_deg).flatten()
    values = np.asarray(values).flatten()
    ok = np.isfinite(lon_deg) & np.isfinite(lat_deg) & np.isfinite(values)
    xi_g, eta_g = grid.projection.geo2cube(lon_deg[ok], lat_deg[ok])
    interp = griddata((xi_g, eta_g), values[ok], (grid.xi.flatten(), grid.eta.flatten()))
    return interp.reshape(grid.xi.shape)


def load_mix(mix_h5: Path, step: str) -> dict[str, np.ndarray | float]:
    with h5py.File(mix_h5, "r") as handle:
        group = handle[step]
        colat_v = np.rad2deg(group["Latitude (Geographic) NORTH"][...])
        return {
            "time_s": float(group.attrs["time"]),
            "mjd": float(group.attrs["MJD"]),
            "mlat": cell_centres(90.0 - colat_v),
            "mlt": cell_centres(mlt_from_lon_rad(group["Longitude (Geographic) NORTH"][...])),
            "hall": group["Hall conductance NORTH"][...],
            "potential": group["Potential NORTH"][...],
            "fac": group["Field-aligned current NORTH"][...] * 1.0e-6,
        }


def load_deltab(deltab_h5: Path, step: str) -> dict[str, np.ndarray | float]:
    with h5py.File(deltab_h5, "r") as handle:
        group = handle[step]
        return {
            "time_s": float(group.attrs["time"]),
            "mjd": float(group.attrs["MJD"]),
            "mlat": group["smlat"][0, ...].T,
            "mlt": ((group["smlon"][0, ...] / 15.0) % 24.0).T,
            "dBe": group["dBp"][0, ...].T * 1.0e-9,
            "dBn": group["dBn"][0, ...].T * 1.0e-9,
            "dBu": group["dBr"][0, ...].T * 1.0e-9,
        }


def interpolate_outputs(mix: dict, deltab: dict, grid, grid2) -> dict[str, np.ndarray]:
    mix_inside = grid.ingrid(mix["mlt"] * 15.0, mix["mlat"], ext_factor=1.5)
    mix_lon = (mix["mlt"][mix_inside] * 15.0).flatten()
    mix_lat = mix["mlat"][mix_inside].flatten()

    db_inside = grid2.ingrid(deltab["mlt"] * 15.0, deltab["mlat"], ext_factor=1.5)
    db_lon = (deltab["mlt"][db_inside] * 15.0).flatten()
    db_lat = deltab["mlat"][db_inside].flatten()

    return {
        "mlt": interp_to_grid(grid, mix_lon, mix_lat, mix["mlt"][mix_inside]),
        "mlat": interp_to_grid(grid, mix_lon, mix_lat, mix["mlat"][mix_inside]),
        "hall": interp_to_grid(grid, mix_lon, mix_lat, mix["hall"][mix_inside]),
        "potential": interp_to_grid(grid, mix_lon, mix_lat, mix["potential"][mix_inside]),
        "fac": interp_to_grid(grid, mix_lon, mix_lat, mix["fac"][mix_inside]),
        "dBe": interp_to_grid(grid2, db_lon, db_lat, deltab["dBe"][db_inside]),
        "dBn": interp_to_grid(grid2, db_lon, db_lat, deltab["dBn"][db_inside]),
        "dBu": interp_to_grid(grid2, db_lon, db_lat, deltab["dBu"][db_inside]),
    }


def add_reference_contours(ax: plt.Axes, grid, interp: dict[str, np.ndarray]) -> None:
    c = ax.contour(
        grid.xi,
        grid.eta,
        interp["mlat"],
        levels=[50, 60, 70, 80],
        linestyles=":",
        colors="gray",
        linewidths=0.4,
    )
    ax.clabel(c, c.levels, inline=True, fontsize=8)
    ax.contour(
        grid.xi,
        grid.eta,
        interp["mlt"],
        levels=[6, 12, 18],
        linestyles=":",
        colors="gray",
        linewidths=0.4,
    )


def add_mlt_cardinals(ax: plt.Axes) -> None:
    ax.text(0.5, 0.965, "12", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
    ax.text(0.965, 0.5, "06", transform=ax.transAxes, ha="right", va="center", fontsize=8, color="0.35")
    ax.text(0.5, 0.035, "00", transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="0.35")
    ax.text(0.035, 0.5, "18", transform=ax.transAxes, ha="left", va="center", fontsize=8, color="0.35")


def draw_column(interp: dict[str, np.ndarray], grid, grid2, output: Path, timestamp: str) -> None:
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(3.4, 15.5))

    blevels = np.linspace(-500e-9, 500e-9, 22)
    jlevels = np.linspace(-2e-6, 2e-6, 22)
    clevels = np.linspace(0, 30, 25)

    panels = [
        ("Electric field", grid, interp["hall"], None, clevels),
        ("FAC", grid, interp["fac"], plt.cm.bwr, jlevels),
        ("Ground $B_e$", grid2, interp["dBe"], plt.cm.bwr, blevels),
        ("Ground $B_n$", grid2, interp["dBn"], plt.cm.bwr, blevels),
        ("Ground $B_u$", grid2, interp["dBu"], plt.cm.bwr, blevels),
        ("Space $B_e$", grid2, None, plt.cm.bwr, blevels),
        ("Space $B_n$", grid2, None, plt.cm.bwr, blevels),
        ("Space $B_u$", grid2, None, plt.cm.bwr, blevels),
    ]

    first_mesh = None
    for i, (label, panel_grid, values, cmap, levels) in enumerate(panels):
        ax = axes[i]
        if values is None:
            ax.text(
                0.5,
                0.5,
                "not in current\ncalcdb output",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="0.35",
            )
        else:
            first_mesh = ax.contourf(panel_grid.xi, panel_grid.eta, values, cmap=cmap, levels=levels, extend="both")
            if i == 0:
                ax.contour(grid.xi, grid.eta, interp["potential"], levels=np.r_[-200:201:10], colors="black", linewidths=0.7)
        add_reference_contours(ax, grid, interp)
        add_mlt_cardinals(ax)
        ax.set_ylabel(label, size=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_title("Gamera output", size=14)
    fig.suptitle(f"{timestamp}", y=0.996, fontsize=8)

    cax_b = fig.add_axes([0.12, 0.02, 0.76, 0.012])
    cax_b.contourf(
        np.vstack((blevels, blevels)) * 1e9,
        np.vstack((blevels * 0, blevels * 0 + 1)),
        np.vstack((blevels, blevels)),
        cmap=plt.cm.bwr,
        levels=blevels,
    )
    cax_b.set_yticks([])
    cax_b.set_xlabel("Magnetic field [nT] / current density [uA m$^{-2}$]", size=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.975, bottom=0.055, hspace=0.08)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_summary(path: Path, mix_h5: Path, deltab_h5: Path, mix_step: str, deltab_step: str, interp: dict, timestamp: str) -> None:
    summary = {
        "source": "public Kaiju/GAMERA candidate run, ReMIX plus calcdb.x outputs",
        "mix_h5": str(mix_h5),
        "deltab_h5": str(deltab_h5),
        "mix_step": mix_step,
        "deltab_step": deltab_step,
        "timestamp": timestamp,
        "figure4_rows_rendered": [
            "Hall conductance with potential contours",
            "FAC",
            "Ground Be",
            "Ground Bn",
            "Ground Bu",
        ],
        "figure4_rows_missing_from_current_output": [
            "Space Be at 1000 km",
            "Space Bn at 1000 km",
            "Space Bu at 1000 km",
        ],
        "ranges": {
            name: [float(np.nanmin(values)), float(np.nanmax(values))]
            for name, values in interp.items()
            if name not in {"mlt", "mlat"}
        },
        "units": {
            "hall": "S",
            "potential": "kV",
            "fac": "A/m^2, converted from ReMIX muA/m^2",
            "dBe_dBn_dBu": "T, converted from calcdb nT",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("gamera_ireland/simulations/lompe_candidate_a"))
    parser.add_argument("--runid", default="lompe_candidate_a")
    parser.add_argument("--mix-step", default="Step#5")
    parser.add_argument("--deltab-step", default="Step#5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gamera_ireland/output/figures/lompe_candidate_a_figure4_gamera_column.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("gamera_ireland/output/figures/lompe_candidate_a_figure4_gamera_column.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mix_h5 = args.run_dir / f"{args.runid}.mix.h5"
    deltab_h5 = args.run_dir / f"{args.runid}.deltab.h5"
    grid, grid2 = build_lompe_grids()
    mix = load_mix(mix_h5, args.mix_step)
    deltab = load_deltab(deltab_h5, args.deltab_step)
    interp = interpolate_outputs(mix, deltab, grid, grid2)
    timestamp = Time(float(mix["mjd"]), format="mjd").isot
    draw_column(interp, grid, grid2, args.output, timestamp)
    write_summary(args.summary, mix_h5, deltab_h5, args.mix_step, args.deltab_step, interp, timestamp)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
