#!/usr/bin/env python3
"""Reproduce the GAMERA truth panels from the LOMPE synthetic-data example.

This script intentionally does not run a LOMPE inversion. It verifies the
coordinate transforms, signs, units, interpolation, and plotting conventions
used by the LOMPE paper's bundled synthetic GAMERA/ReMIX product.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

import lompe


LOMPE_EXAMPLE_DIR = Path("/home/simon/gits/lompe/examples")
DEFAULT_GAMERA_NC = LOMPE_EXAMPLE_DIR / "sample_dataset" / "gamera.nc"


def load_gamera_truth(path: Path) -> dict[str, np.ndarray | str]:
    with xr.open_dataset(path) as ds:
        out = {
            "fac": ds["FAC"].values,
            "potential": ds["potential"].values,
            "En": -ds["Etheta"].values,
            "Ee": ds["Ephi"].values,
            "SigmaPed": ds["SigmaPed"].values,
            "SigmaHall": ds["SigmaHall"].values,
            "dBr": ds["dBg_r"].values,
            "dBn": -ds["dBg_theta"].values,
            "dBe": ds["dBg_phi"].values,
            "dBr_space": ds["dBspace_r"].values,
            "dBn_space": -ds["dBspace_theta"].values,
            "dBe_space": ds["dBspace_phi"].values,
            "mlat": ds["mlat"].values,
            "mlt": ds["mlt"].values,
            "hemi": str(ds["hemi"].values),
            "step": int(ds["step"].values),
            "time": pd.Timestamp(ds["time"].values).isoformat(),
            "creation_date": pd.Timestamp(ds["creation_date"].values).isoformat(),
        }
    out["mlt_grid"], out["mlat_grid"] = np.meshgrid(out["mlt"], out["mlat"])
    return out


def build_lompe_grids(kind: str):
    if kind == "global":
        position = (0, 90)
        orientation = 0.0
        length, width, lres, wres = 15_000e3, 15_000e3, 200e3, 200e3
    elif kind == "local":
        position = (240, 74)
        orientation = 45.0
        length, width, lres, wres = 3_000e3, 3_000e3, 75e3, 75e3
    else:
        raise ValueError("kind must be 'global' or 'local'")

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


def interp_to_grid(grid, lon_deg, lat_deg, values):
    xi_g, eta_g = grid.projection.geo2cube(lon_deg, lat_deg)
    interp = griddata(
        (xi_g, eta_g),
        values.flatten(),
        (grid.xi.flatten(), grid.eta.flatten()),
    )
    return interp.reshape(grid.xi.shape)


def interpolate_truth(truth: dict[str, np.ndarray | str], grid, grid2):
    mlt = truth["mlt_grid"]
    mlat = truth["mlat_grid"]
    inside = grid.ingrid(mlt * 15.0, mlat, ext_factor=1.5)
    lon = mlt[inside].flatten() * 15.0
    lat = mlat[inside].flatten()

    interp = {
        "SigmaHall": interp_to_grid(grid, lon, lat, truth["SigmaHall"][inside].flatten()),
        "SigmaPed": interp_to_grid(grid, lon, lat, truth["SigmaPed"][inside].flatten()),
        "potential": interp_to_grid(grid, lon, lat, truth["potential"][inside].flatten()),
        "FAC": interp_to_grid(grid, lon, lat, truth["fac"][inside].flatten() * 1e-6),
        "mlt": interp_to_grid(grid, lon, lat, mlt[inside].flatten()),
        "mlat": interp_to_grid(grid, lon, lat, mlat[inside].flatten()),
        "En": interp_to_grid(grid, lon, lat, truth["En"][inside].flatten()),
        "Ee": interp_to_grid(grid, lon, lat, truth["Ee"][inside].flatten()),
        "dBr": interp_to_grid(grid2, lon, lat, truth["dBr"][inside].flatten()),
        "dBn": interp_to_grid(grid2, lon, lat, truth["dBn"][inside].flatten()),
        "dBe": interp_to_grid(grid2, lon, lat, truth["dBe"][inside].flatten()),
        "dBr_space": interp_to_grid(grid2, lon, lat, truth["dBr_space"][inside].flatten()),
        "dBn_space": interp_to_grid(grid2, lon, lat, truth["dBn_space"][inside].flatten()),
        "dBe_space": interp_to_grid(grid2, lon, lat, truth["dBe_space"][inside].flatten()),
    }

    if not np.isfinite(interp["SigmaPed"]).all() or not np.isfinite(interp["SigmaHall"]).all():
        raise ValueError("Conductance interpolation produced non-finite values")

    return interp


def draw_truth(kind: str, truth: dict[str, np.ndarray | str], interp, grid, grid2, output: Path):
    fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(8.5, 18), constrained_layout=True)

    panels = [
        (axes[0, 0], grid, interp["SigmaHall"], "Hall conductance + potential", None, np.linspace(0, 30, 25)),
        (axes[0, 1], grid, interp["SigmaPed"], "Pedersen conductance + potential", None, np.linspace(0, 30, 25)),
        (axes[1, 0], grid, interp["FAC"], "FAC", plt.cm.bwr, np.linspace(-2e-6, 2e-6, 22)),
        (axes[1, 1], grid, interp["En"], "$E_n$", plt.cm.bwr, np.linspace(-80e-3, 80e-3, 22)),
        (axes[2, 0], grid2, interp["dBe"], "Ground $B_e$", plt.cm.bwr, np.linspace(-500e-9, 500e-9, 22)),
        (axes[2, 1], grid2, interp["dBn"], "Ground $B_n$", plt.cm.bwr, np.linspace(-500e-9, 500e-9, 22)),
        (axes[3, 0], grid2, interp["dBr"], "Ground $B_u$", plt.cm.bwr, np.linspace(-500e-9, 500e-9, 22)),
        (axes[3, 1], grid, interp["Ee"], "$E_e$", plt.cm.bwr, np.linspace(-80e-3, 80e-3, 22)),
        (axes[4, 0], grid2, interp["dBe_space"], "Space $B_e$", plt.cm.bwr, np.linspace(-500e-9, 500e-9, 22)),
        (axes[4, 1], grid2, interp["dBn_space"], "Space $B_n$", plt.cm.bwr, np.linspace(-500e-9, 500e-9, 22)),
        (axes[5, 0], grid2, interp["dBr_space"], "Space $B_u$", plt.cm.bwr, np.linspace(-500e-9, 500e-9, 22)),
    ]

    used_axes = {id(panel[0]) for panel in panels}

    for ax, panel_grid, values, title, cmap, levels in panels:
        ax.contourf(panel_grid.xi, panel_grid.eta, values, cmap=cmap, levels=levels)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_axis_off()

    for ax in axes.flatten():
        if id(ax) not in used_axes:
            ax.set_axis_off()
            continue
        c = ax.contour(
            grid.xi,
            grid.eta,
            interp["mlat"],
            levels=[50, 60, 70, 80],
            linestyles=":",
            colors="gray",
            linewidths=0.4,
        )
        ax.clabel(c, c.levels, inline=True, fontsize=10)
        ax.contour(
            grid.xi,
            grid.eta,
            interp["mlt"],
            levels=[6, 12, 18],
            linestyles=":",
            colors="gray",
            linewidths=0.4,
        )
        ax.contour(
            grid.xi,
            grid.eta,
            interp["potential"],
            levels=np.r_[-200:201:10],
            colors="black",
            linewidths=0.7,
        )
        ax.set_axis_off()

    fig.suptitle(f"LOMPE synthetic GAMERA truth ({kind}), {truth['time']}", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summary(path: Path, nc_path: Path, kind: str, truth, interp, grid, grid2):
    summary = {
        "source": "LOMPE bundled synthetic GAMERA/ReMIX netCDF product",
        "source_file": str(nc_path),
        "kind": kind,
        "time": truth["time"],
        "creation_date": truth["creation_date"],
        "hemi": truth["hemi"],
        "step": truth["step"],
        "input_shape": {
            "mlat": int(len(truth["mlat"])),
            "mlt": int(len(truth["mlt"])),
        },
        "inner_grid_shape": list(grid.shape),
        "outer_grid_shape": list(grid2.shape),
        "ranges": {
            name: [float(np.nanmin(values)), float(np.nanmax(values))]
            for name, values in interp.items()
            if name not in {"mlt", "mlat"}
        },
        "sign_conventions_from_lompe_notebook": {
            "En": "-Etheta",
            "Ee": "Ephi",
            "dBn": "-dBg_theta",
            "dBe": "dBg_phi",
            "dBr": "dBg_r",
            "FAC": "FAC * 1e-6",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamera-nc", type=Path, default=DEFAULT_GAMERA_NC)
    parser.add_argument("--kind", choices=["global", "local"], default="global")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gamera_ireland/output/figures/lompe_gamera_truth_global.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("gamera_ireland/output/figures/lompe_gamera_truth_global.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    truth = load_gamera_truth(args.gamera_nc)
    grid, grid2 = build_lompe_grids(args.kind)
    interp = interpolate_truth(truth, grid, grid2)
    draw_truth(args.kind, truth, interp, grid, grid2, args.output)
    write_summary(args.summary, args.gamera_nc, args.kind, truth, interp, grid, grid2)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
