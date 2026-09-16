#!/usr/bin/env python3
"""Plot what a LOMPE-like GAMERA candidate run produced."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time


def _steps(handle: h5py.File) -> list[str]:
    return sorted([key for key in handle if key.startswith("Step#")], key=lambda key: int(key.split("#", 1)[1]))


def _cell_centres(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values[:-1, :] + values[1:, :])


def _mlt_from_lon_rad(lon_rad: np.ndarray) -> np.ndarray:
    return (np.rad2deg(lon_rad) / 15.0) % 24.0


def _load_driver(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        t = handle["T"][...]
        density = handle["D"][...]
        vx = handle["Vx"][...] / 1.0e3
        by = handle["By"][...]
        bz = handle["Bz"][...]
    pdyn = 1.6726219e-6 * density * np.abs(vx) ** 2
    return pd.DataFrame({"time_s": t, "density": density, "vx": vx, "by": by, "bz": bz, "pdyn": pdyn})


def _load_mix_series(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as handle:
        for step in _steps(handle):
            group = handle[step]
            rows.append(
                {
                    "step": step,
                    "time_s": float(group.attrs["time"]),
                    "nCPCP": float(group.attrs["nCPCP"]),
                    "sCPCP": float(group.attrs["sCPCP"]),
                }
            )
    return pd.DataFrame(rows)


def _load_mix_maps(path: Path, step: str) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        group = handle[step]
        colat_v = np.rad2deg(group["Latitude (Geographic) NORTH"][...])
        mlat = _cell_centres(90.0 - colat_v)
        mlt = _cell_centres(_mlt_from_lon_rad(group["Longitude (Geographic) NORTH"][...]))
        return {
            "time_s": float(group.attrs["time"]),
            "mjd": float(group.attrs["MJD"]),
            "mlat": mlat,
            "mlt": mlt,
            "potential": group["Potential NORTH"][...],
            "fac": group["Field-aligned current NORTH"][...],
            "pedersen": group["Pedersen conductance NORTH"][...],
            "hall": group["Hall conductance NORTH"][...],
        }


def _load_deltab_maps(path: Path, step: str) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        group = handle[step]
        return {
            "time_s": float(group.attrs["time"]),
            "mjd": float(group.attrs["MJD"]),
            "mlat": group["smlat"][0, ...].T,
            "mlt": ((group["smlon"][0, ...] / 15.0) % 24.0).T,
            "dBn": group["dBn"][0, ...].T,
            "dBp": group["dBp"][0, ...].T,
            "dBr": group["dBr"][0, ...].T,
            "dBh": np.hypot(group["dBn"][0, ...].T, group["dBp"][0, ...].T),
        }


def _plot_mlt_mlat(ax: plt.Axes, mlt: np.ndarray, mlat: np.ndarray, values: np.ndarray, title: str, cmap: str, label: str, symmetric: bool = False):
    vals = np.asarray(values, dtype=float)
    if symmetric:
        vmax = np.nanpercentile(np.abs(vals), 99)
        kwargs = {"vmin": -vmax, "vmax": vmax}
    else:
        kwargs = {}
    mesh = ax.pcolormesh(mlt, mlat, vals, shading="auto", cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.set_xlim(0, 24)
    ax.set_ylim(45, 90)
    ax.set_xlabel("MLT")
    ax.set_ylabel("MLat")
    ax.grid(alpha=0.25, linewidth=0.5)
    plt.colorbar(mesh, ax=ax, label=label)
    return mesh


def make_plot(run_dir: Path, runid: str, mix_step: str, deltab_step: str, output: Path) -> dict:
    bcwind = run_dir / "bcwind.h5"
    mix_h5 = run_dir / f"{runid}.mix.h5"
    deltab_h5 = run_dir / f"{runid}.deltab.h5"

    driver = _load_driver(bcwind)
    series = _load_mix_series(mix_h5)
    mix = _load_mix_maps(mix_h5, mix_step)
    deltab = _load_deltab_maps(deltab_h5, deltab_step)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(driver["time_s"] / 60.0, driver["density"], label="density")
    ax2 = ax.twinx()
    ax2.plot(driver["time_s"] / 60.0, driver["pdyn"], color="tab:red", label="Pdyn")
    ax.set_title("Solar-wind pressure driver")
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Density (cm$^{-3}$)")
    ax2.set_ylabel("Dynamic pressure (nPa)")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(driver["time_s"] / 60.0, driver["bz"], label="Bz")
    ax.plot(driver["time_s"] / 60.0, driver["by"], label="By")
    ax.set_title("IMF driver")
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("nT")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 2]
    ax.plot(series["time_s"] / 60.0, series["nCPCP"], marker="o", label="north")
    ax.plot(series["time_s"] / 60.0, series["sCPCP"], marker="o", label="south")
    ax.axvline(float(mix["time_s"]) / 60.0, color="0.2", linestyle="--", linewidth=1)
    ax.set_title("ReMIX cross-polar cap potential")
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("kV")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 3]
    for field, color in [("dBn", "tab:blue"), ("dBp", "tab:orange"), ("dBr", "tab:green")]:
        mins = []
        maxs = []
        times = []
        with h5py.File(deltab_h5, "r") as handle:
            for step in _steps(handle):
                group = handle[step]
                arr = group[field][0, ...]
                times.append(float(group.attrs["time"]) / 60.0)
                mins.append(np.nanpercentile(arr, 1))
                maxs.append(np.nanpercentile(arr, 99))
        ax.fill_between(times, mins, maxs, color=color, alpha=0.18)
        ax.plot(times, maxs, color=color, linewidth=1, label=field)
    ax.axvline(float(deltab["time_s"]) / 60.0, color="0.2", linestyle="--", linewidth=1)
    ax.set_title("Ground perturbation envelope")
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("nT, 1st-99th percentile")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    _plot_mlt_mlat(axes[1, 0], mix["mlt"], mix["mlat"], mix["potential"], "Potential", "RdBu_r", "kV", True)
    axes[1, 0].contour(mix["mlt"], mix["mlat"], mix["potential"], colors="black", linewidths=0.45, alpha=0.55)
    _plot_mlt_mlat(axes[1, 1], mix["mlt"], mix["mlat"], mix["fac"], "Field-aligned current", "seismic", "ReMIX output", True)
    _plot_mlt_mlat(axes[1, 2], mix["mlt"], mix["mlat"], mix["pedersen"], "Pedersen conductance", "viridis", "mho")
    _plot_mlt_mlat(axes[1, 3], mix["mlt"], mix["mlat"], mix["hall"], "Hall conductance", "viridis", "mho")

    _plot_mlt_mlat(axes[2, 0], deltab["mlt"], deltab["mlat"], deltab["dBn"], "Ground Delta B north", "seismic", "nT", True)
    _plot_mlt_mlat(axes[2, 1], deltab["mlt"], deltab["mlat"], deltab["dBp"], "Ground Delta B east", "seismic", "nT", True)
    _plot_mlt_mlat(axes[2, 2], deltab["mlt"], deltab["mlat"], deltab["dBr"], "Ground Delta B up", "seismic", "nT", True)
    _plot_mlt_mlat(axes[2, 3], deltab["mlt"], deltab["mlat"], deltab["dBh"], "Ground Delta H", "magma", "nT")

    timestamp = Time(float(mix["mjd"]), format="mjd").isot
    fig.suptitle(f"{runid}: public GAMERA/ReMIX/RAIJU candidate, {mix_step} / {deltab_step}, {timestamp}", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)

    return {
        "run_dir": str(run_dir),
        "runid": runid,
        "mix_step": mix_step,
        "deltab_step": deltab_step,
        "timestamp": timestamp,
        "output": str(output),
        "driver": {
            "density_cm3_min_max": [float(driver["density"].min()), float(driver["density"].max())],
            "pdyn_npa_min_max": [float(driver["pdyn"].min()), float(driver["pdyn"].max())],
            "vx_kms": float(driver["vx"].iloc[0]),
            "bz_nt_min_max": [float(driver["bz"].min()), float(driver["bz"].max())],
        },
        "response": {
            "nCPCP_kV_at_step": float(series.loc[series["step"] == mix_step, "nCPCP"].iloc[0]),
            "sCPCP_kV_at_step": float(series.loc[series["step"] == mix_step, "sCPCP"].iloc[0]),
            "max_abs_fac": float(np.nanmax(np.abs(mix["fac"]))),
            "max_pedersen_mho": float(np.nanmax(mix["pedersen"])),
            "max_hall_mho": float(np.nanmax(mix["hall"])),
            "max_abs_dBr_nT": float(np.nanmax(np.abs(deltab["dBr"]))),
            "max_dH_nT": float(np.nanmax(deltab["dBh"])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("gamera_ireland/simulations/lompe_candidate_a"))
    parser.add_argument("--runid", default="lompe_candidate_a")
    parser.add_argument("--mix-step", default="Step#5")
    parser.add_argument("--deltab-step", default="Step#5")
    parser.add_argument("--output", type=Path, default=Path("gamera_ireland/output/figures/lompe_candidate_a_summary.png"))
    parser.add_argument("--summary", type=Path, default=Path("gamera_ireland/output/figures/lompe_candidate_a_summary.json"))
    args = parser.parse_args()

    summary = make_plot(args.run_dir, args.runid, args.mix_step, args.deltab_step, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
