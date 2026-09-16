#!/usr/bin/env python3
"""Compare a from-scratch MAGE/calcdb run with the LOMPE GAMERA sample file.

This does not assume exact reproduction. It compares shared fields on the
Lompe sample mlat/mlt grid and reports the best circular MLT roll for each
candidate time step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from scipy.interpolate import griddata


DEFAULT_TARGET = Path("/home/simon/gits/lompe/examples/sample_dataset/gamera.nc")


def _cell_centres_from_vertices(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values[:-1, :] + values[1:, :])


def _mlt_from_lon_rad(lon_rad: np.ndarray) -> np.ndarray:
    return (np.rad2deg(lon_rad) / 15.0) % 24.0


def _periodic_griddata(mlt: np.ndarray, mlat: np.ndarray, values: np.ndarray, target_mlt: np.ndarray, target_mlat: np.ndarray) -> np.ndarray:
    points = np.column_stack([mlt.ravel(), mlat.ravel()])
    vals = values.ravel()
    points = np.vstack(
        [
            points,
            np.column_stack([points[:, 0] - 24.0, points[:, 1]]),
            np.column_stack([points[:, 0] + 24.0, points[:, 1]]),
        ]
    )
    vals = np.tile(vals, 3)
    return griddata(points, vals, (target_mlt, target_mlat), method="linear")


def _metric(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(candidate) & np.isfinite(target)
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "nrmse": float("nan"), "corr": float("nan"), "scale": float("nan")}
    c = candidate[mask].astype(float)
    t = target[mask].astype(float)
    denom = np.nanstd(t)
    nrmse = np.sqrt(np.nanmean((c - t) ** 2)) / denom if denom else float("nan")
    corr = np.corrcoef(c, t)[0, 1] if np.nanstd(c) and np.nanstd(t) else float("nan")
    scale = np.nanstd(c) / denom if denom else float("nan")
    return {"n": int(mask.sum()), "nrmse": float(nrmse), "corr": float(corr), "scale": float(scale)}


def _best_roll(candidate: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    best = None
    for roll in range(candidate.shape[1]):
        rolled = np.roll(candidate, roll, axis=1)
        metrics = _metric(rolled, target)
        if best is None or metrics["nrmse"] < best["nrmse"]:
            best = {"roll": roll, **metrics}
    return best if best is not None else {"roll": 0, "n": 0, "nrmse": float("nan"), "corr": float("nan"), "scale": float("nan")}


def load_target(path: Path) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    target_mlt, target_mlat = np.meshgrid(ds["mlt"].values, ds["mlat"].values)
    return ds, target_mlt, target_mlat


def load_mix_step(path: Path, step: str, target_mlt: np.ndarray, target_mlat: np.ndarray) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        group = handle[step]
        colat_v = np.rad2deg(group["Latitude (Geographic) NORTH"][...])
        mlat_v = 90.0 - colat_v
        mlt_v = _mlt_from_lon_rad(group["Longitude (Geographic) NORTH"][...])
        mlat = _cell_centres_from_vertices(mlat_v)
        mlt = _cell_centres_from_vertices(mlt_v)
        return {
            "time": float(group.attrs.get("time", np.nan)),
            "mjd": float(group.attrs.get("MJD", np.nan)),
            "FAC": _periodic_griddata(mlt, mlat, group["Field-aligned current NORTH"][...], target_mlt, target_mlat),
            "potential": _periodic_griddata(mlt, mlat, group["Potential NORTH"][...], target_mlt, target_mlat),
            "SigmaPed": _periodic_griddata(mlt, mlat, group["Pedersen conductance NORTH"][...], target_mlt, target_mlat),
            "SigmaHall": _periodic_griddata(mlt, mlat, group["Hall conductance NORTH"][...], target_mlt, target_mlat),
        }


def load_deltab_step(path: Path, step: str, target_mlt: np.ndarray, target_mlat: np.ndarray) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        group = handle[step]
        mlat = group["smlat"][0, ...]
        mlt = (group["smlon"][0, ...] / 15.0) % 24.0
        return {
            "time": float(group.attrs.get("time", np.nan)),
            "dBg_r": _periodic_griddata(mlt, mlat, group["dBr"][0, ...] * 1.0e-9, target_mlt, target_mlat),
            "dBg_theta": _periodic_griddata(mlt, mlat, -group["dBn"][0, ...] * 1.0e-9, target_mlt, target_mlat),
            "dBg_phi": _periodic_griddata(mlt, mlat, group["dBp"][0, ...] * 1.0e-9, target_mlt, target_mlat),
        }


def step_ids(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted([key for key in handle if key.startswith("Step#")], key=lambda item: int(item.split("#", 1)[1]))


def compare(run_dir: Path, runid: str, target: Path) -> dict:
    target_ds, target_mlt, target_mlat = load_target(target)
    mix_path = run_dir / f"{runid}.mix.h5"
    deltab_path = run_dir / f"{runid}.deltab.h5"
    mix_steps = step_ids(mix_path)
    deltab_steps = step_ids(deltab_path) if deltab_path.exists() else []

    rows = []
    for step in mix_steps:
        fields = load_mix_step(mix_path, step, target_mlt, target_mlat)
        row = {"step": step, "time_s": fields.pop("time"), "mjd": fields.pop("mjd"), "fields": {}}
        for name, values in fields.items():
            row["fields"][name] = _best_roll(values, target_ds[name].values)
        rows.append(row)

    for row in rows:
        if row["step"] not in deltab_steps:
            continue
        fields = load_deltab_step(deltab_path, row["step"], target_mlt, target_mlat)
        fields.pop("time", None)
        for name, values in fields.items():
            row["fields"][name] = _best_roll(values, target_ds[name].values)

    summary_fields = ["FAC", "potential", "SigmaPed", "SigmaHall", "dBg_r", "dBg_theta", "dBg_phi"]
    for row in rows:
        vals = [row["fields"][name]["nrmse"] for name in summary_fields if name in row["fields"] and np.isfinite(row["fields"][name]["nrmse"])]
        row["mean_nrmse"] = float(np.mean(vals)) if vals else float("nan")

    return {
        "target": str(target),
        "run_dir": str(run_dir),
        "runid": runid,
        "target_time": str(target_ds["time"].values),
        "target_step": int(target_ds["step"].values),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--runid", required=True)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare(args.run_dir, args.runid, args.target)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
