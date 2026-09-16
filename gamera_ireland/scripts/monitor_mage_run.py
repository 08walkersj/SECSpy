#!/usr/bin/env python3
"""Print concise progress diagnostics for a local MAGE/GAMERA run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def latest_step(path: Path) -> tuple[int, str | None, dict]:
    if not path.exists():
        return 0, None, {}
    with h5py.File(path, "r") as handle:
        steps = sorted(
            [name for name in handle if name.startswith("Step#")],
            key=lambda name: int(name.split("#")[1]),
        )
        if not steps:
            return 0, None, {}
        attrs = {
            key: float(value) if hasattr(value, "item") else value
            for key, value in handle[steps[-1]].attrs.items()
        }
        return len(steps), steps[-1], attrs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--runid", help="Defaults to the run directory name.")
    args = parser.parse_args()

    run_dir = args.run_dir
    runid = args.runid or run_dir.name
    mix = run_dir / f"{runid}.mix.h5"
    gam = run_dir / f"{runid}.gam.h5"
    restarts = sorted(
        path
        for path in run_dir.glob(f"{runid}*.Res.*.h5")
        if path.stem.rsplit(".", 1)[-1].isdigit()
    )

    n_steps, step, attrs = latest_step(mix)
    print(f"run_dir: {run_dir}")
    print(f"runid: {runid}")
    print(f"mix_exists: {mix.exists()}")
    print(f"gam_exists: {gam.exists()}")
    print(f"mix_steps: {n_steps}")
    print(f"latest_step: {step}")
    if attrs:
        print(f"latest_time_s: {attrs.get('time')}")
        print(f"nCPCP_kV: {attrs.get('nCPCP')}")
        print(f"sCPCP_kV: {attrs.get('sCPCP')}")
    print(f"restart_files: {len(restarts)}")
    if restarts:
        print(f"latest_restart: {restarts[-1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
