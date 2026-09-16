#!/usr/bin/env python3
"""Watch a MAGE/GAMERA run and print a live text progress bar."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py


def read_latest_time(mix_path: Path) -> tuple[int, str | None, float | None, float | None, float | None]:
    if not mix_path.exists():
        return 0, None, None, None, None
    with h5py.File(mix_path, "r") as handle:
        steps = sorted(
            [name for name in handle if name.startswith("Step#")],
            key=lambda name: int(name.split("#")[1]),
        )
        if not steps:
            return 0, None, None, None, None
        attrs = handle[steps[-1]].attrs
        return (
            len(steps),
            steps[-1],
            float(attrs.get("time", math.nan)),
            float(attrs.get("nCPCP", math.nan)),
            float(attrs.get("sCPCP", math.nan)),
        )


def is_running(runid: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", runid],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return "voltron" in result.stdout


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    return f"{minutes:d}m {secs:02d}s"


def progress_bar(frac: float, width: int) -> str:
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--runid", help="Defaults to the run directory name.")
    parser.add_argument("--target-s", type=float, default=10800.0)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--stale-s", type=float, default=300.0)
    parser.add_argument("--exit-when-complete", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    runid = args.runid or run_dir.name
    mix_path = run_dir / f"{runid}.mix.h5"

    print(f"Watching {runid} in {run_dir}", flush=True)
    print("Press Ctrl-C to stop watching; this does not stop Voltron.", flush=True)

    start_wall = time.time()
    last_model_time = None
    last_change_wall = None
    previous_sample = None

    while True:
        now = time.time()
        try:
            n_steps, step, model_time, ncpcp, scpcp = read_latest_time(mix_path)
            error = None
        except OSError as exc:
            n_steps, step, model_time, ncpcp, scpcp = 0, None, last_model_time, math.nan, math.nan
            error = str(exc).splitlines()[0]

        if model_time is not None and math.isfinite(model_time):
            if last_model_time is None or model_time != last_model_time:
                last_change_wall = now
                last_model_time = model_time
            frac = model_time / args.target_s
        else:
            frac = 0.0

        speed = None
        if previous_sample and model_time is not None and math.isfinite(model_time):
            prev_wall, prev_model = previous_sample
            dt_wall = now - prev_wall
            dt_model = model_time - prev_model
            if dt_wall > 0 and dt_model >= 0:
                speed = dt_model / dt_wall
        if model_time is not None and math.isfinite(model_time):
            previous_sample = (now, model_time)

        eta_seconds = None
        if speed and speed > 0 and model_time is not None:
            eta_seconds = max(args.target_s - model_time, 0.0) / speed

        terminal_width = shutil.get_terminal_size((120, 20)).columns
        bar_width = max(20, min(50, terminal_width - 90))
        running = is_running(runid)
        stale = last_change_wall is not None and (now - last_change_wall) > args.stale_s
        status = "running" if running else "not running"
        if stale:
            status += ", stale"
        if error:
            status += ", hdf5 busy"

        finish = "unknown"
        if eta_seconds is not None:
            finish_dt = dt.datetime.now().astimezone() + dt.timedelta(seconds=eta_seconds)
            finish = finish_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        line = (
            f"{progress_bar(frac, bar_width)} {100 * frac:6.2f}% "
            f"t={model_time if model_time is not None else float('nan'):8.1f}/{args.target_s:.0f}s "
            f"steps={n_steps:3d} latest={step or '-':>8s} "
            f"speed={speed if speed is not None else float('nan'):5.2f}x "
            f"ETA={format_duration(eta_seconds):>9s} finish={finish} "
            f"n/sCPCP={ncpcp if ncpcp is not None else float('nan'):6.1f}/"
            f"{scpcp if scpcp is not None else float('nan'):6.1f} kV "
            f"({status})"
        )
        print("\r" + line[: terminal_width - 1], end="", flush=True)

        if args.exit_when_complete and model_time is not None and model_time >= args.target_s:
            print()
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
