#!/usr/bin/env python3
"""Prepare GAMERA `bcwind.h5` from Solar_Wind_Tools OMNI time-series data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from gamera_ireland.config import event_window, load_json_config, resolve_project_path
from gamera_ireland.solar_wind import (
    download_omni_with_solar_wind_tools,
    read_omni_subset,
    write_bcwind_from_omni_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json",
        help="Event configuration JSON.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download OMNI first using the local Solar_Wind_Tools repository.",
    )
    parser.add_argument(
        "--f107",
        type=float,
        default=None,
        help="Fallback F10.7 value if unavailable in the OMNI HDF file.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Fallback Kp value if unavailable in the OMNI HDF file.",
    )
    return parser.parse_args()


def write_manifest(config: dict, output_dir: Path, start, end, bcwind: Path, n_rows: int) -> None:
    """Write a small run-preparation manifest."""
    manifest = {
        "run_id": config["run_id"],
        "event_time": config["event_time"],
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "omni_rows_used": n_rows,
        "bcwind_hdf": str(bcwind),
        "notes": [
            "bcwind.h5 follows the dataset names used by kaipy/scripts/preproc/cda2wind.py.",
            "Kaiju/MAGE executables are not invoked by this preparation script.",
            "Run calcdb.x only after a real MAGE/GAMERA/REMIX/RAIJU simulation exists.",
        ],
    }
    path = output_dir / "run_manifest.json"
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    config = load_json_config(args.config)
    _, start, end = event_window(config)

    paths = config["paths"]
    omni_hdf = resolve_project_path(paths["omni_raw_hdf"], PROJECT_DIR)
    bcwind_hdf = resolve_project_path(paths["bcwind_hdf"], PROJECT_DIR)
    run_directory = resolve_project_path(paths["run_directory"], PROJECT_DIR)
    run_directory.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_omni_with_solar_wind_tools(
            config["solar_wind_tools_repo"],
            omni_hdf,
            start,
            end,
        )

    bc_config = config["bcwind"]
    fallback_f107 = args.f107
    if fallback_f107 is None:
        fallback_f107 = bc_config["fallback_f107"]
    fallback_kp = args.kp
    if fallback_kp is None:
        fallback_kp = bc_config["fallback_kp"]

    omni = read_omni_subset(omni_hdf, config["omni_hdf_key"], start, end)
    bcwind = write_bcwind_from_omni_dataframe(
        omni,
        bcwind_hdf,
        include_bx_fit=bc_config["include_bx_fit"],
        coarse_filter=bc_config["coarse_filter"],
        sigma=bc_config["sigma"],
        lfm_start_minutes=bc_config["lfm_start_minutes"],
        gamera_start_minutes=bc_config["gamera_start_minutes"],
        fallback_f107=fallback_f107,
        fallback_kp=fallback_kp,
    )
    write_manifest(config, run_directory, start, end, bcwind, len(omni))
    print(f"Wrote {bcwind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

