#!/usr/bin/env python3
"""Render a frame sequence and MP4 from MAGE/ReMIX/calcdb Ireland outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from gamera_ireland.config import load_json_config  # noqa: E402
from make_ireland_grid import build_grid  # noqa: E402
from plot_mage_ireland_maps import (  # noqa: E402
    load_calcdb_ground,
    load_remix_current_proxy,
    make_figure,
)


def step_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        names = [name for name in handle if name.startswith("Step#")]
    return sorted(names, key=lambda name: int(name.split("#", 1)[1]))


def enlarged_grid(config_path: Path, length_km: float, width_km: float, res_km: float):
    config = load_json_config(config_path)
    grid_config = dict(config["ireland_grid"])
    grid_config["L_km"] = length_km
    grid_config["W_km"] = width_km
    grid_config["Lres_km"] = res_km
    grid_config["Wres_km"] = res_km
    return build_grid(grid_config)[0]


def render_frames(args: argparse.Namespace) -> list[Path]:
    mix_steps = set(step_names(args.mix))
    deltab_steps = set(step_names(args.deltab))
    steps = sorted(mix_steps & deltab_steps, key=lambda name: int(name.split("#", 1)[1]))
    if not steps:
        raise ValueError(f"No common Step# groups in {args.mix} and {args.deltab}")

    node_grid = enlarged_grid(args.config, args.length_km, args.width_km, args.res_km)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    metadata = {
        "mix_h5": str(args.mix),
        "deltab_h5": str(args.deltab),
        "steps": [],
        "grid": {
            "L_km": args.length_km,
            "W_km": args.width_km,
            "res_km": args.res_km,
            "shape": list(node_grid.shape),
        },
    }
    for index, step in enumerate(steps):
        current = load_remix_current_proxy(args.mix, step)
        ground = load_calcdb_ground(args.deltab, step)
        frame = args.frames_dir / f"frame_{index:04d}.png"
        make_figure(current, ground, frame, args.stations, node_grid)
        frames.append(frame)
        metadata["steps"].append(
            {
                "step": step,
                "frame": str(frame),
                "current_timestamp": current["timestamp"],
                "ground_timestamp": ground["timestamp"],
            }
        )
        print(f"Wrote {frame}")

    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.metadata}")
    return frames


def encode_video(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.framerate),
        "-i",
        str(args.frames_dir / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    run_dir = PROJECT_DIR / "gamera_ireland/simulations/may2024_ireland"
    out_dir = PROJECT_DIR / "gamera_ireland/output/figures/may2024_storm1832_video"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, default=run_dir / "may2024_storm1832.mix.h5")
    parser.add_argument("--deltab", type=Path, default=run_dir / "may2024_storm1832.deltab.h5")
    parser.add_argument("--frames-dir", type=Path, default=out_dir / "frames")
    parser.add_argument("--output", type=Path, default=out_dir / "may2024_storm1832_ireland.mp4")
    parser.add_argument("--metadata", type=Path, default=out_dir / "may2024_storm1832_ireland_video.json")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "gamera_ireland/config/may2024_event.json")
    parser.add_argument("--stations", type=Path, default=PROJECT_DIR / "gamera_ireland/config/ireland_reference_stations.csv")
    parser.add_argument("--length-km", type=float, default=900.0)
    parser.add_argument("--width-km", type=float, default=900.0)
    parser.add_argument("--res-km", type=float, default=10.0)
    parser.add_argument("--framerate", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_frames(args)
    encode_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
