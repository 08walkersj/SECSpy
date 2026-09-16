#!/usr/bin/env python3
"""Render ReMIX FAC/Hall-current frames and assemble a video.

This is a direct visualization of MAGE/ReMIX output. It does not run a SECS,
Lompe, or other inversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_DIR / "src"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from plot_global_fac_hall_current import load_low_lat_boundary, make_plot  # noqa: E402
from SECSpy.utils import create_video_from_images  # noqa: E402


def list_steps(mix_h5: Path) -> list[str]:
    with h5py.File(mix_h5, "r") as handle:
        steps = [name for name in handle.keys() if name.startswith("Step#")]
    return sorted(steps, key=lambda name: int(name.split("#", 1)[1]))


def step_time(mix_h5: Path, step: str) -> float:
    with h5py.File(mix_h5, "r") as handle:
        return float(handle[step].attrs["time"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--filename", default="fac_hall_movie")
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--start-step", type=int, default=1, help="First ReMIX step number to render.")
    parser.add_argument("--end-step", type=int, help="Last ReMIX step number to render, inclusive.")
    parser.add_argument("--theta-stride", type=int, default=2)
    parser.add_argument("--phi-stride", type=int, default=8)
    parser.add_argument("--quiver-length-deg", type=float, default=1.15)
    parser.add_argument("--create-gif", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    low_lat_boundary = load_low_lat_boundary(args.xml, fallback=45.0)

    selected = []
    for step in list_steps(args.mix):
        number = int(step.split("#", 1)[1])
        if number < args.start_step:
            continue
        if args.end_step is not None and number > args.end_step:
            continue
        selected.append(step)

    if not selected:
        raise SystemExit("No ReMIX steps selected.")

    manifest = []
    num_digits = max(4, len(str(len(selected))))
    for index, step in enumerate(selected, start=1):
        frame_path = args.frames_dir / f"{args.prefix}{index:0{num_digits}d}.png"
        summary_path = args.frames_dir / f"{args.prefix}{index:0{num_digits}d}.json"
        make_plot(
            mix_h5=args.mix,
            step=step,
            output=frame_path,
            summary_path=summary_path,
            low_lat_boundary_deg=low_lat_boundary,
            theta_stride=args.theta_stride,
            phi_stride=args.phi_stride,
            quiver_length_deg=args.quiver_length_deg,
        )
        manifest.append({"frame": frame_path.name, "step": step, "time_s": step_time(args.mix, step)})

    (args.frames_dir / "frames_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    create_video_from_images(
        args.frames_dir,
        prefix=args.prefix,
        filename=args.filename,
        num_digits=num_digits,
        start_number=1,
        fps=args.fps,
        create_gif=args.create_gif,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
