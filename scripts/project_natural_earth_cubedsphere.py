#!/usr/bin/env python
"""Create a projected Natural Earth cache for a cubed-sphere grid."""

from __future__ import annotations

import argparse

from SECSpy.natural_earth import save_projected_natural_earth


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project bundled Natural Earth layers into secsy cubed-sphere "
            "coordinates and save them as a compact NPZ cache."
        )
    )
    parser.add_argument("output", help="Output NPZ path.")
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument(
        "--orientation",
        type=float,
        default=0.0,
        help="CSprojection orientation argument in degrees.",
    )
    parser.add_argument("--height-km", type=float, required=True)
    parser.add_argument("--width-km", type=float, required=True)
    parser.add_argument("--dx-km", type=float, default=50.0)
    parser.add_argument("--dy-km", type=float, default=50.0)
    parser.add_argument(
        "--geographic-buffer",
        type=float,
        default=2.0,
        help="Degree buffer around the grid lon/lat extent before projection.",
    )
    parser.add_argument(
        "--resolution",
        choices=("110m", "50m", "10m"),
        default="50m",
        help="Bundled Natural Earth resolution.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["land", "ocean", "coastline", "lakes", "borders"],
        help="Natural Earth layers to project.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from secsy.cubedsphere import CSgrid, CSprojection
    except ImportError as exc:
        raise SystemExit(
            "This script needs secsy installed because it constructs the "
            "cubed-sphere grid. The SECSpy projection helpers themselves only "
            "need a grid/projection object with geo2cube()."
        ) from exc

    projection = CSprojection(
        (args.center_lon, args.center_lat),
        args.orientation,
    )
    grid = CSgrid(
        projection,
        args.height_km,
        args.width_km,
        args.dx_km,
        args.dy_km,
    )
    output = save_projected_natural_earth(
        args.output,
        grid,
        layers=args.layers,
        resolution=args.resolution,
        geographic_buffer=args.geographic_buffer,
    )
    print(output)


if __name__ == "__main__":
    main()
