"""Build conditioned hydrologic grids from the raw DEM."""

from __future__ import annotations

import argparse
from pathlib import Path

from watershed.geo import RAW_DEM_PATH
from watershed.hydro import (
    DEFAULT_SMOOTH_SIGMA,
    DEFAULT_STREAM_THRESHOLD,
    DEFAULT_WORKING_RESOLUTION_M,
    build_hydro_products,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flow-direction grid and streams")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DEM_PATH,
        help="Raw DEM GeoTIFF",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_WORKING_RESOLUTION_M,
        help="Working resolution in meters",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA,
        help="Gaussian smoothing sigma in pixels (0 to disable)",
    )
    parser.add_argument(
        "--stream-threshold",
        type=int,
        default=DEFAULT_STREAM_THRESHOLD,
        help="Flow accumulation threshold for stream extraction",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Missing {args.input}. Run: uv run python scripts/download_dem.py"
        )

    manifest = build_hydro_products(
        args.input,
        working_resolution_m=args.resolution,
        smooth_sigma=args.smooth_sigma,
        stream_threshold=args.stream_threshold,
    )
    print("Hydrologic products built:")
    for key, value in manifest["files"].items():
        print(f"  {key}: data/processed/{value}")


if __name__ == "__main__":
    main()
