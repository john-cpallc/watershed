"""Download and mosaic USGS 3DEP elevation for Jefferson County, Alabama."""

from __future__ import annotations

import argparse
import math
import time
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from pyproj import Transformer
import requests

from watershed.geo import (
    COUNTY_DEM_CRS,
    COUNTY_NAME,
    RAW_DEM_PATH,
    build_watershed_outlet_polygon,
    load_county_boundary,
)

IMAGE_SERVER = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)


def _utm_bbox(county_gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    utm = county_gdf.to_crs(COUNTY_DEM_CRS)
    return tuple(utm.total_bounds)


def _download_tile(
    bbox_utm: tuple[float, float, float, float],
    pixel_size_m: float,
    session: requests.Session,
) -> tuple[np.ndarray, rasterio.Affine]:
    minx, miny, maxx, maxy = bbox_utm
    width = max(1, int(math.ceil((maxx - minx) / pixel_size_m)))
    height = max(1, int(math.ceil((maxy - miny) / pixel_size_m)))

    params = {
        "f": "image",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": COUNTY_DEM_CRS.split(":")[1],
        "imageSR": COUNTY_DEM_CRS.split(":")[1],
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataValue": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
    }

    for attempt in range(5):
        response = session.get(IMAGE_SERVER, params=params, timeout=300)
        if response.status_code == 200 and response.content[:4] in (b"II*\x00", b"MM\x00*"):
            with rasterio.open(BytesIO(response.content)) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata if src.nodata is not None else -9999
                data = np.where(data == nodata, np.nan, data)
                return data, src.transform
        time.sleep(2**attempt)

    raise RuntimeError(f"Failed to download DEM tile for bbox {bbox_utm}")


def download_dem(
    output_path: Path = RAW_DEM_PATH,
    *,
    pixel_size_m: float = 10.0,
    tile_size_m: float = 20_000.0,
) -> Path:
    """Download 3DEP elevation in UTM tiles and mosaic to a county-clipped GeoTIFF."""
    county = load_county_boundary()
    build_watershed_outlet_polygon()

    minx, miny, maxx, maxy = _utm_bbox(county)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    datasets = []
    tmp_paths: list[Path] = []

    x_steps = max(1, int(math.ceil((maxx - minx) / tile_size_m)))
    y_steps = max(1, int(math.ceil((maxy - miny) / tile_size_m)))

    print(
        f"Downloading 3DEP DEM for {COUNTY_NAME} County at {pixel_size_m}m "
        f"({x_steps}x{y_steps} tiles)..."
    )

    tile_idx = 0
    for xi in range(x_steps):
        for yi in range(y_steps):
            tile_minx = minx + xi * tile_size_m
            tile_maxx = min(minx + (xi + 1) * tile_size_m, maxx)
            tile_miny = miny + yi * tile_size_m
            tile_maxy = min(miny + (yi + 1) * tile_size_m, maxy)

            data, transform = _download_tile(
                (tile_minx, tile_miny, tile_maxx, tile_maxy),
                pixel_size_m,
                session,
            )
            tile_idx += 1
            tmp = output_path.parent / f"_tile_{tile_idx}.tif"
            profile = {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": COUNTY_DEM_CRS,
                "transform": transform,
                "nodata": np.nan,
            }
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data, 1)
            tmp_paths.append(tmp)
            datasets.append(rasterio.open(tmp))
            print(f"  tile {tile_idx}/{x_steps * y_steps} done")

    mosaic, out_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    county_utm = county.to_crs(COUNTY_DEM_CRS)
    from rasterio.mask import mask as rio_mask

    profile = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": 1,
        "dtype": "float32",
        "crs": COUNTY_DEM_CRS,
        "transform": out_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    mosaic_path = output_path.parent / "_mosaic.tif"
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    with rasterio.open(mosaic_path) as src:
        shapes = [geom for geom in county_utm.geometry]
        clipped, clipped_transform = rio_mask(src, shapes, crop=True, nodata=np.nan)
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped[0], 1)

    for tmp in tmp_paths:
        tmp.unlink(missing_ok=True)
    mosaic_path.unlink(missing_ok=True)

    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download USGS 3DEP DEM for {COUNTY_NAME} County"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="Output pixel size in meters (default: 10). Use 1 for 1m lidar (slow).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DEM_PATH,
        help="Output GeoTIFF path",
    )
    args = parser.parse_args()
    download_dem(args.output, pixel_size_m=args.pixel_size)


if __name__ == "__main__":
    main()
