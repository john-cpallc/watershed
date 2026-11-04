"""Elevation-colored zoomable XYZ topography tiles for the geofence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import Normalize
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from PIL import Image

from watershed.geo import PROCESSED_DIR, WGS84

WEB_MERCATOR = "EPSG:3857"
TILE_SIZE = 256
ELEVATION_CMAP = colormaps["terrain"]


def _meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2**zoom)


def _choose_zoom_range(
    bbox_wgs84: tuple[float, float, float, float],
    pixel_size_m: float,
) -> tuple[int, int]:
    """Pick tile zoom levels that match the working DEM resolution."""
    lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2.0
    max_zoom = 14
    for z in range(8, 16):
        if _meters_per_pixel(lat, z) <= pixel_size_m * 1.5:
            max_zoom = z
            break
    min_zoom = max(7, max_zoom - 5)
    return min_zoom, max_zoom


def _reproject_dem_to_web_mercator(
    dem: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    pixel_size_m: float,
) -> tuple[np.ndarray, rasterio.Affine]:
    src_bounds = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], transform)
    west, south, east, north = transform_bounds(
        crs, WEB_MERCATOR, *src_bounds, densify_pts=21
    )
    width = max(1, int((east - west) / pixel_size_m))
    height = max(1, int((north - south) / pixel_size_m))
    dst_transform = from_bounds(west, south, east, north, width, height)
    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=dem.astype(np.float32),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=WEB_MERCATOR,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst, dst_transform


def _elevation_range(dem: np.ndarray) -> tuple[float, float]:
    valid = dem[~np.isnan(dem)]
    if valid.size == 0:
        return 0.0, 1.0
    return float(valid.min()), float(valid.max())


def _elevation_to_rgb(
    elevation: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Map elevation (meters) to RGB using the terrain colormap."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(np.nan_to_num(elevation, nan=vmin))
    rgba = ELEVATION_CMAP(normalized)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def _tile_image(
    dem_merc: np.ndarray,
    transform: rasterio.Affine,
    tile: mercantile.Tile,
    *,
    vmin: float,
    vmax: float,
) -> Image.Image | None:
    """Render one 256×256 RGBA elevation tile."""
    bounds = mercantile.xy_bounds(tile)
    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    tile_transform = from_bounds(west, south, east, north, TILE_SIZE, TILE_SIZE)

    elevation = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
    reproject(
        source=dem_merc,
        destination=elevation,
        src_transform=transform,
        src_crs=WEB_MERCATOR,
        dst_transform=tile_transform,
        dst_crs=WEB_MERCATOR,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    valid = np.isfinite(elevation)
    if not valid.any():
        return None

    rgb = _elevation_to_rgb(elevation, vmin, vmax)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def build_topography_tiles(
    dem: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    boundary_wgs84: gpd.GeoDataFrame,
    *,
    pixel_size_m: float,
    output_dir: Path | None = None,
) -> dict:
    """Build a zoomable XYZ elevation tile pyramid clipped to the geofence."""
    output_dir = output_dir or (PROCESSED_DIR / "topography")
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    bbox = tuple(boundary_wgs84.total_bounds)
    vmin, vmax = _elevation_range(dem)

    elev_path = output_dir / "elevation.tif"
    with rasterio.open(
        elev_path,
        "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(dem.astype(np.float32), 1)

    dem_merc, merc_transform = _reproject_dem_to_web_mercator(
        dem, transform, crs, pixel_size_m
    )

    min_zoom, max_zoom = _choose_zoom_range(bbox, pixel_size_m)
    tile_count = 0
    for z in range(min_zoom, max_zoom + 1):
        for tile in mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zooms=z):
            image = _tile_image(
                dem_merc,
                merc_transform,
                tile,
                vmin=vmin,
                vmax=vmax,
            )
            if image is None:
                continue
            tile_path = tiles_dir / str(z) / str(tile.x) / f"{tile.y}.png"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(tile_path, optimize=True)
            tile_count += 1

    meta = {
        "tilesize": TILE_SIZE,
        "scheme": "xyz",
        "format": "png",
        "style": "elevation",
        "colormap": "terrain",
        "elevation_min_m": vmin,
        "elevation_max_m": vmax,
        "minzoom": min_zoom,
        "maxzoom": max_zoom,
        "bounds": list(bbox),
        "tile_url": "/api/tiles/topography/{z}/{x}/{y}.png",
        "tile_count": tile_count,
        "elevation_tif": elev_path.name,
    }
    with (output_dir / "tileset.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
