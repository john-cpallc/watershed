"""DEM conditioning, smoothing, and hydrologic grid generation."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds, xy as transform_xy
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, mapping

from watershed.d8 import D8_OFFSETS, DIRMAP, NEIGHBOR_OFFSETS
from watershed.topography import build_topography_tiles
from watershed.geo import (
    COUNTY_DEM_CRS,
    COUNTY_FIPS,
    COUNTY_NAME,
    COUNTY_STATE,
    PROCESSED_DIR,
    WGS84,
    load_county_boundary,
)

DEFAULT_WORKING_RESOLUTION_M = 10
DEFAULT_SMOOTH_SIGMA = 1.0
DEFAULT_STREAM_THRESHOLD = 1000


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = path or (PROCESSED_DIR / "manifest.json")
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def write_manifest(manifest: dict, path: Path | None = None) -> None:
    manifest_path = path or (PROCESSED_DIR / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _pixel_size_m(transform: rasterio.Affine) -> float:
    return float(abs(transform.a))


def resample_raster(
    src_array: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: str,
    dst_shape: tuple[int, int],
    dst_transform: rasterio.Affine,
    dst_crs: str,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    dst = np.empty(dst_shape, dtype=np.float32)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def resample_to_resolution(
    dem: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    target_resolution_m: float,
) -> tuple[np.ndarray, rasterio.Affine]:
    current = _pixel_size_m(transform)
    if abs(current - target_resolution_m) < 0.01:
        return dem, transform

    scale = current / target_resolution_m
    height, width = dem.shape
    dst_width = max(1, int(round(width * scale)))
    dst_height = max(1, int(round(height * scale)))

    bounds = rasterio.transform.array_bounds(height, width, transform)
    dst_transform = from_bounds(*bounds, dst_width, dst_height)
    dst = resample_raster(
        dem,
        transform,
        crs,
        (dst_height, dst_width),
        dst_transform,
        crs,
    )
    return dst, dst_transform


def smooth_dem(dem: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return dem.copy()
    valid = ~np.isnan(dem)
    if not valid.any():
        return dem.copy()
    fill_value = float(np.nanmedian(dem))
    filled = np.where(valid, dem, fill_value)
    smoothed = gaussian_filter(filled, sigma=sigma)
    return np.where(valid, smoothed, np.nan).astype(np.float32)


def _neighbor_minimum(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
  mins = np.full(dem.shape, np.inf, dtype=np.float64)
  for dr, dc in NEIGHBOR_OFFSETS:
      neighbor = _shifted(dem, dr, dc)
      use = valid & ~np.isnan(neighbor)
      mins = np.where(use, np.minimum(mins, neighbor), mins)
  return mins


def fill_depressions(dem: np.ndarray, passes: int = 200) -> np.ndarray:
    """Raise cells iteratively to the minimum neighbor elevation until stable."""
    filled = dem.astype(np.float64).copy()
    valid = ~np.isnan(filled)
    if not valid.any():
        return dem.copy()

    for _ in range(passes):
        mins = _neighbor_minimum(filled, valid)
        raise_mask = valid & (filled < mins)
        if not raise_mask.any():
            break
        filled = np.where(raise_mask, mins, filled)

    return np.where(valid, filled, np.nan).astype(np.float32)


def _shifted(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Return neighbor values at offset (dr, dc) aligned to each source cell."""
    out = np.full(arr.shape, np.nan, dtype=arr.dtype)
    row_a = max(0, -dr)
    row_b = arr.shape[0] - max(0, dr)
    col_a = max(0, -dc)
    col_b = arr.shape[1] - max(0, dc)
    row_a_src = max(0, dr)
    row_b_src = arr.shape[0] - max(0, -dr)
    col_a_src = max(0, dc)
    col_b_src = arr.shape[1] - max(0, -dc)
    out[row_a:row_b, col_a:col_b] = arr[row_a_src:row_b_src, col_a_src:col_b_src]
    return out


def resolve_flats(dem: np.ndarray, scale: float = 1e-4) -> np.ndarray:
    """Break flats with a monotonic row/col drainage bias."""
    valid = ~np.isnan(dem)
    if not valid.any():
        return dem.copy()
    rows, cols = np.indices(dem.shape)
    bias = (rows.astype(np.float64) + cols.astype(np.float64) * 0.1) * scale
    out = dem.astype(np.float64) + bias
    return np.where(valid, out, np.nan).astype(np.float32)


def flow_direction(dem: np.ndarray) -> np.ndarray:
    """Vectorized D8 steepest-descent flow direction (ESRI encoding)."""
    dem = dem.astype(np.float64)
    valid = ~np.isnan(dem)
    rows, cols = np.indices(dem.shape)
    dem = dem.copy()
    dem[valid] += rows[valid] * 1e-3 + cols[valid] * 1e-4

    best_drop = np.zeros(dem.shape, dtype=np.float64)
    best_dir = np.zeros(dem.shape, dtype=np.int32)

    specs = [
        (-1, 0, 64, 1.0),
        (-1, 1, 128, 1.41421356237),
        (0, 1, 1, 1.0),
        (1, 1, 2, 1.41421356237),
        (1, 0, 4, 1.0),
        (1, -1, 8, 1.41421356237),
        (0, -1, 16, 1.0),
        (-1, -1, 32, 1.41421356237),
    ]
    center = dem
    for dr, dc, code, dist in specs:
        neighbor = _shifted(dem, dr, dc)
        mask = valid & ~np.isnan(neighbor)
        drop = np.zeros_like(dem)
        drop[mask] = (center[mask] - neighbor[mask]) / dist
        improve = mask & (drop > best_drop)
        best_drop = np.where(improve, drop, best_drop)
        best_dir = np.where(improve, code, best_dir)
    return best_dir.astype(np.int32)


def flow_accumulation(fdir: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Count upstream cells by propagating from high elevation to low."""
    h, w = fdir.shape
    valid = ~np.isnan(dem)
    acc = np.ones((h, w), dtype=np.int32)

    dr_out = np.zeros((h, w), dtype=np.int32)
    dc_out = np.zeros((h, w), dtype=np.int32)
    for code, (dr, dc) in D8_OFFSETS.items():
        mask = fdir == code
        dr_out[mask] = dr
        dc_out[mask] = dc

    rows = np.arange(h, dtype=np.int32)[:, None]
    cols = np.arange(w, dtype=np.int32)[None, :]
    nrow = rows + dr_out
    ncol = cols + dc_out
    has_ds = (
        (fdir > 0)
        & valid
        & (nrow >= 0)
        & (nrow < h)
        & (ncol >= 0)
        & (ncol < w)
    )

    flat_dem = np.where(valid, dem, -np.inf).ravel()
    order = np.argsort(flat_dem)[::-1]
    for idx in order:
        row, col = divmod(int(idx), w)
        if not valid[row, col] or not has_ds[row, col]:
            continue
        nr = int(nrow[row, col])
        nc = int(ncol[row, col])
        acc[nr, nc] += acc[row, col]
    return acc


def extract_stream_network(
    fdir: np.ndarray,
    acc: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    threshold: int,
) -> dict:
    """Build GeoJSON stream segments along high-accumulation cells."""
    features = []
    h, w = acc.shape
    for row in range(h):
        for col in range(w):
            if acc[row, col] < threshold:
                continue
            direction = int(fdir[row, col])
            if direction not in D8_OFFSETS:
                continue
            dr, dc = D8_OFFSETS[direction]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if acc[nr, nc] < threshold:
                continue
            x1, y1 = transform_xy(transform, row, col, offset="center")
            x2, y2 = transform_xy(transform, nr, nc, offset="center")
            line = LineString([(x1, y1), (x2, y2)])
            if crs != WGS84:
                gdf = gpd.GeoDataFrame(geometry=[line], crs=crs).to_crs(WGS84)
                line = gdf.geometry.iloc[0]
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(line),
                    "properties": {
                        "acc": int(acc[row, col]),
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def write_geotiff(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    nodata: float | None = np.nan,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32" if np.issubdtype(array.dtype, np.floating) else array.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def read_geotiff(path: Path) -> tuple[np.ndarray, rasterio.Affine, str]:
    with rasterio.open(path) as src:
        data = src.read(1)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        data = data.astype(np.float32) if np.issubdtype(data.dtype, np.floating) else data
        return data, src.transform, src.crs.to_string()


def clip_dem_to_county(
    dem_path: Path,
    output_path: Path,
    county_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    county_gdf = county_gdf or load_county_boundary()
    county = county_gdf.to_crs(COUNTY_DEM_CRS)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs.to_string()
        county_reproj = county.to_crs(dem_crs)
        shapes = [geom for geom in county_reproj.geometry]
        out_image, out_transform = rio_mask(src, shapes, crop=True, nodata=np.nan)
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            nodata=np.nan,
            compress="deflate",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out_image[0], 1)
    return output_path


def build_hydro_products(
    raw_dem_path: Path,
    *,
    working_resolution_m: float = DEFAULT_WORKING_RESOLUTION_M,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
    stream_threshold: int = DEFAULT_STREAM_THRESHOLD,
) -> dict:
    """Full pipeline: resample → smooth → condition → flowdir → accumulation → streams."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    county = load_county_boundary()

    dem, transform, crs = read_geotiff(raw_dem_path)
    source_resolution_m = _pixel_size_m(transform)

    pixel_count = dem.shape[0] * dem.shape[1]
    if pixel_count > 12_000_000 and working_resolution_m < 30:
        working_resolution_m = 30.0
        print(
            f"Large DEM ({pixel_count:,} px); using {working_resolution_m}m working resolution"
        )

    dem_work, transform_work = resample_to_resolution(
        dem, transform, crs, working_resolution_m
    )
    dem_raw_path = PROCESSED_DIR / "dem_10m_raw.tif"
    write_geotiff(dem_raw_path, dem_work, transform_work, crs)

    dem_smooth = smooth_dem(dem_work, smooth_sigma)
    dem_smooth_path = PROCESSED_DIR / "dem_10m_smooth.tif"
    write_geotiff(dem_smooth_path, dem_smooth, transform_work, crs)

    print("Building zoomable elevation tiles for geofence…")
    topo_meta = build_topography_tiles(
        dem_smooth,
        transform_work,
        crs,
        county,
        pixel_size_m=_pixel_size_m(transform_work),
    )
    print(f"  topography: {topo_meta['tile_count']} tiles (z{topo_meta['minzoom']}–z{topo_meta['maxzoom']})")

    dem_filled = fill_depressions(dem_smooth)
    dem_conditioned = resolve_flats(dem_filled)
    dem_conditioned_path = PROCESSED_DIR / "dem_conditioned.tif"
    write_geotiff(dem_conditioned_path, dem_conditioned, transform_work, crs)

    valid = ~np.isnan(dem_conditioned)
    fdir = flow_direction(dem_conditioned)
    fdir = np.where(valid, fdir, -1).astype(np.int32)
    acc = flow_accumulation(fdir, dem_conditioned)
    acc = np.where(valid, acc, 0).astype(np.int32)

    fdir_path = PROCESSED_DIR / "fdir.tif"
    acc_path = PROCESSED_DIR / "acc.tif"
    write_geotiff(fdir_path, fdir.astype(np.float32), transform_work, crs, nodata=0)
    write_geotiff(acc_path, acc.astype(np.float32), transform_work, crs, nodata=0)

    streams = extract_stream_network(fdir, acc, transform_work, crs, stream_threshold)
    streams_path = PROCESSED_DIR / "streams.geojson"
    with streams_path.open("w", encoding="utf-8") as f:
        json.dump(streams, f)

    bounds = rasterio.transform.array_bounds(
        dem_smooth.shape[0], dem_smooth.shape[1], transform_work
    )
    manifest = {
        "county_fips": COUNTY_FIPS,
        "county_name": COUNTY_NAME,
        "state": COUNTY_STATE,
        "crs": crs,
        "dem_crs": COUNTY_DEM_CRS,
        "transform": list(transform_work)[:6],
        "width": int(dem_smooth.shape[1]),
        "height": int(dem_smooth.shape[0]),
        "bounds": list(bounds),
        "bbox_wgs84": list(county.total_bounds),
        "source_resolution_m": source_resolution_m,
        "working_resolution_m": working_resolution_m,
        "smooth_method": "gaussian",
        "smooth_sigma": smooth_sigma,
        "dirmap": list(DIRMAP),
        "stream_threshold": stream_threshold,
        "files": {
            "dem_raw": dem_raw_path.name,
            "dem_smooth": dem_smooth_path.name,
            "dem_conditioned": dem_conditioned_path.name,
            "fdir": fdir_path.name,
            "acc": acc_path.name,
            "streams": streams_path.name,
            "topography": "topography/tileset.json",
        },
        "topography": topo_meta,
    }
    write_manifest(manifest)
    return manifest
