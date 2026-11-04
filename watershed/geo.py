"""CRS helpers, boundaries, and geometry utilities."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
PROCESSED_DIR = DATA_DIR / "processed"

WGS84 = "EPSG:4326"
COUNTY_FIPS = "01073"
COUNTY_NAME = "Jefferson"
COUNTY_STATE = "AL"
COUNTY_SLUG = "jefferson_al"
CENSUS_COUNTY_GEOJSON = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)

# UTM 16N covers Jefferson County, Alabama
COUNTY_DEM_CRS = "EPSG:32616"
RAW_DEM_PATH = DATA_DIR / "raw" / f"{COUNTY_SLUG}_dem.tif"
WATERSHED_OUTLET_PATH = BOUNDARIES_DIR / "watershed_outlet.geojson"


def project_root() -> Path:
    return PROJECT_ROOT


def load_county_boundary(county_fips: str = COUNTY_FIPS) -> gpd.GeoDataFrame:
    """Load county polygon from bundled GeoJSON or fetch from Census dataset."""
    local_path = BOUNDARIES_DIR / f"{COUNTY_SLUG}_{county_fips}.geojson"
    if local_path.exists():
        gdf = gpd.read_file(local_path)
    else:
        import requests

        response = requests.get(CENSUS_COUNTY_GEOJSON, timeout=60)
        response.raise_for_status()
        counties = response.json()
        feature = next(
            f
            for f in counties["features"]
            if f["properties"]["STATE"] + f["properties"]["COUNTY"] == county_fips
        )
        gdf = gpd.GeoDataFrame.from_features([feature], crs=WGS84)
        BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_file(local_path, driver="GeoJSON")

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    return gdf.to_crs(WGS84)


def county_bbox(county_fips: str = COUNTY_FIPS) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in WGS84."""
    gdf = load_county_boundary(county_fips)
    return tuple(gdf.total_bounds)


def load_watershed_outlet() -> gpd.GeoDataFrame:
    """Downstream termination mask south of the county."""
    if not WATERSHED_OUTLET_PATH.exists():
        raise FileNotFoundError(f"Missing watershed outlet polygon: {WATERSHED_OUTLET_PATH}")
    gdf = gpd.read_file(WATERSHED_OUTLET_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    return gdf.to_crs(WGS84)


def build_watershed_outlet_polygon() -> gpd.GeoDataFrame:
    """Create a simplified downstream termination polygon south of the county."""
    minx, miny, maxx, maxy = county_bbox()
    # Band south of Jefferson County along the Black Warrior / Cahaba drainages
    outlet = box(minx - 0.2, miny - 0.35, maxx + 0.2, miny + 0.05)
    gdf = gpd.GeoDataFrame(geometry=[outlet], crs=WGS84)
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_file(WATERSHED_OUTLET_PATH, driver="GeoJSON")
    return gdf


def wgs84_to_dem_crs(lon: float, lat: float, dem_crs: str = COUNTY_DEM_CRS) -> tuple[float, float]:
    transformer = Transformer.from_crs(WGS84, dem_crs, always_xy=True)
    return transformer.transform(lon, lat)


def dem_crs_to_wgs84(x: float, y: float, dem_crs: str = COUNTY_DEM_CRS) -> tuple[float, float]:
    transformer = Transformer.from_crs(dem_crs, WGS84, always_xy=True)
    return transformer.transform(x, y)


def geojson_feature(geom, properties: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": properties or {},
    }


def geojson_feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def point_in_watershed_outlet(lon: float, lat: float, outlet_gdf: gpd.GeoDataFrame) -> bool:
    pt = Point(lon, lat)
    return bool(outlet_gdf.contains(pt).any())


def clip_geometry_to_county(geom, county_gdf: gpd.GeoDataFrame):
    county = unary_union(county_gdf.geometry)
    return geom.intersection(county)
