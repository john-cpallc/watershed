# Alabama Rainwater Flow Map

Interactive map of **Jefferson County, Alabama**. Click a point and trace where a raindrop would flow downhill along a **D8 flow-direction grid** until it reaches a **downstream watershed outlet** (Black Warrior / Cahaba drainages).

This project is a hands-on introduction to GIS / hydrology: digital elevation models (DEMs), hydrologic conditioning, flow accumulation, and pour-point tracing.

## What happens under the hood

1. **Download** USGS 3DEP elevation for Jefferson County.
2. **Resample** to a 10 m working grid (bilinear).
3. **Smooth** with a light Gaussian filter to reduce lidar speckle.
4. **Condition** the DEM: fill pits, resolve flats.
5. **Compute** D8 flow direction and flow accumulation.
6. **Export** a zoomable **elevation tile pyramid** (terrain colormap, XYZ PNGs) clipped to the county geofence.
7. **Trace** interactively on the map with the elevation basemap.

Large county grids auto-resample to **30 m** working resolution when the source exceeds ~12M pixels.

## Setup (uv venv)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd watershed
uv venv
uv sync
```

All commands below use the project virtual environment via `uv run`.

## Build data (first time)

```bash
# 1. Download elevation (~10m; use --pixel-size 1 for 1m lidar — much slower)
uv run python scripts/download_dem.py

# 2. Build flow-direction grid, accumulation, and streams
uv run python scripts/build_hydro_grid.py
```

Processed outputs land in `data/processed/`:

| File | Description |
|------|-------------|
| `dem_10m_raw.tif` | Resampled DEM before smoothing |
| `dem_10m_smooth.tif` | Smoothed DEM used for hydrology |
| `fdir.tif` | D8 flow direction grid |
| `acc.tif` | Flow accumulation |
| `streams.geojson` | Extracted stream network |
| `topography/tileset.json` | Zoomable elevation tile metadata (min/max meters, colormap) |
| `topography/tiles/{z}/{x}/{y}.png` | XYZ elevation-colored tiles for the map |
| `topography/elevation.tif` | Elevation raster used for tile coloring |
| `manifest.json` | CRS, resolution, tuning parameters |

## Run the app

```bash
uv run uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000 — click anywhere in Jefferson County to trace a path.

## Learning notebooks

```bash
uv run jupyter lab notebooks/
```

- `01_dem_and_conditioning.ipynb` — DEM, smoothing, pit filling
- `02_flow_direction.ipynb` — D8 vector field and accumulation
- `03_trace_paths.ipynb` — downstream tracing and validation

## Project layout

```
scripts/          Download DEM, build hydro grids
watershed/        geo, hydro, trace Python modules
api/              FastAPI trace server
web/              MapLibre interactive map
notebooks/        Step-by-step GIS learning
data/boundaries/  County + watershed outlet polygons
data/processed/   Generated rasters and GeoJSON
```

## Configuration

County settings live in `watershed/geo.py` (`COUNTY_FIPS`, `COUNTY_NAME`, etc.).

Tune preprocessing via `build_hydro_grid.py` flags:

```bash
uv run python scripts/build_hydro_grid.py --resolution 10 --smooth-sigma 1.0 --stream-threshold 1000
```

Set `smooth_sigma` to `0` to compare unsmoothed routing.

## Limitations

- **D8** routes to one downslope neighbor; real hillslope flow can spread.
- **30 m working grid** (auto-selected for full county) misses small ditches and culverts.
- **County-clipped DEM** — flow often stops at `edge` where the grid hits nodata at the county boundary; the tracer can jump gaps but won't route through storm drains.
- **Urban Birmingham** has DEM gaps and storm-drain routing not modeled.
- **Smoothing** stabilizes paths but can soften fine channel detail.
- **Pure terrain routing** — no rainfall, infiltration, storm drains, or pumping.

To rebuild with improved depression filling after code changes:

```bash
uv run python scripts/build_hydro_grid.py
```

## Data sources

- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) elevation
- [Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) county boundaries
