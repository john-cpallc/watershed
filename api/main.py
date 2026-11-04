"""FastAPI server for raindrop flow tracing."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from watershed.geo import (
    PROCESSED_DIR,
    geojson_feature,
    load_county_boundary,
    project_root,
)
from watershed.hydro import load_manifest
from watershed.trace import FlowTracer

WEB_DIR = project_root() / "web"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

tracer: FlowTracer | None = None
_tracer_manifest_mtime: float | None = None


def get_tracer() -> FlowTracer:
    """Return a FlowTracer, reloading when processed grids are rebuilt."""
    global tracer, _tracer_manifest_mtime

    if not MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Hydrologic grids not built. Run scripts/build_hydro_grid.py",
        )

    mtime = MANIFEST_PATH.stat().st_mtime
    if tracer is None or _tracer_manifest_mtime != mtime:
        tracer = FlowTracer(PROCESSED_DIR)
        _tracer_manifest_mtime = mtime
    return tracer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tracer, _tracer_manifest_mtime
    if MANIFEST_PATH.exists():
        tracer = FlowTracer(PROCESSED_DIR)
        _tracer_manifest_mtime = MANIFEST_PATH.stat().st_mtime
    yield
    tracer = None
    _tracer_manifest_mtime = None


app = FastAPI(title="Alabama Rainwater Flow Map", lifespan=lifespan)


class TraceRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "hydro_ready": MANIFEST_PATH.exists(),
    }


@app.get("/api/layers")
def layers():
    if not MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Hydrologic grids not built. Run scripts/build_hydro_grid.py",
        )

    county = load_county_boundary()
    manifest = load_manifest()

    streams_path = PROCESSED_DIR / manifest["files"]["streams"]
    with streams_path.open(encoding="utf-8") as f:
        streams = json.load(f)

    return {
        "manifest": manifest,
        "county": json.loads(county.to_json()),
        "streams": streams,
        "topography": manifest.get("topography"),
    }


@app.post("/api/trace")
def trace(request: TraceRequest):
    result = get_tracer().trace(request.lon, request.lat)
    feature = geojson_feature(
        result.line_wgs84,
        {
            "length_km": round(result.length_km, 2),
            "cells_traversed": result.cells_traversed,
            "terminated_at": result.terminated_at,
            "snapped": result.snapped,
            "breaches": result.breaches,
            "start": [request.lon, request.lat],
        },
    )
    return feature


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


TOPO_TILES_DIR = PROCESSED_DIR / "topography" / "tiles"

# 256×256 transparent PNG — returned for tile indices outside the geofence pyramid
_TRANSPARENT_TILE = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00"
    b"\x08\x06\x00\x00\x00\\r\xa8f\x00\x00\x00\x0eIDATx\x9cc\x00\x00\x00"
    b"\x02\x00\x01\xe5'\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82"
)


@app.get("/api/tiles/topography/{z}/{x}/{y}.png")
def topography_tile(z: int, x: int, y: int):
    tile_path = TOPO_TILES_DIR / str(z) / str(x) / f"{y}.png"
    if tile_path.is_file():
        return FileResponse(tile_path, media_type="image/png")
    return Response(content=_TRANSPARENT_TILE, media_type="image/png")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
