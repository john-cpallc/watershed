const statusEl = document.getElementById("status");
const cursorEl = document.getElementById("cursor");
const legendEl = document.getElementById("elevation-legend");
const resultEl = document.getElementById("result");
const clearBtn = document.getElementById("clear");

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors",
      },
    },
    layers: [
      {
        id: "osm",
        type: "raster",
        source: "osm",
      },
    ],
  },
  center: [-86.79, 33.52],
  zoom: 9.5,
});

map.addControl(new maplibregl.NavigationControl(), "top-right");

function setStatus(text) {
  statusEl.textContent = text;
}

function showResult(props) {
  resultEl.classList.remove("hidden");
  clearBtn.classList.remove("hidden");
  const term = props.terminated_at.replaceAll("_", " ");
  const breachLine = props.breaches
    ? `<div><strong>Gap jumps:</strong> ${props.breaches}</div>`
    : "";
  resultEl.innerHTML = `
    <div><strong>Length:</strong> ${props.length_km} km</div>
    <div><strong>Cells:</strong> ${props.cells_traversed}</div>
    <div><strong>Ended at:</strong> ${term}</div>
    ${breachLine}
    <div><strong>Snapped to channel:</strong> ${props.snapped ? "yes" : "no"}</div>
  `;
}

function clearPath() {
  if (map.getLayer("rain-path-line")) {
    map.removeLayer("rain-path-line");
  }
  if (map.getLayer("rain-path-start")) {
    map.removeLayer("rain-path-start");
  }
  if (map.getSource("rain-path")) {
    map.removeSource("rain-path");
  }
  if (map.getSource("rain-start")) {
    map.removeSource("rain-start");
  }
  resultEl.classList.add("hidden");
  clearBtn.classList.add("hidden");
}

clearBtn.addEventListener("click", clearPath);

async function loadLayers() {
  const response = await fetch("/api/layers");
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function addGeoJsonSource(id, data) {
  if (map.getSource(id)) {
    map.getSource(id).setData(data);
    return;
  }
  map.addSource(id, { type: "geojson", data });
}

function setupLayers(data) {
  const county = data.county;
  const topo = data.topography;

  if (topo) {
    map.addSource("topography", {
      type: "raster",
      tiles: [window.location.origin + topo.tile_url],
      tileSize: topo.tilesize || 256,
      bounds: topo.bounds,
      minzoom: topo.minzoom,
      maxzoom: topo.maxzoom,
    });
    map.addLayer({
      id: "topography",
      type: "raster",
      source: "topography",
      paint: {
        "raster-opacity": 1,
      },
    });

    if (topo.elevation_min_m != null && topo.elevation_max_m != null) {
      legendEl.classList.remove("hidden");
      legendEl.innerHTML = `
        <div><strong>Elevation</strong> (meters, terrain colors)</div>
        <div>${Math.round(topo.elevation_min_m)} m — low &nbsp;→&nbsp; ${Math.round(topo.elevation_max_m)} m — high</div>
      `;
    }
  }

  addGeoJsonSource("county", county);
  map.addLayer({
    id: "county-fill",
    type: "fill",
    source: "county",
    paint: {
      "fill-color": "#0ea5e9",
      "fill-opacity": topo ? 0 : 0.06,
    },
  });
  map.addLayer({
    id: "county-outline",
    type: "line",
    source: "county",
    paint: {
      "line-color": "#0369a1",
      "line-width": 2,
    },
  });

  addGeoJsonSource("streams", data.streams);
  map.addLayer({
    id: "streams",
    type: "line",
    source: "streams",
    paint: {
      "line-color": "#1d4ed8",
      "line-width": 1.2,
      "line-opacity": 0.7,
    },
  });

  const bbox = data.manifest.bbox_wgs84;
  map.fitBounds(
    [
      [bbox[0], bbox[1]],
      [bbox[2], bbox[3]],
    ],
    { padding: 40, duration: 0 }
  );
}

async function tracePath(lon, lat) {
  setStatus("Tracing flow path…");
  const response = await fetch("/api/trace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lon, lat }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const feature = await response.json();
  const lineFc = { type: "FeatureCollection", features: [feature] };
  addGeoJsonSource("rain-path", lineFc);

  if (!map.getLayer("rain-path-line")) {
    map.addLayer({
      id: "rain-path-line",
      type: "line",
      source: "rain-path",
      paint: {
        "line-color": "#dc2626",
        "line-width": 4,
        "line-opacity": 0.9,
      },
    });
  }

  const startFc = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: {},
      },
    ],
  };
  addGeoJsonSource("rain-start", startFc);
  if (!map.getLayer("rain-path-start")) {
    map.addLayer({
      id: "rain-path-start",
      type: "circle",
      source: "rain-start",
      paint: {
        "circle-radius": 6,
        "circle-color": "#f59e0b",
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 2,
      },
    });
  }

  showResult(feature.properties);
  setStatus("Click another point to trace a new path.");
}

map.on("mousemove", (e) => {
  cursorEl.textContent = `Cursor: ${e.lngLat.lng.toFixed(5)}, ${e.lngLat.lat.toFixed(5)}`;
});

map.on("click", async (e) => {
  try {
    await tracePath(e.lngLat.lng, e.lngLat.lat);
  } catch (err) {
    setStatus(`Trace failed: ${err.message}`);
  }
});

map.on("load", async () => {
  try {
    const layers = await loadLayers();
    setupLayers(layers);
    setStatus("Click anywhere in Jefferson County to trace rainwater flow.");
  } catch (err) {
    setStatus(
      "Hydrologic data not ready. Run: uv run python scripts/download_dem.py && uv run python scripts/build_hydro_grid.py"
    );
  }
});
