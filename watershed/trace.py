"""Downstream raindrop tracing on a D8 flow-direction grid."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from watershed.d8 import D8_OFFSETS, NEIGHBOR_OFFSETS
from watershed.geo import WGS84, load_watershed_outlet, wgs84_to_dem_crs
from watershed.hydro import load_manifest


@dataclass
class TraceResult:
    line_wgs84: LineString
    length_km: float
    cells_traversed: int
    terminated_at: str
    snapped: bool
    breaches: int = 0


MAX_CELL_VISITS = 3
CYCLE_ESCAPE_RADIUS_FACTOR = 2
MAX_UPHILL_ESCAPE_M = 8.0


class FlowTracer:
    def __init__(
        self,
        processed_dir: Path,
        *,
        snap_threshold: int = 5000,
        max_steps: int = 100_000,
        max_breaches: int = 600,
        breach_search_radius: int = 20,
    ):
        self.processed_dir = processed_dir
        self.snap_threshold = snap_threshold
        self.max_steps = max_steps
        self.max_breaches = max_breaches
        self.breach_search_radius = breach_search_radius
        self.manifest = load_manifest(processed_dir / "manifest.json")

        fdir_path = processed_dir / self.manifest["files"]["fdir"]
        acc_path = processed_dir / self.manifest["files"]["acc"]
        dem_key = self.manifest["files"].get("dem_conditioned", self.manifest["files"]["dem_smooth"])
        dem_path = processed_dir / dem_key

        with rasterio.open(fdir_path) as src:
            self.fdir = src.read(1).astype(np.int32)
            self.transform = src.transform
            self.crs = src.crs.to_string()
            self.height, self.width = self.fdir.shape

        with rasterio.open(acc_path) as src:
            self.acc = src.read(1).astype(np.int32)

        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype(np.float64)

        outlet = load_watershed_outlet()
        self._outlet_prep = prep(outlet.union_all())
        self._from_dem = Transformer.from_crs(self.crs, WGS84, always_xy=True)
        self._county_south = float(self.manifest["bbox_wgs84"][1])

    def _cell_center(self, row: int, col: int) -> tuple[float, float]:
        x, y = rasterio.transform.xy(self.transform, row, col, offset="center")
        return float(x), float(y)

    def _to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._from_dem.transform(x, y)
        return float(lon), float(lat)

    def _xy_to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        col, row = ~self.transform * (x, y)
        return int(round(row)), int(round(col))

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def _in_outlet(self, lon: float, lat: float) -> bool:
        return self._outlet_prep.contains(Point(lon, lat))

    def _near_downstream_outlet(self, lon: float, lat: float) -> bool:
        return lat <= self._county_south + 0.08

    def _line(self, coords: list[tuple[float, float]]) -> LineString:
        if len(coords) > 1:
            return LineString(coords)
        return LineString([coords[0], coords[0]])

    def _snap(self, x: float, y: float) -> tuple[float, float, bool]:
        row, col = self._xy_to_rowcol(x, y)
        if not self._in_bounds(row, col) or self.fdir[row, col] < 0:
            return x, y, False

        if self.acc[row, col] >= self.snap_threshold and self.fdir[row, col] > 0:
            cx, cy = self._cell_center(row, col)
            return cx, cy, True

        best_acc = -1
        best: tuple[int, int] | None = None
        for radius in range(1, 21):
            r0 = max(0, row - radius)
            r1 = min(self.height, row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(self.width, col + radius + 1)
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    if self.fdir[nr, nc] <= 0:
                        continue
                    acc_val = int(self.acc[nr, nc])
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best = (nr, nc)
            if best_acc >= self.snap_threshold:
                break

        if best is None or best_acc < self.snap_threshold:
            return x, y, False
        cx, cy = self._cell_center(best[0], best[1])
        return cx, cy, True

    def _dem_value(self, row: int, col: int) -> float | None:
        if not self._in_bounds(row, col):
            return None
        value = float(self.dem[row, col])
        if np.isnan(value):
            return None
        return value

    def _can_enter(self, row: int, col: int) -> bool:
        """True for in-bounds cells with elevation, including D8 sinks (fdir=0)."""
        if not self._in_bounds(row, col):
            return False
        if self.fdir[row, col] < 0:
            return False
        return self._dem_value(row, col) is not None

    def _visit_count(self, visit_counts: dict[tuple[int, int], int], cell: tuple[int, int]) -> int:
        return visit_counts.get(cell, 0)

    def _is_exhausted(self, visit_counts: dict[tuple[int, int], int], cell: tuple[int, int]) -> bool:
        return self._visit_count(visit_counts, cell) >= MAX_CELL_VISITS

    def _steepest_descent_step(
        self,
        row: int,
        col: int,
        visit_counts: dict[tuple[int, int], int],
    ) -> tuple[int, int] | None:
        """One downhill step on the DEM, breaking flats with the same bias as hydro conditioning."""
        current = self._dem_value(row, col)
        if current is None:
            return None

        best: tuple[int, int] | None = None
        best_score = -np.inf
        flat_origin = row + col * 0.1
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            cell = (nr, nc)
            if self._is_exhausted(visit_counts, cell) or not self._can_enter(nr, nc):
                continue
            neighbor = self._dem_value(nr, nc)
            if neighbor is None:
                continue
            dist = 1.41421356237 if dr and dc else 1.0
            drop = current - neighbor
            flat_bias = (nr + nc * 0.1 - flat_origin) * 1e-4
            revisit_penalty = self._visit_count(visit_counts, cell) * 0.05
            score = drop / dist + flat_bias + self.acc[nr, nc] * 1e-7 - revisit_penalty
            if score > best_score:
                best_score = score
                best = cell
        if best is None or best_score <= 0:
            return None
        return best

    def _find_escape_cell(
        self,
        row: int,
        col: int,
        visit_counts: dict[tuple[int, int], int],
        *,
        max_radius: int | None = None,
        allow_uphill_m: float = 0.0,
    ) -> tuple[int, int] | None:
        """Jump to a nearby downhill cell when D8 routing hits a sink or nodata gap."""
        current = self._dem_value(row, col)
        if current is None:
            return None

        best: tuple[int, int] | None = None
        best_score = -np.inf
        flat_origin = row + col * 0.1
        radius_limit = max_radius or self.breach_search_radius
        for radius in range(1, radius_limit + 1):
            r0 = max(0, row - radius)
            r1 = min(self.height, row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(self.width, col + radius + 1)
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    cell = (nr, nc)
                    if cell == (row, col) or self._is_exhausted(visit_counts, cell):
                        continue
                    if not self._can_enter(nr, nc):
                        continue
                    neighbor = self._dem_value(nr, nc)
                    if neighbor is None:
                        continue
                    dist = max(abs(nr - row), abs(nc - col))
                    if dist > radius:
                        continue
                    drop = current - neighbor
                    if drop < -allow_uphill_m:
                        continue
                    flat_bias = (nr + nc * 0.1 - flat_origin) * 1e-4
                    revisit_penalty = self._visit_count(visit_counts, cell) * 0.02
                    score = (
                        drop / dist
                        + flat_bias
                        + self.acc[nr, nc] * 1e-6
                        - revisit_penalty
                    )
                    if score > best_score:
                        best_score = score
                        best = cell
            if best is not None and best_score > -allow_uphill_m:
                return best
        return None

    def _step_toward_accumulation(
        self,
        row: int,
        col: int,
        visit_counts: dict[tuple[int, int], int],
        *,
        max_radius: int | None = None,
        min_acc_gain: float = 1.0,
    ) -> tuple[int, int] | None:
        """When stuck in a visited bowl, step toward higher flow accumulation."""
        current_acc = int(self.acc[row, col])
        best: tuple[int, int] | None = None
        best_score = -np.inf
        radius_limit = max_radius or self.breach_search_radius
        for radius in range(1, radius_limit + 1):
            r0 = max(0, row - radius)
            r1 = min(self.height, row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(self.width, col + radius + 1)
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    cell = (nr, nc)
                    if self._is_exhausted(visit_counts, cell) or not self._can_enter(nr, nc):
                        continue
                    dist = max(abs(nr - row), abs(nc - col))
                    if dist > radius:
                        continue
                    acc_val = int(self.acc[nr, nc])
                    if acc_val <= current_acc:
                        continue
                    gain = acc_val - current_acc
                    if gain < min_acc_gain:
                        continue
                    revisit_penalty = self._visit_count(visit_counts, cell) * gain * 0.25
                    score = gain / dist - revisit_penalty
                    if score > best_score:
                        best_score = score
                        best = cell
            if best is not None:
                return best
        return None

    def _cycle_escape(
        self,
        row: int,
        col: int,
        visit_counts: dict[tuple[int, int], int],
    ) -> tuple[int, int] | None:
        """Aggressive jump out of a routing loop or visited bowl."""
        wide_radius = self.breach_search_radius * CYCLE_ESCAPE_RADIUS_FACTOR
        step = self._find_escape_cell(
            row,
            col,
            visit_counts,
            max_radius=wide_radius,
            allow_uphill_m=MAX_UPHILL_ESCAPE_M,
        )
        if step is not None:
            return step

        step = self._step_toward_accumulation(
            row,
            col,
            visit_counts,
            max_radius=wide_radius,
            min_acc_gain=1.0,
        )
        if step is not None:
            return step

        current = self._dem_value(row, col)
        if current is None:
            return None

        current_acc = int(self.acc[row, col])
        best: tuple[int, int] | None = None
        best_score = -np.inf
        for radius in range(1, wide_radius + 1):
            r0 = max(0, row - radius)
            r1 = min(self.height, row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(self.width, col + radius + 1)
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    cell = (nr, nc)
                    if cell == (row, col) or not self._can_enter(nr, nc):
                        continue
                    dist = max(abs(nr - row), abs(nc - col))
                    if dist > radius:
                        continue
                    visits = self._visit_count(visit_counts, cell)
                    if visits >= MAX_CELL_VISITS:
                        continue
                    neighbor = self._dem_value(nr, nc)
                    if neighbor is None:
                        continue
                    acc_val = int(self.acc[nr, nc])
                    drop = current - neighbor
                    score = (
                        (acc_val - current_acc) * 0.05
                        + min(drop, MAX_UPHILL_ESCAPE_M) / dist
                        - visits * 50.0
                    )
                    if score > best_score:
                        best_score = score
                        best = cell
            if best is not None:
                return best
        return None

    def _next_cell(
        self,
        row: int,
        col: int,
        visit_counts: dict[tuple[int, int], int],
    ) -> tuple[int, int, bool] | None:
        """Return (next_row, next_col, used_breach) or None if stuck."""
        direction = int(self.fdir[row, col])
        if direction in D8_OFFSETS:
            dr, dc = D8_OFFSETS[direction]
            nr, nc = row + dr, col + dc
            cell = (nr, nc)
            if not self._is_exhausted(visit_counts, cell) and self._can_enter(nr, nc):
                return nr, nc, False

        step = self._steepest_descent_step(row, col, visit_counts)
        if step is not None:
            return step[0], step[1], True

        escape = self._find_escape_cell(row, col, visit_counts)
        if escape is not None:
            return escape[0], escape[1], True

        toward_stream = self._step_toward_accumulation(row, col, visit_counts)
        if toward_stream is not None:
            return toward_stream[0], toward_stream[1], True

        return None

    def _at_domain_outlet(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col) or self.fdir[row, col] < 0:
            return True
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if not self._in_bounds(nr, nc) or self.fdir[nr, nc] < 0:
                return True
        return False

    def trace(self, lon: float, lat: float) -> TraceResult:
        x, y = wgs84_to_dem_crs(lon, lat, self.crs)
        sx, sy, snapped = self._snap(x, y)
        start_lon, start_lat = self._to_wgs84(sx, sy)

        coords = [(start_lon, start_lat)]
        row, col = self._xy_to_rowcol(sx, sy)
        terminated_at = "edge"
        steps = 0
        breaches = 0
        visit_counts: dict[tuple[int, int], int] = defaultdict(int)

        if self.fdir[row, col] < 0 if self._in_bounds(row, col) else True:
            return TraceResult(self._line(coords), 0.0, 0, "out_of_bounds", snapped)

        if self._in_outlet(start_lon, start_lat):
            return TraceResult(self._line(coords), 0.0, 0, "outlet", snapped)

        for _ in range(self.max_steps):
            cell = (row, col)
            if self._is_exhausted(visit_counts, cell):
                escape = self._cycle_escape(row, col, visit_counts)
                if escape is None or breaches >= self.max_breaches:
                    terminated_at = "cycle"
                    break
                breaches += 1
                row, col = escape
                coords.append(self._to_wgs84(*self._cell_center(row, col)))
                steps += 1
                continue

            visit_counts[cell] += 1

            lon_c, lat_c = coords[-1]
            if self._in_outlet(lon_c, lat_c):
                terminated_at = "outlet"
                break
            if self._at_domain_outlet(row, col) and self._near_downstream_outlet(lon_c, lat_c):
                terminated_at = "outlet"
                break

            nxt = self._next_cell(row, col, visit_counts)
            if nxt is None:
                escape = self._cycle_escape(row, col, visit_counts)
                if escape is None:
                    terminated_at = "sink" if int(self.fdir[row, col]) <= 0 else "edge"
                    break
                if breaches >= self.max_breaches:
                    terminated_at = "sink" if int(self.fdir[row, col]) <= 0 else "edge"
                    break
                breaches += 1
                row, col = escape
                coords.append(self._to_wgs84(*self._cell_center(row, col)))
                steps += 1
                continue

            nrow, ncol, breached = nxt
            if breached:
                if breaches >= self.max_breaches:
                    terminated_at = "sink" if int(self.fdir[row, col]) <= 0 else "edge"
                    break
                breaches += 1

            row, col = nrow, ncol
            coords.append(self._to_wgs84(*self._cell_center(row, col)))
            steps += 1
        else:
            terminated_at = "max_steps"

        line = self._line(coords)
        return TraceResult(line, self._length_km(line), steps, terminated_at, snapped, breaches)

    def _length_km(self, line: LineString) -> float:
        gdf = gpd.GeoDataFrame(geometry=[line], crs=WGS84).to_crs(self.crs)
        return float(gdf.length.iloc[0] / 1000.0)
