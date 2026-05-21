"""Natural Earth feature projection helpers for cubed-sphere grids."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "natural_earth_npz"
AVAILABLE_LAYERS = (
    "land",
    "ocean",
    "coastline",
    "lakes",
    "rivers_lake_centerlines",
    "borders",
    "states_provinces",
)
AVAILABLE_RESOLUTIONS = ("110m", "50m", "10m")
_POLYGON_MASK_CACHE = {}


def clear_natural_earth_cache():
    """Clear cached Natural Earth projection and land-mask results."""
    _load_bundle.cache_clear()
    _POLYGON_MASK_CACHE.clear()


def _asset_path(resolution):
    resolution = str(resolution)
    if resolution not in AVAILABLE_RESOLUTIONS:
        raise ValueError(
            f"Unknown Natural Earth resolution {resolution!r}; "
            f"choose from {AVAILABLE_RESOLUTIONS}."
        )
    path = ASSET_DIR / f"natural_earth_{resolution}_all.npz"
    if not path.exists():
        raise FileNotFoundError(f"Natural Earth asset bundle not found: {path}")
    return path


@lru_cache(maxsize=None)
def _load_bundle(resolution):
    with np.load(_asset_path(resolution), allow_pickle=True) as bundle:
        return {key: bundle[key] for key in bundle.files}


def available_layers(resolution="50m"):
    """Return the feature layers present in a bundled Natural Earth asset."""
    bundle = _load_bundle(resolution)
    return tuple(
        layer for layer in AVAILABLE_LAYERS if f"{layer}_coords" in bundle
    )


def natural_earth_records(layer, resolution="50m"):
    """Return stored Natural Earth record metadata for a layer."""
    bundle = _load_bundle(resolution)
    key = f"{layer}_records_json"
    if key not in bundle:
        raise ValueError(f"Layer {layer!r} is not available in {resolution!r}.")
    return json.loads(str(bundle[key]))


def iter_natural_earth_parts(layer, resolution="50m", part_types=None):
    """
    Yield Natural Earth geometry parts as ``(lon, lat, part_type)`` tuples.

    Coordinates are geographic degrees. Polygon layers may contain exterior and
    hole parts; line layers contain line parts.
    """
    bundle = _load_bundle(resolution)
    prefix = f"{layer}_"
    try:
        coords = bundle[f"{prefix}coords"]
        offsets = bundle[f"{prefix}offsets"]
        stored_part_types = bundle[f"{prefix}part_types"]
    except KeyError as exc:
        raise ValueError(f"Layer {layer!r} is not available in {resolution!r}.") from exc

    if isinstance(part_types, str):
        part_types = {part_types}
    elif part_types is not None:
        part_types = {str(part_type) for part_type in part_types}

    for index, part_type in enumerate(stored_part_types):
        part_type = str(part_type)
        if part_types is not None and part_type not in part_types:
            continue
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        part = coords[start:stop]
        if len(part):
            yield part[:, 0], part[:, 1], part_type


def _projection_from_grid(grid_or_projection):
    return getattr(grid_or_projection, "projection", grid_or_projection)


def grid_extent(grid):
    """Return a cubed-sphere grid extent as ``(x0, x1, y0, y1)``."""
    try:
        return grid.xi_min, grid.xi_max, grid.eta_min, grid.eta_max
    except AttributeError:
        return (
            np.nanmin(grid.xi),
            np.nanmax(grid.xi),
            np.nanmin(grid.eta),
            np.nanmax(grid.eta),
        )


def finite_segments(x, y, min_points=2):
    """Yield contiguous finite projected coordinate segments."""
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return

    starts = np.where(finite & np.r_[True, ~finite[:-1]])[0]
    stops = np.where(finite & np.r_[~finite[1:], True])[0] + 1
    for start, stop in zip(starts, stops):
        if stop - start >= min_points:
            yield x[start:stop], y[start:stop]


def overlaps_extent(x, y, extent):
    """Return True when a segment bounding box overlaps an extent."""
    x0, x1, y0, y1 = extent
    return (
        np.nanmax(x) >= x0
        and np.nanmin(x) <= x1
        and np.nanmax(y) >= y0
        and np.nanmin(y) <= y1
    )


def _longitude_center(lon):
    lon = np.asarray(lon, dtype=float)
    lon = lon[np.isfinite(lon)]
    if not len(lon):
        return 0.0
    lon_rad = np.deg2rad(lon)
    return np.rad2deg(np.arctan2(np.mean(np.sin(lon_rad)), np.mean(np.cos(lon_rad))))


def _normalize_longitude(lon, center):
    return ((np.asarray(lon, dtype=float) - center + 180.0) % 360.0) + center - 180.0


def geographic_extent(grid, buffer_degrees=2.0):
    """
    Estimate a lon/lat extent around a cubed-sphere grid.

    The longitude interval is unwrapped around the grid's circular mean
    longitude so regional grids near the dateline can still be represented as a
    simple interval.
    """
    try:
        lon = np.asarray(grid.lon, dtype=float)
        lat = np.asarray(grid.lat, dtype=float)
    except AttributeError:
        x0, x1, y0, y1 = grid_extent(grid)
        xi = np.linspace(x0, x1, 25)
        eta = np.linspace(y0, y1, 25)
        bottom = np.column_stack([xi, np.full_like(xi, y0)])
        top = np.column_stack([xi, np.full_like(xi, y1)])
        left = np.column_stack([np.full_like(eta, x0), eta])
        right = np.column_stack([np.full_like(eta, x1), eta])
        boundary = np.vstack([bottom, top, left, right])
        projection = _projection_from_grid(grid)
        lon, lat = projection.cube2geo(boundary[:, 0], boundary[:, 1])

    finite = np.isfinite(lon) & np.isfinite(lat)
    if not finite.any():
        raise ValueError("Could not estimate geographic extent from grid.")

    center = _longitude_center(lon[finite])
    unwrapped_lon = _normalize_longitude(lon[finite], center)
    lat = lat[finite]
    return (
        float(np.nanmin(unwrapped_lon) - buffer_degrees),
        float(np.nanmax(unwrapped_lon) + buffer_degrees),
        float(max(-90.0, np.nanmin(lat) - buffer_degrees)),
        float(min(90.0, np.nanmax(lat) + buffer_degrees)),
        float(center),
    )


def _bbox_overlaps(lon, lat, geo_extent):
    lon_min, lon_max, lat_min, lat_max, center = geo_extent
    lon = _normalize_longitude(lon, center)
    return (
        np.nanmax(lon) >= lon_min
        and np.nanmin(lon) <= lon_max
        and np.nanmax(lat) >= lat_min
        and np.nanmin(lat) <= lat_max
    )


def _points_in_polygon(x, y, polygon_x, polygon_y):
    """Vectorized ray-casting point-in-polygon test."""
    inside = np.zeros(x.shape, dtype=bool)
    x0 = polygon_x[-1]
    y0 = polygon_y[-1]
    for x1, y1 in zip(polygon_x, polygon_y):
        crosses = (y1 > y) != (y0 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersection = (x0 - x1) * (y - y1) / (y0 - y1) + x1
        inside ^= crosses & (x < x_intersection)
        x0 = x1
        y0 = y1
    return inside


def natural_earth_polygon_mask(
    grid_or_projection,
    layer,
    resolution="50m",
    nx=500,
    ny=None,
    geographic_buffer=2.0,
    extent=None,
):
    """
    Return a raster polygon-layer mask on a cubed-sphere grid.

    The mask is evaluated on a regular cubed-sphere ``xi``/``eta`` grid and is
    robust for regional views because no projected polygons are filled.

    Returns
    -------
    xi, eta, mask : numpy.ndarray
        Two-dimensional cubed-sphere coordinates and a boolean land mask.
    """
    projection = _projection_from_grid(grid_or_projection)
    if layer not in {"land", "ocean", "lakes"}:
        raise ValueError(f"Layer {layer!r} is not a polygon mask layer.")
    if extent is None:
        extent = grid_extent(grid_or_projection)
    if ny is None:
        x0, x1, y0, y1 = extent
        aspect = abs((y1 - y0) / (x1 - x0)) if x1 != x0 else 1.0
        ny = max(2, int(round(nx * aspect)))

    x0, x1, y0, y1 = extent
    cache_key = (
        layer,
        id(projection),
        resolution,
        int(nx),
        None if ny is None else int(ny),
        float(geographic_buffer),
        round(float(x0), 12),
        round(float(x1), 12),
        round(float(y0), 12),
        round(float(y1), 12),
    )
    if cache_key in _POLYGON_MASK_CACHE:
        return _POLYGON_MASK_CACHE[cache_key]

    xi_1d = np.linspace(x0, x1, int(nx))
    eta_1d = np.linspace(y0, y1, int(ny))
    xi, eta = np.meshgrid(xi_1d, eta_1d)
    lon, lat = projection.cube2geo(xi, eta)

    geo_extent = geographic_extent(
        grid_or_projection,
        buffer_degrees=geographic_buffer,
    )
    lon_min, lon_max, lat_min, lat_max, center = geo_extent
    test_lon = _normalize_longitude(lon, center)
    valid = (
        np.isfinite(test_lon)
        & np.isfinite(lat)
        & (test_lon >= lon_min)
        & (test_lon <= lon_max)
        & (lat >= lat_min)
        & (lat <= lat_max)
    )

    mask = np.zeros(test_lon.shape, dtype=bool)
    flat_lon = test_lon.ravel()
    flat_lat = lat.ravel()
    flat_valid = valid.ravel()
    flat_mask = mask.ravel()

    for poly_lon, poly_lat, part_type in iter_natural_earth_parts(
        layer,
        resolution=resolution,
        part_types={"polygon_exterior", "polygon_hole"},
    ):
        if not _bbox_overlaps(poly_lon, poly_lat, geo_extent):
            continue
        local_lon = _normalize_longitude(poly_lon, center)
        local_lat = np.asarray(poly_lat, dtype=float)
        local_vertex = (
            np.isfinite(local_lon)
            & np.isfinite(local_lat)
            & (local_lon >= lon_min)
            & (local_lon <= lon_max)
            & (local_lat >= lat_min)
            & (local_lat <= lat_max)
        )
        if not local_vertex.any():
            continue
        poly_lon, poly_lat = _clip_polygon_to_geographic_extent(
            poly_lon,
            poly_lat,
            geo_extent,
        )
        if poly_lon is None:
            continue
        if len(poly_lon) < 3:
            continue

        in_bbox = (
            flat_valid
            & (flat_lon >= np.nanmin(poly_lon))
            & (flat_lon <= np.nanmax(poly_lon))
            & (flat_lat >= np.nanmin(poly_lat))
            & (flat_lat <= np.nanmax(poly_lat))
        )
        if not in_bbox.any():
            continue
        contained = _points_in_polygon(
            flat_lon[in_bbox],
            flat_lat[in_bbox],
            poly_lon,
            poly_lat,
        )
        if part_type == "polygon_hole":
            flat_mask[in_bbox] &= ~contained
        else:
            flat_mask[in_bbox] |= contained

    result = xi, eta, mask
    _POLYGON_MASK_CACHE[cache_key] = result
    return result


def natural_earth_land_mask(
    grid_or_projection,
    resolution="50m",
    nx=500,
    ny=None,
    geographic_buffer=2.0,
    extent=None,
):
    """Return a raster land mask on a cubed-sphere grid."""
    return natural_earth_polygon_mask(
        grid_or_projection,
        "land",
        resolution=resolution,
        nx=nx,
        ny=ny,
        geographic_buffer=geographic_buffer,
        extent=extent,
    )


def _clip_polygon_to_boundary(points, inside, intersect):
    if not len(points):
        return points

    clipped = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                clipped.append(intersect(previous, current))
            clipped.append(current)
        elif previous_inside:
            clipped.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside

    return np.asarray(clipped, dtype=float)


def _clip_polygon_to_geographic_extent(lon, lat, geo_extent):
    lon_min, lon_max, lat_min, lat_max, center = geo_extent
    lon = _normalize_longitude(lon, center)
    points = np.column_stack([lon, lat])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return None, None
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    def x_intersection(edge_x):
        def intersect(start, end):
            dx = end[0] - start[0]
            if np.isclose(dx, 0.0):
                return np.asarray([edge_x, start[1]])
            t = (edge_x - start[0]) / dx
            return start + t * (end - start)

        return intersect

    def y_intersection(edge_y):
        def intersect(start, end):
            dy = end[1] - start[1]
            if np.isclose(dy, 0.0):
                return np.asarray([start[0], edge_y])
            t = (edge_y - start[1]) / dy
            return start + t * (end - start)

        return intersect

    boundaries = (
        (lambda point: point[0] >= lon_min, x_intersection(lon_min)),
        (lambda point: point[0] <= lon_max, x_intersection(lon_max)),
        (lambda point: point[1] >= lat_min, y_intersection(lat_min)),
        (lambda point: point[1] <= lat_max, y_intersection(lat_max)),
    )
    for inside, intersect in boundaries:
        points = _clip_polygon_to_boundary(points, inside, intersect)
        if len(points) < 3:
            return None, None

    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points[:, 0], points[:, 1]


def project_natural_earth(
    grid_or_projection,
    layer,
    resolution="50m",
    extent=None,
    geo_extent=None,
    geographic_buffer=2.0,
    part_types=None,
    min_points=None,
    close_polygons=True,
    require_closed=False,
):
    """
    Project a Natural Earth layer into cubed-sphere coordinates.

    Parameters
    ----------
    grid_or_projection : object
        A cubed-sphere grid with a ``projection`` attribute, or the projection
        itself. The projection must provide ``geo2cube(lon, lat)``.
    layer : str
        Natural Earth layer name, e.g. ``"land"``, ``"ocean"``, ``"lakes"``,
        ``"coastline"``, ``"borders"``, or ``"states_provinces"``.
    resolution : {"110m", "50m", "10m"}, optional
        Bundled Natural Earth resolution.
    extent : tuple, optional
        ``(x0, x1, y0, y1)`` clipping extent used to discard off-grid parts.
        If omitted and a grid is supplied, the grid extent is used.
    geo_extent : tuple or False, optional
        Geographic pre-window as ``(lon_min, lon_max, lat_min, lat_max,
        lon_center)``. If omitted and a grid is supplied, a buffered extent is
        estimated from the grid for polygon layers. Set to False to disable
        geographic pre-windowing.
    geographic_buffer : float, optional
        Degree buffer added around the estimated grid lon/lat extent.
    part_types : iterable, optional
        Restrict parts by stored type, e.g. ``{"polygon_exterior"}``.
    min_points : int, optional
        Minimum projected points per retained segment. Defaults to 3 for
        polygon layers and 2 for line layers.
    close_polygons : bool, optional
        Ensure retained polygon segments end at their first coordinate.
    require_closed : bool, optional
        If True, discard projected polygon segments that are not already closed.
        This avoids artificial fill chords when a polygon ring is broken by the
        cubed-sphere projection.

    Returns
    -------
    list of dict
        Each dict has ``x``, ``y``, and ``part_type`` entries.
    """
    projection = _projection_from_grid(grid_or_projection)
    if extent is None and projection is not grid_or_projection:
        extent = grid_extent(grid_or_projection)

    polygon_layer = layer in {"land", "ocean", "lakes"}
    if min_points is None:
        min_points = 3 if polygon_layer else 2
    if geo_extent is None and polygon_layer and projection is not grid_or_projection:
        geo_extent = geographic_extent(
            grid_or_projection,
            buffer_degrees=geographic_buffer,
        )
    elif geo_extent is False:
        geo_extent = None

    projected = []
    for lon, lat, part_type in iter_natural_earth_parts(
        layer,
        resolution=resolution,
        part_types=part_types,
    ):
        if geo_extent is not None:
            if not _bbox_overlaps(lon, lat, geo_extent):
                continue
            if polygon_layer:
                lon, lat = _clip_polygon_to_geographic_extent(lon, lat, geo_extent)
                if lon is None:
                    continue
        x, y = projection.geo2cube(lon, lat)
        for seg_x, seg_y in finite_segments(x, y, min_points=min_points):
            if extent is not None and not overlaps_extent(seg_x, seg_y, extent):
                continue
            if polygon_layer and close_polygons:
                closed = np.isclose(seg_x[0], seg_x[-1]) and np.isclose(
                    seg_y[0], seg_y[-1]
                )
                if require_closed and not closed:
                    continue
                if not closed:
                    seg_x = np.r_[seg_x, seg_x[0]]
                    seg_y = np.r_[seg_y, seg_y[0]]
            projected.append(
                {
                    "x": np.asarray(seg_x),
                    "y": np.asarray(seg_y),
                    "part_type": part_type,
                }
            )
    return projected


def projected_parts_to_path(parts):
    """Convert projected polygon parts to a Matplotlib ``Path``."""
    from matplotlib.path import Path

    def signed_area(x, y):
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    vertices = []
    codes = []
    for part in parts:
        x = np.asarray(part["x"])
        y = np.asarray(part["y"])
        if len(x) < 3:
            continue
        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

        area = signed_area(x, y)
        is_hole = part.get("part_type") == "polygon_hole"
        if (not is_hole and area > 0) or (is_hole and area < 0):
            x = x[::-1]
            y = y[::-1]

        ring = np.column_stack([x, y])
        vertices.extend(ring)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2))
        codes.append(Path.CLOSEPOLY)

    if not vertices:
        return None
    return Path(vertices, codes)


def save_projected_natural_earth(
    filename,
    grid_or_projection,
    layers=("land", "ocean", "coastline", "lakes", "borders"),
    resolution="50m",
    extent=None,
    geographic_buffer=2.0,
):
    """
    Save projected Natural Earth layers as a compact NPZ cache.

    The cache stores one flattened coordinate array plus offsets and part types
    per layer, mirroring the source asset format but in cubed-sphere ``x, y``.
    """
    filename = Path(filename)
    output = {}
    for layer in layers:
        parts = project_natural_earth(
            grid_or_projection,
            layer,
            resolution=resolution,
            extent=extent,
            geographic_buffer=geographic_buffer,
        )
        coords = []
        offsets = [0]
        part_types = []
        for part in parts:
            xy = np.column_stack([part["x"], part["y"]])
            coords.append(xy)
            offsets.append(offsets[-1] + len(xy))
            part_types.append(part["part_type"])

        output[f"{layer}_coords"] = (
            np.vstack(coords) if coords else np.empty((0, 2), dtype=float)
        )
        output[f"{layer}_offsets"] = np.asarray(offsets, dtype=int)
        output[f"{layer}_part_types"] = np.asarray(part_types, dtype=str)

    output["resolution"] = np.asarray(resolution)
    np.savez_compressed(filename, **output)
    return filename
