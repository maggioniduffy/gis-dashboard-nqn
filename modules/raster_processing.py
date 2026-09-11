"""
Raster -> classified vector pipeline for climate rasters (CHIRPS precipitation).

Opens a single-band GeoTIFF with Rasterio, classifies its values into bins
(quantile or Jenks natural breaks), vectorizes each bin region with
`rasterio.features.shapes`, and returns a GeoDataFrame in EPSG:4326 ready for
both the Folium 2D layer (`folium_layers.py`) and the PyDeck 3D panel
(`pydeck_layers.py`) — neither of which needs to know anything about rasters.
"""

import logging
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows
import streamlit as st
from rasterio.features import geometry_mask, shapes as rasterio_shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, shape

logger = logging.getLogger(__name__)

DISPLAY_CRS = "EPSG:4326"
GDF_COLUMNS = ["geometry", "value", "class"]

# Metric CRS used to compare/resample CHIRPS and ERA5-Land pixel sizes in
# meters (same CRS the hydrology pipeline uses for its own raster math), so
# "which grid is finer" is a plain number comparison instead of comparing
# degrees across two rasters that may not share a datum-consistent pixel
# size (CHIRPS ~0.05°, ERA5-Land ~0.1°, but degrees aren't square meters).
ALIGNMENT_CRS = "EPSG:5343"
COMBINED_GRID_COLUMNS = ["geometry", "precip_value", "temp_value"]


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges splitting `values` into `n_bins` equal-count groups."""
    quantile_positions = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, quantile_positions))
    if len(edges) < 2:
        # All values identical (or nearly so): fabricate a minimal valid range.
        edges = np.array([values.min(), values.max() + 1e-9])
    return edges


def _jenks_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Bin edges via Jenks natural breaks (Fisher-Jenks), using `mapclassify`
    when it's installed. Jenks minimizes within-class variance, which
    typically produces more visually meaningful classes for skewed
    precipitation data than equal-count quantiles.

    Natural-breaks optimization is O(n^2)-ish, so it's run on a capped random
    subsample for large rasters rather than every pixel; the resulting edges
    are then applied to the full raster in `_compute_bin_edges`'s caller.

    Falls back to quantile edges (with a warning) if `mapclassify` isn't
    installed, so a missing optional dependency never breaks the pipeline.
    """
    sample = values
    if sample.size > 20000:
        sample = np.random.default_rng(0).choice(sample, size=20000, replace=False)

    try:
        import mapclassify

        classifier = mapclassify.NaturalBreaks(sample, k=n_bins)
        edges = np.concatenate(([float(values.min())], classifier.bins))
        edges[-1] = float(values.max())
        return np.unique(edges)
    except ImportError:
        logger.warning(
            "mapclassify no está instalado; usando clasificación por cuantiles "
            "en lugar de Jenks natural breaks. Agregue 'mapclassify' a "
            "requirements.txt para habilitar Jenks."
        )
        return _quantile_edges(values, n_bins)


def _compute_bin_edges(
    values: np.ndarray, n_bins: int, method: Literal["quantile", "jenks"]
) -> np.ndarray:
    if method == "quantile":
        return _quantile_edges(values, n_bins)
    elif method == "jenks":
        return _jenks_edges(values, n_bins)
    raise ValueError(f"Unsupported classification method: {method!r} (use 'quantile' or 'jenks')")


def _zonal_mean_per_feature(
    band: np.ndarray,
    valid_mask: np.ndarray,
    transform: rasterio.Affine,
    geometries,
) -> np.ndarray:
    """
    For each polygon in `geometries` (one per vectorized bin-region), computes
    the mean of the ORIGINAL raster values (`band`) covered by that polygon —
    not the bin midpoint, so `value` reflects the actual underlying data.

    Each polygon is rasterized back onto a small window cropped to its own
    bounding box (via `geometry_mask`), rather than over the whole raster, so
    this stays cheap even for rasters with many vectorized regions.
    """
    height, width = band.shape
    means = np.full(len(geometries), np.nan, dtype="float64")

    for i, geom in enumerate(geometries):
        minx, miny, maxx, maxy = geom.bounds
        row_start, col_start = rasterio.transform.rowcol(transform, minx, maxy)
        row_stop, col_stop = rasterio.transform.rowcol(transform, maxx, miny)
        row_start, row_stop = sorted((max(row_start, 0), min(row_stop + 1, height)))
        col_start, col_stop = sorted((max(col_start, 0), min(col_stop + 1, width)))
        if row_start >= row_stop or col_start >= col_stop:
            continue

        window = rasterio.windows.Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
        window_transform = rasterio.windows.transform(window, transform)
        window_band = band[row_start:row_stop, col_start:col_stop]
        window_valid = valid_mask[row_start:row_stop, col_start:col_stop]

        feature_mask = geometry_mask(
            [geom], out_shape=window_band.shape, transform=window_transform, invert=True
        )
        combined_mask = feature_mask & window_valid
        if combined_mask.any():
            means[i] = window_band[combined_mask].mean()

    return means


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"value": [], "class": []}, geometry=gpd.GeoSeries([], crs=DISPLAY_CRS), crs=DISPLAY_CRS
    )[GDF_COLUMNS]


@st.cache_data(show_spinner=False)
def raster_to_classified_gdf(
    raster_path: str,
    n_bins: int = 8,
    method: Literal["quantile", "jenks"] = "quantile",
) -> gpd.GeoDataFrame:
    """
    Opens `raster_path` with Rasterio, classifies its single band into
    `n_bins` bins (quantile or Jenks natural breaks), vectorizes the
    classified array with `rasterio.features.shapes`, and returns a
    GeoDataFrame in EPSG:4326 with columns:
      - geometry: polygon of each vectorized bin region
      - value:    mean of the ORIGINAL raster values covered by that polygon
      - class:    bin index (0-based) that region was assigned to

    Cached with `st.cache_data`. The cache key is built from this function's
    arguments (`raster_path`, `n_bins`, `method`) — since `raster_path`
    already encodes the date range/aggregation via
    `ee_client.get_cache_path()`, switching climate layers or reprocessing
    parameters correctly busts the cache. The Rasterio dataset itself is
    never part of the key (it's opened and closed inside this function) —
    only the resulting GeoDataFrame is cached.
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs

    valid_mask = np.isfinite(band)
    if nodata is not None and np.isfinite(nodata):
        valid_mask &= band != nodata

    if not valid_mask.any():
        logger.warning("Raster '%s' has no valid pixels; returning empty GeoDataFrame.", raster_path)
        return _empty_gdf()

    edges = _compute_bin_edges(band[valid_mask], n_bins, method)

    # np.digitize against the interior edges only (edges[1:-1]): pixels below
    # the first edge get class 0, up to class len(edges)-2 for the top bin.
    class_idx = np.digitize(band, edges[1:-1], right=True).astype("int32")
    class_idx = np.where(valid_mask, class_idx, -1)

    records = []
    for geom_json, class_value in rasterio_shapes(class_idx, mask=valid_mask, transform=transform):
        records.append({"geometry": shape(geom_json), "class": int(class_value)})

    if not records:
        return _empty_gdf()

    gdf = gpd.GeoDataFrame(records, crs=src_crs)
    gdf["value"] = _zonal_mean_per_feature(band, valid_mask, transform, gdf.geometry)
    gdf = gdf.dropna(subset=["value"])
    gdf = gdf.to_crs(DISPLAY_CRS)

    return gdf[GDF_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-source grid alignment (CHIRPS precipitation + ERA5-Land temperature)
# ---------------------------------------------------------------------------


def _empty_combined_grid() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"precip_value": [], "temp_value": []},
        geometry=gpd.GeoSeries([], crs=DISPLAY_CRS),
        crs=DISPLAY_CRS,
    )[COMBINED_GRID_COLUMNS]


def _reproject_band(
    raster_path: str, dst_crs: str
) -> tuple[np.ndarray, rasterio.Affine, int, int, float | None]:
    """
    Reads band 1 of `raster_path` and reprojects it to `dst_crs`, returning
    (band, transform, width, height, nodata). Only the CRS changes here — the
    pixel size in `dst_crs` units still reflects the raster's own native
    resolution, so two rasters at different native resolutions still end up
    with different pixel sizes after this step; matching them is
    `_resample_to_grid`'s job.
    """
    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        nodata = src.nodata if src.nodata is not None else np.nan
        destination = np.full((height, width), nodata, dtype="float64")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
    return destination, transform, width, height, nodata


def _resample_to_grid(
    band: np.ndarray,
    transform: rasterio.Affine,
    nodata: float | None,
    crs: str,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    """Resamples `band` (already in `crs`) onto the exact grid described by (dst_transform, dst_width, dst_height)."""
    dst_nodata = nodata if nodata is not None else np.nan
    destination = np.full((dst_height, dst_width), dst_nodata, dtype="float64")
    reproject(
        source=band,
        destination=destination,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        src_nodata=nodata,
        dst_nodata=dst_nodata,
        resampling=Resampling.bilinear,
    )
    return destination


@st.cache_data(show_spinner=False)
def align_climate_grids(
    precip_raster_path: str,
    temp_raster_path: str,
    alignment_crs: str = ALIGNMENT_CRS,
) -> gpd.GeoDataFrame:
    """
    Aligns a CHIRPS precipitation raster and an ERA5-Land temperature raster
    — downloaded independently by `ee_client.fetch_and_cache_chirps` /
    `fetch_and_cache_era5_temperature` at their own native resolutions
    (~5.5km and ~9-11km respectively) — onto a single common grid, cell by
    cell, so the two variables can be cross-referenced in the combined 3D
    PyDeck panel (height = one variable, color = the other).

    This operates on the RAW rasters, not on the classified GeoDataFrames
    `raster_to_classified_gdf` produces: that function merges runs of
    adjacent same-class pixels into irregular polygons, which is exactly
    what a single-variable choropleth/3D-columns view wants, but there is no
    well-defined "cell-by-cell join" between two such irregular polygon sets
    coming from rasters with different native resolutions. Resampling before
    vectorizing is the only way to get an actual regular grid where each row
    is one physical cell shared by both variables.

    Steps:
      1. Both rasters are reprojected to `alignment_crs` (EPSG:5343, a metric
         CRS), so pixel size becomes directly comparable in meters instead of
         degrees (CHIRPS's ~0.05° and ERA5-Land's ~0.1° aren't square meters
         and aren't even at the same latitude-dependent scale).
      2. Whichever of the two has the SMALLER pixel size after step 1 (in
         practice CHIRPS, ~5.5km vs. ERA5-Land's ~9-11km) is kept as-is and
         used as the target grid — resampling the finer dataset down to the
         coarser one would throw away real resolution for no benefit, while
         resampling the coarser dataset up to the finer one is the standard
         "disaggregate" direction for a continuous field like temperature.
      3. The other raster is resampled directly onto that exact grid
         (same transform/width/height) via `rasterio.warp.reproject` with
         `Resampling.bilinear`, appropriate for a smooth, continuous
         variable like temperature (as opposed to nearest-neighbor, which
         would produce blocky ERA5-Land artifacts at CHIRPS's resolution).
      4. One square polygon per shared cell (centered on the cell, sized to
         the target grid's pixel size) is built directly from the aligned
         arrays — no classification/merging step, since every cell is kept
         individually so height and color can vary independently per cell.
      5. The result is reprojected back to EPSG:4326 (`DISPLAY_CRS`), the CRS
         every other GeoDataFrame in this module and `pydeck_layers.py` /
         `folium_layers.py` already expects.

    Cells where EITHER source is nodata/NaN are dropped, since a combined
    height+color panel needs both values for a cell to mean anything.

    Cached with `st.cache_data`; the cache key is this function's arguments.
    `precip_raster_path`/`temp_raster_path` already encode their own date
    range and aggregation (see `ee_client.get_cache_path` /
    `get_era5_cache_path`), so requesting a different date range or
    aggregation for either source — or switching `alignment_crs` — correctly
    produces a new cache entry instead of reusing a stale alignment.

    Returns a GeoDataFrame in EPSG:4326 with columns:
      - geometry:     square polygon of each shared grid cell
      - precip_value: CHIRPS value for that cell (mm)
      - temp_value:   ERA5-Land value for that cell (°C)
    """
    precip_band, precip_transform, precip_width, precip_height, precip_nodata = _reproject_band(
        precip_raster_path, alignment_crs
    )
    temp_band, temp_transform, temp_width, temp_height, temp_nodata = _reproject_band(
        temp_raster_path, alignment_crs
    )

    # Pixel size in meters along the x axis (transform.a); the finer grid
    # (smaller absolute pixel size) is kept as the target, the coarser one
    # gets resampled onto it.
    precip_pixel_size = abs(precip_transform.a)
    temp_pixel_size = abs(temp_transform.a)

    if precip_pixel_size <= temp_pixel_size:
        target_transform, target_width, target_height = precip_transform, precip_width, precip_height
        target_precip_band = precip_band
        target_temp_band = _resample_to_grid(
            temp_band, temp_transform, temp_nodata, alignment_crs,
            target_transform, target_width, target_height,
        )
        logger.info(
            "align_climate_grids: using CHIRPS grid (%.0fm) as target; resampling "
            "ERA5-Land (%.0fm) onto it.", precip_pixel_size, temp_pixel_size,
        )
    else:
        target_transform, target_width, target_height = temp_transform, temp_width, temp_height
        target_temp_band = temp_band
        target_precip_band = _resample_to_grid(
            precip_band, precip_transform, precip_nodata, alignment_crs,
            target_transform, target_width, target_height,
        )
        logger.info(
            "align_climate_grids: using ERA5-Land grid (%.0fm) as target; resampling "
            "CHIRPS (%.0fm) onto it.", temp_pixel_size, precip_pixel_size,
        )

    precip_valid = np.isfinite(target_precip_band)
    temp_valid = np.isfinite(target_temp_band)
    if precip_nodata is not None and np.isfinite(precip_nodata):
        precip_valid &= target_precip_band != precip_nodata
    if temp_nodata is not None and np.isfinite(temp_nodata):
        temp_valid &= target_temp_band != temp_nodata
    valid_mask = precip_valid & temp_valid

    if not valid_mask.any():
        logger.warning(
            "align_climate_grids: no cells with valid data in both '%s' and '%s'; "
            "returning empty GeoDataFrame.", precip_raster_path, temp_raster_path,
        )
        return _empty_combined_grid()

    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(target_transform, rows, cols, offset="center")
    half_width = abs(target_transform.a) / 2.0
    half_height = abs(target_transform.e) / 2.0
    geometries = [
        box(x - half_width, y - half_height, x + half_width, y + half_height)
        for x, y in zip(xs, ys)
    ]

    combined = gpd.GeoDataFrame(
        {
            "precip_value": target_precip_band[valid_mask],
            "temp_value": target_temp_band[valid_mask],
            "geometry": geometries,
        },
        crs=alignment_crs,
    )
    combined = combined.to_crs(DISPLAY_CRS)

    return combined[COMBINED_GRID_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flow accumulation (Pysheds, per-polygon) -> climate grid alignment
# ---------------------------------------------------------------------------


def _zonal_stat_per_feature(
    band: np.ndarray,
    transform: rasterio.Affine,
    geometries,
    stat: Literal["max", "mean"],
) -> np.ndarray:
    """
    For each polygon in `geometries`, aggregates the finite (non-NaN) values
    of `band` it covers with `stat`. Same cropped-window approach as
    `_zonal_mean_per_feature` (cheap even for many climate cells), but
    parameterized on the aggregate so flow accumulation can use "max" —
    `align_flow_accumulation_to_climate_grid`'s default — while still sharing
    the windowing logic in spirit. Kept separate from `_zonal_mean_per_feature`
    rather than generalizing it in place, so that function's existing
    (cached) callers are untouched.

    A cell with no covered finite pixels gets NaN, which the caller then
    drops — a climate cell outside the flow accumulation raster's footprint
    (i.e. outside the user-drawn polygon) simply has no flow value, rather
    than a fabricated zero.
    """
    height, width = band.shape
    aggregate = np.nanmax if stat == "max" else np.nanmean
    results = np.full(len(geometries), np.nan, dtype="float64")

    for i, geom in enumerate(geometries):
        minx, miny, maxx, maxy = geom.bounds
        top_row, left_col = rasterio.transform.rowcol(transform, minx, maxy)
        bottom_row, right_col = rasterio.transform.rowcol(transform, maxx, miny)

        # Order the two corners first, THEN clamp to the raster. Unlike
        # `_zonal_mean_per_feature`, whose geometries always come from the very
        # raster it samples, the climate cells here can sit entirely outside
        # the flow accumulation crop (anything beyond the drawn polygon);
        # clamping before ordering turns those into bogus windows instead of
        # skipping them.
        row_start = max(0, min(top_row, bottom_row))
        row_stop = min(height, max(top_row, bottom_row) + 1)
        col_start = max(0, min(left_col, right_col))
        col_stop = min(width, max(left_col, right_col) + 1)
        if row_start >= row_stop or col_start >= col_stop:
            continue

        window = rasterio.windows.Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
        window_transform = rasterio.windows.transform(window, transform)
        window_band = band[row_start:row_stop, col_start:col_stop]

        feature_mask = geometry_mask(
            [geom], out_shape=window_band.shape, transform=window_transform, invert=True
        )
        covered = window_band[feature_mask]
        covered = covered[np.isfinite(covered)]
        if covered.size:
            results[i] = aggregate(covered)

    return results


def align_flow_accumulation_to_climate_grid(
    flow_accum_array: np.ndarray,
    dem_transform: rasterio.Affine,
    dem_crs,
    climate_gdf: gpd.GeoDataFrame,
    aggregation: Literal["max", "mean"] = "max",
    value_column: str = "flow_value",
) -> gpd.GeoDataFrame:
    """
    Aggregates a Pysheds flow accumulation raster (fine, DEM resolution —
    `hydrology.compute_runoff_overlay`'s on-the-fly crop for ONE user-drawn
    polygon) onto `climate_gdf`'s existing grid (coarser, CHIRPS resolution),
    adding it as a new `value_column` so it can drive the 3D panel's column
    height/color the same way `precip_value`/`temp_value` already do.

    This does NOT re-run Pysheds or re-vectorize/re-classify any climate
    raster: `flow_accum_array`/`dem_transform`/`dem_crs` are expected to come
    straight from `hydrology.compute_runoff_overlay`'s result dict
    ("flow_accumulation"/"flow_transform"/"flow_crs"), and `climate_gdf` from
    an already-computed `raster_to_classified_gdf`/`align_climate_grids` call
    — this function is purely the join/resample between two results that
    already exist.

    `aggregation="max"` (the default) takes, for each climate cell, the
    highest flow accumulation among the DEM pixels it covers — appropriate
    for a RISK indicator, where a single high-accumulation stream cell
    crossing a coarse climate cell matters more than diluting it into an
    average. Pass `aggregation="mean"` for a smoother, less spike-sensitive
    view instead.

    Because `flow_accum_array` only covers the user-drawn polygon's bounding
    box (Pysheds runs on-the-fly per polygon, never province-wide), cells
    outside that footprint get NaN in `value_column` — they are KEPT, not
    dropped. That matters: this is the same grid the 3D panel uses for its
    precipitation x temperature cross, so dropping rows here would silently
    shrink that cross down to the drawn polygon. Restricting to the drawn
    area is the *renderer's* job instead, and it happens naturally —
    `pydeck_layers.build_pydeck_layer` drops rows whose driving column is
    null, so selecting the "flow" variable shows only the polygon's cells
    while "precip"/"temp" still span the whole province.

    NaN (rather than 0) is what marks "no flow data here", so a cell with
    genuinely zero accumulation stays distinguishable from one the polygon
    never covered.

    Returns a copy of `climate_gdf`, in `climate_gdf`'s original CRS, with one
    added `value_column` (NaN outside the flow accumulation footprint).
    Returns `climate_gdf` unchanged if it's empty/None.
    """
    if climate_gdf is None or climate_gdf.empty:
        return climate_gdf

    original_crs = climate_gdf.crs
    # Reproject the climate cells into the DEM's own CRS (rather than
    # resampling the accumulation array into the climate grid's CRS): the
    # accumulation array is irregular-footprint and already small (one
    # polygon's bounding box), so it's cheaper and avoids a second resampling
    # step to instead evaluate each climate polygon directly against it.
    climate_projected = climate_gdf.to_crs(dem_crs)

    aggregated = _zonal_stat_per_feature(
        flow_accum_array, dem_transform, climate_projected.geometry, aggregation
    )

    climate_projected = climate_projected.copy()
    climate_projected[value_column] = aggregated

    return climate_projected.to_crs(original_crs)
