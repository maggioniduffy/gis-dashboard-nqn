"""
On-the-fly runoff analysis for a user-drawn polygon.

Crops the provincial DEM (a single continuous GeoTIFF, see
`ee_client.PROVINCIAL_DEM_PATH`) to the polygon and runs Pysheds D8 routing
and flow accumulation over it. `rasterio.mask.mask(crop=True)` only reads the
window covering the polygon's bounding box, so the size of the provincial
raster on disk does not affect the cost of a crop.
"""

import os

import geopandas as gpd
import numpy as np
import rasterio
from pysheds.grid import Grid
from pysheds.sview import Raster, ViewFinder
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import box, shape

# Pysheds keeps several full-size float64 copies of the crop in memory; this
# caps a single analysis at ~22,500 km² on a 30m DEM (~3GB peak).
MAX_CROP_CELLS = 25_000_000
HIGH_RUNOFF_PERCENTILE = 95

# Metric CRS used for the Rational Method's area/intersection math (same CRS
# `raster_processing.ALIGNMENT_CRS` uses to compare CHIRPS/ERA5-Land pixel
# sizes), so polygon area and precipitation-cell overlap are both computed in
# real square meters instead of degrees.
RATIONAL_METHOD_CRS = "EPSG:5343"

# Default runoff coefficient (C) for the Rational Method when the caller
# doesn't supply `land_use_coefficient`. 0.3 is a commonly cited value for
# bare/natural soil and steppe-type vegetation with flat-to-moderate slope in
# classic runoff-coefficient reference tables (e.g. Chow, Maidment & Mays,
# "Applied Hydrology", Table 15.1.2; also close to the Témez/ASCE tables used
# in Spanish-language hydrology references). It is a coarse, single-value
# simplification standing in for a real land-use/soil-type classification —
# pass `land_use_coefficient` explicitly whenever one is available (e.g. from
# a land-cover raster) instead of relying on this default.
DEFAULT_RUNOFF_COEFFICIENT = 0.3

# The Rational Method (Q = C * I * A) is a classic engineering approximation
# for PEAK flow in small basins, where the assumption of a spatially/temporally
# uniform rainfall intensity across the whole time of concentration is
# reasonable. Textbooks typically cap its applicability at ~2.5-3 km²; above
# that, distributed/hydrograph methods are preferred. This module still
# computes Q above that threshold (drawing an arbitrary polygon shouldn't
# hard-fail), but flags it via `"method_valid_for_basin_size"` in the result
# so the UI can warn the user instead of presenting the number as reliable.
RATIONAL_METHOD_MAX_AREA_KM2 = 2.5


class DemCoverageError(ValueError):
    """The drawn polygon can't be analyzed with the available DEM (message is user-facing)."""


class PrecipCoverageError(ValueError):
    """The drawn polygon doesn't overlap any precipitation cell (message is user-facing)."""


def get_dem_bounds_wgs84(dem_path: str) -> tuple | None:
    """(west, south, east, north) of the DEM in EPSG:4326, or None if it isn't available."""
    if not os.path.exists(dem_path):
        return None
    with rasterio.open(dem_path) as dem:
        return transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)


def downsample_runoff_mask(mask: np.ndarray, max_dimension: int = 768) -> np.ndarray:
    """Reduce una máscara booleana preservando los píxeles positivos de cada bloque."""
    row_factor = max(1, int(np.ceil(mask.shape[0] / max_dimension)))
    col_factor = max(1, int(np.ceil(mask.shape[1] / max_dimension)))
    padded_rows = int(np.ceil(mask.shape[0] / row_factor) * row_factor)
    padded_cols = int(np.ceil(mask.shape[1] / col_factor) * col_factor)
    padded_mask = np.pad(
        mask,
        ((0, padded_rows - mask.shape[0]), (0, padded_cols - mask.shape[1])),
        constant_values=False,
    )
    return padded_mask.reshape(
        padded_rows // row_factor,
        row_factor,
        padded_cols // col_factor,
        col_factor,
    ).any(axis=(1, 3))


def _crop_dem(dem_path: str, geometry: dict, max_cells: int):
    """
    Validates the polygon against the DEM extent and returns the masked crop.
    Raises `DemCoverageError` instead of Rasterio's cryptic errors.
    """
    if not os.path.exists(dem_path):
        raise DemCoverageError(
            "No hay un DEM provincial disponible. Use el botón «Actualizar DEM "
            "provincial» del panel lateral para descargarlo."
        )

    with rasterio.open(dem_path) as dem:
        dem_geometry = transform_geom("EPSG:4326", dem.crs, geometry)
        polygon = shape(dem_geometry)
        dem_extent = box(*dem.bounds)
        if not polygon.intersects(dem_extent):
            raise DemCoverageError(
                "El polígono dibujado está fuera de la cobertura del DEM provincial "
                "(rectángulo amarillo). Dibuje un área dentro de la provincia de Neuquén."
            )

        covered = polygon.intersection(dem_extent)
        window = from_bounds(*covered.bounds, transform=dem.transform)
        n_cells = int(np.ceil(window.width) * np.ceil(window.height))
        if n_cells > max_cells:
            cell_km2 = abs(dem.res[0] * dem.res[1]) / 1e6
            raise DemCoverageError(
                f"El área dibujada es demasiado grande (~{n_cells * cell_km2:,.0f} km²). "
                f"Dibuje un área menor a ~{max_cells * cell_km2:,.0f} km²."
            )

        cropped_dem, cropped_transform = mask(dem, [dem_geometry], crop=True, filled=False)
        partial_coverage = not dem_extent.contains(polygon)
        return cropped_dem, cropped_transform, dem.crs, partial_coverage


def compute_runoff_overlay(
    geometry: dict,
    dem_path: str,
    max_cells: int = MAX_CROP_CELLS,
) -> dict:
    """
    Runs the D8 runoff analysis for `geometry` (GeoJSON in EPSG:4326) and
    returns an RGBA overlay of high-accumulation cells ready for Folium, plus
    the raw flow accumulation grid so other modules (e.g. the 3D panel's
    `raster_processing.align_flow_accumulation_to_climate_grid`) can reuse
    this same Pysheds run instead of recomputing it:
      {"overlay": ndarray, "bounds": [[south, west], [north, east]],
       "partial_coverage": bool, "flow_accumulation": ndarray,
       "flow_transform": Affine, "flow_crs": CRS}
    """
    cropped_dem, cropped_transform, dem_crs, partial_coverage = _crop_dem(
        dem_path, geometry, max_cells
    )

    dem_data = cropped_dem[0].astype(np.float64)
    valid_cells = ~np.ma.getmaskarray(dem_data)
    if not valid_cells.any():
        raise DemCoverageError(
            "El área dibujada no contiene datos de elevación (cae fuera del límite "
            "provincial o sobre vacíos del DEM)."
        )
    dem_values = np.where(valid_cells, dem_data.filled(np.nan), np.nan)

    viewfinder = ViewFinder(
        affine=cropped_transform,
        shape=dem_values.shape,
        nodata=np.nan,
        mask=valid_cells,
        crs=dem_crs,
    )
    grid = Grid(viewfinder=viewfinder)
    dem_raster = Raster(dem_values, viewfinder=viewfinder)

    pit_filled_dem = grid.fill_pits(dem_raster)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    conditioned_dem = grid.resolve_flats(flooded_dem)
    flow_direction = grid.flowdir(conditioned_dem, routing="d8")
    flow_accumulation = grid.accumulation(flow_direction, routing="d8")

    accumulation_values = np.asarray(flow_accumulation)
    high_threshold = np.nanpercentile(
        accumulation_values[valid_cells], HIGH_RUNOFF_PERCENTILE
    )
    high_runoff_mask = valid_cells & (accumulation_values >= high_threshold)

    # Masked (NaN outside the DEM crop's valid cells) copy of the raw
    # accumulation grid, exported alongside the display overlay so a caller
    # can align it onto a coarser climate grid later without re-running
    # Pysheds — `align_flow_accumulation_to_climate_grid` treats NaN as "no
    # data here" the same way `raster_to_classified_gdf` does for CHIRPS.
    flow_accumulation_export = np.where(valid_cells, accumulation_values, np.nan)

    display_mask = downsample_runoff_mask(high_runoff_mask)
    runoff_overlay = np.zeros((*display_mask.shape, 4), dtype=np.uint8)
    runoff_overlay[display_mask] = [0, 220, 255, 210]

    west, south, east, north = array_bounds(
        high_runoff_mask.shape[0],
        high_runoff_mask.shape[1],
        cropped_transform,
    )
    overlay_bounds = transform_bounds(dem_crs, "EPSG:4326", west, south, east, north)

    return {
        "overlay": runoff_overlay,
        "bounds": [
            [overlay_bounds[1], overlay_bounds[0]],
            [overlay_bounds[3], overlay_bounds[2]],
        ],
        "partial_coverage": partial_coverage,
        "flow_accumulation": flow_accumulation_export,
        "flow_transform": cropped_transform,
        "flow_crs": dem_crs,
    }


# ---------------------------------------------------------------------------
# Rational Method peak flow (Q = C * I * A), driven by real CHIRPS precipitation
# ---------------------------------------------------------------------------

# Candidate column names to look for the precipitation value in `precip_gdf`
# when `precip_value_column` isn't given explicitly: "value" matches
# `raster_processing.raster_to_classified_gdf`'s output (a plain classified
# CHIRPS layer), "precip_value" matches `raster_processing.align_climate_grids`'s
# output (the CHIRPS+ERA5-Land combined grid).
_PRECIP_VALUE_COLUMN_CANDIDATES = ("value", "precip_value")


def _resolve_precip_value_column(precip_gdf: gpd.GeoDataFrame, precip_value_column: str | None) -> str:
    if precip_value_column is not None:
        if precip_value_column not in precip_gdf.columns:
            raise ValueError(f"precip_gdf has no column named '{precip_value_column}'.")
        return precip_value_column

    for candidate in _PRECIP_VALUE_COLUMN_CANDIDATES:
        if candidate in precip_gdf.columns:
            return candidate

    raise ValueError(
        "Could not find a precipitation value column in precip_gdf "
        f"(looked for {_PRECIP_VALUE_COLUMN_CANDIDATES!r}); pass `precip_value_column` explicitly."
    )


def calculate_peak_flow(
    geometry: dict,
    precip_gdf: gpd.GeoDataFrame,
    land_use_coefficient: float | None = None,
    precip_value_column: str | None = None,
) -> dict:
    """
    Estimates peak runoff for the user-drawn `geometry` with the Rational
    Method (Q = C * I * A / 360, with A in hectares and I in mm/h, giving Q in
    m^3/s), using real CHIRPS precipitation instead of a topography-only
    proxy.

    This is a classic, coarse engineering approximation intended for SMALL
    basins (textbooks generally cap it around 2.5-3 km^2), where rainfall
    intensity can reasonably be treated as uniform over the whole basin and
    over the time of concentration. It runs entirely independently of the
    Pysheds D8 routing / flow accumulation pipeline in this module — it does
    not replace that analysis, only complements it with a precipitation-aware
    number.

    Important simplification: a rigorous application of the Rational Method
    derives I from an IDF (intensity-duration-frequency) curve evaluated at
    the sub-basin's time of concentration, for a chosen return period. This
    function does not have IDF curves available, so it instead uses the
    area-weighted CHIRPS value from `precip_gdf` directly as a stand-in for
    "I" (mm/h). Depending on how `precip_gdf` was aggregated upstream (see
    `ee_client.fetch_and_cache_chirps`'s `aggregation` parameter), this is a
    rough order-of-magnitude proxy rather than a true design-storm intensity —
    treat the resulting Q as indicative, not a design value.

    Args:
        geometry: GeoJSON geometry dict in EPSG:4326 (WGS84), same format as
            `compute_runoff_overlay`'s `geometry` argument (e.g. straight from
            a Folium `Draw` result).
        precip_gdf: GeoDataFrame of precipitation cells/regions, as returned
            by `raster_processing.raster_to_classified_gdf` (column "value")
            or `raster_processing.align_climate_grids` (column
            "precip_value"). Any CRS is accepted; it's reprojected internally.
        land_use_coefficient: Runoff coefficient C (0-1). If omitted, uses
            `DEFAULT_RUNOFF_COEFFICIENT` (0.3), a generic value for natural/
            steppe soil — see that constant's docstring comment for its
            source and caveats. Pass an explicit value whenever a real
            land-use/soil classification is available.
        precip_value_column: Name of the precipitation column in `precip_gdf`.
            Auto-detected ("value" then "precip_value") when omitted.

    Raises:
        PrecipCoverageError: `geometry` doesn't intersect any cell of
            `precip_gdf` (polygon drawn outside CHIRPS coverage, or an empty
            `precip_gdf`). This is raised separately from `DemCoverageError`
            so callers can let the Pysheds topography-only analysis keep
            working even when this precipitation-based estimate can't run.

    Returns:
        dict with:
          - "Q_m3_s": peak flow estimate (m^3/s)
          - "C": runoff coefficient used
          - "C_is_default": whether C came from DEFAULT_RUNOFF_COEFFICIENT
          - "I_mm_h": area-weighted precipitation intensity used (mm/h)
          - "A_km2" / "A_ha": basin area in km^2 / hectares
          - "method_valid_for_basin_size": False if A_km2 exceeds
            RATIONAL_METHOD_MAX_AREA_KM2 (the UI should warn, not hide, the
            number in that case)
    """
    if precip_gdf is None or precip_gdf.empty:
        raise PrecipCoverageError(
            "No hay datos de precipitación (CHIRPS) disponibles para calcular el "
            "caudal pico con el Método Racional."
        )

    value_column = _resolve_precip_value_column(precip_gdf, precip_value_column)

    # Reproject both the polygon and the precipitation cells to a metric CRS
    # so area and intersection-area math is in real square meters, not degrees.
    polygon_wgs84 = shape(geometry)
    polygon_projected = (
        gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326").to_crs(RATIONAL_METHOD_CRS).iloc[0]
    )
    precip_crs = precip_gdf.crs if precip_gdf.crs is not None else "EPSG:4326"
    precip_projected = gpd.GeoDataFrame(
        precip_gdf[[value_column, "geometry"]], geometry="geometry", crs=precip_crs
    ).to_crs(RATIONAL_METHOD_CRS)

    area_m2 = polygon_projected.area
    area_ha = area_m2 / 10_000
    area_km2 = area_m2 / 1_000_000

    # Only cells that actually overlap the polygon contribute, weighted by how
    # much of each cell's area falls inside it (not just whichever cell the
    # centroid lands in), so a polygon straddling several CHIRPS cells gets a
    # proper area-weighted average instead of a single sampled value.
    candidates = precip_projected[precip_projected.intersects(polygon_projected)]
    if candidates.empty:
        raise PrecipCoverageError(
            "El polígono dibujado está fuera de la cobertura de precipitación "
            "(CHIRPS). No se puede estimar el caudal pico con el Método Racional "
            "para esta área (el análisis de topografía puede seguir funcionando)."
        )

    intersection_areas = candidates.geometry.intersection(polygon_projected).area
    covered_area = intersection_areas.sum()
    if covered_area <= 0:
        raise PrecipCoverageError(
            "El polígono dibujado está fuera de la cobertura de precipitación "
            "(CHIRPS). No se puede estimar el caudal pico con el Método Racional "
            "para esta área (el análisis de topografía puede seguir funcionando)."
        )

    intensity_mm_h = float((candidates[value_column] * intersection_areas).sum() / covered_area)

    runoff_coefficient = (
        land_use_coefficient if land_use_coefficient is not None else DEFAULT_RUNOFF_COEFFICIENT
    )

    # Metric Rational Method: Q [m^3/s] = C * I [mm/h] * A [ha] / 360.
    peak_flow_m3_s = runoff_coefficient * intensity_mm_h * area_ha / 360

    return {
        "Q_m3_s": peak_flow_m3_s,
        "C": runoff_coefficient,
        "C_is_default": land_use_coefficient is None,
        "I_mm_h": intensity_mm_h,
        "A_km2": area_km2,
        "A_ha": area_ha,
        "method_valid_for_basin_size": area_km2 <= RATIONAL_METHOD_MAX_AREA_KM2,
    }
