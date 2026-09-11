"""
Google Earth Engine client for the Neuquén basin dashboard.

This is the ONLY module that imports `ee` directly. Everything else in the app
(app.py, raster_processing.py, folium_layers.py, pydeck_layers.py) talks to
Earth Engine only through the functions below, so authentication, quota
handling and the getDownloadURL-vs-Drive-export fallback logic live in one
place.

All failures are raised as `EarthEngineError` with a user-facing Spanish
message, so app.py can catch a single exception type and show `st.error()`
instead of letting the Streamlit app crash.
"""

import json
import logging
import os
import zipfile
from datetime import date, datetime, timezone
from typing import Literal

import ee
import geopandas as gpd
import rasterio
import requests
import streamlit as st
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

EE_PROJECT_ID = "gis-nqn"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
ERA5_LAND_MONTHLY_COLLECTION = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA5_TEMPERATURE_BAND = "temperature_2m"
KELVIN_TO_CELSIUS_OFFSET = 273.15

# Official administrative boundaries hosted by Earth Engine, used to clip CHIRPS
# to the actual Neuquén province outline (see load_neuquen_ee_boundary).
GAUL_LEVEL1_COLLECTION = "FAO/GAUL/2015/level1"
NEUQUEN_ADM0_NAME = "Argentina"
NEUQUEN_ADM1_NAME = "Neuquen"

# Local on-disk cache for downloaded rasters, keyed by aggregation/date-range
# so the same CHIRPS request is never re-fetched from Earth Engine twice.
CACHE_DIR = "./data/cache/"

# Provincial DEM: one continuous mosaic covering all of Neuquén, downloaded
# once and reused by the on-the-fly hydrology pipeline (modules/hydrology.py).
# A sidecar "argendem_neuquen.json" stores its source/download date and any
# pending Drive export, so the task survives Streamlit reruns and restarts.
PROVINCIAL_DEM_PATH = os.path.join(CACHE_DIR, "argendem_neuquen.tif")
DEM_EXPORT_FOLDER = "gis_nqn_dem"
DEM_EXPORT_FILE_PREFIX = "argendem_neuquen"
# Metric CRS used for hydrological calculations (POSGAR 2007 / Argentina 1),
# so D8 routing works on square cells of `scale` meters.
DEM_CRS = "EPSG:5343"
DEM_NODATA = -32768
DEM_SOURCES = {
    "AW3D30": {"asset": "JAXA/ALOS/AW3D30/V3_2", "band": "DSM", "is_collection": True},
    "SRTM": {"asset": "USGS/SRTMGL1_003", "band": "elevation", "is_collection": False},
}
DRIVE_FILES_API = "https://www.googleapis.com/drive/v3/files"


class EarthEngineError(RuntimeError):
    """Raised for any Earth Engine failure that the UI should surface to the user."""


@st.cache_resource(show_spinner=False)
def init_earth_engine(project: str = EE_PROJECT_ID) -> bool:
    """
    Initializes the Earth Engine client for `project`.

    Wrapped in st.cache_resource so it only actually runs once per Streamlit
    session (ee.Initialize/ee.Authenticate are not free and don't need to be
    repeated on every rerun).

    If no valid credentials are stored locally, `ee.Initialize()` raises; in
    that case this falls back to `ee.Authenticate()`, which opens a browser/
    device-code flow to obtain and persist credentials, then retries
    `ee.Initialize()`. If that also fails (no interactive terminal available,
    revoked access, wrong project, etc.) an `EarthEngineError` with a clear,
    actionable message is raised instead of letting the raw `ee` exception
    propagate into the Streamlit UI.
    """
    try:
        ee.Initialize(project=project)
        logger.info("Earth Engine initialized for project '%s'.", project)
        return True
    except Exception as init_error:
        logger.warning(
            "ee.Initialize() failed (%s); attempting ee.Authenticate().", init_error
        )
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
            logger.info("Earth Engine authenticated and initialized for '%s'.", project)
            return True
        except Exception as auth_error:
            raise EarthEngineError(
                "No se pudo inicializar Google Earth Engine. Verifique que el proyecto "
                f"'{project}' tenga la API de Earth Engine habilitada y que existan "
                "credenciales válidas. En una terminal interactiva ejecute "
                "`earthengine authenticate` (o `python -c \"import ee; ee.Authenticate()\"`) "
                "y vuelva a intentarlo."
            ) from auth_error


def load_neuquen_ee_geometry(geojson_path: str = "cuenca_neuquen.geojson") -> "ee.Geometry":
    """
    Loads `geojson_path` with GeoPandas, dissolves every feature into one
    geometry, and converts it to an `ee.Geometry` usable as an Earth Engine
    `region` argument.

    IMPORTANT: `cuenca_neuquen.geojson` in this project is actually a
    collection of ~2000 individual water-body polygons (it is byte-identical
    to `cuerpos_agua_neuquen.geojson`), not a single basin/province boundary.
    Dissolving that many small, scattered polygons directly produces a
    (multi)polygon with hundreds of thousands of vertices, which:
      1. Exceeds Earth Engine's ~10MB request-payload limit on its own — this
         hits BOTH `getDownloadURL` and the `Export.image.toDrive()` fallback,
         since both send the full region geometry in the request body; and
      2. Would clip CHIRPS to only the footprint of individual lakes/lagoons,
         leaving the rest of the territory (where most of the precipitation
         signal actually matters) empty.
    The convex hull of the dissolved geometry is used instead: a single,
    low-vertex polygon that safely fits the request-size limit and covers the
    full extent of interest, which is what a CHIRPS clip region needs at
    kilometer-scale resolution anyway. If a real basin/province boundary file
    becomes available, point `geojson_path` at it and this convex-hull step
    stops mattering (a proper boundary is already low-vertex and contiguous).
    """
    if not os.path.exists(geojson_path):
        raise EarthEngineError(
            f"No se encontró el archivo de geometría '{geojson_path}' para recortar CHIRPS."
        )

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise EarthEngineError(f"El archivo '{geojson_path}' no contiene geometrías.")

    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # union_all() replaces the deprecated unary_union in recent GeoPandas/Shapely;
    # fall back to unary_union for older installs.
    dissolved = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    coverage_geometry = dissolved.convex_hull

    return ee.Geometry(coverage_geometry.__geo_interface__)


def load_neuquen_ee_boundary() -> "ee.Geometry":
    """
    Returns the official Neuquén province outline from Earth Engine's hosted
    FAO GAUL administrative boundaries.

    This is the preferred clip region, because the project's local
    `cuenca_neuquen.geojson` is a scattered water-bodies dataset whose extent
    reaches well past the province (east into Río Negro and La Pampa); clipping
    CHIRPS to its convex hull produces a large trapezoid that doesn't resemble
    Neuquén at all. The GAUL outline is a real provincial boundary, so the
    resulting layer is both geographically correct and recognizable on the map.

    Raises `EarthEngineError` if the province can't be found, letting the
    caller fall back to the local geojson.
    """
    boundary = (
        ee.FeatureCollection(GAUL_LEVEL1_COLLECTION)
        .filter(ee.Filter.eq("ADM0_NAME", NEUQUEN_ADM0_NAME))
        .filter(ee.Filter.eq("ADM1_NAME", NEUQUEN_ADM1_NAME))
    )
    try:
        if boundary.size().getInfo() == 0:
            raise EarthEngineError(
                "No se encontró el límite provincial de Neuquén en "
                f"'{GAUL_LEVEL1_COLLECTION}'."
            )
    except EarthEngineError:
        raise
    except Exception as lookup_error:
        raise EarthEngineError(
            f"Error consultando el límite provincial en Earth Engine: {lookup_error}"
        ) from lookup_error

    return boundary.geometry()


def _years_between(start_date: str, end_date: str) -> float:
    """Number of (fractional) years spanned by [start_date, end_date), floored to avoid /0."""
    delta_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
    return max(delta_days / 365.25, 1e-6)


def _is_size_limit_error(message: str) -> bool:
    """True if an Earth Engine error message refers to getDownloadURL's size ceiling."""
    return any(
        token in message.lower()
        for token in ("size", "too large", "bytes", "limit", "dimensions")
    )


def _fetch_image_download_or_export(
    image: "ee.Image",
    geometry: "ee.Geometry",
    scale: int,
    crs: str,
    export_description: str,
    export_folder: str,
    error_label: str,
) -> dict:
    """
    Shared `getDownloadURL()` → `Export.image.toDrive()` fallback used by every
    Earth Engine data source in this module (CHIRPS precipitation, ERA5-Land
    temperature, and the provincial DEM export path). Tries a direct download
    URL first; if Earth Engine rejects it for exceeding getDownloadURL's
    internal size ceiling (~32-50MB depending on format), falls back to an
    asynchronous Drive export instead. Any other Earth Engine failure (quota,
    auth, a genuinely invalid request) is raised as `EarthEngineError` rather
    than silently falling back.

    `error_label` is a short Spanish description of the dataset (e.g.
    "CHIRPS", "ERA5-Land temperatura") used only to build user-facing error
    messages.

    Returns one of:
      {"method": "download_url", "url": <str>}
      {"method": "drive_export", "task": <ee.batch.Task>, "task_id": <str>}
    """
    download_params = {
        "region": geometry,
        "scale": scale,
        "crs": crs,
        "format": "GEO_TIFF",
    }

    try:
        url = image.getDownloadURL(download_params)
        logger.info("%s: direct getDownloadURL at scale=%sm.", export_description, scale)
        return {"method": "download_url", "url": url}
    except ee.EEException as size_error:
        message = str(size_error)
        if not _is_size_limit_error(message):
            raise EarthEngineError(
                f"Earth Engine rechazó la solicitud de descarga de {error_label}: {message}"
            ) from size_error

        logger.warning(
            "%s: getDownloadURL excedió el límite de tamaño (%s). "
            "Usando Export.image.toDrive() como alternativa.",
            export_description, message,
        )
    except Exception as quota_error:
        # Covers EE quota/auth errors that aren't ee.EEException (e.g. HTTP 429
        # surfaced by the underlying API client).
        raise EarthEngineError(
            f"Earth Engine devolvió un error al solicitar {error_label} (posible problema de "
            f"cuota o autenticación): {quota_error}"
        ) from quota_error

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=export_description,
        folder=export_folder,
        region=geometry,
        scale=scale,
        crs=crs,
        fileFormat="GeoTIFF",
        maxPixels=1e13,
    )
    task.start()
    logger.info(
        "%s: fallback a Export.image.toDrive() (task id=%s, folder=%s).",
        export_description, task.id, export_folder,
    )
    return {"method": "drive_export", "task": task, "task_id": task.id}


def fetch_chirps_precipitation(
    geometry: "ee.Geometry",
    start_date: str,
    end_date: str,
    scale: int = 5000,
    aggregation: Literal["sum", "mean"] = "sum",
    export_folder: str = "gis_nqn_chirps",
) -> dict:
    """
    Builds a single-band CHIRPS precipitation image over `geometry` for
    [start_date, end_date) and returns instructions for retrieving it.

    aggregation:
      - "sum":  accumulated precipitation over the whole period (mm) — used
                for "Precipitación acumulada 2016-2026".
      - "mean": average ANNUAL rainfall over the same period (mm/year) — the
                daily images are summed first (total accumulation) and then
                divided by the number of years spanned, so the result reads
                as a yearly average rather than a daily one. Used for
                "Promedio lluvias".

    Tries `image.getDownloadURL()` first (a direct GeoTIFF link). Earth
    Engine's getDownloadURL has an internal size ceiling (~32-50MB depending
    on format); when the requested area/scale exceeds it, Earth Engine raises
    an `ee.EEException` mentioning the size limit. That case is caught here
    and the function falls back to `ee.batch.Export.image.toDrive()`, which
    has no such size limit but delivers the file asynchronously into Google
    Drive instead of returning a URL. Which path was taken is always logged.

    Returns one of:
      {"method": "download_url", "url": <str>}
      {"method": "drive_export", "task": <ee.batch.Task>, "task_id": <str>}
    """
    if aggregation not in ("sum", "mean"):
        raise ValueError(f"aggregation must be 'sum' or 'mean', got {aggregation!r}")

    collection = ee.ImageCollection(CHIRPS_COLLECTION).filterDate(start_date, end_date)

    if aggregation == "sum":
        image = collection.sum()
    else:
        n_years = _years_between(start_date, end_date)
        image = collection.sum().divide(n_years)

    image = image.clip(geometry).rename("precipitation").toFloat()

    description = f"chirps_{aggregation}_{start_date}_{end_date}".replace("-", "")
    return _fetch_image_download_or_export(
        image, geometry, scale, "EPSG:4326",
        export_description=description,
        export_folder=export_folder,
        error_label="CHIRPS",
    )


def fetch_era5_temperature(
    geometry: "ee.Geometry",
    start_date: str,
    end_date: str,
    scale: int = 5000,
    aggregation: Literal["mean", "sum"] = "mean",
    export_folder: str = "gis_nqn_era5",
) -> dict:
    """
    Builds a single-band ERA5-Land monthly 2m air temperature image over
    `geometry` for [start_date, end_date) and returns instructions for
    retrieving it, mirroring `fetch_chirps_precipitation`.

    Source: "ECMWF/ERA5_LAND/MONTHLY_AGGR", band "temperature_2m", one image
    per month. The band is delivered in Kelvin; it is converted to Celsius
    (`image.subtract(273.15)`) on each monthly image BEFORE aggregating, so
    both "mean" and "sum" operate on Celsius values.

    aggregation:
      - "mean": average monthly temperature over the period (°C) — the usual
                choice for a temperature layer.
      - "sum":  sum of the monthly mean temperatures over the period (°C);
                rarely meaningful on its own, provided only for symmetry with
                `fetch_chirps_precipitation`'s aggregation parameter.

    Tries `image.getDownloadURL()` first and falls back to
    `ee.batch.Export.image.toDrive()` when Earth Engine reports the request
    exceeds getDownloadURL's size ceiling (see `_fetch_image_download_or_export`,
    shared with CHIRPS and the provincial DEM).

    Returns one of:
      {"method": "download_url", "url": <str>}
      {"method": "drive_export", "task": <ee.batch.Task>, "task_id": <str>}
    """
    if aggregation not in ("mean", "sum"):
        raise ValueError(f"aggregation must be 'mean' or 'sum', got {aggregation!r}")

    collection = (
        ee.ImageCollection(ERA5_LAND_MONTHLY_COLLECTION)
        .filterDate(start_date, end_date)
        .select(ERA5_TEMPERATURE_BAND)
        .map(lambda monthly_image: monthly_image.subtract(KELVIN_TO_CELSIUS_OFFSET))
    )

    image = collection.mean() if aggregation == "mean" else collection.sum()
    image = image.clip(geometry).rename("temperature").toFloat()

    description = f"era5_temp_{aggregation}_{start_date}_{end_date}".replace("-", "")
    return _fetch_image_download_or_export(
        image, geometry, scale, "EPSG:4326",
        export_description=description,
        export_folder=export_folder,
        error_label="ERA5-Land temperatura",
    )


def _stream_to_file(response: requests.Response, path: str) -> None:
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def download_raster_from_url(url: str, output_path: str) -> str:
    """
    Downloads `url` (an Earth Engine getDownloadURL link) to `output_path`
    using `requests`, streaming to disk to avoid loading large rasters fully
    into memory.

    Earth Engine's GeoTIFF download links are normally a single .tif, but for
    some request shapes (e.g. multi-band results) Earth Engine wraps the
    output in a .zip; this detects that case (via Content-Type or a `.zip`
    URL suffix), extracts the first .tif member, and writes it to
    `output_path` so callers always get back a plain GeoTIFF path regardless
    of how Earth Engine packaged it.

    The file is written to a temporary path and only renamed onto
    `output_path` once complete, so an interrupted download never replaces a
    previously cached raster with a truncated one.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".part"

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
    except requests.RequestException as download_error:
        raise EarthEngineError(
            f"Falló la descarga del raster desde Earth Engine: {download_error}"
        ) from download_error

    content_type = response.headers.get("Content-Type", "")
    is_zip = "zip" in content_type.lower() or url.lower().split("?")[0].endswith(".zip")

    try:
        if is_zip:
            tmp_zip_path = output_path + ".zip.tmp"
            _stream_to_file(response, tmp_zip_path)
            try:
                with zipfile.ZipFile(tmp_zip_path) as zf:
                    tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
                    if not tif_names:
                        raise EarthEngineError(
                            "La respuesta de Earth Engine fue un .zip sin ningún GeoTIFF."
                        )
                    with zf.open(tif_names[0]) as src, open(tmp_path, "wb") as dst:
                        while chunk := src.read(1024 * 1024):
                            dst.write(chunk)
            finally:
                if os.path.exists(tmp_zip_path):
                    os.remove(tmp_zip_path)
        else:
            _stream_to_file(response, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info("Downloaded raster to '%s' (%d bytes).", output_path, os.path.getsize(output_path))
    return output_path


def get_cache_path(
    aggregation: str,
    start_date: str,
    end_date: str,
    cache_dir: str = CACHE_DIR,
    region_key: str = "gaul",
) -> str:
    """
    Deterministic on-disk path for a given CHIRPS request.

    `region_key` identifies which clip region produced the file (the official
    province boundary vs. the local geojson fallback), so changing the region
    yields a different filename instead of silently reusing a raster covering
    a different area.
    """
    filename = f"chirps_{aggregation}_{start_date}_{end_date}_{region_key}.tif"
    return os.path.join(cache_dir, filename)


def fetch_and_cache_chirps(
    geojson_path: str,
    start_date: str,
    end_date: str,
    aggregation: Literal["sum", "mean"] = "sum",
    scale: int = 5000,
    cache_dir: str = CACHE_DIR,
) -> str:
    """
    High-level entry point meant to be called directly from the Streamlit app.

    Returns the local path to a cached CHIRPS GeoTIFF for the requested
    (aggregation, start_date, end_date), downloading it from Earth Engine only
    if it isn't already present under `cache_dir` — so re-selecting the same
    climate layer in the sidebar never re-hits Earth Engine.

    Raises `EarthEngineError` (never a raw `ee`/`requests` exception) on any
    failure — init/auth problems, quota errors, or a Drive-export fallback
    that can't be returned synchronously — so app.py can catch this single
    type and show `st.error()` without crashing.
    """
    # Disk cache is checked BEFORE touching Earth Engine at all (for either
    # possible clip region), so an already-downloaded date range never costs a
    # single EE call.
    for cached_region in ("gaul", "local"):
        cached_path = get_cache_path(
            aggregation, start_date, end_date, cache_dir, cached_region
        )
        if os.path.exists(cached_path):
            logger.info("Using cached CHIRPS raster: %s", cached_path)
            return cached_path

    init_earth_engine()

    # Prefer the official province outline; fall back to the local geojson
    # (convex hull) only if Earth Engine's boundary lookup fails. The region
    # used is part of the cache filename, so the two never get mixed up.
    try:
        geometry = load_neuquen_ee_boundary()
        region_key = "gaul"
    except EarthEngineError as boundary_error:
        logger.warning(
            "No se pudo obtener el límite provincial oficial (%s); "
            "usando la geometría local '%s'.",
            boundary_error, geojson_path,
        )
        geometry = load_neuquen_ee_geometry(geojson_path)
        region_key = "local"

    cache_path = get_cache_path(aggregation, start_date, end_date, cache_dir, region_key)

    result = fetch_chirps_precipitation(
        geometry, start_date, end_date, scale=scale, aggregation=aggregation
    )

    if result["method"] == "download_url":
        return download_raster_from_url(result["url"], cache_path)

    # Drive export: the file lands asynchronously in the user's Google Drive,
    # not on the local filesystem, so there is nothing to return synchronously.
    raise EarthEngineError(
        "El área o la resolución solicitada excede el límite de descarga directa de "
        f"Earth Engine. Se inició una exportación a Google Drive (tarea '{result['task_id']}', "
        "carpeta 'gis_nqn_chirps'). Espere a que la tarea termine (revise la pestaña "
        "'Tasks' en code.earthengine.google.com o su Google Drive), coloque el archivo "
        f"resultante en '{cache_path}', y vuelva a seleccionar la capa; o reduzca el área "
        "dibujada / aumente el parámetro `scale` para que quepa en una descarga directa."
    )


def get_era5_cache_path(
    aggregation: str,
    start_date: str,
    end_date: str,
    cache_dir: str = CACHE_DIR,
    region_key: str = "gaul",
) -> str:
    """
    Deterministic on-disk path for a given ERA5-Land temperature request.

    Uses a "temp_" prefix (as opposed to CHIRPS's "chirps_") so temperature
    and precipitation rasters never collide in `./data/cache/` even when they
    share the same date range/aggregation/region, e.g. "temp_mean_2016-01-01_
    2026-01-01_gaul.tif".
    """
    filename = f"temp_{aggregation}_{start_date}_{end_date}_{region_key}.tif"
    return os.path.join(cache_dir, filename)


def fetch_and_cache_era5_temperature(
    geojson_path: str,
    start_date: str,
    end_date: str,
    aggregation: Literal["mean", "sum"] = "mean",
    scale: int = 5000,
    cache_dir: str = CACHE_DIR,
) -> str:
    """
    High-level entry point meant to be called directly from the Streamlit app,
    mirroring `fetch_and_cache_chirps`.

    Returns the local path to a cached ERA5-Land temperature GeoTIFF (°C) for
    the requested (aggregation, start_date, end_date), downloading it from
    Earth Engine only if it isn't already present under `cache_dir`.

    Raises `EarthEngineError` (never a raw `ee`/`requests` exception) on any
    failure, so app.py can catch this single type and show `st.error()`
    without crashing.
    """
    # Disk cache is checked BEFORE touching Earth Engine at all (for either
    # possible clip region), same as fetch_and_cache_chirps.
    for cached_region in ("gaul", "local"):
        cached_path = get_era5_cache_path(
            aggregation, start_date, end_date, cache_dir, cached_region
        )
        if os.path.exists(cached_path):
            logger.info("Using cached ERA5-Land temperature raster: %s", cached_path)
            return cached_path

    init_earth_engine()

    try:
        geometry = load_neuquen_ee_boundary()
        region_key = "gaul"
    except EarthEngineError as boundary_error:
        logger.warning(
            "No se pudo obtener el límite provincial oficial (%s); "
            "usando la geometría local '%s'.",
            boundary_error, geojson_path,
        )
        geometry = load_neuquen_ee_geometry(geojson_path)
        region_key = "local"

    cache_path = get_era5_cache_path(aggregation, start_date, end_date, cache_dir, region_key)

    result = fetch_era5_temperature(
        geometry, start_date, end_date, scale=scale, aggregation=aggregation
    )

    if result["method"] == "download_url":
        return download_raster_from_url(result["url"], cache_path)

    # Drive export: the file lands asynchronously in the user's Google Drive,
    # not on the local filesystem, so there is nothing to return synchronously.
    raise EarthEngineError(
        "El área o la resolución solicitada excede el límite de descarga directa de "
        f"Earth Engine. Se inició una exportación a Google Drive (tarea '{result['task_id']}', "
        "carpeta 'gis_nqn_era5'). Espere a que la tarea termine (revise la pestaña "
        "'Tasks' en code.earthengine.google.com o su Google Drive), coloque el archivo "
        f"resultante en '{cache_path}', y vuelva a seleccionar la capa; o reduzca el área "
        "dibujada / aumente el parámetro `scale` para que quepa en una descarga directa."
    )


# ---------------------------------------------------------------------------
# Provincial DEM (single continuous mosaic for the hydrology pipeline)
# ---------------------------------------------------------------------------
# (_is_size_limit_error lives above, next to _fetch_image_download_or_export,
# and is reused here.)


def fetch_provincial_dem(
    geometry: "ee.Geometry",
    source: Literal["AW3D30", "SRTM"] = "AW3D30",
    scale: int = 30,
    export_folder: str = DEM_EXPORT_FOLDER,
) -> dict:
    """
    Builds a single continuous DEM over `geometry` and returns instructions
    for retrieving it, mirroring `fetch_chirps_precipitation`.

    source:
      - "AW3D30": JAXA ALOS World 3D 30m (ImageCollection of 1°x1° tiles,
                  band "DSM"), merged server-side with `.mosaic()`.
      - "SRTM":   USGS SRTM 1 arc-second (single global image, band "elevation").

    The image is resampled bilinearly, clipped to `geometry`, reprojected to
    `DEM_CRS` at `scale` meters, and stored as Int16 with `DEM_NODATA` for
    voids and pixels outside the province.

    Tries `getDownloadURL()` first. A full province at 30m is several hundred
    MB, far above getDownloadURL's ~32-48MB ceiling, so in practice this falls
    back to `ee.batch.Export.image.toDrive()`. The export is only *started*
    here — this function never waits for it — so the Streamlit app is not
    blocked; the task id is returned so the caller can check on it later.
    Which method was used is always logged.

    Returns one of:
      {"method": "download_url", "url": <str>}
      {"method": "drive_export", "task": <ee.batch.Task>, "task_id": <str>}
    """
    if source not in DEM_SOURCES:
        raise ValueError(f"source must be one of {list(DEM_SOURCES)}, got {source!r}")
    config = DEM_SOURCES[source]

    if config["is_collection"]:
        image = (
            ee.ImageCollection(config["asset"])
            .select(config["band"])
            .map(lambda tile: tile.resample("bilinear"))
            .mosaic()
        )
    else:
        image = ee.Image(config["asset"]).select(config["band"]).resample("bilinear")

    image = image.clip(geometry).unmask(DEM_NODATA).toInt16().rename("elevation")

    try:
        url = image.getDownloadURL({
            "region": geometry,
            "scale": scale,
            "crs": DEM_CRS,
            "format": "GEO_TIFF",
        })
        logger.info("Provincial DEM (%s, %sm): direct getDownloadURL.", source, scale)
        return {"method": "download_url", "url": url}
    except ee.EEException as size_error:
        message = str(size_error)
        if not _is_size_limit_error(message):
            raise EarthEngineError(
                f"Earth Engine rechazó la solicitud del DEM provincial: {message}"
            ) from size_error
        logger.warning(
            "Provincial DEM (%s, %sm): getDownloadURL exceeded the size limit (%s). "
            "Falling back to Export.image.toDrive().",
            source, scale, message,
        )
    except Exception as quota_error:
        raise EarthEngineError(
            "Earth Engine devolvió un error al solicitar el DEM provincial (posible "
            f"problema de cuota o autenticación): {quota_error}"
        ) from quota_error

    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f"{DEM_EXPORT_FILE_PREFIX}_{source.lower()}_{scale}m",
            folder=export_folder,
            fileNamePrefix=DEM_EXPORT_FILE_PREFIX,
            region=geometry,
            scale=scale,
            crs=DEM_CRS,
            fileFormat="GeoTIFF",
            maxPixels=1e13,
            # One file for the whole province (EE shards large exports by
            # default); cloud-optimized so windowed reads stay cheap.
            fileDimensions=32768,
            formatOptions={"cloudOptimized": True, "noData": DEM_NODATA},
        )
        task.start()
    except Exception as export_error:
        raise EarthEngineError(
            "No se pudo iniciar la exportación del DEM provincial a Google Drive "
            f"(posible problema de cuota o autenticación): {export_error}"
        ) from export_error

    logger.info(
        "Provincial DEM (%s, %sm): Export.image.toDrive() started (task id=%s, folder=%s).",
        source, scale, task.id, export_folder,
    )
    return {"method": "drive_export", "task": task, "task_id": task.id}


def _dem_meta_path(output_path: str) -> str:
    return os.path.splitext(output_path)[0] + ".json"


def _read_dem_meta(output_path: str) -> dict:
    meta_path = _dem_meta_path(output_path)
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable DEM metadata file '%s'.", meta_path)
        return {}


def _write_dem_meta(output_path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(_dem_meta_path(output_path), "w") as f:
        json.dump(meta, f, indent=2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def describe_dem(path: str = PROVINCIAL_DEM_PATH) -> dict:
    """File size, CRS, extent (native and EPSG:4326) and grid of a local DEM."""
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as dem:
        return {
            "path": path,
            "size_bytes": os.path.getsize(path),
            "crs": dem.crs.to_string() if dem.crs else None,
            "bounds": tuple(dem.bounds),
            "bounds_wgs84": transform_bounds(dem.crs, "EPSG:4326", *dem.bounds),
            "shape": dem.shape,
            "resolution": dem.res,
            "dtype": dem.dtypes[0],
            "nodata": dem.nodata,
        }


def _finalize_provincial_dem(output_path: str) -> dict:
    """
    Validates a freshly downloaded DEM and makes sure its nodata value is set,
    so masked reads in the hydrology pipeline exclude voids/outside pixels.
    """
    try:
        with rasterio.open(output_path) as dem:
            nodata_missing = dem.nodata is None
        if nodata_missing:
            with rasterio.open(output_path, "r+") as dem:
                dem.nodata = DEM_NODATA
        return describe_dem(output_path)
    except rasterio.errors.RasterioIOError as read_error:
        raise EarthEngineError(
            f"El DEM descargado en '{output_path}' no es un GeoTIFF válido: {read_error}"
        ) from read_error


def download_drive_export(
    file_prefix: str,
    output_path: str,
    created_after: str | None = None,
) -> str:
    """
    Downloads a finished `Export.image.toDrive()` result from the user's
    Google Drive into `output_path`, using the same stored Earth Engine
    credentials (they carry the Drive scope). Streams to a temporary file and
    renames it into place only when complete.
    """
    try:
        credentials = Credentials(None, **ee.oauth.get_credentials_arguments())
        session = AuthorizedSession(credentials)
        query = (
            f"name contains '{file_prefix}' and trashed = false "
            "and mimeType != 'application/vnd.google-apps.folder'"
        )
        if created_after:
            query += f" and createdTime >= '{created_after}'"
        listing = session.get(
            DRIVE_FILES_API,
            params={
                "q": query,
                "orderBy": "createdTime desc",
                "fields": "files(id,name,size,createdTime)",
            },
            timeout=60,
        )
        listing.raise_for_status()
        files = [f for f in listing.json().get("files", []) if f["name"].lower().endswith(".tif")]
        if not files:
            raise EarthEngineError(
                f"La exportación terminó pero no se encontró '{file_prefix}*.tif' en Google Drive."
            )
        if len(files) > 1 and created_after:
            raise EarthEngineError(
                f"La exportación generó {len(files)} archivos en Drive "
                f"({', '.join(f['name'] for f in files)}); se esperaba uno solo."
            )
        drive_file = files[0]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tmp_path = output_path + ".part"
        try:
            with session.get(
                f"{DRIVE_FILES_API}/{drive_file['id']}",
                params={"alt": "media"},
                stream=True,
                timeout=600,
            ) as response:
                response.raise_for_status()
                _stream_to_file(response, tmp_path)
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except EarthEngineError:
        raise
    except Exception as drive_error:
        raise EarthEngineError(
            "No se pudo descargar el DEM exportado desde Google Drive (carpeta "
            f"'{DEM_EXPORT_FOLDER}'). Descárguelo manualmente y guárdelo como "
            f"'{output_path}'. Detalle: {drive_error}"
        ) from drive_error

    logger.info(
        "Downloaded Drive export '%s' to '%s' (%d bytes).",
        drive_file["name"], output_path, os.path.getsize(output_path),
    )
    return output_path


def get_provincial_dem_status(output_path: str = PROVINCIAL_DEM_PATH) -> dict:
    """
    Local-only view of the provincial DEM state (no Earth Engine calls), cheap
    enough to render on every Streamlit rerun:
      {"cached": bool, "cached_at": iso str | None, "source": str | None,
       "pending_task": {"task_id", "started_at", "source", "scale"} | None}
    """
    meta = _read_dem_meta(output_path)
    cached = os.path.exists(output_path)
    cached_at = meta.get("downloaded_at")
    if cached and not cached_at:
        # File placed manually (e.g. downloaded from Drive by hand).
        cached_at = datetime.fromtimestamp(
            os.path.getmtime(output_path), timezone.utc
        ).isoformat(timespec="seconds")
    return {
        "cached": cached,
        "cached_at": cached_at if cached else None,
        "source": meta.get("source") if cached else None,
        "pending_task": meta.get("pending_task"),
    }


def _load_province_geometry(geojson_path: str) -> "ee.Geometry":
    try:
        return load_neuquen_ee_boundary()
    except EarthEngineError as boundary_error:
        logger.warning(
            "Official province boundary unavailable (%s); using local geometry '%s'.",
            boundary_error, geojson_path,
        )
        return load_neuquen_ee_geometry(geojson_path)


def update_provincial_dem(
    source: Literal["AW3D30", "SRTM"] = "AW3D30",
    scale: int = 30,
    force: bool = False,
    output_path: str = PROVINCIAL_DEM_PATH,
    geojson_path: str = "cuenca_neuquen.geojson",
) -> dict:
    """
    High-level entry point for the "Actualizar DEM provincial" button.

    Only talks to Earth Engine when there is no cached DEM (or `force=True`)
    and no export already in flight. The previously cached file is never
    touched until a new one has fully arrived, so the app keeps working with
    the last good DEM if Earth Engine fails.

    Returns {"state": "CACHED" | "COMPLETED" | "PENDING", ...}; raises
    `EarthEngineError` on any failure.
    """
    meta = _read_dem_meta(output_path)
    if meta.get("pending_task") and not force:
        return {"state": "PENDING", **meta["pending_task"]}
    if os.path.exists(output_path) and not force:
        logger.info("Using cached provincial DEM: %s", output_path)
        return {"state": "CACHED", "path": output_path}

    init_earth_engine()
    geometry = _load_province_geometry(geojson_path)
    result = fetch_provincial_dem(geometry, source=source, scale=scale)

    if result["method"] == "download_url":
        download_raster_from_url(result["url"], output_path)
        info = _finalize_provincial_dem(output_path)
        _write_dem_meta(output_path, {
            "source": source, "scale": scale, "method": "download_url",
            "downloaded_at": _utc_now(), "pending_task": None,
        })
        return {"state": "COMPLETED", "method": "download_url", **info}

    pending_task = {
        "task_id": result["task_id"],
        "started_at": _utc_now(),
        "source": source,
        "scale": scale,
    }
    _write_dem_meta(output_path, {**meta, "pending_task": pending_task})
    return {"state": "PENDING", "method": "drive_export", **pending_task}


def check_provincial_dem_export(output_path: str = PROVINCIAL_DEM_PATH) -> dict:
    """
    Polls the pending Drive export once (non-blocking). When it has finished,
    downloads the file from Drive into `output_path` and clears the pending
    task. Returns {"state": READY | RUNNING | COMPLETED | NONE, ...}; raises
    `EarthEngineError` if the task failed or Earth Engine is unreachable.
    """
    meta = _read_dem_meta(output_path)
    pending_task = meta.get("pending_task")
    if not pending_task:
        return {"state": "NONE"}

    init_earth_engine()
    try:
        status = ee.data.getTaskStatus(pending_task["task_id"])[0]
    except Exception as status_error:
        raise EarthEngineError(
            f"No se pudo consultar el estado de la exportación en Earth Engine: {status_error}"
        ) from status_error

    state = status.get("state", "UNKNOWN")
    if state in ("FAILED", "CANCELLED", "UNKNOWN"):
        _write_dem_meta(output_path, {**meta, "pending_task": None})
        raise EarthEngineError(
            f"La exportación del DEM provincial terminó en estado {state}: "
            f"{status.get('error_message', 'sin detalle')}. Vuelva a intentar la actualización."
        )
    if state != "COMPLETED":
        return {"state": state, **pending_task}

    download_drive_export(DEM_EXPORT_FILE_PREFIX, output_path, pending_task["started_at"])
    info = _finalize_provincial_dem(output_path)
    _write_dem_meta(output_path, {
        "source": pending_task["source"], "scale": pending_task["scale"],
        "method": "drive_export", "task_id": pending_task["task_id"],
        "export_started_at": pending_task["started_at"],
        "downloaded_at": _utc_now(), "pending_task": None,
    })
    return {"state": "COMPLETED", "method": "drive_export", **info}
