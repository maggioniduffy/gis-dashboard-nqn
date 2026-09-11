import os
import json
import logging
from datetime import datetime

import folium
import geopandas as gpd
import numpy as np
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from modules.data_loader import load_additional_layer, load_cuenca_geojson
from modules.ee_client import (
    DEM_SOURCES,
    EE_PROJECT_ID,
    PROVINCIAL_DEM_PATH,
    EarthEngineError,
    check_provincial_dem_export,
    fetch_and_cache_chirps,
    fetch_and_cache_era5_temperature,
    get_provincial_dem_status,
    update_provincial_dem,
)
from modules.folium_layers import add_climate_layer_to_map
from modules.hydrology import (
    DemCoverageError,
    PrecipCoverageError,
    calculate_peak_flow,
    compute_runoff_overlay,
    get_dem_bounds_wgs84,
)
from modules.map_builder import create_cuenca_map
from modules.pydeck_layers import (
    BASEMAP_TILE_URLS,
    DEFAULT_BASEMAP,
    build_isometric_view_state,
    build_pydeck_layer,
    gdf_to_pydeck_df,
    render_3d_panel_live,
)
from modules.raster_processing import (
    align_climate_grids,
    align_flow_accumulation_to_climate_grid,
    raster_to_classified_gdf,
)
from modules.sidebar import render_sidebar


RUNOFF_OVERLAY_VERSION = 4
CUENCA_GEOJSON_PATH = "cuenca_neuquen.geojson"

# Earth Engine parameters for each selectable sidebar option. All three cover
# the same 2016-2026 period; "source" picks which Earth Engine dataset (and
# therefore which fetch_and_cache_* function in modules/ee_client.py) backs
# the layer, "aggregation"/"color_scheme" differ per layer as before, and
# "value_label"/"unit" drive the Folium tooltip and the 3D panel's tooltip so
# a temperature layer doesn't show up mislabeled as "Precipitación (mm)".
CLIMATE_LAYER_CONFIG = {
    "Precipitación acumulada 2016-2026": {
        "source": "chirps",
        "aggregation": "sum",
        "color_scheme": "accumulated",
        "start_date": "2016-01-01",
        "end_date": "2026-01-01",
        "value_label": "Precipitación",
        "unit": "mm",
    },
    "Promedio lluvias": {
        "source": "chirps",
        "aggregation": "mean",
        "color_scheme": "average",
        "start_date": "2016-01-01",
        "end_date": "2026-01-01",
        "value_label": "Precipitación",
        "unit": "mm",
    },
    "Temperatura promedio 2016-2026": {
        "source": "era5",
        "aggregation": "mean",
        "color_scheme": "temperature",
        "start_date": "2016-01-01",
        "end_date": "2026-01-01",
        "value_label": "Temperatura",
        "unit": "°C",
    },
}


logger = logging.getLogger(__name__)

# Human-readable name for each step of `run_polygon_analysis`, reused by the
# debug log and by the UI's "what completed / what didn't" summary so both
# call the same thing by the same name.
ANALYSIS_STEP_LABELS = {
    "runoff": "Escorrentía (Pysheds D8)",
    "peak_flow": "Caudal pico (Método Racional)",
    "grid_3d": "Grilla 3D (precipitación × temperatura × flujo)",
}


def load_combined_climate_grid() -> gpd.GeoDataFrame:
    """
    THE single precipitation/temperature grid the whole polygon analysis reads
    from — both the Rational Method (as its rainfall intensity source) and the
    3D panel (as the grid flow accumulation is joined onto).

    Having one shared grid is the point: the "I" behind a reported Q is then
    literally the same cell value the user sees extruded in the 3D panel,
    instead of coming from a second, independently-built vectorization of the
    same CHIRPS raster that can drift from it (`raster_to_classified_gdf`
    merges runs of same-class pixels into irregular regions, while
    `align_climate_grids` keeps one square polygon per physical cell — same
    source raster, different geometry AND different per-feature values).

    This adds NO new cache layer: `fetch_and_cache_chirps` /
    `fetch_and_cache_era5_temperature` hit the existing on-disk raster cache
    and `align_climate_grids` is `st.cache_data`-memoized, so calling this
    from several places within one rerun costs a cache lookup rather than a
    re-download or a re-alignment.

    Note that `st.cache_data` hands back an equal COPY, not the same object
    (that's its documented mutation-safety behaviour, verified here). So
    "both calculations read the same grid" is NOT guaranteed by the cache —
    it's guaranteed by `run_polygon_analysis` taking the grid as a single
    parameter and passing that one object to both consumers. Call this once
    per rerun and thread the result through; don't call it again deeper down.
    """
    precip_config = CLIMATE_LAYER_CONFIG["Promedio lluvias"]
    temp_config = CLIMATE_LAYER_CONFIG["Temperatura promedio 2016-2026"]

    precip_raster_path = fetch_and_cache_chirps(
        geojson_path=CUENCA_GEOJSON_PATH,
        start_date=precip_config["start_date"],
        end_date=precip_config["end_date"],
        aggregation=precip_config["aggregation"],
    )
    temp_raster_path = fetch_and_cache_era5_temperature(
        geojson_path=CUENCA_GEOJSON_PATH,
        start_date=temp_config["start_date"],
        end_date=temp_config["end_date"],
        aggregation=temp_config["aggregation"],
    )
    return align_climate_grids(precip_raster_path, temp_raster_path)


def run_polygon_analysis(
    geometry: dict,
    dem_path: str,
    climate_grid: gpd.GeoDataFrame | None,
) -> dict:
    """
    Everything that happens when the user draws a polygon, in one pass and in
    a fixed order:

      1. "runoff"   - Pysheds D8 routing + flow accumulation  (topography only)
      2. "peak_flow"- Rational Method Q = C*I*A               (precipitation only)
      3. "grid_3d"  - climate grid + step 1's flow accumulation joined on

    Steps 1 and 2 are INDEPENDENT and each has its own error boundary: a
    polygon outside the DEM still gets a peak flow estimate, and a polygon
    outside CHIRPS coverage still gets its runoff overlay. Step 3 degrades
    rather than fails — without step 1 it still returns the precip/temp grid,
    only without the "flow" height variable.

    `climate_grid` must be `load_combined_climate_grid()`'s result (or None if
    it couldn't be loaded); it is read here and NOT re-derived, so steps 2 and
    3 are guaranteed to be looking at the same cells.

    Every step logs its outcome at INFO/WARNING with the same step labels the
    UI shows, so a confusing panel can be traced in the server log.

    Returns:
        {
          "runoff": dict | None,          # compute_runoff_overlay's result
          "peak_flow": dict | None,       # calculate_peak_flow's result
          "grid_3d": GeoDataFrame | None, # grid for the 3D panel
          "has_flow_variable": bool,      # is "flow" selectable in the panel?
          "errors": {step: message | None},
          "completed": [step, ...],
        }
    Never raises for the expected coverage failures; unexpected exceptions in
    step 3 are caught and reported rather than taking down steps 1 and 2.
    """
    result = {
        "runoff": None,
        "peak_flow": None,
        "grid_3d": None,
        "has_flow_variable": False,
        "errors": {step: None for step in ANALYSIS_STEP_LABELS},
        "completed": [],
    }

    def _succeed(step: str, detail: str) -> None:
        result["completed"].append(step)
        logger.info("polygon analysis | %s | OK | %s", ANALYSIS_STEP_LABELS[step], detail)

    def _fail(step: str, message: str) -> None:
        result["errors"][step] = message
        logger.warning("polygon analysis | %s | FAILED | %s", ANALYSIS_STEP_LABELS[step], message)

    logger.info("polygon analysis | start | climate_grid=%s cells",
                0 if climate_grid is None else len(climate_grid))

    # --- Step 1: topography (Pysheds D8) ---
    try:
        result["runoff"] = compute_runoff_overlay(geometry, dem_path)
        _succeed("runoff", f"overlay {result['runoff']['overlay'].shape}, "
                           f"partial_coverage={result['runoff']['partial_coverage']}")
    except DemCoverageError as coverage_error:
        _fail("runoff", str(coverage_error))

    # --- Step 2: precipitation (Rational Method) — independent of step 1 ---
    if climate_grid is None:
        _fail("peak_flow", "No hay grilla climática disponible (CHIRPS/ERA5-Land).")
    else:
        try:
            # Reads `climate_grid`'s own "precip_value" column; calculate_peak_flow
            # auto-detects it, so no second precipitation source is involved.
            result["peak_flow"] = calculate_peak_flow(geometry, climate_grid)
            _succeed("peak_flow", f"Q={result['peak_flow']['Q_m3_s']:.3f} m3/s, "
                                  f"I={result['peak_flow']['I_mm_h']:.2f} mm/h, "
                                  f"A={result['peak_flow']['A_km2']:.3f} km2")
        except PrecipCoverageError as precip_error:
            _fail("peak_flow", str(precip_error))

    # --- Step 3: 3D grid — needs step 1 for the "flow" variable, degrades without it ---
    if climate_grid is None:
        _fail("grid_3d", "No hay grilla climática disponible (CHIRPS/ERA5-Land).")
    elif result["runoff"] is None:
        # Pysheds failed: the panel still works with precipitation x temperature,
        # just without the flow accumulation variable.
        result["grid_3d"] = climate_grid
        _fail("grid_3d", "Sin acumulación de flujo: el análisis de topografía no se completó. "
                         "El panel 3D queda con precipitación y temperatura únicamente.")
    else:
        try:
            joined_grid = align_flow_accumulation_to_climate_grid(
                result["runoff"]["flow_accumulation"],
                result["runoff"]["flow_transform"],
                result["runoff"]["flow_crs"],
                climate_grid,
            )
            # The join keeps every climate cell and marks the ones outside the
            # polygon with NaN, so the precipitation x temperature cross still
            # spans the whole province; only the "flow" variable is restricted
            # to the drawn area (build_pydeck_layer drops the NaN rows).
            covered_cells = int(joined_grid["flow_value"].notna().sum())
            result["grid_3d"] = joined_grid
            if covered_cells == 0:
                # The drawn polygon is smaller than / misses every climate cell.
                _fail("grid_3d", "El polígono no cubre ninguna celda climática completa; "
                                 "el panel 3D queda con precipitación y temperatura únicamente.")
            else:
                result["has_flow_variable"] = True
                _succeed("grid_3d", f"{len(joined_grid)} celdas en la grilla, "
                                    f"{covered_cells} con flow_value")
        except Exception as grid_error:  # noqa: BLE001 - reported, never fatal for steps 1-2
            result["grid_3d"] = climate_grid
            _fail("grid_3d", f"Error inesperado alineando la acumulación de flujo: {grid_error}")
            logger.exception("polygon analysis | grid_3d raised")

    logger.info("polygon analysis | done | completed=%s failed=%s",
                result["completed"],
                [step for step, msg in result["errors"].items() if msg])
    return result


def render_analysis_status(analysis: dict) -> None:
    """
    Estado del análisis del polígono + caudal pico estimado.

    Un paso fallido nunca oculta a los que sí corrieron: se listan todos los
    errores y después se muestra lo que haya podido calcularse.
    """
    for step, label in ANALYSIS_STEP_LABELS.items():
        error_message = analysis["errors"][step]
        if error_message:
            st.warning(f"⚠️ {label}: {error_message}")

    if "runoff" in analysis["completed"]:
        st.success("Riesgo de escorrentía calculado exitosamente")
        if analysis["runoff"]["partial_coverage"]:
            st.warning(
                "Parte del polígono cae fuera del DEM provincial; "
                "el análisis cubre solo el área con datos."
            )
        st.caption("Cian: píxeles del percentil 95 o superior de acumulación de flujo.")

    # El "I" de este caudal sale de la misma grilla climática que alimenta el
    # panel 3D (ver load_combined_climate_grid), no de una lectura aparte.
    peak_flow_result = analysis["peak_flow"]
    if not peak_flow_result:
        return

    metric_col, breakdown_col = st.columns([1, 2])
    with metric_col:
        st.metric(
            "Caudal pico estimado (Método Racional)",
            f"{peak_flow_result['Q_m3_s']:.2f} m³/s",
        )
    with breakdown_col:
        st.caption(
            f"Q = C × I × A / 360 · C = {peak_flow_result['C']:.2f}"
            f"{' (default, estepa/suelo natural)' if peak_flow_result['C_is_default'] else ''}"
            f" · I = {peak_flow_result['I_mm_h']:.2f} mm/h (CHIRPS, prom. histórico)"
            f" · A = {peak_flow_result['A_km2']:.2f} km²"
        )
        if not peak_flow_result["method_valid_for_basin_size"]:
            st.warning(
                "El área dibujada supera ~2.5 km²: el Método Racional está "
                "pensado para cuencas pequeñas y este valor es solo orientativo."
            )


def render_polygon_3d_panel(analysis_grid, height_variable: str) -> None:
    """
    Panel 3D poblado con la grilla del análisis en curso.

    Precipitación y Temperatura se cruzan sobre la grilla provincial completa;
    solo "Acumulación de flujo" queda acotada al área dibujada, porque es la
    única columna con NaN fuera de ella (build_pydeck_layer descarta esas filas).
    """
    if analysis_grid is None or analysis_grid.empty:
        return

    with st.expander("🧊 Panel 3D (PyDeck)", expanded=True):
        layer_spec = build_pydeck_layer(analysis_grid, height_variable)
        if layer_spec is None:
            st.info(
                "La variable seleccionada no tiene datos para este polígono. "
                "Elegí otra en «Variable → Altura de columnas» del panel lateral."
            )
            return

        basemap = st.selectbox(
            "Mapa base",
            options=list(BASEMAP_TILE_URLS.keys()),
            index=list(BASEMAP_TILE_URLS.keys()).index(DEFAULT_BASEMAP),
            help="Mapa base dibujado debajo de las columnas 3D.",
            key="polygon_basemap_select",
        )
        render_3d_panel_live(
            layer_spec.df, layer_spec.view_state, basemap=basemap,
            tooltip_label=layer_spec.height_label,
            unit=layer_spec.height_unit,
            color_tooltip_label=layer_spec.color_label,
            color_unit=layer_spec.color_unit,
            elevation_scale=layer_spec.elevation_scale,
        )
        if layer_spec.is_single_variable:
            st.caption(
                f"Altura y color de columnas: {layer_spec.height_label} "
                f"({layer_spec.height_unit}) · acotado al polígono dibujado "
                f"({len(layer_spec.df)} celdas), verde→rojo de menor a mayor riesgo."
            )
        else:
            st.caption(
                f"Altura de columnas: {layer_spec.height_label} "
                f"({layer_spec.height_unit}) · "
                f"Color de columnas: {layer_spec.color_label} "
                f"({layer_spec.color_unit}) · grilla provincial completa."
            )


def _format_local_time(iso_timestamp: str) -> str:
    return datetime.fromisoformat(iso_timestamp).astimezone().strftime("%d/%m/%Y %H:%M")


def render_dem_panel():
    """
    Sidebar controls for the provincial DEM. Earth Engine is only contacted
    when a button is clicked, never on a regular rerun; the status shown here
    comes from the local cache alone, so the app works offline.
    """
    with st.sidebar:
        st.markdown("---")
        st.subheader("⛰️ DEM Provincial")
        status = get_provincial_dem_status()
        pending_task = status["pending_task"]

        if status["cached"]:
            source_txt = f" · {status['source']}" if status["source"] else ""
            st.success(f"DEM cacheado ({_format_local_time(status['cached_at'])}{source_txt})")
        else:
            st.warning("No hay DEM provincial cacheado: el análisis de escorrentía no está disponible.")

        if pending_task:
            st.info(
                "Export a Drive pendiente, revisar en unos minutos "
                f"(iniciado {_format_local_time(pending_task['started_at'])})."
            )
            if st.button("Verificar export"):
                try:
                    with st.spinner("Consultando el estado del export en Earth Engine..."):
                        result = check_provincial_dem_export()
                except EarthEngineError as ee_error:
                    st.error(f"⚠️ {ee_error}")
                else:
                    if result["state"] == "COMPLETED":
                        st.rerun()
                    st.info(f"Estado del export: {result['state']}. Vuelva a verificar en unos minutos.")

        source = st.selectbox("Fuente del DEM", options=list(DEM_SOURCES.keys()), index=0)
        force = st.checkbox(
            "Forzar actualización",
            value=False,
            help="Vuelve a descargar el DEM aunque ya exista uno cacheado o un export pendiente.",
        )
        if st.button("Actualizar DEM provincial"):
            if (status["cached"] or pending_task) and not force:
                st.info("Ya hay un DEM cacheado o un export en curso. Marque «Forzar actualización» para volver a pedirlo.")
            else:
                try:
                    with st.spinner("Descargando de Earth Engine..."):
                        update_provincial_dem(source=source, force=force)
                except EarthEngineError as ee_error:
                    fallback_txt = " Se sigue usando el último DEM cacheado." if status["cached"] else ""
                    st.error(f"⚠️ {ee_error}{fallback_txt}")
                else:
                    st.rerun()


@st.cache_data(show_spinner=False)
def calculate_accurate_metrics() -> tuple[float | None, float | None]:
    """
    Total water-body area (km²) and river length (km), both computed on the
    metric CRS (EPSG:5343) rather than read off the raw GeoJSON — reprojecting
    first is what makes these figures accurate instead of a degrees-based
    approximation. Returns raw floats (`None` on a read/compute failure) so
    every call site formats them for its own context instead of inheriting a
    hard-coded label/unit string.
    """
    area_km2 = None
    try:
        file_cuerpos = "cuerpos_agua_neuquen.geojson"
        if os.path.exists(file_cuerpos):
            gdf_cuerpos = gpd.read_file(file_cuerpos)
            if not gdf_cuerpos.empty:
                gdf_cuerpos_proj = gdf_cuerpos.to_crs(epsg=5343)
                area_km2 = float(gdf_cuerpos_proj.geometry.area.sum() / 1e6)
    except Exception:
        area_km2 = None

    length_km = None
    try:
        file_rios = "rios_neuquen.geojson"
        if os.path.exists(file_rios):
            gdf_rios = gpd.read_file(file_rios)
            if not gdf_rios.empty:
                gdf_rios_proj = gdf_rios.to_crs(epsg=5343)
                length_km = float(gdf_rios_proj.geometry.length.sum() / 1000)
    except Exception:
        length_km = None

    return area_km2, length_km

# 1. Configuración principal de la aplicación Streamlit
st.set_page_config(
    page_title="Monitoreo Cuenca Neuquén",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Estilos CSS: tema claro/técnico. La tipografía serif (Source Serif 4,
# estilo Claude) se define a nivel de tema en .streamlit/config.toml
# (theme.font / theme.headingFont), que es lo que realmente llega a todos
# los widgets nativos de Streamlit (botones, inputs, tablas, sidebar) — el
# CSS inyectado acá solo alcanza el contenedor principal, por eso se
# mantiene como refuerzo/fallback. JetBrains Mono se reserva para lecturas
# numéricas/técnicas (KPIs, chips de metadata); el contraste entre ambas
# tipografías es lo que le da el aspecto de instrumento técnico en vez de
# una página de texto plano.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"], .stApp, .stApp * {
        font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    }

    .main .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    /* Encabezado: título + bajada + fila de chips de metadata técnica
       (fuentes de datos, CRS, proyecto de Earth Engine). */
    .app-header h1 {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.3rem 0;
    }
    .app-subtitle {
        color: #5b6b7a;
        font-size: 0.95rem;
        max-width: 800px;
        line-height: 1.55;
        margin-bottom: 0.6rem;
    }
    .app-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1.1rem;
    }
    .tech-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #0f766e;
        background: #e6f4f2;
        border: 1px solid #bfe4df;
        border-radius: 999px;
        padding: 3px 11px;
        white-space: nowrap;
    }

    /* Tarjetas KPI: valor en monospace (lectura tipo instrumento), etiqueta
       muda en versalitas pequeñas. */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e3e8ec;
        border-radius: 12px;
        padding: 14px 18px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 30, 0.06);
    }
    .metric-card h4 {
        margin: 0;
        color: #5b6b7a;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        color: #0f766e;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card small {
        display: block;
        margin-top: 2px;
        color: #8c98a4;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Expanders con el mismo tratamiento de tarjeta que los KPIs, para que
       los paneles 3D/tablas se sientan parte del mismo sistema visual. */
    div[data-testid="stExpander"] {
        border: 1px solid #e3e8ec;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 30, 0.05);
    }

    hr {
        margin: 0.9rem 0;
        border-color: #e3e8ec !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 0.92rem;
        letter-spacing: 0.01em;
    }
    </style>
""", unsafe_allow_html=True)


def _render_header():
    """
    Título + bajada + fila de chips técnicos (fuentes de datos, CRS de
    referencia, proyecto de Earth Engine). Reemplaza el `st.title` simple
    original: da contexto técnico de un vistazo sin ocupar una sección
    aparte, en línea con el resto del dashboard.
    """
    st.markdown(
        f"""
        <div class="app-header">
          <h1>🌊 Monitoreo Ambiental — Cuenca Neuquén</h1>
          <div class="app-subtitle">
            Plataforma de visualización geoespacial e hidrológica: precipitación (CHIRPS) y
            temperatura (ERA5-Land) vía Google Earth Engine, análisis de escorrentía sobre DEM
            provincial, y capas vectoriales de cuerpos de agua, ríos y estaciones meteorológicas
            de la <strong>Cuenca del Neuquén</strong>.
          </div>
          <div class="app-meta-row">
            <span class="tech-chip">GEE · {EE_PROJECT_ID}</span>
            <span class="tech-chip">CRS vis · EPSG:4326</span>
            <span class="tech-chip">CRS cálculo · EPSG:5343</span>
            <span class="tech-chip">DEM · AW3D30 / SRTM</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Header (título + bajada + chips) va primero: es lo primero que se
    # renderiza en la página, antes del sidebar y del panel DEM.
    _render_header()

    # Renderizar el menú lateral (Sidebar) y obtener filtros
    controls = render_sidebar()

    # Métricas espaciales exactas (EPSG:5343), compartidas entre el sidebar y
    # el KPI row principal más abajo — se calculan una sola vez acá.
    area_km2, length_km = calculate_accurate_metrics()
    with st.sidebar:
        st.markdown("---")
        st.subheader("📏 Métricas Espaciales (EPSG:5343)")
        st.metric(
            label="Cuerpos de Agua",
            value=f"{area_km2:,.2f} km²" if area_km2 is not None else "N/D",
        )
        st.metric(
            label="Ríos",
            value=f"{length_km:,.2f} km" if length_km is not None else "N/D",
        )

    render_dem_panel()

    # Carga de datos GeoJSON con almacenamiento en caché (@st.cache_data)
    with st.spinner("Cargando capas geográficas de la Cuenca Neuquén..."):
        gdf_cuenca = load_cuenca_geojson("cuenca_neuquen.geojson")
        gdf_lagos = load_additional_layer("lagos_neuquen.geojson")
        gdf_rios = load_additional_layer("rios_neuquen.geojson")
        gdf_puntos = load_additional_layer("puntos_neuquen.geojson")

    # KPI row: mismas .metric-card definidas en el CSS de arriba, con datos
    # reales en vez de placeholders — área/longitud recién calculadas, CRS
    # nativo del GeoJSON cargado, y el filtro de estación activo.
    n_features = len(gdf_cuenca) if gdf_cuenca is not None else 0
    crs_name = str(gdf_cuenca.crs) if gdf_cuenca is not None and gdf_cuenca.crs else "N/D"
    filtro_txt = controls["selected_estacion"]
    if len(filtro_txt) > 24:
        filtro_txt = filtro_txt[:22] + "…"

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(
            '<div class="metric-card"><h4>Cuerpos de agua</h4>'
            f'<p>{area_km2:,.1f}<small>km²</small></p></div>'
            if area_km2 is not None
            else '<div class="metric-card"><h4>Cuerpos de agua</h4><p>N/D</p></div>',
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            '<div class="metric-card"><h4>Longitud de ríos</h4>'
            f'<p>{length_km:,.1f}<small>km</small></p></div>'
            if length_km is not None
            else '<div class="metric-card"><h4>Longitud de ríos</h4><p>N/D</p></div>',
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            f'<div class="metric-card"><h4>Geometrías de cuenca</h4>'
            f'<p>{n_features}<small>{crs_name}</small></p></div>',
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            f'<div class="metric-card"><h4>Filtro de estación</h4>'
            f'<p style="font-size:1rem;">{filtro_txt}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Renderizado del mapa interactivo con streamlit-folium
    if gdf_cuenca is not None and not gdf_cuenca.empty:
        st.subheader("📍 Mapa Interactivo")

        # Slots FIJOS para los dos iframes de la página (el mapa de Folium y el
        # panel 3D de deck.gl), reservados acá arriba antes de cualquier trabajo
        # condicional.
        #
        # Por qué: Streamlit identifica cada componente por su posición en el
        # árbol de elementos (su "delta path"), no por el orden en que se
        # escribe el código. Todo lo que se renderiza condicionalmente más
        # abajo — los st.spinner del cálculo, los st.error de la capa
        # climática, los st.warning de estado del análisis — aparece en unos
        # reruns y no en otros, y corre la posición de todo lo que viene
        # después. Cuando el iframe del mapa cambia de posición, el frontend
        # desmonta el que ya estaba y monta uno nuevo: eso es exactamente el
        # "el mapa 2D desaparece después de cualquier cálculo".
        #
        # Un st.container() reserva su lugar en el árbol en el momento en que
        # se crea, y todo lo que se escriba después con `with` cae en ese lugar
        # reservado. Así la posición de ambos iframes queda fija sin importar
        # cuántos elementos transitorios aparezcan en el medio.
        map_container = st.container()
        analysis_results_container = st.container()
        panel_3d_container = st.container()

        # st_folium guarda el resultado del dibujo en session_state antes de
        # iniciar este rerun, permitiendo calcular la capa sin reiniciar el mapa.
        last_drawing = st.session_state.get("folium_map_component", {}).get(
            "last_active_drawing"
        )

        if last_drawing:
            geometry = last_drawing.get("geometry", last_drawing)
            # The DEM's mtime is part of the key so a refreshed DEM recomputes the overlay.
            dem_mtime = (
                os.path.getmtime(PROVINCIAL_DEM_PATH)
                if os.path.exists(PROVINCIAL_DEM_PATH)
                else None
            )
            drawing_id = json.dumps({"geometry": geometry, "dem": dem_mtime}, sort_keys=True)

            if (
                st.session_state.get("dem_drawing_id") != drawing_id
                or "runoff_overlay" not in st.session_state
                or st.session_state.get("runoff_overlay_version")
                != RUNOFF_OVERLAY_VERSION
            ):
                st.session_state.dem_drawing_id = drawing_id

                # UN solo flujo por polígono dibujado: Pysheds -> Método Racional
                # -> grilla 3D, en ese orden, leyendo todos la MISMA grilla
                # climática cacheada (load_combined_climate_grid). Antes esto
                # eran dos bloques separados que vectorizaban CHIRPS por su
                # cuenta y podían quedar desincronizados entre sí.
                climate_grid = None
                climate_grid_error = None
                try:
                    with st.spinner("Cargando grilla climática (CHIRPS × ERA5-Land)..."):
                        climate_grid = load_combined_climate_grid()
                except EarthEngineError as ee_error:
                    climate_grid_error = (
                        f"No se pudo obtener precipitación/temperatura de Earth Engine: {ee_error}"
                    )
                    logger.warning("polygon analysis | climate grid unavailable | %s", ee_error)
                except Exception as grid_error:  # noqa: BLE001 - Pysheds debe poder seguir igual
                    climate_grid_error = f"Error inesperado cargando la grilla climática: {grid_error}"
                    logger.exception("polygon analysis | climate grid raised")

                with st.spinner("Analizando el polígono (escorrentía, caudal y grilla 3D)..."):
                    analysis = run_polygon_analysis(
                        geometry, PROVINCIAL_DEM_PATH, climate_grid
                    )

                # Un fallo al traer la grilla climática es más informativo que el
                # "no hay grilla disponible" genérico que reporta el orquestador.
                if climate_grid_error:
                    for step in ("peak_flow", "grid_3d"):
                        analysis["errors"][step] = climate_grid_error

                st.session_state.polygon_analysis = analysis

                # La capa 2D de Folium sigue leyendo estas claves, así que se
                # derivan del resultado unificado en vez de calcularse aparte.
                runoff = analysis["runoff"]
                if runoff is not None:
                    st.session_state.runoff_overlay = runoff["overlay"]
                    st.session_state.runoff_overlay_bounds = runoff["bounds"]
                    st.session_state.runoff_overlay_version = RUNOFF_OVERLAY_VERSION
                    st.session_state.dem_partial_coverage = runoff["partial_coverage"]
                else:
                    st.session_state.pop("runoff_overlay", None)
                    st.session_state.pop("runoff_overlay_bounds", None)
                st.session_state.dem_crop_error = analysis["errors"]["runoff"]

                # Gate del selector del sidebar: la opción "flow" solo existe si
                # esta iteración realmente produjo celdas con flow_value.
                st.session_state.flow_variable_available = analysis["has_flow_variable"]

        # Crear una nueva instancia es necesario porque st_folium procesa el mapa
        # al renderizarlo. Se conserva el estado aleatorio para que la capa de ríos
        # no cambie entre reruns y el componente pueda conservarse en el frontend.
        random_state = np.random.get_state()
        np.random.seed(42)
        try:
            cuenca_map = create_cuenca_map(
                gdf_cuenca, gdf_rios, gdf_lagos, gdf_puntos, controls
            )
        finally:
            np.random.set_state(random_state)

        # Capa climática (precipitación CHIRPS o temperatura ERA5-Land, según
        # "source" en CLIMATE_LAYER_CONFIG, descargadas vía Google Earth Engine),
        # clasificada y vectorizada desde el raster resultante. Se descarga/procesa
        # una sola vez aquí y el mismo GeoDataFrame + colormap se reutiliza tanto
        # para el GeoJson 2D como para el panel 3D de PyDeck más abajo, evitando
        # duplicar la lógica de descarga/clasificación.
        selected_climate_layer = controls.get("selected_climate_layer", "Ninguna")
        gdf_climate = None
        climate_colormap = None
        if selected_climate_layer != "Ninguna":
            climate_config = CLIMATE_LAYER_CONFIG[selected_climate_layer]
            try:
                # fetch_and_cache_chirps/fetch_and_cache_era5_temperature ya evitan
                # volver a pedirle a Earth Engine si ese rango de fechas/agregación
                # ya está en ./data/cache/.
                if climate_config["source"] == "era5":
                    raster_path = fetch_and_cache_era5_temperature(
                        geojson_path=CUENCA_GEOJSON_PATH,
                        start_date=climate_config["start_date"],
                        end_date=climate_config["end_date"],
                        aggregation=climate_config["aggregation"],
                    )
                else:
                    raster_path = fetch_and_cache_chirps(
                        geojson_path=CUENCA_GEOJSON_PATH,
                        start_date=climate_config["start_date"],
                        end_date=climate_config["end_date"],
                        aggregation=climate_config["aggregation"],
                    )
                gdf_climate = raster_to_classified_gdf(raster_path, n_bins=8, method="quantile")
            except EarthEngineError as ee_error:
                st.error(f"⚠️ No se pudo obtener la capa climática de Earth Engine: {ee_error}")
            except Exception as processing_error:
                st.error(f"❌ Error inesperado procesando la capa climática: {processing_error}")

            if gdf_climate is not None and not gdf_climate.empty:
                climate_colormap = add_climate_layer_to_map(
                    cuenca_map,
                    gdf_climate,
                    value_column="value",
                    layer_name=selected_climate_layer,
                    color_scheme=climate_config["color_scheme"],
                    tooltip_label=f"{climate_config['value_label']} ({climate_config['unit']}):",
                )
            elif gdf_climate is not None:
                st.warning(f"⚠️ La capa climática '{selected_climate_layer}' no generó datos vectorizables.")

        # Permitir seleccionar un área de recorte, limitando el dibujo a polígonos y rectángulos.
        # Mostrar el límite del DEM para guiar la selección de un área con solapamiento.
        dem_bounds_wgs84 = get_dem_bounds_wgs84(PROVINCIAL_DEM_PATH)
        if dem_bounds_wgs84 is not None:
            folium.Rectangle(
                bounds=[
                    [dem_bounds_wgs84[1], dem_bounds_wgs84[0]],
                    [dem_bounds_wgs84[3], dem_bounds_wgs84[2]],
                ],
                color="#ffb703",
                weight=2,
                fill=False,
                tooltip="Cobertura del DEM provincial",
            ).add_to(cuenca_map)

        # Añadir el resultado como una capa dinámica evita reconstruir el mapa.
        runoff_feature_group = None
        if "runoff_overlay" in st.session_state:
            runoff_feature_group = folium.FeatureGroup(
                name="Zonas de alta acumulación"
            )
            folium.raster_layers.ImageOverlay(
                image=st.session_state.runoff_overlay,
                bounds=st.session_state.runoff_overlay_bounds,
                opacity=1,
                origin="upper",
                name="Zonas de alta acumulación",
            ).add_to(runoff_feature_group)

        Draw(
            draw_options={
                "polyline": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": {
                    "shapeOptions": {
                        "color": "#3388ff",
                        "weight": 3,
                        "fillOpacity": 0,
                    },
                },
                "polygon": {
                    "allowIntersection": False,
                    "shapeOptions": {
                        "color": "#3388ff",
                        "weight": 3,
                        "fillOpacity": 0,
                    },
                },
            },
            # El cian queda reservado para el resultado de escorrentía.
            edit_options={"edit": True, "remove": True},
        ).add_to(cuenca_map)

        # La capa dinámica se inserta sin cambiar el script base del mapa.
        # Va dentro de map_container (reservado arriba) para que el iframe
        # conserve siempre la misma posición en el árbol de elementos.
        with map_container:
            st_folium(
                cuenca_map,
                width=1200,
                height=600,
                key="folium_map_component",
                returned_objects=["last_active_drawing"],
                feature_group_to_add=runoff_feature_group,
            )

        # Resultados del análisis unificado del polígono, todos provenientes de
        # la MISMA llamada a run_polygon_analysis(): estado por paso, caudal
        # estimado y panel 3D, en ese orden y sin que el usuario tenga que
        # accionar nada entre uno y otro.
        analysis = st.session_state.get("polygon_analysis") if last_drawing else None
        if analysis:
            # Cada sección va a su slot reservado: los mensajes de estado varían
            # en cantidad según qué pasos fallaron, y sin slots fijos esa
            # variación correría la posición del iframe del panel 3D.
            with analysis_results_container:
                render_analysis_status(analysis)
            with panel_3d_container:
                render_polygon_3d_panel(
                    analysis["grid_3d"], controls.get("cross_height_var", "precip")
                )

        # Panel 3D: reacciona al mismo selectbox de capa climática que la capa 2D de arriba,
        # reutilizando el mismo GeoDataFrame clasificado y el mismo colormap (sin duplicar
        # la descarga/clasificación de datos, ni volver a llamar a Earth Engine).
        if selected_climate_layer != "Ninguna":
            with st.expander(f"🧊 Vista 3D - {selected_climate_layer} (PyDeck)", expanded=True):
                if gdf_climate is not None and not gdf_climate.empty and climate_colormap is not None:
                    # Rotación, inclinación y escala de elevación viven como sliders HTML
                    # DENTRO del panel embebido (ver render_3d_panel_live): moverlos actualiza
                    # solo el canvas WebGL vía deck.gl `setProps`, sin rerun de Streamlit ni
                    # recarga del resto del dashboard. Solo el mapa base queda como control de
                    # Streamlit porque cambiarlo sí implica reconstruir el panel embebido.
                    basemap = st.selectbox(
                        "Mapa base",
                        options=list(BASEMAP_TILE_URLS.keys()),
                        index=list(BASEMAP_TILE_URLS.keys()).index(DEFAULT_BASEMAP),
                        help="Mapa base de CARTO dibujado debajo de las columnas 3D.",
                    )
                    climate_df_3d = gdf_to_pydeck_df(gdf_climate, "value", climate_colormap)
                    view_state_3d = build_isometric_view_state(gdf_climate)
                    render_3d_panel_live(
                        climate_df_3d, view_state_3d, basemap=basemap,
                        tooltip_label=climate_config["value_label"],
                        unit=climate_config["unit"],
                    )
                else:
                    st.info("No hay datos climáticos disponibles para la vista 3D.")

        # Cruce 3D provincial (precipitación x temperatura), SIN polígono dibujado.
        # Cuando sí hay un polígono, el panel 3D ya se renderiza arriba con la
        # grilla de esa iteración (incluida la acumulación de flujo), así que
        # este bloque se omite para no mostrar dos paneles con la misma variable.
        if controls.get("enable_climate_cross") and not analysis:
            with st.expander("🧊 Cruce 3D - Precipitación x Temperatura", expanded=True):
                gdf_combined_climate = None
                try:
                    # La MISMA lectura cacheada que usa el análisis del polígono
                    # (load_combined_climate_grid), no una carga paralela.
                    gdf_combined_climate = load_combined_climate_grid()
                except EarthEngineError as ee_error:
                    st.error(f"⚠️ No se pudo obtener precipitación/temperatura de Earth Engine: {ee_error}")
                except Exception as processing_error:
                    st.error(f"❌ Error inesperado alineando las grillas climáticas: {processing_error}")

                if gdf_combined_climate is not None and not gdf_combined_climate.empty:
                    # Sin polígono no existe la variable "flow", así que el
                    # selector del sidebar solo ofrece precip/temp acá.
                    cross_height_var = controls.get("cross_height_var", "precip")
                    layer_spec = build_pydeck_layer(gdf_combined_climate, cross_height_var)

                    if layer_spec is None:
                        st.info(
                            "La variable seleccionada no tiene datos en la grilla provincial. "
                            "Dibujá un polígono para habilitar la acumulación de flujo."
                        )
                    else:
                        cross_basemap = st.selectbox(
                            "Mapa base",
                            options=list(BASEMAP_TILE_URLS.keys()),
                            index=list(BASEMAP_TILE_URLS.keys()).index(DEFAULT_BASEMAP),
                            help="Mapa base de CARTO dibujado debajo de las columnas 3D.",
                            key="cross_basemap_select",
                        )
                        render_3d_panel_live(
                            layer_spec.df, layer_spec.view_state, basemap=cross_basemap,
                            tooltip_label=layer_spec.height_label,
                            unit=layer_spec.height_unit,
                            color_tooltip_label=layer_spec.color_label,
                            color_unit=layer_spec.color_unit,
                            elevation_scale=layer_spec.elevation_scale,
                        )
                        st.caption(
                            f"Altura de columnas: {layer_spec.height_label} "
                            f"({layer_spec.height_unit})"
                            if layer_spec.is_single_variable else
                            f"Altura de columnas: {layer_spec.height_label} "
                            f"({layer_spec.height_unit}) · "
                            f"Color de columnas: {layer_spec.color_label} "
                            f"({layer_spec.color_unit})."
                        )
                elif gdf_combined_climate is not None:
                    st.info(
                        "No hay celdas con precipitación y temperatura válidas simultáneamente "
                        "para cruzar."
                    )

        # Tabla de atributos expandible
        with st.expander("📄 Ver Atributos de los Polígonos de la Cuenca"):
            df_display = gdf_cuenca.drop(columns=["geometry"], errors="ignore")
            st.dataframe(df_display, use_container_width=True)

        # Ficha técnica: fuentes de datos, CRS y ubicación del caché — documenta
        # de dónde sale cada capa sin tener que leer el código fuente.
        with st.expander("ℹ️ Fuentes de datos y metodología"):
            st.markdown(
                f"""
| Capa | Fuente | Resolución nativa | Notas |
|---|---|---|---|
| Precipitación | CHIRPS Daily (`UCSB-CHG/CHIRPS/DAILY`) | ~5.5 km | Acumulada o promedio anual 2016–2026, vía Earth Engine. |
| Temperatura | ERA5-Land Monthly (`ECMWF/ERA5_LAND/MONTHLY_AGGR`) | ~9–11 km | Banda `temperature_2m`, convertida de Kelvin a °C. |
| DEM provincial | AW3D30 (JAXA) / SRTM (USGS) | 30 m | Mosaico único, usado para el análisis de escorrentía. |
| Límite provincial | FAO GAUL 2015, nivel 1 | — | Recorte oficial de Neuquén; con fallback al polígono local. |
| Cuerpos de agua / ríos / puntos | IGN (`*_neuquen.geojson`) | — | 2.161 cuerpos de agua y ríos digitalizados. |

**CRS**: EPSG:4326 para visualización (Folium/PyDeck), EPSG:5343 (POSGAR 2007 / Argentina 1) para
áreas, longitudes y el pipeline hidrológico D8, de modo que las celdas del raster sean cuadradas
en metros.

**Proyecto de Earth Engine**: `{EE_PROJECT_ID}`. Los rasters descargados se cachean en
`./data/cache/`, por rango de fechas/agregación — una misma capa climática no vuelve a pedirse
a Earth Engine si ya está en disco.

**Cruce Precipitación × Temperatura**: CHIRPS (~5.5 km) y ERA5-Land (~9–11 km) tienen resoluciones
nativas distintas; el panel 3D combinado las alinea celda a celda reproyectando ambas a EPSG:5343
y resampleando la más gruesa (ERA5-Land) sobre la grilla de la más fina (CHIRPS) con interpolación
bilineal antes de vectorizar.
                """
            )
    else:
        st.info("💡 Asegúrese de colocar el archivo **cuenca_neuquen.geojson** en el directorio raíz de la aplicación para visualizar la cuenca.")




if __name__ == "__main__":
    main()
