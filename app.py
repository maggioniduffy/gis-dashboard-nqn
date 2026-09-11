import os
import json
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
    PROVINCIAL_DEM_PATH,
    EarthEngineError,
    check_provincial_dem_export,
    fetch_and_cache_chirps,
    fetch_and_cache_era5_temperature,
    get_provincial_dem_status,
    update_provincial_dem,
)
from modules.folium_layers import add_climate_layer_to_map, build_colormap
from modules.hydrology import DemCoverageError, compute_runoff_overlay, get_dem_bounds_wgs84
from modules.map_builder import create_cuenca_map
from modules.pydeck_layers import (
    BASEMAP_TILE_URLS,
    DEFAULT_BASEMAP,
    build_isometric_view_state,
    gdf_to_pydeck_df,
    gdf_to_pydeck_df_dual,
    render_3d_panel_live,
)
from modules.raster_processing import align_climate_grids, raster_to_classified_gdf
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
def calculate_accurate_metrics():
    area_str = "N/A"
    length_str = "N/A"
    
    try:
        file_cuerpos = "cuerpos_agua_neuquen.geojson"
        if os.path.exists(file_cuerpos):
            gdf_cuerpos = gpd.read_file(file_cuerpos)
            if not gdf_cuerpos.empty:
                gdf_cuerpos_proj = gdf_cuerpos.to_crs(epsg=5343)
                area_km2 = gdf_cuerpos_proj.geometry.area.sum() / 1e6
                area_str = f"Total Area: {area_km2:,.2f} km²"
    except Exception:
        area_str = "Error calculating area"

    try:
        file_rios = "rios_neuquen.geojson"
        if os.path.exists(file_rios):
            gdf_rios = gpd.read_file(file_rios)
            if not gdf_rios.empty:
                gdf_rios_proj = gdf_rios.to_crs(epsg=5343)
                length_km = gdf_rios_proj.geometry.length.sum() / 1000
                length_str = f"Total Length: {length_km:,.2f} km"
    except Exception:
        length_str = "Error calculating length"
        
    return area_str, length_str

# 1. Configuración principal de la aplicación Streamlit
st.set_page_config(
    page_title="Monitoreo Cuenca Neuquén",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Estilos CSS personalizados para tarjetas e interfaz
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }
    .metric-card h4 {
        margin: 0;
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 6px 0 0 0;
        color: #00c896;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Renderizar el menú lateral (Sidebar) y obtener filtros
    controls = render_sidebar()

    # Añadir métricas exactas al sidebar
    area_str, length_str = calculate_accurate_metrics()
    with st.sidebar:
        st.markdown("---")
        st.subheader("📏 Métricas Espaciales (EPSG:5343)")
        st.metric(label="Cuerpos de Agua", value=area_str)
        st.metric(label="Ríos", value=length_str)

    render_dem_panel()

    # Título principal y descripción del Dashboard
    st.title("🌊 Dashboard de Monitoreo Ambiental - Cuenca Neuquén")
    st.markdown(
        "Plataforma interactiva para la visualización de capas geográficas, análisis hidrológico "
        "y monitoreo de estaciones meteorológicas en la **Cuenca del Neuquén**."
    )

    # Carga de datos GeoJSON con almacenamiento en caché (@st.cache_data)
    with st.spinner("Cargando capas geográficas de la Cuenca Neuquén..."):
        gdf_cuenca = load_cuenca_geojson("cuenca_neuquen.geojson")
        gdf_lagos = load_additional_layer("lagos_neuquen.geojson")
        gdf_rios = load_additional_layer("rios_neuquen.geojson")
        gdf_puntos = load_additional_layer("puntos_neuquen.geojson")

    # Métricas del Dashboard
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     n_features = len(gdf_cuenca) if gdf_cuenca is not None else 0
    #     st.markdown(f'<div class="metric-card"><h4>Cuerpos de Agua</h4><p>{n_features}</p></div>', unsafe_allow_html=True)

    # with col2:
    #     crs_name = str(gdf_cuenca.crs) if gdf_cuenca is not None and gdf_cuenca.crs else "N/A"
    #     st.markdown(f'<div class="metric-card"><h4>Sistema de Coordenadas</h4><p>{crs_name}</p></div>', unsafe_allow_html=True)

    # with col3:
    #     st.markdown('<div class="metric-card"><h4>Estaciones Activas</h4><p>5 Estaciones</p></div>', unsafe_allow_html=True)

    # with col4:
    #     filtro_txt = controls['selected_estacion']
    #     if len(filtro_txt) > 20:
    #         filtro_txt = filtro_txt[:18] + "..."
    #     st.markdown(f'<div class="metric-card"><h4>Filtro de Estación</h4><p style="color: #58a6ff;">{filtro_txt}</p></div>', unsafe_allow_html=True)

    # st.markdown("<br>", unsafe_allow_html=True)

    # Renderizado del mapa interactivo con streamlit-folium
    if gdf_cuenca is not None and not gdf_cuenca.empty:
        st.subheader("📍 Mapa Interactivo")
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
                try:
                    with st.spinner("Calculando riesgo de escorrentía..."):
                        runoff = compute_runoff_overlay(geometry, PROVINCIAL_DEM_PATH)

                    st.session_state.runoff_overlay = runoff["overlay"]
                    st.session_state.runoff_overlay_bounds = runoff["bounds"]
                    st.session_state.runoff_overlay_version = RUNOFF_OVERLAY_VERSION
                    st.session_state.dem_crop_error = None
                    st.session_state.dem_partial_coverage = runoff["partial_coverage"]
                except DemCoverageError as coverage_error:
                    st.session_state.pop("runoff_overlay", None)
                    st.session_state.pop("runoff_overlay_bounds", None)
                    st.session_state.dem_crop_error = str(coverage_error)

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
        st_folium(
            cuenca_map,
            width=1200,
            height=600,
            key="folium_map_component",
            returned_objects=["last_active_drawing"],
            feature_group_to_add=runoff_feature_group,
        )

        if last_drawing:
            if st.session_state.get("dem_crop_error"):
                st.warning(st.session_state.dem_crop_error)
            else:
                st.success("Riesgo de escorrentía calculado exitosamente")
                if st.session_state.get("dem_partial_coverage"):
                    st.warning(
                        "Parte del polígono cae fuera del DEM provincial; "
                        "el análisis cubre solo el área con datos."
                    )
                st.caption(
                    "Cian: píxeles del percentil 95 o superior de acumulación de flujo."
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

        # Panel 3D combinado: cruza precipitación (CHIRPS) y temperatura (ERA5-Land)
        # en una sola grilla vía align_climate_grids(), independiente del selector
        # de "Capa Climática" de arriba — siempre descarga/cachea ambas fuentes
        # (Promedio lluvias + Temperatura promedio) cuando el checkbox está activo.
        if controls.get("enable_climate_cross"):
            with st.expander("🧊 Cruce 3D - Precipitación x Temperatura", expanded=True):
                gdf_combined_climate = None
                try:
                    precip_cross_config = CLIMATE_LAYER_CONFIG["Promedio lluvias"]
                    temp_cross_config = CLIMATE_LAYER_CONFIG["Temperatura promedio 2016-2026"]

                    # Reusa el mismo caché en disco que las capas individuales de
                    # arriba: si "Promedio lluvias" o "Temperatura promedio" ya se
                    # pidieron en esta sesión (o en una anterior), esto no vuelve a
                    # llamar a Earth Engine.
                    precip_raster_path = fetch_and_cache_chirps(
                        geojson_path=CUENCA_GEOJSON_PATH,
                        start_date=precip_cross_config["start_date"],
                        end_date=precip_cross_config["end_date"],
                        aggregation=precip_cross_config["aggregation"],
                    )
                    temp_raster_path = fetch_and_cache_era5_temperature(
                        geojson_path=CUENCA_GEOJSON_PATH,
                        start_date=temp_cross_config["start_date"],
                        end_date=temp_cross_config["end_date"],
                        aggregation=temp_cross_config["aggregation"],
                    )
                    gdf_combined_climate = align_climate_grids(precip_raster_path, temp_raster_path)
                except EarthEngineError as ee_error:
                    st.error(f"⚠️ No se pudo obtener precipitación/temperatura de Earth Engine: {ee_error}")
                except Exception as processing_error:
                    st.error(f"❌ Error inesperado alineando las grillas climáticas: {processing_error}")

                if gdf_combined_climate is not None and not gdf_combined_climate.empty:
                    # cross_height_var decide qué columna maneja la altura; la otra
                    # queda para el color. Cada una con su propia etiqueta/unidad/
                    # esquema de color, igual que sus capas individuales de arriba.
                    if controls.get("cross_height_var") == "Temperatura":
                        height_column, height_label, height_unit = "temp_value", "Temperatura", "°C"
                        color_column, color_label, color_unit, color_scheme = (
                            "precip_value", "Precipitación", "mm", "average",
                        )
                    else:
                        height_column, height_label, height_unit = "precip_value", "Precipitación", "mm"
                        color_column, color_label, color_unit, color_scheme = (
                            "temp_value", "Temperatura", "°C", "temperature",
                        )

                    cross_colormap = build_colormap(
                        gdf_combined_climate, color_column, color_scheme,
                        caption=f"{color_label} ({color_unit})",
                    )
                    cross_basemap = st.selectbox(
                        "Mapa base",
                        options=list(BASEMAP_TILE_URLS.keys()),
                        index=list(BASEMAP_TILE_URLS.keys()).index(DEFAULT_BASEMAP),
                        help="Mapa base de CARTO dibujado debajo de las columnas 3D.",
                        key="cross_basemap_select",
                    )
                    cross_df_3d = gdf_to_pydeck_df_dual(
                        gdf_combined_climate, height_column, color_column, cross_colormap
                    )
                    cross_view_state = build_isometric_view_state(gdf_combined_climate)
                    render_3d_panel_live(
                        cross_df_3d, cross_view_state, basemap=cross_basemap,
                        tooltip_label=height_label, unit=height_unit,
                        color_tooltip_label=color_label, color_unit=color_unit,
                    )
                    st.caption(
                        f"Altura de columnas: {height_label} ({height_unit}) · "
                        f"Color de columnas: {color_label} ({color_unit})."
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
    else:
        st.info("💡 Asegúrese de colocar el archivo **cuenca_neuquen.geojson** en el directorio raíz de la aplicación para visualizar la cuenca.")




if __name__ == "__main__":
    main()
