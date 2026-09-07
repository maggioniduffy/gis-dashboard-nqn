import os
import json

import folium
import geopandas as gpd
import numpy as np
import rasterio
import streamlit as st
from folium.plugins import Draw
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, transform_geom
from streamlit_folium import st_folium
from pysheds.grid import Grid
from pysheds.sview import Raster, ViewFinder

from modules.data_loader import load_additional_layer, load_cuenca_geojson
from modules.map_builder import create_cuenca_map
from modules.sidebar import render_sidebar


DEM_PATH = "AP/AP_27847_PLR_F6470_RT1.dem.tif"
RUNOFF_OVERLAY_VERSION = 3


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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_features = len(gdf_cuenca) if gdf_cuenca is not None else 0
        st.markdown(f'<div class="metric-card"><h4>Cuerpos de Agua</h4><p>{n_features}</p></div>', unsafe_allow_html=True)

    with col2:
        crs_name = str(gdf_cuenca.crs) if gdf_cuenca is not None and gdf_cuenca.crs else "N/A"
        st.markdown(f'<div class="metric-card"><h4>Sistema de Coordenadas</h4><p>{crs_name}</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card"><h4>Estaciones Activas</h4><p>5 Estaciones</p></div>', unsafe_allow_html=True)

    with col4:
        filtro_txt = controls['selected_estacion']
        if len(filtro_txt) > 20:
            filtro_txt = filtro_txt[:18] + "..."
        st.markdown(f'<div class="metric-card"><h4>Filtro de Estación</h4><p style="color: #58a6ff;">{filtro_txt}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
            drawing_id = json.dumps(geometry, sort_keys=True)

            if (
                st.session_state.get("dem_drawing_id") != drawing_id
                or "runoff_overlay" not in st.session_state
                or st.session_state.get("runoff_overlay_version")
                != RUNOFF_OVERLAY_VERSION
            ):
                st.session_state.dem_drawing_id = drawing_id
                try:
                    with st.spinner("Calculando riesgo de escorrentía..."):
                        with rasterio.open(DEM_PATH) as dem:
                            dem_geometry = transform_geom("EPSG:4326", dem.crs, geometry)
                            cropped_dem, cropped_transform = mask(
                                dem, [dem_geometry], crop=True, filled=False
                            )
                            dem_crs = dem.crs

                        dem_data = cropped_dem[0].astype(np.float64)
                        valid_cells = ~np.ma.getmaskarray(dem_data)
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
                        flow_accumulation = grid.accumulation(
                            flow_direction, routing="d8"
                        )

                        accumulation_values = np.asarray(flow_accumulation)
                        high_threshold = np.nanpercentile(
                            accumulation_values[valid_cells], 95
                        )
                        high_runoff_mask = valid_cells & (
                            accumulation_values >= high_threshold
                        )

                        display_mask = downsample_runoff_mask(high_runoff_mask)
                        runoff_overlay = np.zeros(
                            (*display_mask.shape, 4), dtype=np.uint8
                        )
                        runoff_overlay[display_mask] = [0, 220, 255, 210]

                        west, south, east, north = array_bounds(
                            high_runoff_mask.shape[0],
                            high_runoff_mask.shape[1],
                            cropped_transform,
                        )
                        overlay_bounds = transform_bounds(
                            dem_crs, "EPSG:4326", west, south, east, north
                        )

                    st.session_state.runoff_overlay = runoff_overlay
                    st.session_state.runoff_overlay_bounds = [
                        [overlay_bounds[1], overlay_bounds[0]],
                        [overlay_bounds[3], overlay_bounds[2]],
                    ]
                    st.session_state.runoff_overlay_version = RUNOFF_OVERLAY_VERSION
                    st.session_state.dem_crop_error = None
                except ValueError:
                    st.session_state.pop("runoff_overlay", None)
                    st.session_state.pop("runoff_overlay_bounds", None)
                    st.session_state.dem_crop_error = (
                        "El área seleccionada no se superpone con el DEM. "
                        "Dibuje dentro del rectángulo amarillo (Cobertura del DEM)."
                    )

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

        # Permitir seleccionar un área de recorte, limitando el dibujo a polígonos y rectángulos.
        # Mostrar el límite del DEM para guiar la selección de un área con solapamiento.
        with rasterio.open(DEM_PATH) as dem:
            dem_bounds_wgs84 = transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)
        folium.Rectangle(
            bounds=[
                [dem_bounds_wgs84[1], dem_bounds_wgs84[0]],
                [dem_bounds_wgs84[3], dem_bounds_wgs84[2]],
            ],
            color="#ffb703",
            weight=2,
            fill=False,
            tooltip="Cobertura del DEM",
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
                st.caption(
                    "Cian: píxeles del percentil 95 o superior de acumulación de flujo."
                )

        # Tabla de atributos expandible
        with st.expander("📄 Ver Atributos de los Polígonos de la Cuenca"):
            df_display = gdf_cuenca.drop(columns=["geometry"], errors="ignore")
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("💡 Asegúrese de colocar el archivo **cuenca_neuquen.geojson** en el directorio raíz de la aplicación para visualizar la cuenca.")




if __name__ == "__main__":
    main()
