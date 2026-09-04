import os

import geopandas as gpd
import streamlit as st
from streamlit_folium import folium_static

from modules.data_loader import load_additional_layer, load_cuenca_geojson
from modules.map_builder import create_cuenca_map
from modules.sidebar import render_sidebar

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
        cuenca_map = create_cuenca_map(gdf_cuenca, gdf_rios, gdf_lagos, gdf_puntos, controls)

        # Mostrar mapa Folium de forma directa y fluida
        folium_static(cuenca_map, width=1200, height=600)

        # Tabla de atributos expandible
        with st.expander("📄 Ver Atributos de los Polígonos de la Cuenca"):
            df_display = gdf_cuenca.drop(columns=["geometry"], errors="ignore")
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("💡 Asegúrese de colocar el archivo **cuenca_neuquen.geojson** en el directorio raíz de la aplicación para visualizar la cuenca.")




if __name__ == "__main__":
    main()
