import os
import streamlit as st
import geopandas as gpd

@st.cache_data(show_spinner=False)
def load_cuenca_geojson(file_path: str = "cuenca_neuquen.geojson") -> gpd.GeoDataFrame | None:
    """
    Carga el archivo GeoJSON de la Cuenca del Neuquén utilizando el caché de Streamlit.
    Maneja adecuadamente las excepciones de archivo no encontrado y errores de lectura.
    """
    if not os.path.exists(file_path):
        st.error(f"⚠️ Error: No se encontró el archivo de cuenca en '{file_path}'. Por favor verifique la ubicación.")
        return None

    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            st.warning(f"⚠️ El archivo '{file_path}' está vacío.")
            return None

        # Asegurar proyección EPSG:4326 para mapas web (Folium)
        if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Simplificar geometrías para optimizar renderizado interactivo en el navegador
        gdf["geometry"] = gdf.geometry.simplify(tolerance=0.0005, preserve_topology=True)

        return gdf
    except FileNotFoundError:
        st.error(f"⚠️ Error: No se pudo localizar el archivo '{file_path}'.")
        return None
    except Exception as e:
        st.error(f"❌ Ocurrió un error inesperado al leer '{file_path}': {e}")
        return None

@st.cache_data(show_spinner=False)
def load_additional_layer(file_path: str) -> gpd.GeoDataFrame | None:
    """
    Carga opcional de capas vectoriales adicionales (como ríos o lagos) con caché.
    """
    if not os.path.exists(file_path):
        return None

    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            return None

        if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        gdf["geometry"] = gdf.geometry.simplify(tolerance=0.0005, preserve_topology=True)
        return gdf
    except Exception:
        return None
