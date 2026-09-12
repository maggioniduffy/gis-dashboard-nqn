import os

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def generate_mock_flow_data() -> pd.DataFrame:
    """Generates a mock Pandas DataFrame containing 1 year of daily flow rate data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Realistic seasonal fluctuations
    time_idx = np.arange(365)
    # Peak flow in late winter / spring
    seasonality = np.sin(2 * np.pi * (time_idx - 150) / 365)
    
    # Base flows and amplitudes
    limay_flow = 300 + 150 * seasonality + np.random.normal(0, 20, 365)
    neuquen_flow = 150 + 80 * seasonality + np.random.normal(0, 15, 365)
    
    df = pd.DataFrame({
        'Fecha': dates,
        'Limay - Confluencia': np.maximum(limay_flow, 10),
        'Neuquén - Paso de los Indios': np.maximum(neuquen_flow, 5)
    })
    
    return df

@st.cache_data(show_spinner=False)
def load_cuenca_geojson(file_path: str = "cuenca_neuquen.geojson") -> gpd.GeoDataFrame | None:
    """
    Carga el archivo GeoJSON de la Cuenca del Neuquén utilizando el caché de Streamlit.
    Maneja adecuadamente las excepciones de archivo no encontrado y errores de lectura.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: No se encontró el archivo de cuenca en '{file_path}'. Por favor verifique la ubicación.")
        return None

    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            st.warning(f"El archivo '{file_path}' está vacío.")
            return None

        # Asegurar proyección EPSG:4326 para mapas web (Folium)
        if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Simplificar geometrías para optimizar renderizado interactivo en el navegador
        gdf["geometry"] = gdf.geometry.simplify(tolerance=0.0005, preserve_topology=True)

        return gdf
    except FileNotFoundError:
        st.error(f"Error: No se pudo localizar el archivo '{file_path}'.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al leer '{file_path}': {e}")
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
