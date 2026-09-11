import streamlit as st

def render_sidebar():
    """
    Renderiza el menú lateral (Sidebar) con títulos, filtros y controles de capas.
    """
    with st.sidebar:
        st.title("Monitoreo Cuenca Neuquén")
        st.markdown("---")
        
        st.subheader("🌦️ Filtros")
        
        # Filtro requerido para Estaciones Meteorológicas
        estaciones_options = [
            "Todas las estaciones",
            "Estación Chos Malal (Norte)",
            "Estación Zapala (Centro)",
            "Estación Neuquén Capital (Confluencia)",
            "Estación San Martín de los Andes (Sur)",
            "Estación Añelo (Vaca Muerta)"
        ]
        
        selected_estacion = st.selectbox(
            label="Estaciones Meteorológicas",
            options=estaciones_options,
            index=0,
            help="Filtre las estaciones meteorológicas desplegadas en la cuenca."
        )

        st.markdown("---")

        st.subheader("🌦️ Capa Climática")
        climate_layer_options = [
            "Ninguna",
            "Precipitación acumulada 2016-2026",
            "Promedio lluvias",
            "Temperatura promedio 2016-2026",
        ]
        selected_climate_layer = st.selectbox(
            label="Capa climática (CHIRPS / ERA5-Land vía Earth Engine)",
            options=climate_layer_options,
            index=0,
            help="Descarga y superpone una capa de precipitación (CHIRPS) o temperatura "
                 "(ERA5-Land) clasificada en el mapa 2D y en el panel 3D.",
        )

        st.markdown("---")

        st.subheader("🧊 Cruce 3D Precipitación x Temperatura")
        enable_climate_cross = st.checkbox(
            "Cruzar precipitación y temperatura en el panel 3D",
            value=False,
            help="Descarga precipitación (CHIRPS) y temperatura (ERA5-Land) para el mismo "
                 "período, las alinea celda a celda (ver align_climate_grids) y las cruza en "
                 "un panel 3D aparte: una variable define la altura de las columnas y la otra "
                 "su color. Independiente del selector de 'Capa Climática' de arriba.",
        )
        cross_height_var = st.selectbox(
            "Variable → Altura de columnas",
            options=["Precipitación", "Temperatura"],
            index=0,
            disabled=not enable_climate_cross,
            help="La otra variable define el color de las columnas.",
        )

        st.markdown("---")

        st.subheader("🗺️ Capas del Mapa")
        show_cuenca = st.checkbox("Mostrar Polígonos de Cuenca", value=True)
        show_rios = st.checkbox("Mostrar Ríos", value=True)
        show_lagos = st.checkbox("Mostrar Lagos y Cuerpos de Agua", value=True)
        show_stations = st.checkbox("Mostrar Estaciones Meteorológicas", value=True)

        st.markdown("---")
        st.info("💡 **Tip**: Seleccione un polígono en el mapa para inspeccionar sus atributos geográficos.")

        return {
            "selected_estacion": selected_estacion,
            "selected_climate_layer": selected_climate_layer,
            "enable_climate_cross": enable_climate_cross,
            "cross_height_var": cross_height_var,
            "show_cuenca": show_cuenca,
            "show_rios": show_rios,
            "show_lagos": show_lagos,
            "show_stations": show_stations
        }
