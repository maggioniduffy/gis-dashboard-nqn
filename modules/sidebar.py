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

        st.subheader("🗺️ Capas del Mapa")
        show_cuenca = st.checkbox("Mostrar Polígonos de Cuenca", value=True)
        show_rios = st.checkbox("Mostrar Ríos", value=True)
        show_lagos = st.checkbox("Mostrar Lagos y Cuerpos de Agua", value=True)
        show_stations = st.checkbox("Mostrar Estaciones Meteorológicas", value=True)

        st.markdown("---")
        st.info("💡 **Tip**: Seleccione un polígono en el mapa para inspeccionar sus atributos geográficos.")

        return {
            "selected_estacion": selected_estacion,
            "show_cuenca": show_cuenca,
            "show_rios": show_rios,
            "show_lagos": show_lagos,
            "show_stations": show_stations
        }
