import streamlit as st

from modules.pydeck_layers import height_variable_options


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

        st.subheader("🧊 Panel 3D")

        # A drawn polygon renders the 3D panel on its own (see the polygon
        # analysis section in app.py), so the selector below has to stay usable
        # in that case too — gating it on this checkbox alone used to leave the
        # panel stuck on "Precipitación" with no way to switch variables.
        polygon_panel_active = st.session_state.get("polygon_analysis") is not None

        enable_climate_cross = st.checkbox(
            "Mostrar el panel 3D sin polígono dibujado",
            value=False,
            help="Cruza precipitación (CHIRPS) y temperatura (ERA5-Land) sobre toda la "
                 "provincia en un panel 3D: una variable define la altura de las columnas "
                 "y la otra su color. Si ya dibujaste un polígono, el panel aparece igual "
                 "debajo del mapa y este checkbox no hace falta.",
        )

        # "Acumulación de flujo (riesgo)" only appears once the unified polygon
        # analysis has actually produced cells with a flow_value (see
        # `run_polygon_analysis` in app.py, which sets this flag). Without that
        # check the option would be selectable with nothing behind it — Pysheds
        # runs on-the-fly per drawn polygon, there's no province-wide flow
        # accumulation to fall back on.
        height_var_options = height_variable_options(
            include_flow=st.session_state.get("flow_variable_available", False)
        )

        selected_height_label = st.selectbox(
            "Variable → Altura de columnas",
            options=list(height_var_options),
            index=0,
            disabled=not (enable_climate_cross or polygon_panel_active),
            help="Precipitación y Temperatura se cruzan entre sí: la elegida define la "
                 "altura de las columnas y la otra su color. 'Acumulación de flujo' usa "
                 "una sola variable (altura y color) y solo aparece si ya se dibujó un "
                 "polígono y corrió el análisis de Pysheds.",
        )
        # Returned as the stable "precip"/"temp"/"flow" key rather than the
        # display label, so app.py can hand it straight to build_pydeck_layer()
        # without re-matching translated strings.
        cross_height_var = height_var_options[selected_height_label]

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
