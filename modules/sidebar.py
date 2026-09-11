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

        enable_climate_cross = st.checkbox(
            "Mostrar el panel 3D sin polígono dibujado",
            value=False,
            help="Cruza precipitación (CHIRPS) y temperatura (ERA5-Land) sobre toda la "
                 "provincia en un panel 3D: una variable define la altura de las columnas "
                 "y la otra su color. Si ya dibujaste un polígono, el panel aparece igual "
                 "debajo del mapa y este checkbox no hace falta.",
        )

        # Reserved slot for the "Variable → Altura de columnas" selectbox. It is
        # NOT filled here: whether "Acumulación de flujo (riesgo)" is available
        # depends on the polygon analysis, which app.py runs further down in the
        # same rerun. Filling it here read the previous rerun's state, so right
        # after drawing a polygon the selector showed up disabled and without
        # the flow option (st_folium doesn't trigger a second rerun to fix it).
        # app.py fills this slot via `render_height_variable_selector` once the
        # analysis is done; the container keeps its position in the sidebar.
        height_selector_slot = st.container()

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
            "height_selector_slot": height_selector_slot,
            "show_cuenca": show_cuenca,
            "show_rios": show_rios,
            "show_lagos": show_lagos,
            "show_stations": show_stations
        }


def render_height_variable_selector(
    slot, polygon_panel_active: bool, flow_available: bool, enable_climate_cross: bool
) -> str:
    """
    Fills the sidebar slot reserved by `render_sidebar` with the
    "Variable → Altura de columnas" selectbox and returns the selected
    height_variable key ("precip"/"temp"/"flow").

    Must be called AFTER the polygon analysis of the current rerun, so that
    `polygon_panel_active`/`flow_available` describe the polygon currently on
    the map rather than the previous rerun's.

    `flow_available=False` leaves "Acumulación de flujo (riesgo)" out of the
    options entirely: Pysheds runs on-the-fly per drawn polygon, so without
    one there is no flow accumulation to show.
    """
    height_var_options = height_variable_options(include_flow=flow_available)

    with slot:
        # With a `key`, Streamlit keeps this widget's identity when its options
        # change, so the user's choice survives the flow option appearing;
        # if "flow" was selected and then disappears (polygon deleted),
        # Streamlit falls back to `index=0` (Precipitación) on its own.
        selected_height_label = st.selectbox(
            "Variable → Altura de columnas",
            options=list(height_var_options),
            index=0,
            key="height_variable_select",
            disabled=not (enable_climate_cross or polygon_panel_active),
            help="Precipitación y Temperatura se cruzan entre sí: la elegida define la "
                 "altura de las columnas y la otra su color. 'Acumulación de flujo' cruza "
                 "las tres variables sobre el polígono dibujado: altura = flujo, color = "
                 "precipitación y temperatura en el tooltip. Solo aparece si ya se dibujó "
                 "un polígono y corrió el análisis de Pysheds.",
        )
        if not flow_available:
            st.caption("Dibujá un polígono en el mapa para habilitar «Acumulación de flujo (riesgo)».")

    # Returned as the stable key rather than the display label, so app.py can
    # hand it straight to build_pydeck_layer() without re-matching strings.
    return height_var_options[selected_height_label]
