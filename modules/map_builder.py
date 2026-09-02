import folium
from folium import plugins
import geopandas as gpd

# Estaciones meteorológicas de demostración para el filtro de sidebar
METEO_STATIONS = [
    {"name": "Estación Chos Malal (Norte)", "lat": -37.3789, "lon": -70.2709, "temp": "14.2 °C", "hum": "45%", "precip": "0.0 mm", "status": "Activa"},
    {"name": "Estación Zapala (Centro)", "lat": -38.9026, "lon": -70.0657, "temp": "12.5 °C", "hum": "52%", "precip": "1.2 mm", "status": "Activa"},
    {"name": "Estación Neuquén Capital (Confluencia)", "lat": -38.9516, "lon": -68.0591, "temp": "18.1 °C", "hum": "38%", "precip": "0.0 mm", "status": "Activa"},
    {"name": "Estación San Martín de los Andes (Sur)", "lat": -40.1579, "lon": -71.3534, "temp": "8.7 °C", "hum": "70%", "precip": "4.5 mm", "status": "Activa"},
    {"name": "Estación Añelo (Vaca Muerta)", "lat": -38.3532, "lon": -68.7884, "temp": "16.4 °C", "hum": "40%", "precip": "0.0 mm", "status": "Activa"}
]

def create_cuenca_map(
    gdf_cuenca: gpd.GeoDataFrame | None,
    gdf_rios: gpd.GeoDataFrame | None = None,
    gdf_lagos: gpd.GeoDataFrame | None = None,
    gdf_puntos: gpd.GeoDataFrame | None = None,
    controls: dict = None
) -> folium.Map:
    """
    Construye un mapa interactivo con Folium incorporando las capas vectoriales GeoJSON y estaciones.
    """
    if controls is None:
        controls = {
            "selected_estacion": "Todas las estaciones",
            "show_cuenca": True,
            "show_rios": True,
            "show_lagos": True,
            "show_stations": True
        }

    # Coordenadas por defecto (Provincia de Neuquén)
    center_lat, center_lon = -38.95, -70.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None,
        control_scale=True
    )

    # Capas de mapa base
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark Matter', default=True).add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    # 1. Capa de Polígonos de Cuenca Neuquén
    if gdf_cuenca is not None and not gdf_cuenca.empty and controls.get("show_cuenca", True):
        # Ajustar vista del mapa a los límites de la cuenca
        bounds = gdf_cuenca.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Estilo de polígonos
        style_cuenca = {
            'fillColor': '#00c896',
            'color': '#00ffb7',
            'weight': 2,
            'fillOpacity': 0.35
        }

        highlight_cuenca = {
            'fillColor': '#00ffb7',
            'color': '#ffffff',
            'weight': 3,
            'fillOpacity': 0.65
        }

        fields = [c for c in ['nam', 'gna', 'fna', 'objeto', 'gid'] if c in gdf_cuenca.columns]
        aliases = [c.upper() for c in fields]

        geojson_layer = folium.GeoJson(
            gdf_cuenca,
            name="Polígonos Cuenca Neuquén",
            style_function=lambda x: style_cuenca,
            highlight_function=lambda x: highlight_cuenca,
            tooltip=folium.GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
                localize=True
            ),
            popup=folium.GeoJsonPopup(fields=fields)
        )
        geojson_layer.add_to(m)

    # 2. Capa de Lagos
    if gdf_lagos is not None and not gdf_lagos.empty and controls.get("show_lagos", True):
        folium.GeoJson(
            gdf_lagos,
            name="Lagos y Embalses",
            style_function=lambda x: {
                'fillColor': '#0077b6',
                'color': '#90e0ef',
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(fields=['nam'] if 'nam' in gdf_lagos.columns else None)
        ).add_to(m)

    # 3. Capa de Ríos
    if gdf_rios is not None and not gdf_rios.empty and controls.get("show_rios", True):
        folium.GeoJson(
            gdf_rios,
            name="Tramos de Ríos",
            style_function=lambda x: {
                'color': '#48cae4',
                'weight': 2.5,
                'opacity': 0.85
            },
            tooltip=folium.GeoJsonTooltip(fields=['nam'] if 'nam' in gdf_rios.columns else None)
        ).add_to(m)

    # 3b. Capa de Puntos de Interés IGN
    if gdf_puntos is not None and not gdf_puntos.empty:
        puntos_group = folium.FeatureGroup(name="Puntos Hidrológicos IGN")
        for idx, row in gdf_puntos.iterrows():
            if row.geometry and row.geometry.geom_type == 'Point':
                name = row.get('nam') or row.get('fna') or 'Punto IGN'
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color="#ffb703",
                    fill=True,
                    fill_color="#fb8500",
                    fill_opacity=0.9,
                    tooltip=name,
                    popup=folium.Popup(f"<b>Punto IGN:</b> {name}", max_width=200)
                ).add_to(puntos_group)
        puntos_group.add_to(m)

    # 4. Capa de Estaciones Meteorológicas
    if controls.get("show_stations", True):
        stations_group = folium.FeatureGroup(name="Estaciones Meteorológicas")
        selected = controls.get("selected_estacion")

        for st_data in METEO_STATIONS:
            if selected != "Todas las estaciones" and st_data["name"] != selected:
                continue

            popup_content = f"""
            <div style="font-family: Arial, sans-serif; min-width: 170px;">
                <h4 style="margin:0 0 6px 0; color:#00c896;">{st_data['name']}</h4>
                <hr style="margin:4px 0; border:0; border-top:1px solid #ddd;">
                <p style="margin:3px 0;"><b>Temperatura:</b> {st_data['temp']}</p>
                <p style="margin:3px 0;"><b>Humedad:</b> {st_data['hum']}</p>
                <p style="margin:3px 0;"><b>Precipitación:</b> {st_data['precip']}</p>
                <p style="margin:3px 0;"><b>Estado:</b> <span style="color:#00c896; font-weight:bold;">{st_data['status']}</span></p>
            </div>
            """

            is_selected = selected == st_data["name"]

            folium.Marker(
                location=[st_data["lat"], st_data["lon"]],
                popup=folium.Popup(popup_content, max_width=260),
                tooltip=st_data["name"],
                icon=folium.Icon(
                    color="red" if is_selected else "green",
                    icon="cloud",
                    prefix="fa"
                )
            ).add_to(stations_group)

        stations_group.add_to(m)

    # Herramientas interactivas
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position='bottomleft').add_to(m)
    folium.LayerControl(position='topright').add_to(m)

    return m
