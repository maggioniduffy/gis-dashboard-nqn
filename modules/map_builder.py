import folium
from folium import plugins
import geopandas as gpd
import numpy as np

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

    # Capas de mapa base. Folium's built-in 'CartoDB positron'/'dark_matter'
    # presets point at basemaps.cartocdn.com, which now requires a registered
    # API key even for anonymous/low-volume access — the tiles still return
    # HTTP 200 but with the real map replaced by an "API KEY REQUIRED"
    # watermark (see carto.com/basemaps/apikey; same issue fixed for the
    # PyDeck panel in modules/pydeck_layers.py). Esri's Canvas styles serve
    # real tiles anonymously and match the app's new light theme.
    #
    # `show=False` on every layer but the default one is required: base
    # layers all default to `show=True`, and since they're mutually-exclusive
    # radio entries in the LayerControl but NOT lazily added, every `show=True`
    # layer actually gets painted onto the map — the last one added ends up
    # visually on top regardless of which radio is checked. (The previous
    # code passed a `default=True` kwarg here, which isn't a real
    # `folium.TileLayer` parameter — it silently landed in **kwargs and did
    # nothing, which is how the map ended up defaulting to the dark layer.)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Claro (Esri)",
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="Calles (OSM)",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Oscuro (Esri)",
        show=False,
    ).add_to(m)

    # 1. Capa de Polígonos de Cuenca Neuquén
    if gdf_cuenca is not None and not gdf_cuenca.empty and controls.get("show_cuenca", True):
        # Ajustar vista del mapa a los límites de la cuenca
        bounds = gdf_cuenca.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Estilo de polígonos. Los tonos neón (#00c896/#00ffb7) del tema oscuro
        # original se ven ilegibles sobre un basemap claro; el teal se oscurece
        # para que el contorno se distinga del fill y del fondo gris claro.
        style_cuenca = {
            'fillColor': '#0f766e',
            'color': '#0b4f4a',
            'weight': 1.5,
            'fillOpacity': 0.25
        }

        highlight_cuenca = {
            'fillColor': '#14b8a6',
            'color': '#0b4f4a',
            'weight': 3,
            'fillOpacity': 0.55
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
        # Reproject to EPSG:5343 temporarily to calculate area in km2
        gdf_lagos = gdf_lagos.copy()
        gdf_lagos_proj = gdf_lagos.to_crs(epsg=5343)
        
        name_col_lagos = 'nam' if 'nam' in gdf_lagos.columns else ('fna' if 'fna' in gdf_lagos.columns else ('FNA' if 'FNA' in gdf_lagos.columns else None))
        
        areas_temp = gdf_lagos_proj.geometry.area / 1e6
        if name_col_lagos:
            # Group by name to calculate the TOTAL area of the whole lake (all tiles/squares combined)
            gdf_lagos['Area_km2'] = areas_temp.groupby(gdf_lagos[name_col_lagos]).transform('sum').round(2)
            # If some lakes don't have a name, default to their individual tile area
            gdf_lagos['Area_km2'] = gdf_lagos['Area_km2'].fillna(areas_temp.round(2))
        else:
            gdf_lagos['Area_km2'] = areas_temp.round(2)
        fields_lagos = []
        aliases_lagos = []
        if name_col_lagos:
            fields_lagos.append(name_col_lagos)
            aliases_lagos.append("Nombre:")
        fields_lagos.append('Area_km2')
        aliases_lagos.append("Área (km²):")

        folium.GeoJson(
            gdf_lagos,
            name="Lagos y Embalses",
            # Contorno oscurecido (antes #90e0ef, un cian pálido pensado para
            # fondo oscuro) para que el borde se distinga del fill y del
            # basemap claro en vez de perderse en él.
            style_function=lambda x: {
                'fillColor': '#0077b6',
                'color': '#023e8a',
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(fields=fields_lagos, aliases=aliases_lagos) if fields_lagos else None
        ).add_to(m)

    # 3. Capa de Ríos
    if gdf_rios is not None and not gdf_rios.empty and controls.get("show_rios", True):
        gdf_rios = gdf_rios.copy()
        name_col_rios = 'nam' if 'nam' in gdf_rios.columns else ('fna' if 'fna' in gdf_rios.columns else ('FNA' if 'FNA' in gdf_rios.columns else None))

        # We assign the same mock flow to all segments/tiles of the same river
        if name_col_rios:
            unique_rivers = gdf_rios[name_col_rios].dropna().unique()
            river_flows = {river: np.round(np.random.uniform(10.0, 800.0), 1) for river in unique_rivers}
            gdf_rios['Ultimo_Caudal_m3s'] = gdf_rios[name_col_rios].map(river_flows)
            # Fill remaining unnamed segments with random flows
            mask = gdf_rios['Ultimo_Caudal_m3s'].isna()
            if mask.any():
                gdf_rios.loc[mask, 'Ultimo_Caudal_m3s'] = np.round(np.random.uniform(10.0, 800.0, size=mask.sum()), 1)
        else:
            gdf_rios['Ultimo_Caudal_m3s'] = np.round(np.random.uniform(10.0, 800.0, size=len(gdf_rios)), 1)
        fields_rios = []
        aliases_rios = []
        if name_col_rios:
            fields_rios.append(name_col_rios)
            aliases_rios.append("Nombre:")
        fields_rios.append('Ultimo_Caudal_m3s')
        aliases_rios.append("Caudal (m³/s):")

        folium.GeoJson(
            gdf_rios,
            name="Tramos de Ríos",
            # #48cae4 (cian pálido) es casi invisible sobre un basemap claro;
            # un azul más saturado mantiene el contraste sin competir con el
            # cian reservado para el overlay de escorrentía.
            style_function=lambda x: {
                'color': '#0369a1',
                'weight': 3,  # Fino mantiene la lectura "técnica"; sigue siendo fácil de hover.
                'opacity': 0.85
            },
            tooltip=folium.GeoJsonTooltip(fields=fields_rios, aliases=aliases_rios) if fields_rios else None
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
                <h4 style="margin:0 0 6px 0; color:#0f766e;">{st_data['name']}</h4>
                <hr style="margin:4px 0; border:0; border-top:1px solid #ddd;">
                <p style="margin:3px 0;"><b>Temperatura:</b> {st_data['temp']}</p>
                <p style="margin:3px 0;"><b>Humedad:</b> {st_data['hum']}</p>
                <p style="margin:3px 0;"><b>Precipitación:</b> {st_data['precip']}</p>
                <p style="margin:3px 0;"><b>Estado:</b> <span style="color:#0f766e; font-weight:bold;">{st_data['status']}</span></p>
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

    # No LayerControl here on purpose: the caller adds it as the map's LAST
    # child (see `add_layer_control`). Folium does collect every layer at
    # render time, but it emits the control's JS at the position where it was
    # added — so a layer added afterwards (the climate GeoJson in app.py) is
    # referenced by the control before its `var` is assigned, Leaflet gets
    # `undefined`, throws in `_addLayer`, and the st_folium iframe never
    # reports its height: the 2D map collapses to 0px.
    return m


def add_layer_control(m: folium.Map) -> None:
    """Adds the LayerControl. Must be called after every other layer is on `m`."""
    folium.LayerControl(position='topright').add_to(m)
