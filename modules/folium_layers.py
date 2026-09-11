"""
Folium visualization for a classified climate GeoDataFrame (output of
`raster_processing.raster_to_classified_gdf`): builds a branca colormap
matching the requested color scheme, styles a GeoJson layer by value through
that colormap, adds the colormap as a visible legend, and lets the layer
register itself in the map's existing `folium.LayerControl`.
"""

import branca.colormap as bcm
import folium
import geopandas as gpd

# Color ramps matched to the project's reference screenshots:
#   - "accumulated": yellow -> green -> blue, for total accumulated precipitation.
#   - "average":     red -> orange -> green -> blue, for average annual rainfall.
#   - "temperature": blue -> yellow -> red, standard cold-to-hot ramp for ERA5-Land °C.
#   - "risk":        green -> yellow -> red, standard low-to-high risk ramp for
#                     the Pysheds flow accumulation variable in the 3D panel.
COLOR_SCHEMES = {
    "accumulated": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
    "average": ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850", "#4575b4"],
    "temperature": ["#2166ac", "#67a9cf", "#fee090", "#fc8d59", "#b2182b"],
    "risk": ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"],
}


def build_colormap(
    gdf: gpd.GeoDataFrame, value_column: str, color_scheme: str, caption: str
) -> bcm.LinearColormap:
    """
    Builds a `branca.colormap.LinearColormap` spanning the observed
    min/max of `value_column` in `gdf`, using one of the named
    `COLOR_SCHEMES`. Returning this object (rather than only using it
    internally) is what lets `pydeck_layers.py` sample the exact same colors
    for the 3D panel, keeping both views visually consistent.
    """
    if color_scheme not in COLOR_SCHEMES:
        raise ValueError(f"Unknown color_scheme {color_scheme!r}. Options: {list(COLOR_SCHEMES)}")

    values = gdf[value_column].dropna()
    vmin = float(values.min()) if not values.empty else 0.0
    vmax = float(values.max()) if not values.empty else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6  # avoid a degenerate (zero-width) colormap range

    return bcm.LinearColormap(
        colors=COLOR_SCHEMES[color_scheme], vmin=vmin, vmax=vmax, caption=caption
    )


def add_climate_layer_to_map(
    map_obj: folium.Map,
    gdf: gpd.GeoDataFrame,
    value_column: str,
    layer_name: str,
    color_scheme: str,
    tooltip_label: str = "Precipitación (mm):",
) -> bcm.LinearColormap | None:
    """
    Adds a classified climate layer to `map_obj`:
      1. builds the branca colormap for `color_scheme` over `value_column`;
      2. adds a `folium.GeoJson` layer named `layer_name`, styled per-feature
         by sampling that colormap;
      3. attaches the colormap itself to the map, which renders it as a
         visible legend (branca's `add_to` draws an HTML/SVG gradient bar);
      4. the GeoJson layer is automatically discoverable by an existing
         `folium.LayerControl` even if that control was already added to the
         map earlier (e.g. inside `map_builder.create_cuenca_map`) — Folium's
         LayerControl scans the map's children at render time (when the map
         is actually drawn/saved), not at the moment `LayerControl.add_to()`
         was called, so layers added afterwards still show up as toggleable.

    `tooltip_label` is the field label shown on hover (e.g. "Temperatura (°C):"
    for an ERA5-Land layer); defaults to the original precipitation wording so
    existing CHIRPS call sites keep working unchanged.

    Returns the built colormap so the caller can reuse it when building the
    PyDeck 3D panel for the same layer/value column (see
    `pydeck_layers.gdf_to_pydeck_df`), or `None` if `gdf` is empty/None.
    """
    if gdf is None or gdf.empty:
        return None

    colormap = build_colormap(gdf, value_column, color_scheme, caption=layer_name)

    def style_function(feature):
        value = feature["properties"].get(value_column)
        if value is None:
            return {"fillColor": "#00000000", "color": "#00000000", "weight": 0, "fillOpacity": 0}
        color = colormap(value)
        return {"fillColor": color, "color": color, "weight": 0.4, "fillOpacity": 0.7}

    folium.GeoJson(
        gdf,
        name=layer_name,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[value_column], aliases=[tooltip_label]),
    ).add_to(map_obj)

    colormap.add_to(map_obj)
    return colormap
