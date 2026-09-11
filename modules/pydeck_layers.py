"""
3D panel for a classified climate GeoDataFrame (output of
`raster_processing.raster_to_classified_gdf`).

Reuses the exact same branca colormap the Folium layer used (see
`folium_layers.build_colormap` / `add_climate_layer_to_map`, which returns the
colormap it built) so the 2D map and this 3D panel render identical colors
for identical precipitation values.

Rendered as a self-contained deck.gl page embedded via
`st.components.v1.html` (see `render_3d_panel_live`), NOT `st.pydeck_chart`:
orientation (bearing/pitch) and elevation exaggeration are controlled by
plain HTML sliders inside that embedded page, wired to deck.gl's `setProps`
in client-side JS. That keeps every orientation/height change entirely on
the client — it never triggers a Streamlit rerun, so the surrounding
dashboard (and the map itself) never reloads.
"""

import json
from dataclasses import dataclass
from typing import Literal

import branca.colormap as bcm
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

# The only cross-module import in `modules/` (everything else is wired
# together by app.py). It's here on purpose: the color ramps in
# `folium_layers.COLOR_SCHEMES` are shared with the 2D map, and rebuilding
# them locally would let the 2D and 3D views drift apart over time — the
# whole reason `build_colormap` returns its colormap object in the first place.
from modules.folium_layers import build_colormap

# Slider midpoint: `elevation_scale=50` means "no exaggeration" (1x) relative to
# the automatically computed height (see _compute_elevation_auto_scale).
NEUTRAL_ELEVATION_SCALE = 50.0

# Tallest column, as a fraction of the map's own extent. Keeps the extrusion
# readable as terrain-like relief instead of kilometric needles: raw CHIRPS
# accumulation reaches ~13.000 mm, so extruding it 1:1 (or worse, x50) would
# produce columns hundreds of km tall over a ~1.000 km wide province.
MAX_COLUMN_HEIGHT_FRACTION = 0.08

METERS_PER_DEGREE = 111320.0

# Raster basemap tiles for the embedded deck.gl `TileLayer`, no API key
# needed. Originally these were CARTO's `basemaps.cartocdn.com` XYZ styles,
# but CARTO now requires a registered API key even for anonymous/low-volume
# access — every tile still returns HTTP 200, but with the real map replaced
# by a "API KEY REQUIRED" watermark placeholder, which is why the panel
# looked "empty" (see carto.com/basemaps/apikey). Switched to two providers
# that still serve real tiles anonymously: Esri's `Canvas` styles (plain
# light/dark backgrounds, no street clutter competing with the extruded
# columns) for "Claro"/"Oscuro", and standard OpenStreetMap tiles for
# "Calles" (the only one of the three with street-level detail). Esri's tile
# path takes {y} before {x} — deck.gl's TileLayer substitutes {z}/{x}/{y} by
# NAME wherever they appear in the template, not by position, so this is
# fine to write in either order.
# "Claro" is the default: a dark style paints land almost pure black, so at
# provincial zoom the geography of Neuquén would be invisible underneath the
# extruded columns, defeating the purpose of having a basemap at all.
BASEMAP_TILE_URLS = {
    "Claro": "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "Calles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Oscuro": "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
}
DEFAULT_BASEMAP = "Claro"

DEFAULT_BEARING = 30.0
DEFAULT_PITCH = 45.0

# Which grid column drives column HEIGHT (and which one drives COLOR) for each
# choice of the panel's "Variable → Altura de columnas" selector.
#
# "precip"/"temp" are the two halves of `raster_processing.align_climate_grids`'s
# combined grid, so each one takes the OTHER as its color variable — that's the
# "altura = una variable, color = la otra" cross panel.
#
# "flow" is different: Pysheds flow accumulation has no natural second variable
# to pair against (it's joined onto the climate grid on demand by
# `raster_processing.align_flow_accumulation_to_climate_grid`, only over the
# polygon the user drew), so it drives BOTH height and color through the
# green->yellow->red "risk" ramp. `build_pydeck_layer` detects that
# height_column == color_column and drops the redundant second tooltip line.
#
# Units note: D8 flow accumulation counts UPSTREAM CELLS, not a physical
# discharge — hence "celdas". For a precipitation-driven flow estimate in m^3/s
# see `hydrology.calculate_peak_flow` (Rational Method), which is a separate
# calculation shown next to the 2D map.
HEIGHT_VARIABLE_SPECS = {
    "precip": {
        "selector_label": "Precipitación",
        "height_column": "precip_value",
        "height_label": "Precipitación",
        "height_unit": "mm",
        "color_column": "temp_value",
        "color_label": "Temperatura",
        "color_unit": "°C",
        "color_scheme": "temperature",
    },
    "temp": {
        "selector_label": "Temperatura",
        "height_column": "temp_value",
        "height_label": "Temperatura",
        "height_unit": "°C",
        "color_column": "precip_value",
        "color_label": "Precipitación",
        "color_unit": "mm",
        "color_scheme": "average",
    },
    "flow": {
        "selector_label": "Acumulación de flujo (riesgo)",
        "height_column": "flow_value",
        "height_label": "Acumulación de flujo",
        "height_unit": "celdas",
        "color_column": "flow_value",
        "color_label": "Acumulación de flujo",
        "color_unit": "celdas",
        "color_scheme": "risk",
    },
}

# Order the selector shows them in. "flow" is last because it's conditional:
# it only belongs in the list once Pysheds has actually run for a drawn
# polygon (see `height_variable_options`).
HEIGHT_VARIABLE_ORDER = ("precip", "temp", "flow")
FLOW_HEIGHT_VARIABLE = "flow"


def height_variable_options(include_flow: bool) -> dict[str, str]:
    """
    Ordered `{selector label: height_variable key}` for the sidebar's
    "Variable → Altura de columnas" selectbox.

    `include_flow=False` leaves "Acumulación de flujo (riesgo)" out entirely
    (not merely disabled), which is what the sidebar passes when no polygon
    has been drawn yet: Pysheds runs on-the-fly per polygon, so there is no
    province-wide flow accumulation to fall back on and offering the option
    would just produce an empty panel.
    """
    return {
        HEIGHT_VARIABLE_SPECS[key]["selector_label"]: key
        for key in HEIGHT_VARIABLE_ORDER
        if include_flow or key != FLOW_HEIGHT_VARIABLE
    }

# deck.gl pure-JS bundle (exposes DeckGL/TileLayer/BitmapLayer/ColumnLayer
# under the global `deck` namespace), loaded from a CDN inside the embedded
# HTML page — pinned so the panel doesn't break under an unrelated upstream
# release.
DECKGL_JS_URL = "https://unpkg.com/deck.gl@8.9.35/dist.min.js"


def _hex_to_rgb(hex_color: str) -> list[int]:
    """Converts a '#rrggbb'(aa) string, as returned by calling a branca colormap, to [r, g, b]."""
    hex_color = hex_color.lstrip("#")[:6]
    return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]


def _infer_cell_size_deg(gdf: gpd.GeoDataFrame) -> float:
    """
    Recovers the source raster's cell size (in degrees) from the vectorized
    polygons. `rasterio.features.shapes` traces polygon edges exactly along
    raster grid lines, so every vertex coordinate is a multiple of the cell
    size; the smallest positive gap between distinct vertex coordinates is
    therefore the cell size itself.
    """
    x_coords = [
        np.asarray(geom.exterior.coords)[:, 0]
        for geom in gdf.geometry.explode(index_parts=True)
    ]
    unique_x = np.unique(np.concatenate(x_coords))
    gaps = np.diff(unique_x)
    gaps = gaps[gaps > 1e-12]
    if gaps.size == 0:
        # Degenerate single-column raster: fall back to the full width.
        minx, _, maxx, _ = gdf.total_bounds
        return max(maxx - minx, 1e-6)
    return float(gaps.min())


def _explode_regions_to_cells(
    gdf: gpd.GeoDataFrame, value_column: str, cell_size_deg: float
) -> gpd.GeoDataFrame:
    """
    Rebuilds the original raster grid as one point per cell centre.

    `raster_to_classified_gdf` merges every run of adjacent same-class pixels
    into a single polygon (median ~3 cells, but up to thousands), which is
    exactly what the Folium layer wants but is wrong for a ColumnLayer: one
    column per merged region draws a handful of scattered needles instead of a
    grid surface. Here the regular lattice of cell centres is regenerated
    across the layer's bounds and spatially joined back onto the regions, so
    each 5 km cell gets its own column carrying its region's value.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    half = cell_size_deg / 2.0
    xs = np.arange(minx + half, maxx, cell_size_deg)
    ys = np.arange(miny + half, maxy, cell_size_deg)
    mesh_x, mesh_y = np.meshgrid(xs, ys)

    cell_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(mesh_x.ravel(), mesh_y.ravel()), crs=gdf.crs
    )
    # Inner join: cells outside the clipped basin extent simply drop out.
    return cell_points.sjoin(gdf[[value_column, "geometry"]], how="inner", predicate="within")


@st.cache_data(show_spinner=False)
def _gdf_to_pydeck_df_cached(
    _gdf: gpd.GeoDataFrame, value_column: str, _colormap: bcm.LinearColormap, cache_key: tuple
) -> pd.DataFrame:
    """
    Does the actual long-format conversion. `_gdf`/`_colormap` are prefixed
    with an underscore so Streamlit doesn't try to hash them directly (a
    GeoDataFrame and a branca colormap aren't reliably hashable); `cache_key`
    is a small hashable fingerprint of both, which is what actually drives
    cache invalidation (e.g. switching climate layers busts this cache).
    """
    cell_size_deg = _infer_cell_size_deg(_gdf)
    cells = _explode_regions_to_cells(_gdf, value_column, cell_size_deg)
    if cells.empty:
        return pd.DataFrame(columns=["lon", "lat", "value", "color", "cell_radius_m"])

    values = cells[value_column].astype(float)

    # The colormap is a plain Python callable, so it's sampled once per DISTINCT
    # value (one per classified region, a few hundred) and then broadcast back
    # over the tens of thousands of cells, instead of being called per row.
    color_by_value = {
        value: _hex_to_rgb(_colormap(value)) + [200] for value in values.unique()
    }

    latitudes = cells.geometry.y.to_numpy()

    # Cells are square in degrees, so on the ground they are narrower in
    # longitude than in latitude. The narrower side sets the column size so
    # neighbouring columns tile without overlapping.
    mean_lat_rad = np.radians(float(np.mean(latitudes)))
    cell_width_m = cell_size_deg * METERS_PER_DEGREE * np.cos(mean_lat_rad)

    return pd.DataFrame({
        "lon": cells.geometry.x.to_numpy(),
        "lat": latitudes,
        "value": values.to_numpy(),
        "color": values.map(color_by_value).tolist(),
        "cell_radius_m": cell_width_m / 2.0,
    })


def gdf_to_pydeck_df(
    gdf: gpd.GeoDataFrame, value_column: str, colormap: bcm.LinearColormap
) -> pd.DataFrame:
    """
    Converts a classified GeoDataFrame into the long-format DataFrame PyDeck
    needs: one row per grid cell with its centroid (`lon`, `lat`), its
    `value`, and a `color` ([r, g, b, a]) sampled from `colormap` — the SAME
    colormap object the Folium layer used for that value column, so the 3D
    panel matches the 2D map's colors exactly.
    """
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=["lon", "lat", "value", "color"])

    # Small hashable fingerprint of the otherwise-unhashable gdf/colormap, so
    # switching climate layers correctly invalidates the cache instead of
    # silently reusing a previous layer's positions/colors.
    cache_key = (
        len(gdf),
        value_column,
        float(gdf[value_column].sum()),
        tuple(colormap.colors),
        colormap.vmin,
        colormap.vmax,
    )
    return _gdf_to_pydeck_df_cached(gdf, value_column, colormap, cache_key)


@st.cache_data(show_spinner=False)
def _gdf_to_pydeck_df_dual_cached(
    _gdf: gpd.GeoDataFrame,
    height_column: str,
    color_column: str,
    _colormap: bcm.LinearColormap,
    cache_key: tuple,
) -> pd.DataFrame:
    """
    Does the actual long-format conversion for `gdf_to_pydeck_df_dual`. Unlike
    `_gdf_to_pydeck_df_cached`, `_gdf` here is expected to already be a
    regular per-cell grid (the output of `raster_processing.align_climate_grids`,
    one row per shared cell — not merged bin-regions), so there is no
    `_explode_regions_to_cells` step: each row's own polygon centroid is used
    directly as that cell's position.
    """
    centroids = _gdf.geometry.centroid
    color_values = _gdf[color_column].astype(float)

    # Same one-call-per-distinct-value trick as _gdf_to_pydeck_df_cached.
    color_by_value = {
        value: _hex_to_rgb(_colormap(value)) + [200] for value in color_values.unique()
    }

    latitudes = centroids.y.to_numpy()

    # NOT `_infer_cell_size_deg`: that function finds the smallest gap between
    # vertex x-coordinates across the WHOLE gdf, which only works when
    # neighboring cells share EXACT floating-point-identical edges — true for
    # `raster_to_classified_gdf` (traced directly off the raster's own
    # EPSG:4326 grid, never reprojected). `align_climate_grids` instead builds
    # each cell's box in a metric CRS and reprojects to EPSG:4326 afterwards;
    # reprojection is nonlinear, so neighboring cells' shared edges end up a
    # few floating-point ULPs apart in degrees instead of identical, and
    # `_infer_cell_size_deg` picks up that near-zero noise as "the" cell size
    # (columns a fraction of a millimeter wide — invisible). Reading the
    # width straight off one cell's own bounds sidesteps that entirely: it
    # doesn't depend on any two polygons agreeing on a coordinate.
    bounds = _gdf.geometry.iloc[0].bounds  # (minx, miny, maxx, maxy)
    cell_size_deg = bounds[2] - bounds[0]
    mean_lat_rad = np.radians(float(np.mean(latitudes)))
    cell_width_m = cell_size_deg * METERS_PER_DEGREE * np.cos(mean_lat_rad)

    return pd.DataFrame({
        "lon": centroids.x.to_numpy(),
        "lat": latitudes,
        "value": _gdf[height_column].astype(float).to_numpy(),
        "color_value": color_values.to_numpy(),
        "color": color_values.map(color_by_value).tolist(),
        "cell_radius_m": cell_width_m / 2.0,
    })


def gdf_to_pydeck_df_dual(
    gdf: gpd.GeoDataFrame,
    height_column: str,
    color_column: str,
    colormap: bcm.LinearColormap,
) -> pd.DataFrame:
    """
    Converts a combined grid GeoDataFrame — the output of
    `raster_processing.align_climate_grids`, with one row per cell shared by
    two aligned climate rasters (e.g. `precip_value`/`temp_value`) — into the
    long-format DataFrame PyDeck needs, decoupling elevation from color:
    `value` (column height) comes from `height_column`, `color` comes from
    sampling `colormap` over `color_column`. This is what lets the combined
    3D panel show "altura = una variable, color = la otra" instead of both
    being driven by the same column like the single-variable `gdf_to_pydeck_df`.
    """
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=["lon", "lat", "value", "color_value", "color"])

    cache_key = (
        len(gdf),
        height_column,
        color_column,
        float(gdf[height_column].sum()),
        float(gdf[color_column].sum()),
        tuple(colormap.colors),
        colormap.vmin,
        colormap.vmax,
    )
    return _gdf_to_pydeck_df_dual_cached(gdf, height_column, color_column, colormap, cache_key)


def _compute_elevation_auto_scale(df: pd.DataFrame) -> float:
    """
    "Natural" elevation_scale=50 baseline for `df`: the multiplier that makes
    the tallest column reach MAX_COLUMN_HEIGHT_FRACTION of the map's own
    extent, so the panel stays legible regardless of whether values are
    ~1.000 mm (annual average) or ~13.000 mm (decade accumulation).

    Returns 0.0 when `df` is empty or has no positive value (callers treat
    that as "nothing to render").
    """
    if df is None or df.empty:
        return 0.0

    max_value = float(df["value"].max())
    if max_value <= 0:
        return 0.0

    # Map extent in meters, used to pick a column height that reads as relief.
    lat_span_m = (float(df["lat"].max()) - float(df["lat"].min())) * METERS_PER_DEGREE
    lon_span_m = (
        (float(df["lon"].max()) - float(df["lon"].min()))
        * METERS_PER_DEGREE
        * np.cos(np.radians(float(df["lat"].mean())))
    )
    extent_m = max(lat_span_m, lon_span_m, 1.0)
    return (extent_m * MAX_COLUMN_HEIGHT_FRACTION) / max_value


def build_isometric_view_state(
    gdf: gpd.GeoDataFrame,
    pitch: float = DEFAULT_PITCH,
    bearing: float = DEFAULT_BEARING,
    zoom: float | None = None,
) -> pdk.ViewState:
    """
    Centers the 3D view on `gdf`'s bounds with an isometric-looking initial
    pitch/bearing. Only `latitude`/`longitude`/`zoom` end up used by
    `render_3d_panel_live` (pitch/bearing there come from the embedded page's
    own sliders instead), but this keeps one shared "fit the data" formula.

    `zoom=None` (the default) fits the whole layer in view instead of using a
    fixed zoom: a hard-coded zoom that happens to suit one extent shows only a
    fraction of the province for another. The formula is the standard Web
    Mercator fit (the world spans 360 degrees at zoom 0), nudged out slightly
    so some surrounding basemap stays visible around the data.
    """
    if gdf is None or gdf.empty:
        return pdk.ViewState(
            latitude=-38.95, longitude=-70.0, zoom=zoom or 6.0, pitch=pitch, bearing=bearing
        )

    minx, miny, maxx, maxy = gdf.total_bounds

    if zoom is None:
        span_deg = max(maxx - minx, maxy - miny, 1e-6)
        zoom = float(np.clip(np.log2(360.0 / span_deg) + 0.6, 3.0, 12.0))

    return pdk.ViewState(
        latitude=(miny + maxy) / 2,
        longitude=(minx + maxx) / 2,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )


def _build_live_panel_html(
    records: list[dict],
    radius: float,
    auto_scale: float,
    latitude: float,
    longitude: float,
    zoom: float,
    tile_url: str,
    height: int,
    tooltip_label: str = "Precipitación",
    unit: str = "mm",
    color_tooltip_label: str | None = None,
    color_unit: str | None = None,
    elevation_scale: float = NEUTRAL_ELEVATION_SCALE,
) -> str:
    """
    Standalone HTML page: deck.gl basemap + ColumnLayer, driven by its own
    sliders. `tooltip_label`/`unit` describe the column HEIGHT (`value`);
    `color_tooltip_label`/`color_unit` are optional and describe a SEPARATE
    variable driving column COLOR (`color_value`, present when `records`
    comes from `gdf_to_pydeck_df_dual`) — used by the combined precipitation
    x temperature panel so the tooltip shows both variables instead of just
    the one driving height.

    `elevation_scale` is only the STARTING position of the elevation slider
    (NEUTRAL_ELEVATION_SCALE = "no exaggeration"); the user can move it freely
    afterwards without a Streamlit rerun. NEUTRAL_ELEVATION_SCALE itself stays
    hardcoded in the JS as the divisor that defines what 1x means.
    """
    # json.dumps is the only thing that touches `records`/`tile_url`/labels on
    # their way into the <script> tag, so this is safe against '</script>' or
    # quote characters showing up in a basemap URL, a stray value, or a label.
    data_json = json.dumps(records)
    tile_url_json = json.dumps(tile_url)
    tooltip_label_json = json.dumps(tooltip_label)
    unit_json = json.dumps(unit)
    color_tooltip_label_json = json.dumps(color_tooltip_label)
    color_unit_json = json.dumps(color_unit)

    return f"""
<div id="deck-root" style="position:relative;width:100%;height:{height}px;
     background:#eef1f4;border-radius:12px;overflow:hidden;
     border:1px solid #e3e8ec;">
  <div id="map" style="width:100%;height:100%;"></div>
  <div id="controls" style="position:absolute;top:12px;left:12px;z-index:1;
       background:rgba(255,255,255,0.92);padding:10px 14px;border-radius:10px;
       border:1px solid #e3e8ec;box-shadow:0 2px 8px rgba(15,23,30,0.12);
       color:#1b2430;font:13px/1.4 'Inter',-apple-system,sans-serif;min-width:220px;">
    <div style="margin-bottom:8px;">
      <label>Rotación (bearing): <span id="bearing-val">{DEFAULT_BEARING:.0f}</span>°</label><br>
      <input id="bearing-slider" type="range" min="0" max="360" step="5"
             value="{DEFAULT_BEARING:.0f}" style="width:100%;accent-color:#0f766e;">
    </div>
    <div style="margin-bottom:8px;">
      <label>Inclinación (pitch): <span id="pitch-val">{DEFAULT_PITCH:.0f}</span>°</label><br>
      <input id="pitch-slider" type="range" min="0" max="90" step="5"
             value="{DEFAULT_PITCH:.0f}" style="width:100%;accent-color:#0f766e;">
    </div>
    <div>
      <label>Escala de elevación: <span id="elev-val">{elevation_scale:.0f}</span></label><br>
      <input id="elev-slider" type="range" min="1" max="100" step="1"
             value="{elevation_scale:.0f}" style="width:100%;accent-color:#0f766e;">
    </div>
  </div>
</div>
<script src="{DECKGL_JS_URL}"></script>
<script>
(function () {{
  const DATA = {data_json};
  const RADIUS = {radius};
  const AUTO_SCALE = {auto_scale};
  const NEUTRAL = {NEUTRAL_ELEVATION_SCALE};
  const HEIGHT_LABEL = {tooltip_label_json};
  const HEIGHT_UNIT = {unit_json};
  const COLOR_LABEL = {color_tooltip_label_json};
  const COLOR_UNIT = {color_unit_json};

  let bearing = {DEFAULT_BEARING};
  let pitch = {DEFAULT_PITCH};
  let elevationSliderValue = {elevation_scale};
  let viewState = {{
    longitude: {longitude},
    latitude: {latitude},
    zoom: {zoom},
    pitch: pitch,
    bearing: bearing,
  }};

  // Created ONCE: its `id`/`data` (a tile URL template) never change, so
  // deck.gl's own layer diffing keeps already-fetched tiles cached across
  // every setProps call below — basemap tiles are never re-requested just
  // because the camera or the column heights changed.
  const basemapLayer = new deck.TileLayer({{
    id: 'basemap',
    data: {tile_url_json},
    minZoom: 0,
    maxZoom: 19,
    tileSize: 256,
    renderSubLayers: (props) => {{
      const {{bbox: {{west, south, east, north}}}} = props.tile;
      return new deck.BitmapLayer(props, {{
        data: null,
        image: props.data,
        bounds: [west, south, east, north],
      }});
    }},
  }});

  function buildColumnLayer() {{
    const appliedScale = AUTO_SCALE * (elevationSliderValue / NEUTRAL);
    return new deck.ColumnLayer({{
      id: 'climate-columns',
      data: DATA,
      diskResolution: 4,
      angle: 45,
      // Circumradius of a square whose side is the cell width, shrunk
      // slightly so adjacent cells read as separate columns.
      radius: RADIUS * Math.sqrt(2) * 0.98,
      elevationScale: appliedScale,
      getPosition: (d) => [d.lon, d.lat],
      getElevation: (d) => d.value,
      getFillColor: (d) => d.color,
      extruded: true,
      pickable: true,
      autoHighlight: true,
    }});
  }}

  const deckgl = new deck.DeckGL({{
    container: 'map',
    viewState: viewState,
    controller: true,
    layers: [basemapLayer, buildColumnLayer()],
    getTooltip: ({{object}}) => {{
      if (!object) return null;
      let text = `${{HEIGHT_LABEL}}: ${{object.value.toFixed(1)}} ${{HEIGHT_UNIT}}`;
      if (COLOR_LABEL !== null && object.color_value !== undefined) {{
        text += ` | ${{COLOR_LABEL}}: ${{object.color_value.toFixed(1)}} ${{COLOR_UNIT}}`;
      }}
      return {{text}};
    }},
    onViewStateChange: ({{viewState: nextViewState}}) => {{
      // Mouse drag/scroll on the canvas itself: keep it in sync too, without
      // touching `layers` (so the basemap/columns are untouched by panning).
      viewState = nextViewState;
      deckgl.setProps({{viewState}});
    }},
  }});

  function applyCamera() {{
    viewState = {{...viewState, bearing, pitch}};
    // Only `viewState` changes here — `layers` is left out of this setProps
    // call entirely, so deck.gl repaints the WebGL canvas with the existing
    // layers under a new camera instead of touching them at all.
    deckgl.setProps({{viewState}});
  }}

  function applyElevation() {{
    // Only the ColumnLayer is rebuilt (with a new elevationScale, same DATA
    // array reference); `basemapLayer` is untouched, so tiles already on
    // screen never reload.
    deckgl.setProps({{layers: [basemapLayer, buildColumnLayer()]}});
  }}

  document.getElementById('bearing-slider').addEventListener('input', (e) => {{
    bearing = Number(e.target.value);
    document.getElementById('bearing-val').textContent = bearing;
    applyCamera();
  }});
  document.getElementById('pitch-slider').addEventListener('input', (e) => {{
    pitch = Number(e.target.value);
    document.getElementById('pitch-val').textContent = pitch;
    applyCamera();
  }});
  document.getElementById('elev-slider').addEventListener('input', (e) => {{
    elevationSliderValue = Number(e.target.value);
    document.getElementById('elev-val').textContent = elevationSliderValue;
    applyElevation();
  }});
}})();
</script>
"""


def render_3d_panel_live(
    df: pd.DataFrame,
    view_state: pdk.ViewState,
    basemap: str = DEFAULT_BASEMAP,
    height: int = 620,
    tooltip_label: str = "Precipitación",
    unit: str = "mm",
    color_tooltip_label: str | None = None,
    color_unit: str | None = None,
    elevation_scale: float = NEUTRAL_ELEVATION_SCALE,
) -> None:
    """
    Renders the 3D climate panel as a self-contained deck.gl page embedded
    via `st.components.v1.html`, or a placeholder message if there's nothing
    to show.

    Orientation (bearing/pitch) and elevation exaggeration are controlled by
    HTML `<input type="range">` sliders INSIDE the embedded page (see
    `_build_live_panel_html`), never by `st.slider` — so moving them never
    triggers a Streamlit script rerun: only the WebGL canvas inside this
    iframe repaints, and the surrounding dashboard/map never reloads.

    `basemap` is a key of BASEMAP_TILE_URLS. `tooltip_label`/`unit` describe
    the variable driving column HEIGHT (`value`); `color_tooltip_label`/
    `color_unit` are optional and describe a separate variable driving
    column COLOR (`color_value`) — pass both for the combined precipitation
    x temperature panel built from `gdf_to_pydeck_df_dual`, so the tooltip
    shows both variables. Left as `None` (the default) for a single-variable
    panel built from `gdf_to_pydeck_df`, where height and color already come
    from the same column.
    """
    auto_scale = _compute_elevation_auto_scale(df)
    if auto_scale <= 0:
        st.info("No hay datos suficientes para renderizar el panel 3D.")
        return

    radius = float(df["cell_radius_m"].iloc[0]) if "cell_radius_m" in df.columns else 2500.0
    record_columns = ["lon", "lat", "value", "color"]
    if "color_value" in df.columns:
        record_columns.append("color_value")
    records = df[record_columns].to_dict(orient="records")
    tile_url = BASEMAP_TILE_URLS.get(basemap, BASEMAP_TILE_URLS[DEFAULT_BASEMAP])

    html = _build_live_panel_html(
        records=records,
        radius=radius,
        auto_scale=auto_scale,
        latitude=view_state.latitude,
        longitude=view_state.longitude,
        zoom=view_state.zoom,
        tile_url=tile_url,
        height=height,
        tooltip_label=tooltip_label,
        unit=unit,
        color_tooltip_label=color_tooltip_label,
        color_unit=color_unit,
        elevation_scale=elevation_scale,
    )
    components.html(html, height=height, scrolling=False)


@dataclass
class PydeckLayerSpec:
    """
    Everything `render_3d_panel_live` needs for one height-variable choice,
    plus the colormap itself so a caller can draw a matching legend.

    `color_label`/`color_unit` are None when height and color come from the
    SAME column (the "flow" case), which is how the renderer knows to skip
    the redundant second tooltip line.
    """

    df: pd.DataFrame
    view_state: pdk.ViewState
    colormap: bcm.LinearColormap
    height_label: str
    height_unit: str
    color_label: str | None
    color_unit: str | None
    elevation_scale: float

    @property
    def is_single_variable(self) -> bool:
        """True when one variable drives both height and color."""
        return self.color_label is None


def build_pydeck_layer(
    gdf: gpd.GeoDataFrame,
    height_variable: Literal["precip", "temp", "flow"],
    elevation_scale: float = NEUTRAL_ELEVATION_SCALE,
) -> PydeckLayerSpec | None:
    """
    Builds the full render spec for the 3D panel from an aligned grid
    GeoDataFrame and a choice of height variable, centralizing the
    column/label/unit/colormap mapping that would otherwise be an if/elif
    chain at the call site.

    `gdf` is expected to be a per-cell grid (one row per cell), i.e. the
    output of `raster_processing.align_climate_grids` — optionally with a
    `flow_value` column joined on by
    `raster_processing.align_flow_accumulation_to_climate_grid` when
    `height_variable="flow"`.

    Returns None when there's nothing renderable: an empty/None `gdf`, or a
    `gdf` missing the columns this `height_variable` needs (e.g. "flow"
    requested but no flow accumulation was ever joined on, because no polygon
    has been drawn). Callers should treat None as "show a message instead of
    a panel" rather than as an error — it's a normal state, not a failure.

    Raises:
        ValueError: `height_variable` isn't one of HEIGHT_VARIABLE_SPECS.
    """
    if height_variable not in HEIGHT_VARIABLE_SPECS:
        raise ValueError(
            f"Unknown height_variable {height_variable!r}. "
            f"Options: {list(HEIGHT_VARIABLE_SPECS)}"
        )

    spec = HEIGHT_VARIABLE_SPECS[height_variable]
    if gdf is None or gdf.empty:
        return None

    height_column = spec["height_column"]
    color_column = spec["color_column"]
    if height_column not in gdf.columns or color_column not in gdf.columns:
        return None

    # Rows where either driving column is null would render as zero-height or
    # uncolored columns; drop them so the panel only shows real cells. The two
    # columns are the same one in the "flow" case, hence the dedup.
    driving_columns = [height_column]
    if color_column != height_column:
        driving_columns.append(color_column)
    renderable = gdf.dropna(subset=driving_columns)
    if renderable.empty:
        return None

    colormap = build_colormap(
        renderable,
        color_column,
        spec["color_scheme"],
        caption=f"{spec['color_label']} ({spec['color_unit']})",
    )
    df = gdf_to_pydeck_df_dual(renderable, height_column, color_column, colormap)

    # One variable driving both height and color (flow accumulation) means the
    # color tooltip would just repeat the height tooltip — left as None so
    # `render_3d_panel_live` emits a single line.
    single_variable = height_column == color_column

    return PydeckLayerSpec(
        df=df,
        view_state=build_isometric_view_state(renderable),
        colormap=colormap,
        height_label=spec["height_label"],
        height_unit=spec["height_unit"],
        color_label=None if single_variable else spec["color_label"],
        color_unit=None if single_variable else spec["color_unit"],
        elevation_scale=elevation_scale,
    )
