import geopandas as gpd
import folium
import numpy as np

gdf_rios = gpd.read_file('rios_neuquen.geojson')
gdf_rios = gdf_rios.copy()
gdf_rios['Ultimo_Caudal_m3s'] = np.round(np.random.uniform(10.0, 800.0, size=len(gdf_rios)), 1)
name_col_rios = 'nam' if 'nam' in gdf_rios.columns else None

fields_rios = [name_col_rios, 'Ultimo_Caudal_m3s']
aliases_rios = ['Nombre:', 'Caudal (m³/s):']

m = folium.Map()
folium.GeoJson(
    gdf_rios.head(10),
    tooltip=folium.GeoJsonTooltip(fields=fields_rios, aliases=aliases_rios)
).add_to(m)
m.save('test.html')
print("Success")
