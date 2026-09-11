import geopandas as gpd
import matplotlib.pyplot as plt

# Ruta al archivo .shp basada en tu estructura de carpetas
# Fijate que agregamos "Polygon" antes del .shp
ruta_areas = "datos_ign/areas_de_aguas_continentales_perenne/areas_de_aguas_continentales_perennePolygon.shp"
print("Cargando capa de áreas perennes...")
gdf_areas = gpd.read_file(ruta_areas)

print("\n--- Columnas disponibles ---")
print(gdf_areas.columns.tolist())

print("\nPloteando mapa (cerrá la ventana del mapa para que el script termine)...")
gdf_areas.plot(color='gray')
plt.show()