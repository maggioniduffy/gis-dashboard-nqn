import glob
import geopandas as gpd
import pandas as pd

print("Buscando todos los archivos .shp en 'datos_ign'...")
todos_los_shps = glob.glob("datos_ign/**/*.shp", recursive=True)

# Listas para acumular por tipo de geometría
poligonos_nqn = []
lineas_nqn = []
puntos_nqn = []

# Límite geográfico estricto para la región de Neuquén / Cuenca
MIN_LON, MAX_LON = -72.5, -67.0
MIN_LAT, MAX_LAT = -41.5, -35.5

for ruta in todos_los_shps:
  print(f"Leyendo: {ruta}")
  try:
    gdf = gpd.read_file(ruta)

    if gdf.empty:
      continue

    # Reproyectar a EPSG:4326 para filtro espacial en grados
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
      gdf = gdf.to_crs(epsg=4326)

    # Filtro espacial directo por la región de la cuenca (sin restringir por texto)
    filtrado = gdf.cx[MIN_LON:MAX_LON, MIN_LAT:MAX_LAT].copy()

    if not filtrado.empty:
      geom_tipo = filtrado.geom_type.iloc[0]
      if "Polygon" in geom_tipo:
        poligonos_nqn.append(filtrado)
      elif "Line" in geom_tipo:
        lineas_nqn.append(filtrado)
      elif "Point" in geom_tipo:
        puntos_nqn.append(filtrado)

  except Exception as e:
    print(f"  Error leyendo {ruta}: {e}")

# Unir y exportar resultados si se encontraron elementos
print("\nConsolidando y exportando resultados...")

if poligonos_nqn:
  final_polys = pd.concat(poligonos_nqn, ignore_index=True)
  if "gid" in final_polys.columns:
    final_polys = final_polys.drop_duplicates(subset=["gid"])
  final_polys.to_file("cuerpos_agua_neuquen.geojson", driver="GeoJSON")
  print(
      f"-> Guardados {len(final_polys)} polígonos en"
      " 'cuerpos_agua_neuquen.geojson'"
  )

if lineas_nqn:
  final_lines = pd.concat(lineas_nqn, ignore_index=True)
  final_lines.to_file("rios_neuquen.geojson", driver="GeoJSON")
  print(f"-> Guardadas {len(final_lines)} líneas en 'rios_neuquen.geojson'")

if puntos_nqn:
  final_points = pd.concat(puntos_nqn, ignore_index=True)
  final_points.to_file("puntos_neuquen.geojson", driver="GeoJSON")
  print(f"-> Guardados {len(final_points)} puntos en 'puntos_neuquen.geojson'")

print("¡Proceso finalizado con éxito!")