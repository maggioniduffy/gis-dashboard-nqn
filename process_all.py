import glob
import geopandas as gpd
import pandas as pd

print("Buscando todos los archivos .shp en 'datos_ign'...")
todos_los_shps = glob.glob("datos_ign/**/*.shp", recursive=True)

# Listas para acumular por tipo de geometría
poligonos_nqn = []
lineas_nqn = []
puntos_nqn = []

# Palabras clave amplias para atrapar todo lo de la cuenca y la provincia de Neuquén
patron_nqn = (
    "Limay|Neuquén|Negro|Traful|Aluminé|Lácar|Malleo|Ñorquín|Agrio|Chos"
    " Malal|Cipolletti|Ramos Mexia|Alicurá|Piedra del Águila|Nahuel"
    " Huapi|Colón|Collón Curá|Gnecco|Urrutia"
)

for ruta in todos_los_shps:
  print(f"Leyendo: {ruta}")
  try:
    gdf = gpd.read_file(ruta)

    # Buscar columna de nombre disponible
    col_nombre = next(
        (c for c in ["nam", "NAM", "FNA", "fna", "TLA"] if c in gdf.columns), None
    )

    if col_nombre:
      # Filtrar por la región
      filtrado = gdf[
          gdf[col_nombre].str.contains(patron_nqn, na=False, case=False)
      ]
    else:
      # Si no tiene nombre, la dejamos pasar o la omitimos (las del IGN casi siempre tienen)
      filtrado = gdf.head(0)

    if not filtrado.empty:
      # Clasificar según el tipo de geometría principal
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