import geopandas as gpd

# 1. Definir las rutas
ruta_lagos = "datos_ign/areas_de_aguas_continentales_perenne/areas_de_aguas_continentales_perennePolygon.shp"
ruta_rios = "datos_ign/lineas_de_aguas_continentales_BI020/lineas_de_aguas_continentales_BI020Line.shp"
print("Cargando capas completas...")
lagos = gpd.read_file(ruta_lagos)
rios = gpd.read_file(ruta_rios)

# 2. Filtrar por texto (Buscamos cuenca del Limay / Neuquén)
# Cambiar 'nam' si la columna de nombres se llama distinto en tu dataset
columna_nombre = "nam" 

print("Filtrando datos de Neuquén/Limay...")
lagos_nqn = lagos[lagos[columna_nombre].str.contains('Limay|Neuquén|Ramos Mexia|Nahuel Huapi|Alicurá', na=False, case=False)]
rios_nqn = rios[rios[columna_nombre].str.contains('Limay|Neuquén|Negro', na=False, case=False)]

print(f"Encontrados: {len(lagos_nqn)} lagos/embalses y {len(rios_nqn)} tramos de río.")

# 3. Exportar a GeoJSON para usar en el frontend (Streamlit / Next.js)
print("Exportando a GeoJSON...")
lagos_nqn.to_file("lagos_neuquen.geojson", driver="GeoJSON")
rios_nqn.to_file("rios_neuquen.geojson", driver="GeoJSON")

print("¡Archivos listos para Antigravity!")