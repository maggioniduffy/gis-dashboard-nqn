# Dashboard Geoespacial - Cuenca de Neuquén

Herramienta interactiva para el procesamiento, análisis y visualización cartográfica de la red hídrica, embalses y cuerpos de agua de la cuenca y provincia de Neuquén utilizando datos geográficos oficiales.

## Objetivo

Analizar y visualizar de manera integrada la información geoespacial de la red hidrográfica de Neuquén (ríos, lagos, embalses perennes e intermitentes) obtenida de las capas SIG oficiales del Instituto Geográfico Nacional (IGN), facilitando la exploración ambiental mediante un dashboard interactivo.

## Stack Tecnológico

* **Python:** Lenguaje base para el procesamiento de datos y la lógica de la aplicación.
* **GeoPandas:** Manipulación, filtrado y análisis de geometrías espaciales y archivos vectoriales (Shapefiles / GeoJSON).
* **Streamlit:** Framework para el desarrollo rápido de la interfaz web del dashboard.
* **Folium / streamlit-folium:** Renderizado de mapas interactivos basados en Leaflet directamente en la interfaz de usuario.

## Instalación y Ejecución Local (Ubuntu)

### 1. Posicionarse en el directorio del proyecto

Abre tu terminal y navega hasta la carpeta de trabajo:

```bash
cd ~/Desktop/Documents/code/GIS

```

### 2. Crear y activar el entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Instalar las dependencias necesarias

```bash
pip install geopandas matplotlib streamlit folium streamlit-folium

```

### 4. Procesar los datos espaciales

Ejecuta el script de unificación y filtrado para generar los archivos `.geojson` correspondientes a la región de Neuquén a partir de las carpetas del IGN:

```bash
python procesar_todo.py

```

### 5. Iniciar la aplicación web

Levanta el servidor local de Streamlit para visualizar el mapa interactivo en tu navegador:

```bash
streamlit run app.py

```