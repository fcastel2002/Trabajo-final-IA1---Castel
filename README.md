# Trabajo-final-IA1---Castel
# Proyecto Final - Inteligencia Artificial 1

Este repositorio contiene el código fuente del proyecto final de la cátedra de Inteligencia Artificial 1 (IA1) del año 2024 en la Facultad de Ingeniería de la Universidad Nacional de Cuyo.

## Descripción

El proyecto aborda el reconocimiento de objetos y audio mediante la implementación de algoritmos de aprendizaje supervisado y no supervisado. Se desarrollaron dos agentes principales:

1. **Agente de Reconocimiento de Objetos**: Utiliza el algoritmo *K-means* para la clasificación de imágenes de piezas metálicas, como tornillos, clavos, tuercas y arandelas.

2. **Agente de Reconocimiento de Audio**: Emplea el algoritmo *K-Nearest Neighbors (K-NN)* para la clasificación de palabras específicas a partir de características extraídas de señales de audio.

## Estructura del Repositorio

- **`source/`**: Contiene el código fuente principal del proyecto, organizado en módulos para cada agente y sus respectivas funcionalidades.

- **`anexos/`**: Esta carpeta contiene las diferentes carpetas con imagenes y audios
- **`runtime_files/`** : Esta carpeta contiene todos los archivos utilizados durantes la ejecución del programa general, como la base de datos de los audios `FINAL_DB` y los centroides obtenidos para el clasificador de imágenes `centroides.csv`. También contiene arhivos secundarios generados durante la etapa de aprendizaje.

- **`README.md`**: Este archivo, que proporciona una visión general del proyecto y su estructura.

## Uso
Para personalizar el aprendizaje del agente y utilizar una base de datos diferente, puede ejecutar InterfazAudio y/o InterfazImages, esto lanzará las interfaces especificas de cada agente donde en ellas se encuentran las opciones de aprendizaje correspondientes.
Si desea utilizar el programa del proyecto ejecute InterfazProyecto.
- python .\InterfazAudios.py
- python .\Interfazimages.py
- python .\InterfazProyecto.py

## Requisitos

- **Python 3.8 o superior**: Lenguaje de programación utilizado para el desarrollo del proyecto.

- **Bibliotecas de Python**:
  - `numpy`
  - `scikit-learn`
  - `librosa`
  - `matplotlib`
  - `flask`
  - `tkinter`
  - `pandas`
  - `librosa`
  - `pydub`
  - `umap-learn`
  - `opencv-python`
  - `sounddevice`
  - `PIL`
  - `soundfile`

Estas dependencias pueden instalarse ejecutando:

```bash
pip install -r requirements.txt
