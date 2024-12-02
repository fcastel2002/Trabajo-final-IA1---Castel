import pandas as pd
import numpy as np

# Cargar archivo CSV con las varianzas por etiqueta
varianzas = pd.read_csv('varianza_por_clase.csv', index_col=0)

# Parámetros configurables
umbral_baja_varianza = 0.05  # Define un límite para considerar varianza como baja
umbral_diferencia = 0.2  # Diferencia mínima entre etiquetas para una buena separación

# Analizar las características con varianza baja dentro de cada etiqueta


def analizar_varianza_baja(varianzas, umbral):
    varianza_baja = varianzas[varianzas < umbral].dropna(axis=1, how='all')
    print("\nCaracterísticas con varianza baja dentro de las etiquetas:")
    print(varianza_baja)
    return varianza_baja

# Analizar las diferencias de varianza entre etiquetas


def analizar_separacion(varianzas, umbral):
    separacion = varianzas.max() - varianzas.min()
    buenas_separaciones = separacion[separacion > umbral]
    print("\nCaracterísticas con buena separación entre etiquetas:")
    print(buenas_separaciones)
    return buenas_separaciones

# Generar un informe de análisis


def generar_informe(varianza_baja, buenas_separaciones):
    print("\n--- Informe de análisis ---")
    print(f"Número de características con varianza baja: {
          len(varianza_baja.columns)}")
    print(f"Número de características con buena separación: {
          len(buenas_separaciones)}")
    print("Revisar estas características como potenciales eliminaciones o ajustes.")


# Aplicar análisis
varianza_baja = analizar_varianza_baja(varianzas, umbral_baja_varianza)
buenas_separaciones = analizar_separacion(varianzas, umbral_diferencia)
generar_informe(varianza_baja, buenas_separaciones)

# Guardar resultados a archivos CSV para análisis posterior
varianza_baja.to_csv('caracteristicas_varianza_baja.csv')
buenas_separaciones.to_csv(
    'caracteristicas_buenas_separaciones.csv', header=True)
