import os
import cv2
import csv
import logging  # Add this import

# Configure logging at the beginning of the file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def obtener_imagenes_por_verdura(ruta_base, carpetas, indice):
    try:
        imagenes_por_verdura = []
        for carpeta in carpetas:
            ruta_carpeta = os.path.join(ruta_base, carpeta)
            if not os.path.exists(ruta_carpeta):
                logging.warning(f'La carpeta {ruta_carpeta} no existe.')
                continue

            archivos = os.listdir(ruta_carpeta)
            if not archivos:
                logging.warning(f'La carpeta {ruta_carpeta} está vacía.')
                continue

            imagen_path = os.path.join(ruta_carpeta, archivos[indice % len(archivos)])
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                logging.error(f'No se pudo leer la imagen {imagen_path}.')
                continue

            imagenes_por_verdura.append(imagen)
        return imagenes_por_verdura
    except Exception as e:
        logging.error(f'Error obteniendo imágenes: {e}')
        return []

def crear_archivo_csv(ruta_archivo, encabezados):
    try:
        if not os.path.exists(ruta_archivo):
            with open(ruta_archivo, 'w', newline='') as archivo_csv:
                escritor = csv.writer(archivo_csv)
                escritor.writerow(encabezados)
            logging.info(f'Archivo CSV creado: {ruta_archivo}')
        else:
            logging.info(f'El archivo CSV ya existe: {ruta_archivo}')
    except Exception as e:
        logging.error(f'Error creando archivo CSV: {e}')

def agregar_fila_csv(ruta_archivo, datos):
    try:
        with open(ruta_archivo, 'a', newline='') as archivo_csv:
            escritor = csv.writer(archivo_csv)
            escritor.writerow(datos)
        logging.info(f'Fila agregada al CSV: {datos}')
    except Exception as e:
        logging.error(f'Error agregando fila al CSV: {e}')
