import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
from CaracteristicasImages import calcular_momentos_hu, calcular_color_promedio, guardar_momentos_hu
import subprocess
import os
# Add imports for PIL and Tkinter image display
from PIL import Image, ImageTk
import tkinter as tk
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import euclidean_distances
import csv

DISPLAY_IMAGES = False  # Cambiar a False para no visualizar las imágenes

def aumentar_saturacion_brillo(imagen, mask, factor_saturacion=1.03, factor_brillo=1.7):
    # Convertir la imagen a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Aumentar la saturación y el brillo solo en la región del contorno
    imagen_hsv[:, :, 1] = np.where(mask == 255, np.clip(
        imagen_hsv[:, :, 1] * factor_saturacion, 0, 255), imagen_hsv[:, :, 1])
    imagen_hsv[:, :, 2] = np.where(mask == 255, np.clip(
        imagen_hsv[:, :, 2] * factor_brillo, 0, 255), imagen_hsv[:, :, 2])

    # Convertir de nuevo a espacio de color BGR
    imagen_bgr = cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2BGR)
    return imagen_bgr


def calcular_color_promedio_normalizado(imagen, mask):
    # Normalizar la imagen
    imagen_normalizada = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
    return calcular_color_promedio(imagen_normalizada, mask)


class Analisis:
    def __init__(self):
        self.resultados_prediccion = []

    def seleccionar_carpeta(self):
        try:
            # Intentar usar zenity para seleccionar carpeta
            cmd = ['zenity', '--file-selection',
                   '--directory',
                   '--title=Seleccione la carpeta con imágenes',
                   '--filename=' + os.path.expanduser('../anexos/uploads')]

            resultado = subprocess.run(cmd, capture_output=True, text=True)

            if (resultado.returncode == 0):
                return resultado.stdout.strip()

            return self._usar_tkinter_fallback_carpeta()

        except FileNotFoundError:
            return self._usar_tkinter_fallback_carpeta()

    def _usar_tkinter_fallback_carpeta(self):
        ventana = tk.Tk()
        ventana.withdraw()
        carpeta = filedialog.askdirectory(
            initialdir='../anexos/',
            title='Seleccione la carpeta con imágenes'
        )
        ventana.destroy()
        return carpeta

    def procesar_carpeta(self, ruta_carpeta, filtros_a_aplicar, momentos_elegidos, ruta_csv):
        resultados = []
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_completa = os.path.join(ruta_carpeta, archivo)
                features = self.procesar_imagen(
                    ruta_completa, filtros_a_aplicar, momentos_elegidos, ruta_csv)
                if features is not None:
                    hu_momentos, mean_color = features
                    resultados.append((archivo, hu_momentos, mean_color))
        return resultados

    def seleccionar_imagen(self):
        try:
            # Intentar usar zenity (para GNOME)
            cmd = ['zenity', '--file-selection',
                   '--title=Seleccione la imagen a analizar',
                   '--file-filter=*.png *.jpg *.jpeg',
                   '--filename=' + os.path.expanduser('../anexos/imagenes_mias')]

            resultado = subprocess.run(cmd, capture_output=True, text=True)

            if resultado.returncode == 0:
                return resultado.stdout.strip()

            # Si zenity falla, intentar kdialog (para KDE)
            cmd = ['kdialog', '--getopenfilename',
                   os.path.expanduser('~/Pictures'),
                   'Image Files (*.png *.jpg *.jpeg)']

            resultado = subprocess.run(cmd, capture_output=True, text=True)

            if resultado.returncode == 0:
                return resultado.stdout.strip()

            # Si ambos fallan, usar el diálogo tk por defecto
            return self._usar_tkinter_fallback()

        except FileNotFoundError:
            # Si no se encuentra zenity ni kdialog, usar tkinter
            return self._usar_tkinter_fallback()

    def _usar_tkinter_fallback(self):
        ventana = tk.Tk()
        ventana.withdraw()
        archivo = filedialog.askopenfilename(
            initialdir='../anexos',
            title='Seleccione la imagen a analizar',
            filetypes=[('Image Files', '*.png *.jpg *.jpeg')]
        )
        ventana.destroy()
        return archivo

    def mostrar_imagen(self, titulo, imagen):
        if DISPLAY_IMAGES:
            # Redimensionar la imagen para mostrarla
            max_width = 800
            max_height = 600
            height, width = imagen.shape[:2]
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            imagen_redimensionada = cv2.resize(imagen, new_size)
            # Convertir la imagen a RGB para mostrarla con PIL
            imagen_rgb = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(imagen_rgb)
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
            ventana = tk.Toplevel()
            ventana.title(titulo)
            etiqueta_imagen = tk.Label(ventana, image=imagen_tk)
            etiqueta_imagen.image = imagen_tk
            etiqueta_imagen.pack()
            ventana.wait_window()
           
    def procesar_imagen(self, ruta_imagen, filtros_a_aplicar, momentos_elegidos, ruta_csv):
        import logging
        # Configurar logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        try:
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                logging.error(f"No se pudo leer la imagen: {ruta_imagen}")
                return None
            imagen_original = imagen.copy()
            imagen_procesada = imagen.copy()

            for filtro in filtros_a_aplicar:
                logging.info(f'Aplicando filtro: {filtro}')
                if filtro == 'gaussian':
                    imagen_procesada = cv2.GaussianBlur(
                        imagen_procesada, (17, 17), 0)
                    self.mostrar_imagen('Filtro Gaussian', imagen_procesada)
                elif filtro == 'gris':
                    imagen_procesada = cv2.cvtColor(
                        imagen_procesada, cv2.COLOR_BGR2GRAY)
                    self.mostrar_imagen('Filtro Gris', imagen_procesada)
                elif filtro == 'binarizedADAPTIVE':
                    # imagen_procesada = cv2.threshold(imagen_procesada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    imagen_procesada = cv2.adaptiveThreshold(
                         imagen_procesada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 6)
                    
                    self.mostrar_imagen('Binarización Adaptativa', imagen_procesada)
                elif filtro == 'morfologico':
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_CROSS, (11, 11))
                    imagen_procesada = cv2.morphologyEx(imagen_procesada, cv2.MORPH_DILATE, kernel)
                    self.mostrar_imagen('Filtro Morfológico', imagen_procesada)

                    imagen_procesada = cv2.morphologyEx(
                        imagen_procesada, cv2.MORPH_OPEN, kernel)
                    self.mostrar_imagen('Filtro Morfológico', imagen_procesada)
                    imagen_procesada = cv2.morphologyEx(imagen_procesada, cv2.MORPH_CLOSE, kernel)
                    self.mostrar_imagen('Filtro Morfológico', imagen_procesada)
                elif filtro == 'contornos':
                    contornos = cv2.findContours(
                        imagen_procesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(
                        contornos) == 2 else contornos[1]
                    if contornos:
                        max_contorno = max(contornos, key=cv2.contourArea)
                        mask = np.zeros(
                            imagen_original.shape[:2], dtype=np.uint8)
                        cv2.drawContours(
                            mask, [max_contorno], -1, 255, thickness=cv2.FILLED)
                        # Dibujar los contornos sobre la imagen original
                        imagen_con_contornos = cv2.drawContours(
                            imagen_original.copy(), [max_contorno], -1, (255, 0, 0), 5)
                        self.mostrar_imagen('Contornos', imagen_con_contornos)
                        
                        # Calcular momentos de Hu y color promedio
                        hu_momentos = calcular_momentos_hu(
                            max_contorno, momentos_elegidos)
                        imagen_pospro = aumentar_saturacion_brillo(
                            imagen_original, mask)
                        self.mostrar_imagen('Saturación y Brillo Aumentados', imagen_pospro)
                        
                        mean_color = calcular_color_promedio(
                            imagen_pospro, mask)
                        # Retornar características para predicción
                        return hu_momentos, mean_color
                    else:
                        logging.warning(
                            "No se encontraron contornos en la imagen.")
                        return None
                else:
                    logging.error(f'Filtro "{filtro}" no reconocido')

        except Exception as e:
            logging.error(f'Error al procesar la imagen: {e}')
            return None
        
    def mostrar_resultados(self,resultados, predictor):
        try:
            clustering_df = pd.read_csv('../runtime_files/resultados_clustering.csv')
            cluster_label_map = dict(zip(clustering_df['Cluster'], clustering_df['label']))
            max_cluster = clustering_df['Cluster'].max()
            etiquetas = [cluster_label_map.get(i, f"Cluster {i}") for i in range(max_cluster + 1)]
        except Exception as e:
            print(f"Error al cargar etiquetas: {e}")
            etiquetas = ['zanahoria', 'berenjena', 'papa', 'camote']
        print("\nResultados de la clasificación:")
        print("="*50)
        print(f"{'Nombre de archivo':<30} {'Clasificación':<20}")
        print("-"*50)

        clasificaciones = {}
        for archivo, hu_momentos, mean_color in resultados:
            features = np.concatenate((hu_momentos, mean_color))
            cluster = predictor.predecir_cluster(features)
            if cluster is not None:
                etiqueta = etiquetas[cluster]
                self.resultados_prediccion.append((archivo, etiqueta))
                print(f"{archivo:<30} {etiqueta:<20}")
                guardar_prediccion(archivo, etiqueta, hu_momentos[0], hu_momentos[1],
                                   mean_color[0], mean_color[1], mean_color[2])
                if etiqueta not in clasificaciones:
                    clasificaciones[etiqueta] = 0
                clasificaciones[etiqueta] += 1


        print("\nResumen:")
        print("="*30)
        for etiqueta, cantidad in clasificaciones.items():
            print(f"{etiqueta}: {cantidad} imágenes")


    def run(self):
        ruta_carpeta = self.seleccionar_carpeta()
        if not ruta_carpeta:
            print("No se seleccionó ninguna carpeta.")
            return

        filtros_a_aplicar = ['gaussian', 'gris',
                            'binarizedADAPTIVE', 'morfologico', 'contornos']
        momentos_elegidos = [2, 3]
        ruta_csv = '../runtime_files/predicciones.csv'

        resultados = self.procesar_carpeta(
            ruta_carpeta, filtros_a_aplicar, momentos_elegidos, ruta_csv)
        if not resultados:
            print("No se encontraron imágenes para procesar en la carpeta.")
            return

        predictor = Predictor()
        self.mostrar_resultados(resultados, predictor)


class Estandarizacion:
    def __init__(self, features_df):
        self.mean = features_df.mean()
        self.std = features_df.std()

    def estandarizar(self, features):
        return (features - self.mean) / self.std


class Predictor:
    def __init__(self):
        self.centroides_df = pd.read_csv('../runtime_files/centroides.csv')
        # Omitir columna 'Cluster'
        self.centroides = self.centroides_df.iloc[:, 1:].values
        print(self.centroides)
        self.clusters = self.centroides_df['Cluster'].values
        print(self.clusters)
        # Cargar el StandardScaler y PCA entrenados

        with open('../runtime_files/umap_images.pkl', 'rb') as f:
            self.pca = pickle.load(f)
        with open('../runtime_files/scaler_images.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
    def predecir_cluster(self, features):
        try:
            # Convertir las características a un DataFrame
            features_df = pd.DataFrame([features])

            # Estandarizar utilizando el scaler entrenado
           
            features_estandarizadas = self.scaler.transform(features_df)
            #print(f"features std:{features_estandarizadas}")
            pd.DataFrame(features_estandarizadas).to_csv(
                '../runtime_files/features_estandarizadas.csv', index=True)
            # Aplicar PCA utilizando el modelo entrenado
            features_pca = self.pca.transform(features_estandarizadas)
            # Calcular distancias y encontrar el clúster más cercano
            #print(features_pca)
            distances = euclidean_distances(features_pca, self.centroides)

            idx_min = np.argmin(distances)
            #print(f"Distancias: {distances}")
            return self.clusters[idx_min]
        except Exception as e:
            print(f"Error al predecir el clúster: {e}")
            return None

def guardar_prediccion(image_path, prediccion, hu2, hu3, mean_b, mean_g, mean_r):
    nombre_archivo = os.path.basename(image_path)
    nombre_archivo = nombre_archivo.split('_')[0]
    etiqueta = prediccion
    ruta_csv = '../runtime_files/predicciones.csv'
    encabezado = ['Nombre_archivo', 'Hu2', 'Hu3', 'Mean_B', 'Mean_G', 'Mean_R', 'Etiqueta']
    
    # Check if the CSV file exists; if not, create it with the header
    if not os.path.exists(ruta_csv):
        with open(ruta_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(encabezado)
    
    # Append the prediction data
    with open(ruta_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([nombre_archivo, hu2, hu3, mean_b, mean_g, mean_r, etiqueta])


