from PrediccionImage import *
from AudioDatabase import *
from FiltradoAudio import *
from KNNPredictor import *
from AudiosRaw import *
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, simpledialog
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from PIL import Image, ImageTk  # Added import for image handling
import os
import shutil  # Added imports for file operations
import pickle
from server import *
import threading


def run_flask_server():
    # Ejecutar el servidor Flask
    app.run(host='0.0.0.0', port=5000)


class UserInterface:

    def __init__(self, carpeta_crudos, carpeta_prefiltrados):
        flask_thread = threading.Thread(target=run_flask_server)
        flask_thread.daemon = True  # El hilo se cerrará cuando el programa principal termine
        flask_thread.start()

        self.carpeta_crudos = carpeta_crudos
        self.carpeta_prefiltrados = carpeta_prefiltrados
        self.database = DataBase(
            self.carpeta_prefiltrados, '../runtime_files/audios_features.csv')
        self.filtrado = Filtrado(
            self.carpeta_crudos, self.carpeta_prefiltrados)
        self.grabacion = AudiosRaw(self.carpeta_crudos)

        self.resultados_prediccion = []  # Store prediction results
        self.resultados_imagenes = []  # Store image prediction results

        self.window = tk.Tk()
        self.window.title("Interfaz de usuario")
        self.window.geometry("400x400")
        self.window.resizable(False, False)
        self.create_widgets()
        self.window.mainloop()

    def create_widgets(self):
        self.label = tk.Label(self.window, text="Menu principal")
        self.label.pack(pady=10)

        # Create a labeled frame for Audio Classifier
        audio_frame = ttk.LabelFrame(self.window, text="Clasificador de Audio")
        audio_frame.pack(pady=10, padx=10, fill="x")
        
        # Move audio-related buttons into the audio_frame
        self.button5 = ttk.Button(
            audio_frame, text="Iniciar grabación", command=self.iniciar_grabacion)
        self.button5.pack(pady=5)
    
        self.button1 = ttk.Button(
            audio_frame, text="Filtrar y normalizar audios", command=self.filtrado.procesar_audios)
        self.button1.pack(pady=5)
    
        self.button2 = ttk.Button(
            audio_frame, text="Extraer características", command=self.database.extraer_caracteristicas)
        self.button2.pack(pady=5)
    
        self.button3 = ttk.Button(audio_frame, text="Predecir con KNN")
        self.button3.pack(pady=5)
        self.button3.config(command=self.ingresar_valor_k)
        
        # Create a labeled frame for Image Classifier
        image_frame = ttk.LabelFrame(self.window, text="Clasificador de Imágenes")
        image_frame.pack(pady=10, padx=10, fill="x")
        
        # Move image-related button into the image_frame
        self.button6 = ttk.Button(
            image_frame, text="Predecir imagenes", command=self.ejecutar_prediccion_imagenes)
        self.button6.pack(pady=5)
        
        # Add the Exit button at the bottom
        self.button4 = ttk.Button(
            self.window, text="Salir", command=self.window.quit)
        self.button4.pack(pady=10)
        
        self.button_help = ttk.Button(
            self.window, text="Help", command=self.show_help)
        self.button_help.pack(pady=10)
    
    def show_help(self):
        help_text = (
            "Primero seleccione la carpeta donde se encuentran las imágenes que quiere predecir, puede subir la imagen desde el servidor web.\n" 
            "Luego grabe un audio, filtre, extraiga características y por último, al predecir con KNN, "
            "elija un valor de K. El programa mostrará la imagen correspondiente a la predicción."
            "\n Posibles errores: \n"
            " - RuntimeWarning: invalid value encountered in multiply output = gain * data > Este 'warning' hará imposible una clasificación \n"
            "Para evitarlo evite silencios prolongados en la grabación. \n"
            " - No se seleccionó ningún archivo: Asegúrese de seleccionar un archivo válido\n"
            " - El archivo no es permitido: Asegúrese de que el archivo sea una imagen"
            
        )
        messagebox.showinfo("Ayuda", help_text)

    def iniciar_grabacion(self):
        self.borrar_archivos()
        self.grabacion.iniciar()

    def borrar_archivos(self):
        carpetas = ["../anexos/predict_audios",
                    "../anexos/predict_audios_filt"]
        for carpeta in carpetas:
            if os.path.exists(carpeta):
                shutil.rmtree(carpeta)
                os.makedirs(carpeta)

    def prediccion_knn(self, k):
        self.knn = KNN(k)
        features, etiquetas = self.leer_db_audios()
        # Asegurar arrays numpy
        self.knn.ajustar(features.values, etiquetas.values)
        puntos, labels = self.leer_features_nuevas()  # Nuevos puntos a predecir
        puntos = puntos.values  # Convertimos a array numpy
        with open('../runtime_files/scaler_umap.pkl', 'rb') as file:
            scaler = pickle.load(file)
        puntos = scaler.transform(puntos)

        print(f"Puntos shape: {puntos.shape}")

        # Cargar modelo umap_audio previamente ajustado
        with open("../runtime_files/umap_audio.pkl", "rb") as file:
            umap_audio = pickle.load(file)

        # Transformar los puntos con UMAP
        puntos_reducidos = umap_audio.transform(puntos)
        print(f"Puntos reducidos shape: {puntos_reducidos.shape}")

        # Predecir etiquetas para todos los puntos reducidos
        predicciones = self.knn.predecir(puntos_reducidos)

        # Mostrar resultados
        self.resultados_prediccion = list(zip(labels, predicciones))
        resultado = "\n".join([f"Etiqueta: {label}, Predicción: {
            pred}" for label, pred in zip(labels, predicciones)])
        messagebox.showinfo("Predicción KNN", f"Predicciones:\n{resultado}")

        # Mostrar imágenes correspondientes a las predicciones
        try:
            for label in predicciones:
                imagen_path = self.obtener_imagen_por_etiqueta(label)
                if imagen_path:
                    self.mostrar_imagen(imagen_path)
        except Exception as e:
            print(f"Error al mostrar imagen: {e}")

    def obtener_imagen_por_etiqueta(self, etiqueta):
        # Buscar la ruta de la imagen correspondiente a la etiqueta
        for archivo, label in self.resultados_imagenes:
            if label == etiqueta:
                return '../anexos/img_uploads/'+archivo  # Asume que 'archivo' es la ruta de la imagen
        return None

    def mostrar_imagen(self, path):
        
        ventana_imagen = tk.Toplevel(self.window)
        ventana_imagen.title(f"Imagen para {path}")
        img = Image.open(path)
        img = img.resize((200, 200))  # Ajusta el tamaño según sea necesario
        photo = ImageTk.PhotoImage(img)
        label_img = tk.Label(ventana_imagen, image=photo)
        label_img.image = photo  # Mantener una referencia
        label_img.pack()

    def leer_features_nuevas(self):
        df = pd.read_csv("../runtime_files/audios_features.csv")
        etiquetas = df["Etiqueta"]
        features = df.drop(columns=["Etiqueta"])

        return features, etiquetas

    def leer_db_audios(self):
        df = pd.read_csv("../runtime_files/audio_database_3D.csv")
        etiquetas = df["Etiqueta"]
        features = df.drop(columns=["Etiqueta"])
        return features, etiquetas

    def ingresar_valor_k(self):
        k_value = tk.simpledialog.askinteger(
            "Input", "Ingrese el valor de K:", minvalue=1, maxvalue=21)
        if k_value is not None:
            self.prediccion_knn(k_value)

    def ejecutar_prediccion_imagenes(self):
        analisis = Analisis()
        analisis.run()
        self.resultados_imagenes = analisis.resultados_prediccion
        # Optionally, display or process the results
        resultado = "\n".join([f"Archivo: {archivo}, Clasificación: {etiqueta}"
                              for archivo, etiqueta in self.resultados_imagenes])
        messagebox.showinfo("Predicción de Imágenes",
                            f"Resultados:\n{resultado}")


interface = UserInterface("../anexos/predict_audios",
                          "../anexos/predict_audios_filt")
