import os
import cv2
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from ArchivosImages import *
import sys
import logging  # Add this import
from CaracteristicasImages import calcular_momentos_hu, calcular_color_promedio, guardar_momentos_hu

# Configure logging at the beginning of the file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def aumentar_saturacion_brillo(imagen, mask, factor_saturacion=1.03, factor_brillo=1.7):
    # Convertir la imagen a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Aumentar la saturación y el brillo solo en la región del contorno
    imagen_hsv[:, :, 1] = np.where(mask == 255, np.clip(imagen_hsv[:, :, 1] * factor_saturacion, 0, 255), imagen_hsv[:, :, 1])
    imagen_hsv[:, :, 2] = np.where(mask == 255, np.clip(imagen_hsv[:, :, 2] * factor_brillo, 0, 255), imagen_hsv[:, :, 2])
    
    # Convertir de nuevo a espacio de color BGR
    imagen_bgr = cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2BGR)
    return imagen_bgr

def calcular_color_promedio_normalizado(imagen, mask):
    # Normalizar la imagen
    imagen_normalizada = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
    return calcular_color_promedio(imagen_normalizada, mask)

class ProcesadorImagen:
    def __init__(self):
        self.ventana_principal = tk.Tk()
        self.ventana_principal.withdraw()  # Ocultar la ventana principal
        self.seleccionar_nueva_carpeta()
        self.continuar_procesamiento = True  # Flag to control processing
        self.encabezado = None
    def seleccionar_nueva_carpeta(self):
        self.ruta_imagenes = self.obtener_ruta_imagenes()
        self.etiqueta = os.path.basename(self.ruta_imagenes)  # Set label based on folder name
        self.imagenes = self.cargar_imagenes(self.ruta_imagenes)
        self.indice_actual = 0

    def obtener_ruta_imagenes(self):
        # Código para solicitar al usuario la ruta de las imágenes
        ruta = filedialog.askdirectory(initialdir='../anexos/imagenes_correctas/',title='Seleccione la ruta de las imágenes')
        return ruta

    def cargar_imagenes(self, ruta):
        # Código para cargar imágenes desde la ruta especificada
        imagenes = []
        for archivo in os.listdir(ruta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagen = cv2.imread(os.path.join(ruta, archivo))
                if imagen is not None:
                    imagen = cv2.resize(imagen, (800, 600))  # Redimensionar a tamaño estándar
                    imagenes.append(imagen)
        return imagenes

    def procesar_siguiente_imagen(self, filtros_a_aplicar, momentos_elegidos, ruta_csv):
        if self.indice_actual >= len(self.imagenes):
            logging.info('All images have been processed.')
            return None, None

        try:
            imagen = self.imagenes[self.indice_actual]
            imagen_original = imagen.copy()
            imagen_procesada = imagen.copy()
            imagenes_progreso = [imagen_procesada.copy()]  # Imagen original

            for filtro in filtros_a_aplicar:
                logging.info(f'Applying filter: {filtro}')
                if filtro == 'gaussian':
                    imagen_procesada = cv2.GaussianBlur(imagen_procesada, (13,13), 0)

                elif filtro == 'gris':
                    imagen_procesada = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2GRAY)
                elif filtro == 'binarizedADAPTIVE':
                    imagen_procesada = cv2.adaptiveThreshold(imagen_procesada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 6)

                elif filtro == 'morfologico':
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11,11))
                    imagen_procesada = cv2.morphologyEx(imagen_procesada, cv2.MORPH_DILATE, kernel)

                elif filtro == 'contornos':
                    contornos = cv2.findContours(imagen_procesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
                    if contornos:
                        max_contorno = max(contornos, key=cv2.contourArea)
                        imagen_procesada = imagen_original.copy()
                        cv2.drawContours(imagen_procesada, [max_contorno], -1, (0, 255, 0), 4)
                        
                        # Calculate Hu moments and mean color
                        hu_momentos = calcular_momentos_hu(max_contorno, momentos_elegidos)
                        mask = np.zeros(imagen_original.shape[:2], dtype=np.uint8)
                        cv2.drawContours(mask, [max_contorno], -1, 255, thickness=cv2.FILLED)
                        
                        # Increase saturation and brightness within the contour
                        imagen_saturada_brillo = aumentar_saturacion_brillo(imagen_original, mask)
                        
                        mean_color = calcular_color_promedio_normalizado(imagen_saturada_brillo, mask)
                        
                        # Ask for label and save to CSV
                        etiqueta = self.etiqueta  # Use folder name as label
                        guardar_momentos_hu(ruta_csv, etiqueta, hu_momentos, mean_color,self.encabezado)
                        imagenes_progreso.append(imagen_saturada_brillo.copy())
                    else:
                        logging.warning("No se encontraron contornos en la imagen.")
                else:
                    logging.error(f'Filtro "{filtro}" no reconocido')
                    continue  # Skip unrecognized filters
                imagenes_progreso.append(imagen_procesada.copy())
                
            self.indice_actual += 1
            return imagenes_progreso, mean_color  # Return mean_color along with images
        except Exception as e:
            logging.error(f'Error processing image at index {self.indice_actual}: {e}')
            self.indice_actual += 1
            return None, None

    def mostrar_imagenes(self, imagenes, mean_color=None):
        if imagenes is None:
            print("No hay más imágenes para procesar")
            return False

        # Crear una nueva ventana Toplevel para mostrar las imágenes
        ventana = tk.Toplevel(self.ventana_principal)
        ventana.title(f"Imagen {self.indice_actual} - Progreso de filtros")

        frame = tk.Frame(ventana)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.imagen_refs = []  # Lista para mantener referencias a las imágenes

        for idx, imagen in enumerate(imagenes):
            try:
                # Convertir y normalizar imagen
                if isinstance(imagen.dtype, np.float64):
                    imagen = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
                    imagen = np.uint8(imagen)

                # Convertir espacio de color
                if len(imagen.shape) == 2:
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
                else:
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

                # Redimensionar si es necesario
                altura, ancho = imagen.shape[:2]
                max_size = 200  # Tamaño más pequeño para mostrar más imágenes
                if altura > max_size or ancho > max_size:
                    ratio = min(max_size / altura, max_size / ancho)
                    nuevo_ancho = int(ancho * ratio)
                    nueva_altura = int(altura * ratio)
                    imagen = cv2.resize(imagen, (nuevo_ancho, nueva_altura))

                # Crear imagen Tkinter
                imagen_pil = Image.fromarray(imagen)
                imagen_tk = ImageTk.PhotoImage(image=imagen_pil)

                # Guardar referencias
                self.imagen_refs.append(imagen_tk)

                # Crear y posicionar label
                label = Label(frame, image=self.imagen_refs[-1])
                label.grid(row=idx//3, column=idx%3, padx=5, pady=5)

            except Exception as e:
                logging.error(f'Error displaying images: {e}')
                continue

        # Mostrar el color promedio
        if mean_color is not None:
            mean_color_hex = f'#{int(mean_color[2]):02x}{int(mean_color[1]):02x}{int(mean_color[0]):02x}'
            mean_color_label = Label(ventana, text=f"Mean Color: B={mean_color[0]}, G={mean_color[1]}, R={mean_color[2]}", bg=mean_color_hex, fg='white')
            mean_color_label.pack(pady=5)

        # Botón para cerrar la ventana
        def cerrar_ventana():
            self.ventana_principal.destroy()  # Cerrar la ventana principal
            sys.exit(0)  # Terminar el programa completamente

        btn_cerrar = tk.Button(ventana, text="Cerrar", command=cerrar_ventana)
        btn_cerrar.pack(pady=5)

        # === Cambio: Agregar botón 'Next' ===
        def siguiente_imagen():
            ventana.destroy()

        btn_next = tk.Button(ventana, text="Next", command=siguiente_imagen)
        btn_next.pack(pady=5)
        # ======================================

        # Asegurarse de que la ventana aparezca en primer plano
        ventana.focus_force()
        ventana.grab_set()
        ventana.wait_window()

        return True

if __name__ == '__main__':
    procesador = ProcesadorImagen()
    filtros_a_aplicar = ['gaussian','gris','binarizedADAPTIVE','morfologico','contornos']
    momentos_elegidos = [0,1,2,3,4,5,6]  # Example: 2nd, 4th, and 6th Hu moments
    procesador.encabezado = ['Nombre'] + [f'Hu{i}' for i in momentos_elegidos] + ['Mean_B', 'Mean_G', 'Mean_R']
    ruta_csv = 'resultados.csv'

    def procesar_imagenes():
        try:
            while procesador.continuar_procesamiento:
                while True:
                    imagenes_progreso, mean_color = procesador.procesar_siguiente_imagen(filtros_a_aplicar, momentos_elegidos, ruta_csv)
                    #if not imagenes_progreso or not procesador.mostrar_imagenes(imagenes_progreso, mean_color):
                    #    break
                    if not imagenes_progreso:
                        break
                # Offer to select a new folder after processing all images
                respuesta = tk.messagebox.askyesno("Nueva carpeta", "¿Desea seleccionar una nueva carpeta?")
                if respuesta:
                    procesador.seleccionar_nueva_carpeta()
                else:
                    procesador.continuar_procesamiento = False
                    break
                        
        except KeyboardInterrupt:
            logging.info("Programa terminado por el usuario")
        except Exception as e:
            logging.error(f'Error: {e}')
        finally:
            procesador.ventana_principal.quit()  # Exit the main loop

    # Run the image processing in a separate thread to avoid blocking the main loop
    import threading
    threading.Thread(target=procesar_imagenes).start()
    procesador.ventana_principal.mainloop()