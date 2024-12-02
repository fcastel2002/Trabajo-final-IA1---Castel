import tkinter as tk
from tkinter import messagebox
import threading

# Importar clases y funciones de los módulos existentes
from DataBaseImg import ProcesadorImagen
from AnalisisCaracteristicasImages import realizar_analisis_caracteristicas
from KMeansClustering import KmeansClustering, run_clustering
from prediccion import Analisis, Predictor
def main():
    # Crear la ventana principal
    root = tk.Tk()
    root.title('Interfaz de Procesamiento de Imágenes')

    # Definir funciones para cada operación
    def procesar_imagenes():
        def run_procesamiento():
            procesador = ProcesadorImagen()
            filtros_a_aplicar = ['gaussian', 'gris', 'binarizedADAPTIVE', 'morfologico', 'contornos']
            momentos_elegidos = [0, 1, 2, 3, 4, 5, 6]
            procesador.encabezado = ['Nombre'] + [f'Hu{i}' for i in momentos_elegidos] + ['Mean_B', 'Mean_G', 'Mean_R']
            ruta_csv = 'resultados.csv'


            while procesador.continuar_procesamiento:
                while True:
                    imagenes_progreso, mean_color = procesador.procesar_siguiente_imagen(
                        filtros_a_aplicar, momentos_elegidos, ruta_csv)
                    if not imagenes_progreso:
                        break
                respuesta = tk.messagebox.askyesno("Nueva carpeta", "¿Desea seleccionar una nueva carpeta?")
                if respuesta:
                    procesador.seleccionar_nueva_carpeta()
                else:
                    procesador.continuar_procesamiento = False
                    break

        threading.Thread(target=run_procesamiento).start()

    def realizar_analisis():
        try:
            realizar_analisis_caracteristicas()
            messagebox.showinfo("Éxito", "Análisis de características completado.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar el análisis: {e}")

    def realizar_clustering():
        try:
            run_clustering()
            messagebox.showinfo("Éxito", "Clustering completado.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar el clustering: {e}")

    def predecir():
        def run_prediccion():
            analisis = Analisis()
            ruta_carpeta = analisis.seleccionar_carpeta()
            if not ruta_carpeta:
                messagebox.showwarning("Advertencia", "No se seleccionó ninguna carpeta.")
                return

            filtros_a_aplicar = ['gaussian', 'gris', 'binarizedADAPTIVE', 'morfologico', 'contornos']
            momentos_elegidos = [2, 3]
            ruta_csv = 'predicciones.csv'

            resultados = analisis.procesar_carpeta(ruta_carpeta, filtros_a_aplicar, momentos_elegidos, ruta_csv)
            if not resultados:
                messagebox.showwarning("Advertencia", "No se encontraron imágenes para procesar en la carpeta.")
                return

            predictor = Predictor()
            analisis.mostrar_resultados(resultados, predictor)

        threading.Thread(target=run_prediccion).start()

    # Crear botones para cada funcionalidad
    btn_procesar = tk.Button(root, text='Procesar Imágenes', command=procesar_imagenes, width=30, height=2)
    btn_analisis = tk.Button(root, text='Análisis de Características', command=realizar_analisis, width=30, height=2)
    btn_clustering = tk.Button(root, text='Realizar Clustering', command=realizar_clustering, width=30, height=2)
    btn_predecir = tk.Button(root, text='Predicción', command=predecir, width=30, height=2)
    btn_salir = tk.Button(root, text='Salir', command=root.quit, width=30, height=2)

    # Ubicar los botones en la ventana
    btn_procesar.pack(pady=5)
    btn_analisis.pack(pady=5)
    btn_clustering.pack(pady=5)
    btn_predecir.pack(pady=5)
    btn_salir.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    main()