from AudioDatabase import *
from FiltradoAudio import *
from AnalisisCaracteristicasAudio import *
from KNNPredictor import *
import numpy as np
import os


class Interfaz:

    def __init__(self, carpeta_crudos, carpeta_prefiltrados="../anexos/test_filt"):
        # Inicialización de Filtrado y DataBase
        self.filtrado = Filtrado(carpeta_crudos, carpeta_prefiltrados)
        self.database_ = DataBase(carpeta_prefiltrados, 'caracteristicas.csv')
        self.PCA_analisis = AnalisisPCA(self.database_)
        self.knn = KNN(k=5)
        self.eps_ = 0
        self.min_samples_ = 0

    def menu(self):
        while True:
            print("\n--- Menú Principal ---")
            print("1. Filtrar y normalizar audios")
            print("2. Extraer características")
            print("3. Análisis PCA")
            print("4. Predicción KNN")
            print("5. Persistir base de datos filtrada")
            print("5. Salir")

            opcion = input("Seleccione una opción: ")

            if opcion == "1":
                print("\n--- Filtrar y Normalizar Audios ---")
                try:
                    self.filtrado.procesar_audios()

                except Exception as e:
                    print(f"Error al filtrar y normalizar audios: {e}")

            elif opcion == "2":
                print("\n--- Extraer Características ---")
                try:
                    self.database_.extraer_caracteristicas()
                    self.database_.estandarizar_database()
                    # print("Características extraídas y estandarizadas.")
                except Exception as e:
                    print(
                        f"Error al extraer y estandarizar características: {e}")

            elif opcion == "3":
                print("\n--- Análisis PCA ---")
                try:
                    eps_input = input(
                        f"Distancia minima para outliers, anterior ({self.eps_}):")
                    eps = float(eps_input) if eps_input else self.eps_
                    min_samples_input = int(input(
                        f"Min numero de muestras, anterior ({self.min_samples_}):"))
                    min_samples = int(
                        min_samples_input) if min_samples_input else self.min_samples_
                    self.eps_ = float(eps) if eps else self.eps_
                    self.min_samples_ = int(
                        min_samples) if min_samples else self.min_samples_
                    self.PCA_analisis.analisis_pca_con_outliers(
                        eps, min_samples)
                    self.PCA_analisis.analisis_UMAP()
                    # self.PCA_analisis.calcular_importancia_int()
                except Exception as e:
                    print(f"Error al realizar análisis PCA: {e}")
            elif opcion == "4":
                print("--- Prediccion KNN ---  ")
                # df_filtrado = self.PCA_analisis.get_filtered_dataframe()
                self.entrenar_y_predecir_knn("FINAL_DB.csv")
            elif opcion == "5":
                print("Persistiendo base de datos filtrada...")
                self.PCA_analisis.guardar_dataframe_filtrado(
                    "FINAL_DB.csv", self.eps_, self.min_samples_)

            elif opcion == "6":
                print("Saliendo del programa...")
                break

            else:
                print("Opción no válida. Por favor, seleccione nuevamente.")

    def entrenar_y_predecir_knn(self, db_file):
        try:
            # Leer datos del DataFrame filtrado
            df = pd.read_csv(db_file)
            etiquetas = df['Etiqueta'].values
            datos = df.drop('Etiqueta', axis=1).values

            precisiones = []

            # Realizar el proceso de predicción 200 veces
            for _ in range(100):
                # Barajar los datos aleatoriamente
                indices = np.arange(len(datos))
                np.random.shuffle(indices)

                # Dividir en entrenamiento (70%) y prueba (30%)
                limite = int(len(datos) * 0.7)
                datos_entrenamiento = datos[indices[:limite]]
                etiquetas_entrenamiento = etiquetas[indices[:limite]]
                datos_prueba = datos[indices[limite:]]
                etiquetas_prueba = etiquetas[indices[limite:]]

                # Ajustar el modelo KNN
                self.knn.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

                # Realizar predicciones
                predicciones = self.knn.predecir(datos_prueba)

                # Calcular precisión
                precision = np.mean(predicciones == etiquetas_prueba)
                precisiones.append(precision)

            # Calcular y mostrar la precisión promedio
            precision_promedio = np.mean(precisiones)
            print(f"\nPrecisión promedio del modelo KNN después de 100 iteraciones: {
                  precision_promedio * 100:.2f}%\n")

        except Exception as e:
            print(f"Error al entrenar o probar el modelo KNN: {e}")


# Rutas de las carpetas
carpeta_crudos = "../anexos/test_raw_v2"
carpeta_prefiltrados = '../anexos/test_filt'

# Crear instancia de Interfaz y ejecutar el menú
interfaz = Interfaz(carpeta_crudos, carpeta_prefiltrados)
interfaz.menu()
