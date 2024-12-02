import numpy as np
from collections import Counter
from scipy.spatial import distance


class KNN:
    def __init__(self, k=3, p=2):
        """
        Inicializa el clasificador KNN.

        Args:
            k (int): Número de vecinos más cercanos a considerar.
            p (int): Orden de la distancia de Minkowski (p=2 para euclidiana, p=1 para Manhattan).
        """
        self.k = k
        self.p = p
        self.data = None
        self.labels = None

    def ajustar(self, data, labels):
        """
        Guarda los datos y etiquetas para el modelo.

        Args:
            data (np.ndarray): Datos de entrenamiento.
            labels (np.ndarray): Etiquetas de entrenamiento.
        """
        if len(data) != len(labels):
            raise ValueError(
                "El número de datos y etiquetas debe ser el mismo.")
        self.data = data
        self.labels = labels

    def _calcular_distancia(self, punto1, punto2):
        """
        Calcula la distancia de Minkowski entre dos puntos.

        Args:
            punto1 (np.ndarray): Primer punto.
            punto2 (np.ndarray): Segundo punto.

        Returns:
            float: Distancia de Minkowski.
        """
        return distance.minkowski(punto1, punto2, self.p)

    def predecir(self, puntos):
        """
        Predice las etiquetas para los puntos dados.

        Args:
            puntos (np.ndarray): Puntos a predecir.

        Returns:
            list: Etiquetas predichas.
        """
        predicciones = []
        for punto in puntos:
            distancias = np.array(
                [self._calcular_distancia(punto, x) for x in self.data])
            indices_mas_cercanos = np.argsort(distancias)[:self.k]
            etiquetas_mas_cercanas = [self.labels[i]
                                      for i in indices_mas_cercanos]
            etiqueta_comun = Counter(
                etiquetas_mas_cercanas).most_common(1)[0][0]
            predicciones.append(etiqueta_comun)
        return predicciones
