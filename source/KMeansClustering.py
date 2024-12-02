import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from ArchivosImages import *
from sklearn.preprocessing import StandardScaler


class KmeansClustering:
    def __init__(self, k=4):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((data_point - centroids) ** 2, axis=1))

    def fit(self, X, iteraciones_max=500, min_silhouette=0.65):
        # Inicialización mejorada usando KMeans++
        self.centroids = self.kmeans_plus_plus_init(X)

        for _ in range(iteraciones_max):
            y = []
            for data_point in X:
                distances = KmeansClustering.euclidean_distance(
                    data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            cluster_centers = []

            for i in range(self.k):
                indices = np.where(y == i)[0]
                if len(indices) == 0:
                    # Re-inicializar centroides vacíos a un punto aleatorio
                    new_centroid = X[np.random.randint(0, X.shape[0])]
                    cluster_centers.append(new_centroid)
                    print(f"Reinicializando centroid {
                          i} a un nuevo punto aleatorio.")
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0))

            if np.linalg.norm(self.centroids - cluster_centers) < 1e-6:
                break
            else:
                self.centroids = np.array(cluster_centers)

            # Evaluar la calidad del clustering
            quality = self.evaluate_clustering(X, y)
            if quality >= min_silhouette:
                print(f"Umbral de Silhouette alcanzado: {quality}")
                break
        print(self.centroids)
        return y

    def kmeans_plus_plus_init(self, X):
        """Inicializa los centroides usando el método KMeans++"""
        centroids = []
        centroids.append(X[np.random.randint(0, X.shape[0])])
        for _ in range(1, self.k):
            # Corrige el cálculo de dist_sq para obtener la distancia mínima cuadrada por punto de datos
            dist_sq = np.array(
                [np.min(self.euclidean_distance(x, centroids))**2 for x in X])

            probabilities = dist_sq / dist_sq.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for idx, cum_prob in enumerate(cumulative_probabilities):
                if r < cum_prob:
                    centroids.append(X[idx])
                    break
        return np.array(centroids)

    def plot_clusters(self, X, y):
        """Visualiza los clusters y centroides"""
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', marker='x', s=200, linewidths=3,
                    label='Centroides')
        plt.title('Clusters K-means con Centroides')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.show()

    def plot_clusters_3d(self, X, y):
        """Visualiza los clusters y centroides en 3D"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                             c=y, cmap='viridis')
        ax.scatter(self.centroids[:, 0],
                   self.centroids[:, 1],
                   self.centroids[:, 2],
                   c='red', marker='x', s=200, linewidths=3,
                   label='Centroides')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('Clusters K-means 3D con Centroides')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.show()

    def evaluate_clustering(self, X, y):
        """Evalúa la calidad del clustering usando el coeficiente de silueta"""
        if len(np.unique(y)) > 1:
            score = silhouette_score(X, y)
        else:
            score = -1  # Valor inválido para indicar que no se puede calcular el Silhouette Score
        return score


def run_clustering():
    # Cargar datos completos del CSV
    df_resultados = pd.read_csv('umap_resultados.csv')
    caracteristicas = ['UMAP1', 'UMAP2', 'UMAP3']
    X = df_resultados[caracteristicas].values

    # Crear y ajustar el modelo
    kmeans = KmeansClustering(k=4)
    clusters = kmeans.fit(X)

    # Añadir la columna de clusters al dataframe original
    df_resultados['Cluster'] = clusters
    df_resultados['label'] = df_resultados['Nombre']

    # Guardar resultados del clustering
    df_resultados.to_csv('resultados_clustering.csv', index=False)

    # Guardar centroides en un archivo separado
    centroides_df = pd.DataFrame(
        kmeans.centroids,
        columns=caracteristicas
    )
    centroides_df.index.name = 'Cluster'
    centroides_df.to_csv('centroides.csv')

    # Visualizar los clusters con centroides en 3D
    kmeans.plot_clusters_3d(X, clusters)

    # Evaluar y mostrar la calidad del clustering
    quality = kmeans.evaluate_clustering(X, clusters)
    if quality != -1:
        print(f"Clustering quality (Silhouette Score): {quality}")
    else:
        print("No se puede calcular el Silhouette Score con un solo cluster.")
    
    # Mostrar resultados de la clusterización
    print("Resultados de la clusterización:")
    print(clusters)
    print("Cantidad de puntos por clúster:")
    print(pd.Series(clusters).value_counts())
    
    # Mostrar cantidad de etiquetas por clúster
    cluster_label_counts = pd.crosstab(df_resultados['label'], df_resultados['Cluster'])
    print("Cantidad de etiquetas por clúster:")
    print(cluster_label_counts)


if __name__ == "__main__":
    run_clustering()
