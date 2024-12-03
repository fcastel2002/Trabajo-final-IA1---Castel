import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import os
import pickle
from sklearn.manifold import TSNE
import umap


class AnalisisPCA:

    def __init__(self, database):
        self.database = database
        self.pca = None
        self.feature_names = []

    def filtrar_caracteristicas_por_correlacion(self, df, umbral=0.2, metodo="mutual_info"):
        """
        Filtra características basado en correlación lineal y no lineal.

        Args:
            df: DataFrame con las características.
            umbral: valor mínimo de correlación promedio (default 0.01).
            metodo: método de correlación no lineal, "mutual_info" o "spearman" (default "mutual_info").

        Returns:
            DataFrame filtrado.
        """
        try:
            # Calcular matriz de correlación lineal
            corr_matrix = df.corr().abs()
            avg_corr = corr_matrix.mean()

            if metodo == "mutual_info":
                # Calcular mutual information (no lineal)
                mi_scores = []
                for col in df.columns:
                    # Mutual info entre cada característica y las demás
                    otros = df.drop(columns=[col])
                    mi = mutual_info_regression(otros, df[col])
                    mi_scores.append(mi.mean())
                mi_scores = pd.Series(mi_scores, index=df.columns)

            elif metodo == "spearman":
                # Calcular correlación de Spearman
                corr_spearman = df.corr(method="spearman").abs()
                avg_corr = corr_spearman.mean()

            else:
                raise ValueError(f"Método desconocido: {metodo}")

            # Combinar correlaciones
            avg_corr_combined = avg_corr + mi_scores if metodo == "mutual_info" else avg_corr
            avg_corr_combined /= avg_corr_combined.max()  # Normalizar

            # Filtrar características
            features_to_keep = avg_corr_combined[avg_corr_combined >= umbral].index
            print(
                f"\nCaracterísticas eliminadas por baja correlación (umbral={umbral}):")
            print(set(df.columns) - set(features_to_keep))

            return df[features_to_keep]

        except Exception as e:
            print("Error al filtrar características:", e)
            return df

    def calcular_importancia(self, pca, feature_names):
        """
        Calcula la importancia de las características basado en PCA.

        Args:
            pca: Objeto PCA ajustado
            feature_names: Nombres de las características

        Returns:
            DataFrame con la importancia de las características
        """
        # Calcular importancia usando varianza explicada y loadings
        loadings = np.abs(pca.components_)
        importancia = np.sum(
            loadings * pca.explained_variance_ratio_[:, np.newaxis], axis=0)

        # Normalizar importancia para que sume 1
        importancia = importancia / np.sum(importancia)

        # Crear DataFrame con resultados
        importancia_df = pd.DataFrame({
            'Característica': feature_names,
            'Importancia': importancia
        }).sort_values(by='Importancia', ascending=False)

        return importancia_df

    def calcular_y_filtrar_por_importancia(self, features, umbral_importancia=0.0167):
        try:
            # Estandarizar datos
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Calcular PCA
            pca_temp = PCA()
            pca_temp.fit(features_scaled)

            # Calcular importancia
            importancia_df = self.calcular_importancia(
                pca_temp, features.columns)

            # Filtrar características
            features_importantes = importancia_df[importancia_df['Importancia']
                                                  >= umbral_importancia]['Característica']

            print(f"\nCaracterísticas eliminadas por baja importancia (umbral={
                  umbral_importancia}):")
            print(set(features.columns) - set(features_importantes))
            print("\nImportancia de características:")
            print(importancia_df)

            # Visualizar importancia
            plt.figure(figsize=(10, 6))
            plt.barh(importancia_df['Característica'],
                     importancia_df['Importancia'], color='skyblue')
            plt.xlabel('Importancia relativa')
            plt.title('Importancia de características')
            plt.axvline(x=umbral_importancia, color='r', linestyle='--',
                        label=f'Umbral ({umbral_importancia:.3f})')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.grid(axis='x')
            plt.tight_layout()
            plt.show()

            return features[features_importantes]

        except Exception as e:
            print("Error al calcular y filtrar por importancia:", e)
            return features

    def analisis_pca_con_outliers(self, distancia_min=20, min_muestras=15):
        try:
            # Cargar y preparar datos
            df = pd.read_csv(self.database.csv_file)
            df['Orden'] = df.index  # Añadir columna de orden
            etiquetas = df['Etiqueta']
            original_df = df.copy()

            features = original_df.drop(['Etiqueta', 'Orden'], axis=1)

            try:  # Estandarizar datos
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                with open('scaler.pkl', 'wb') as file:
                    pickle.dump(scaler, file)
            except Exception as e:
                print("Error al estandarizar datos: ", e)
                return None

            # DBSCAN para detección de outliers por etiqueta
            outliers_global = []

            for etiqueta in df['Etiqueta'].unique():
                # Filtrar los datos para la etiqueta actual
                mask = df['Etiqueta'] == etiqueta
                features_by_label = features_scaled[mask]

                # Aplicar DBSCAN
                dbscan = DBSCAN(eps=distancia_min, min_samples=min_muestras)
                etiquetas_dbscan = dbscan.fit_predict(features_by_label)

                # Identificar índices de outliers dentro de la etiqueta
                outliers = np.where(etiquetas_dbscan == -1)[0]

                # Mapear índices locales a índices globales
                outliers_global.extend(df[mask].iloc[outliers].index)

            # Eliminar outliers detectados
            outliers_global = np.array(outliers_global)
            clean_df = original_df.drop(index=outliers_global).reset_index(
                drop=True) if len(outliers_global) > 0 else original_df.copy()

            print(f"Outliers eliminados (índices): {outliers_global}")
            try:
                # Aplicar PCA para determinar dimensiones relevantes
                clean_features = clean_df.drop(['Etiqueta', 'Orden'], axis=1)
                clean_features_scaled = scaler.fit_transform(clean_features)
                cl_features_df = pd.DataFrame(
                    clean_features_scaled, columns=clean_features.columns)

                cl_features_df['Etiqueta'] = clean_df['Etiqueta']
                cl_features_df['Orden'] = clean_df['Orden']
                varianza_por_clase = cl_features_df.groupby(
                    'Etiqueta').var()

                varianza_por_clase.to_csv('varianza_por_clase.csv')

                pca_full = PCA()
                pca_full.fit(clean_features_scaled)
            except Exception as e:
                print("Error al aplicar PCA: ", e)
                return None
            # Varianza explicada acumulada
            varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
            print("\nVarianza explicada acumulada por número de componentes:")
            for i, var in enumerate(varianza_acumulada):
                print(f"{i+1} componentes: {var:.2f}")

            plt.plot(varianza_acumulada)
            plt.xlabel('Número de Componentes')
            plt.ylabel('Varianza Explicada Acumulada')
            plt.title('Curva de Varianza Explicada Acumulada')
            plt.show()

            # Generar un nuevo dataframe con 7 componentes principales
            try:
                pca_comp = 5
                pca_7d = PCA(n_components=pca_comp)
                principalComponents_7d = pca_7d.fit_transform(
                    clean_features_scaled)

                PCA7D_df = pd.DataFrame(data=principalComponents_7d, columns=[
                                        f'PC{i+1}' for i in range(pca_comp)])
                PCA7D_df['Etiqueta'] = clean_df['Etiqueta']
                varianza_pca = PCA7D_df.groupby('Etiqueta').var()
                varianza_pca.to_csv('varianza_pca.csv')
                # Calcular varianza explicada acumulada para las componentes elegidas
                varianza_acumulada_7d = np.cumsum(
                    pca_7d.explained_variance_ratio_)
                print("\nVarianza explicada acumulada por las componentes elegidas:")
                for i, var in enumerate(varianza_acumulada_7d):
                    print(f"{i+1} componentes: {var:.2f}")

            except Exception as e:
                print(
                    "Error al generar un nuevo dataframe con 7 componentes principales: ", e)
                return None
            explained_variance_7d = pca_7d.explained_variance_ratio_
            print("\nVarianza explicada por las 7 componentes principales:",
                  explained_variance_7d)

            principalDf_7d = pd.DataFrame(data=principalComponents_7d,
                                          columns=[f'PC{i+1}' for i in range(pca_comp)])
            finalDf_7d = pd.concat(
                [principalDf_7d, clean_df['Etiqueta'], clean_df['Orden']], axis=1)

            with open('pca_model.pkl', 'wb') as file:
                pickle.dump(pca_7d, file)

            # Crear dataframe para visualización 3D
            try:
                pca_3d = PCA(n_components=3)
                principalComponents_3d = pca_3d.fit_transform(
                    clean_features_scaled)
            except Exception as e:
                print("Error al crear dataframe para visualización 3D: ", e)
                return None

            explained_variance_3d = pca_3d.explained_variance_ratio_
            print("\nVarianza explicada por las 3 componentes principales:",
                  explained_variance_3d)

            principalDf_3d = pd.DataFrame(
                data=principalComponents_3d, columns=['PC1', 'PC2', 'PC3'])
            finalDf_3d = pd.concat(
                [principalDf_3d, clean_df['Etiqueta'], clean_df['Orden']], axis=1)

            # Graficar PCA 3D
            targets = ['berenjena', 'camote', 'papa', 'zanahoria']
            colors = ['r', 'g', 'b', 'y']

            fig_3d = plt.figure(figsize=(8, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.set_xlabel('Principal Component 1', fontsize=15)
            ax_3d.set_ylabel('Principal Component 2', fontsize=15)
            ax_3d.set_zlabel('Principal Component 3', fontsize=15)
            ax_3d.set_title('3 Componentes PCA (Sin Outliers)', fontsize=20)

            for target, color in zip(targets, colors):
                indicesToKeep = finalDf_3d['Etiqueta'] == target
                ax_3d.scatter(finalDf_3d.loc[indicesToKeep, 'PC1'],
                              finalDf_3d.loc[indicesToKeep, 'PC2'],
                              finalDf_3d.loc[indicesToKeep, 'PC3'], c=color, s=50)

            ax_3d.legend(targets)
            ax_3d.grid()
            plt.show()

            return clean_df, finalDf_7d, finalDf_3d, outliers

        except Exception as e:
            print("Error al realizar análisis PCA con outliers: ", e)
            return None

    def get_filtered_dataframe(self, eps, min_samples, test):
        try:
            # Perform PCA analysis to filter out problematic points
            if test:
                # df, filtered_df, finalDF3d, puntos_problematicos_totales = self.analisis_pca()
                pass
            else:
                # df, filtered_df, finalDF3d, puntos_problematicos_totales = self.analisis_pca_con_outliers(
                #     eps, min_samples)
                filtered_df = self.analisis_UMAP()

            # Extract features and labels from the filtered DataFrame
            features = filtered_df.drop(['Etiqueta', 'Orden'], axis=1)
            etiquetas = filtered_df['Etiqueta']

            return pd.concat([features, etiquetas], axis=1)
        except Exception as e:
            print("Error al obtener el DataFrame filtrado: ", e)
            return None

    def calcular_importancia_int(self):
        try:
            if not hasattr(self, 'pca') or not hasattr(self, 'feature_names'):
                print("Debe ejecutar `analisis_pca` antes de calcular la importancia.")
                return

            # Calcular la importancia total para cada característica
            importancia_df = self.calcular_importancia(
                self.pca, self.feature_names)

            print("\nImportancia de las características:")
            print(importancia_df)

            # Gráfico de importancia
            plt.figure(figsize=(10, 6))
            plt.barh(importancia_df['Característica'],
                     importancia_df['Importancia'], color='skyblue')
            plt.xlabel('Importancia acumulada')
            plt.title('Importancia de características basada en PCA')
            plt.gca().invert_yaxis()
            plt.grid(axis='x')
            plt.show()

        except Exception as e:
            print("Error al calcular la importancia: ", e)
            return None

    def guardar_dataframe_filtrado(self, output_path, eps=15, min_samples=20, test=False):
        """
        Guarda el DataFrame filtrado en un archivo CSV.

        Args:
            output_path: Ruta del archivo CSV de salida.
        """
        try:
            filtered_df = self.get_filtered_dataframe(eps, min_samples, test)
            if os.path.exists(output_path):
                os.remove(output_path)
            if filtered_df is not None:
                filtered_df.to_csv(output_path, index=False)
                print(f"DataFrame filtrado guardado en {output_path}")
            else:
                print("No se pudo obtener el DataFrame filtrado.")
        except Exception as e:
            print("Error al guardar el DataFrame filtrado: ", e)

    def analisis_UMAP(self, n_components=3, n_neighbors=15, min_dist=0.15):
        """
        Performs UMAP analysis on the dataset.

        Args:
            n_components: Number of dimensions to reduce to (default=2).
            n_neighbors: Number of neighbors for UMAP (default=15).
            min_dist: Minimum distance between points in the low-dimensional space (default=0.1).

        Returns:
            DataFrame with UMAP components, labels, and order.
        """
        try:
            # Load and prepare data
            df = pd.read_csv(self.database.csv_file)
            df['Orden'] = df.index  # Add 'Orden' column
            etiquetas = df['Etiqueta']
            features = df.drop(['Etiqueta', 'Orden'], axis=1)

            # Standardize data
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            with open('scaler_umap.pkl', 'wb') as file:
                pickle.dump(scaler, file)

            # Apply UMAP
            umap_model = umap.UMAP(
                n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            umap_components = umap_model.fit_transform(features_scaled)
            with open('umap_model.pkl', 'wb') as file:
                pickle.dump(umap_model, file)

            # Create DataFrame with UMAP components
            umap_columns = [f'Dim{i+1}' for i in range(n_components)]
            umap_df = pd.DataFrame(data=umap_components, columns=umap_columns)
            final_df = pd.concat(
                [umap_df, df['Etiqueta'], df['Orden']], axis=1)

            targets = df['Etiqueta'].unique()
            colors = sns.color_palette("hsv", len(targets))
            # Visualize UMAP with 2 components if n_components is 2
            if n_components == 2:
                plt.figure(figsize=(8, 8))
                ax = plt.gca()
                ax.set_xlabel('Dim1')
                ax.set_ylabel('Dim2')
                ax.set_title('Visualización UMAP 2D')

                for target, color in zip(targets, colors):
                    indicesToKeep = final_df['Etiqueta'] == target
                    ax.scatter(final_df.loc[indicesToKeep, 'Dim1'],
                               final_df.loc[indicesToKeep, 'Dim2'],
                               c=[color], s=50, label=target)

                ax.legend()
                plt.show()
            else:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('Dim1')
                ax.set_ylabel('Dim2')
                ax.set_zlabel('Dim3')
                ax.set_title('Visualización UMAP 3D')
                for target, color in zip(targets, colors):
                    indicesToKeep = final_df['Etiqueta'] == target
                    ax.scatter(final_df.loc[indicesToKeep, 'Dim1'],
                               final_df.loc[indicesToKeep, 'Dim2'],
                               final_df.loc[indicesToKeep, 'Dim3'],
                               c=[color], s=50)

                ax.legend(targets)
                plt.show()

            return final_df

        except Exception as e:
            print("Error al realizar análisis UMAP: ", e)
            return None
