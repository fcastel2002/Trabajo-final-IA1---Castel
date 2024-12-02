import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from umap import UMAP

def realizar_analisis_caracteristicas():
    data = pd.read_csv('../runtime_files/resultados.csv')
    print(data.head())
    print(data.columns)
    data.columns = data.columns.str.strip()
    features_header = ['Hu2', 'Hu3']+['Mean_B', 'Mean_G', 'Mean_R']
    missing_columns = [
        col for col in features_header if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

    X = data[features_header].values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    with open('../runtime_files/scaler_images.pkl', 'wb') as f:
        pickle.dump(scaler, f)

        # Guardar las features escaladas en un archivo .csv
    scaled_features_df = pd.DataFrame(data=X_scaled, columns=features_header)
    scaled_features_df.to_csv('../runtime_files/scaled_features.csv', index=False)

    umap = UMAP(n_components=3)
    X_umap = umap.fit_transform(X_scaled)
    with open('../runtime_files/umap_images.pkl', 'wb') as f:
        pickle.dump(umap, f)
    umap_df = pd.DataFrame(data=X_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])
    umap_df['Nombre'] = data['Nombre']  # Add class labels to the UMAP dataframe

    # print("Variancia explicada por cada componente:")
    # for i, var in enumerate(umap.explained_variance_ratio_):
    #     print(f"PC{i+1}: {var:.2f}")

    # Codificar los nombres de las verduras
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['Nombre'])

    # Calculate variance along each UMAP component for each vegetable
    print("\nVariance along each UMAP Component for each vegetable:")
    for vegetable in data['Nombre'].unique():
        umap_variance = umap_df[umap_df['Nombre'] == vegetable][['UMAP1', 'UMAP2', 'UMAP3']].var()
        print(f"{vegetable}:")
        print(umap_variance)

    # Display feature loadings for UMAP1, UMAP2, and UMAP3
    # Note: UMAP does not have feature loadings like PCA, so this part is omitted

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'], umap_df['UMAP3'], c=labels, cmap='viridis', s=50)

    # Add legend to show names corresponding to each color
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(
        label / max(labels)), markersize=10) for label in unique_labels]
    legend1 = ax.legend(handles, label_encoder.inverse_transform(
        unique_labels), title='Verduras')
    ax.add_artist(legend1)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')

    # Guardar los resultados del UMAP en un archivo .csv
    umap_df.to_csv('../runtime_files/umap_resultados.csv', index=False)

    plt.show()
