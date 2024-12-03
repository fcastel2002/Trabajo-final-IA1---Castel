import csv
import os
import pandas as pd
import numpy as np
from CaracteristicasAudios import AudioFeatures
import soundfile as sf


class DataBase:

    def __init__(self, carpeta_prefiltrados, output_csv):
        self.carpeta_prefiltrados = carpeta_prefiltrados
        self.csv_file = output_csv
        self.csv_file_std = 'caracteristicas_std.csv'
        self.csv_file_normalized = 'caracteristicas_normalized.csv'
        self.header = []

    def get_header(self, segmentos, n_mfcc, n_formantes, n_f_amp, n_zcr, n_rms, n_flt, n_spc_bw):

        for i in n_zcr:
            self.header.append(f'zcr_{i}')
        for i in n_rms:
            self.header.append(f'rms_{i}')

        for k in range(segmentos):
            for j in n_mfcc:
                self.header.append(f'mfcc_{j}_{k}')
                # self.header.append(f'delta_mfcc_{j}_{k}')
                # self.header.append(f'delta_delta_mfcc_{j}_{k}')
        # for k in n_formantes:
        #     self.header.append(f'formantes_{k}')

        # for k in range(n_formantes-1):
        #     self.header.append(f'formantes_diff_{k}')

        # for k in n_f_amp:
        #     self.header.append(f'amp_formantes_{k}')

        for k in n_spc_bw:
            self.header.append(f'spc_bw_{k}')
        for k in n_flt:
            self.header.append(f'flatness_{k}')

        for i in range(segmentos):
            self.header.append(f'energy_entropy_{i}')
        for i in range(segmentos):
            self.header.append(f'power_spectrum_mean_{i}')
        for i in range(segmentos):
            self.header.append(f'spectral_centroid_{i}')

        for k in range(segmentos):
            for j in n_mfcc:
                self.header.append(f'mfcc_max_{j}_{k}')
                self.header.append(f'mfcc_min_{j}_{k}')
                self.header.append(f'mfcc_std_{j}_{k}')

        self.header.append('duracion')
        self.header.append('Etiqueta')

    def extraer_caracteristicas(self):
        try:
            if os.path.exists(self.csv_file):
                print(f"Eliminando archivo existente: {self.csv_file}")
                os.remove(self.csv_file)
        except PermissionError as e:
            print(f"Error de permiso al eliminar el archivo: {e}")
            return
        except Exception as e:
            print(f"Error inesperado al eliminar el archivo: {e}")
        self.header = []

        flag = False
        n_segmentos = 4
        n_mfcc = 13
        n_formantes = 0
        # zcr_index = list(range(n_segmentos))
        rms_index = list(range(n_segmentos))
        mfcc_index = [i for i in range(n_mfcc) if i not in [9]]
        # afts_index = [0, 1, 2, 3, 4]
        # fts_index = [1, 2, 3, 4]
        # flatness_index = [0, 1, 2]
        # spc_bw_index = [0, 1, 2]
        # mfcc_index = []
        # rms_index = []
        zcr_index = []
        afts_index = []
        fts_index = []
        flatness_index = []
        spc_bw_index = []

        for root, dirs, files in os.walk(self.carpeta_prefiltrados):
            for file in files:
                if file.endswith(".wav"):
                    print(f"Extrayendo características de {file}")
                    ruta = os.path.join(root, file)
                    print(ruta)
                    # Crear instancia de Caracteristicas

                    # Ajusta parámetros según necesidad
                    caracteristicas = AudioFeatures(
                        ruta, n_segmentos=n_segmentos,
                        n_mfcc=n_mfcc,
                        n_formantes=n_formantes,
                        zcr_indices=zcr_index,
                        mfcc_indices=mfcc_index,
                        formantes_indices=fts_index,
                        amp_formantes_indices=afts_index,
                        rms_indices=rms_index, flat_indices=flatness_index, spc_bw_indices=spc_bw_index)
                    features = caracteristicas.extraer()

                    etiqueta = file.split("_")[0]
                    self.guardar_en_csv(features, etiqueta,
                                        flag, n_segmentos, mfcc_index,
                                        fts_index, afts_index, rms_index, zcr_index, flatness_index, spc_bw_index)
                    flag = True

    def guardar_en_csv(self, features, etiqueta, flag, n_s, n_m, n_f, n_f_amp, n_rms, n_zcr, flatness_index, spc_bw_index):
        try:
            # if not all(isinstance(f, (int, float)) for f in features):
            #     print(f"Advertencia: Características no numéricas detectadas en {
            #           etiqueta}. Revisar.")
            #     return

            mode = 'a' if os.path.exists(self.csv_file) else 'w'
            with open(self.csv_file, mode=mode, newline='') as file:
                writer = csv.writer(file)
                if mode == 'w':
                    # Write the header only if the file is being created
                    self.get_header(segmentos=n_s, n_mfcc=n_m, n_formantes=n_f,
                                    n_f_amp=n_f_amp, n_rms=n_rms, n_zcr=n_zcr, n_flt=flatness_index, n_spc_bw=spc_bw_index)
                    writer.writerow(self.header)
                features = list(features)
                features.append(etiqueta)
                writer.writerow(features)
        except Exception as e:
            print("Error al guardar en CSV: ", e)

    def estandarizar_database(self):
        try:
            if os.path.exists(self.csv_file_std):
                os.remove(self.csv_file_std)

            # Cargar base de datos original
            data = pd.read_csv(self.csv_file)
            etiquetas = data.pop('Etiqueta')
            data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Estandarizar características
            data = (data - data.mean()) / data.std()

            # Agregar etiquetas nuevamente
            data['Etiqueta'] = etiquetas

            # Guardar archivo estandarizado
            with open(self.csv_file_std, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)
                writer.writerows(data.values)
        except Exception as e:
            print("Error al estandarizar la base de datos: ", e)

    # def normalizar_database(self):
    #     try:
    #         csv_file_normalized = 'caracteristicas_normalized.csv'
    #         if os.path.exists(csv_file_normalized):
    #             os.remove(csv_file_normalized)

    #         # Cargar base de datos original
    #         data = pd.read_csv(self.csv_file)
    #         etiquetas = data.pop('Etiqueta')

    #         # Normalizar características
    #         data = (data - data.min()) / (data.max() - data.min())

    #         # Agregar etiquetas nuevamente
    #         data['Etiqueta'] = etiquetas

    #         # Guardar archivo normalizado
    #         with open(csv_file_normalized, 'w', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(self.header)
    #             writer.writerows(data.values)
    #     except Exception as e:
    #         print("Error al normalizar la base de datos: ", e)
