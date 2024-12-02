import numpy as np
import librosa as lb
import os
from scipy.signal import find_peaks
from scipy.fft import fft
import soundfile as sf  # Add this import


class AudioFeatures:

    def __init__(self, ruta, n_segmentos, n_mfcc, n_formantes,
                 zcr_indices, mfcc_indices, formantes_indices,
                 amp_formantes_indices, rms_indices, flat_indices, spc_bw_indices):
        self.ruta = ruta
        # Cargar audio ya filtrado y normalizado
        self.audio, self.fs = lb.load(ruta, sr=None)
        self.audio_name = os.path.basename(ruta)

        self.n_segmentos = n_segmentos
        self.n_mfcc = n_mfcc
        self.n_formantes = n_formantes

        # Índices para seleccionar características específicas
        self.zcr_indices = zcr_indices if zcr_indices is not None else range(
            n_segmentos)
        self.mfcc_indices = mfcc_indices if mfcc_indices is not None else range(
            n_mfcc)
        self.formantes_indices = formantes_indices if formantes_indices is not None else range(
            n_formantes)
        self.amp_formantes_indices = amp_formantes_indices if amp_formantes_indices is not None else range(
            n_formantes)
        self.rms_indices = rms_indices if rms_indices is not None else range(
            n_segmentos)
        self.flatness_indices = flat_indices if flat_indices is not None else range(
            n_segmentos)
        self.spc_bw_indices = spc_bw_indices if spc_bw_indices is not None else range(
            n_segmentos)

        # Inicializar variables de características
        self.segmentos = None
        self.zcr_ = np.array([])
        self.mfcc_ = np.array([])
        self.delta_mfcc_ = np.array([])
        self.delta_delta_mfcc_ = np.array([])
        self.formantes_ = np.array([])
        self.amp_formantes_ = np.array([])
        self.spectral_bandwidth_ = np.array([])
        self.flatness_ = np.array([])
        self.rms_ = np.array([])
        self.duracion = 0  # Inicializar la duración
        self.energy_entropy_ = []
        self.power_spectrum_mean_ = []
        self.spectral_centroid_ = []
        self.mfcc_max_ = []
        self.mfcc_min_ = []
        self.mfcc_std_ = []

        self.ventana_ms = 65  # tamaño de ventana en milisegundos
        self.umbral_percentil = 60  # percentil para el umbral de energía
        self.min_segment_ms = 80  # duración mínima de segmento en milisegundos

    def validar_segmento(self, segmento):
        if len(segmento) == 0 or np.max(np.abs(segmento)) == 0:
            return False  # Segmento vacío o silencioso
        return True

    def recortar_audio(self):
        try:
            # Calcular tamaños de ventana en muestras
            tamano_ventana = int(self.ventana_ms * self.fs / 1000)
            min_segment = int(self.min_segment_ms * self.fs / 1000)

            # Calcular energía RMS
            rms = lb.feature.rms(
                y=self.audio, frame_length=tamano_ventana, hop_length=tamano_ventana)[0]
            umbral_energia = np.percentile(rms, self.umbral_percentil)

            # Encontrar regiones activas por encima del umbral
            regiones_activas = np.where(rms > umbral_energia)[0]

            if len(regiones_activas) == 0:
                print("Advertencia: No se encontraron segmentos de alta energía.")
                return

            # Agrupar regiones activas consecutivas en segmentos
            inicio_muestras = []
            fin_muestras = []

            for i in range(len(regiones_activas)):
                if i == 0 or regiones_activas[i] != regiones_activas[i - 1] + 1:
                    inicio_muestras.append(
                        regiones_activas[i] * tamano_ventana)
                if i == len(regiones_activas) - 1 or regiones_activas[i] != regiones_activas[i + 1] - 1:
                    fin_muestras.append(
                        (regiones_activas[i] + 1) * tamano_ventana)

            # Filtrar segmentos que no cumplen con la longitud mínima
            segmentos = [
                (inicio, fin) for inicio, fin in zip(inicio_muestras, fin_muestras)
                if fin - inicio >= min_segment
            ]

            if not segmentos:
                print(
                    "Advertencia: No se encontraron segmentos que cumplan la longitud mínima.")
                return

            # Concatenar los segmentos activos en un solo audio recortado
            audio_recortado = np.concatenate(
                [self.audio[inicio:fin] for inicio, fin in segmentos])
            # print(f"Audio recortado a {len(audio_recortado)} muestras.")

            # Reemplazar el audio original por el recortado
            self.audio = audio_recortado
            # Calcular duración del audio recortado
            self.duracion = len(self.audio) / self.fs
            self.guardar_audio_recortado()
        except Exception as e:
            print(f"Error al recortar audio: {e}")

    def guardar_audio_recortado(self, output_folder="../anexos/test_recortados"):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, self.audio_name)
            sf.write(output_path, self.audio, self.fs)
            # print(f"Audio recortado guardado en: {output_path}")
        except Exception as e:
            print(f"Error al guardar audio recortado: {e}")

    def segmentar_audio(self):
        try:
            self.segmentos = [np.array(segment) for segment in np.array_split(
                self.audio, self.n_segmentos)]
        except Exception as e:
            print(f"Error al segmentar audio: {e}")

    def get_zero_crossing_rate(self):
        try:
            self.zcr_ = np.array([np.mean(lb.feature.zero_crossing_rate(
                segmento)) for segmento in self.segmentos])
        except Exception as e:
            print(f"Error al calcular zero crossing rate: {e}")

    def calcular_mfcc(self):
        try:
            self.mfcc_ = []
            self.delta_mfcc_ = []
            self.delta_delta_mfcc_ = []
            for segmento in self.segmentos:
                mfcc = lb.feature.mfcc(
                    y=segmento, sr=self.fs, n_mfcc=self.n_mfcc, n_fft=1024)
                delta_mfcc = lb.feature.delta(mfcc, width=3)
                delta_delta_mfcc = lb.feature.delta(mfcc, order=2, width=3)
                self.mfcc_.append(mfcc)
                self.delta_mfcc_.append(delta_mfcc)
                self.delta_delta_mfcc_.append(delta_delta_mfcc)
            self.mfcc_ = np.array(self.mfcc_)
            self.delta_mfcc_ = np.array(self.delta_mfcc_)
            self.delta_delta_mfcc_ = np.array(self.delta_delta_mfcc_)
        except Exception as e:
            print(f"Error al calcular MFCC: {e}")
            self.mfcc_ = np.array([])
            self.delta_mfcc_ = np.array([])
            self.delta_delta_mfcc_ = np.array([])

    def get_formantes(self):
        try:
            n_coeff = 2 * self.n_formantes + 2
            lpc_coeffs = lb.lpc(self.audio, order=n_coeff)

            # Respuesta en frecuencia de los coeficientes LPC
            freqs = np.linspace(0, self.fs / 2, len(self.audio) // 2)
            response = np.abs(fft(lpc_coeffs, n=len(self.audio) // 2))

            # Encuentra picos en la respuesta (formantes)
            peaks, _ = find_peaks(response)
            self.formantes_ = freqs[peaks][:self.n_formantes]
            # Amplitudes de los formantes
            self.amplitudes_formantes_ = response[peaks][:self.n_formantes]
            # Calcular diferencias relativas entre formantes
            if len(self.formantes_) > 1:
                self.formantes_diff_ = np.diff(
                    self.formantes_) / self.formantes_[:-1]
            else:
                self.formantes_diff_ = np.array([])
        except Exception as e:
            print(f"Error al calcular formantes: {e}")
            self.formantes_ = np.array([])
            self.amplitudes_formantes_ = np.array([])

    def calcular_spectral_bandwidth(self):
        try:
            self.spectral_bandwidth_ = np.array([
                np.mean(lb.feature.spectral_bandwidth(
                    y=segmento, sr=self.fs, n_fft=1024))
                for segmento in self.segmentos
            ])
        except Exception as e:
            print(f"Error al calcular spectral bandwidth: {e}")
            self.spectral_bandwidth_ = np.array([])

    def calcular_flatness(self):
        try:
            self.flatness_ = np.array([
                np.mean(lb.feature.spectral_flatness(y=segmento, n_fft=1024))
                for segmento in self.segmentos
            ])
        except Exception as e:
            print(f"Error al calcular spectral flatness: {e}")
            self.flatness_ = np.array([])

    def calcular_rms(self):
        try:
            self.rms_ = np.array([np.mean(lb.feature.rms(y=segmento))
                                 for segmento in self.segmentos])
        except Exception as e:
            print(f"Error al calcular RMS: {e}")
            self.rms_ = np.array([])

    def calcular_energy_entropy(self):
        self.energy_entropy_ = [
            self._entropy(segmento) for segmento in self.segmentos
        ]

    def _entropy(self, segmento):
        psd = np.abs(np.fft.fft(segmento))**2
        psd_norm = psd / np.sum(psd)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        return entropy

    def calcular_power_spectrum_mean(self):
        self.power_spectrum_mean_ = [
            np.mean(np.abs(np.fft.fft(segmento))**2)
            for segmento in self.segmentos
        ]

    def calcular_spectral_centroid(self):
        self.spectral_centroid_ = [
            np.mean(lb.feature.spectral_centroid(y=segmento, sr=self.fs))
            for segmento in self.segmentos
        ]

    def calcular_mfcc_stats(self):
        self.mfcc_max_ = []
        self.mfcc_min_ = []
        self.mfcc_std_ = []
        for mfcc in self.mfcc_:
            self.mfcc_max_.append(np.max(mfcc, axis=1))
            self.mfcc_min_.append(np.min(mfcc, axis=1))
            self.mfcc_std_.append(np.std(mfcc, axis=1))

    def extraer(self):
        try:
            self.recortar_audio()
            self.segmentar_audio()

            # Extraer características por segmento
            features = []

            # ZCR para cada segmento
            self.get_zero_crossing_rate()
            for idx in self.zcr_indices:
                features.append(self.zcr_[idx])

            # RMS para cada segmento
            self.calcular_rms()
            for idx in self.rms_indices:
                features.append(self.rms_[idx])

            # MFCC y deltas por segmento
            self.calcular_mfcc()
            for seg_idx in range(self.n_segmentos):
                for idx in self.mfcc_indices:
                    features.append(np.mean(self.mfcc_[seg_idx][idx]))

            # Formantes y sus diferencias relativas
            self.get_formantes()
            if len(self.formantes_diff_) > 0:
                for diff in self.formantes_diff_:
                    features.append(diff)

            for idx in self.amp_formantes_indices:
                if idx < len(self.amplitudes_formantes_):
                    features.append(self.amplitudes_formantes_[idx])

            # Spectral Bandwidth por segmento
            self.calcular_spectral_bandwidth()
            for idx in self.spc_bw_indices:
                if idx < len(self.spectral_bandwidth_):
                    features.append(self.spectral_bandwidth_[idx])

            # Flatness por segmento
            self.calcular_flatness()
            for idx in self.flatness_indices:
                if idx < len(self.flatness_):
                    features.append(self.flatness_[idx])

            # Añadir la duración del audio recortado a las características
            features.append(self.duracion)

            self.calcular_energy_entropy()
            self.calcular_power_spectrum_mean()
            self.calcular_spectral_centroid()
            self.calcular_mfcc_stats()

            # Añadir energy entropy
            features.extend(self.energy_entropy_)

            # Añadir power spectrum mean
            features.extend(self.power_spectrum_mean_)

            # Añadir spectral centroid
            features.extend(self.spectral_centroid_)

            # Añadir mfcc max, min, std
            for seg_idx in range(self.n_segmentos):
                for idx in self.mfcc_indices:
                    features.append(self.mfcc_max_[seg_idx][idx])
                    features.append(self.mfcc_min_[seg_idx][idx])
                    features.append(self.mfcc_std_[seg_idx][idx])

            features = np.array(features)
            if not np.all(np.isfinite(features)):
                print(f"Advertencia: Características incompletas en {
                      self.audio_name}. Rellenando valores faltantes.")
                features = np.nan_to_num(features)

            return features

        except Exception as e:
            print(f"Error al extraer características: {e}")
            return None
