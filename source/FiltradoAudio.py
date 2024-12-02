import numpy as np
import librosa as lb
import os
import soundfile as sf
from scipy import signal  # Añadir esta importación
import pyloudnorm as pyln
import audioread as ar


class Filtrado:

    def __init__(self, directorio_crudos, directorio_prefiltrados):
        self.directorio_crudos = directorio_crudos
        self.directorio_prefiltrados = directorio_prefiltrados
        self.f0_promedio = None

    def filtrar_y_normalizar_v2(self, audio, sr, filename):
        try:
            # Paso 1: Pre-énfasis
            audio_filtrado = lb.effects.preemphasis(audio, coef=0.99)

            # Paso 2: Filtro paso banda dinámico
            nyquist = sr / 2
            filter_order = 4
            lowcut = 50
            # Ajustar highcut si supera el Nyquist
            highcut = min(8000, nyquist - 1)

            if lowcut >= highcut:
                raise ValueError(f"Los valores de lowcut ({lowcut}) y highcut ({
                                 highcut}) son inválidos para sr={sr}.")

            b, a = signal.butter(
                filter_order, [lowcut / nyquist, highcut / nyquist], btype='band')

            try:
                audio_filtrado = signal.filtfilt(b, a, audio_filtrado)
            except ValueError as e:
                print(f"Error en filtfilt: {e}")
                return None

            # Paso 3: Reducción de ruido
            D = lb.stft(audio_filtrado, n_fft=2048, hop_length=512)
            S = np.abs(D)
            noise_threshold = np.mean(S, axis=1, keepdims=True) * 1.5
            noise_threshold = np.repeat(noise_threshold, S.shape[1], axis=1)
            mask = (S > noise_threshold) * (S / noise_threshold)
            mask = np.clip(mask, 0, 1)
            mask = lb.decompose.nn_filter(
                mask, aggregate=np.median, metric='cosine')
            S_clean = S * mask
            phase = np.angle(D)
            D_clean = S_clean * np.exp(1.0j * phase)
            audio_filtrado = lb.istft(D_clean)

            # Paso 6: Normalización de Loudness
            meter = pyln.Meter(sr)  # Inicializar el medidor de loudness
            loudness_actual = meter.integrated_loudness(audio_filtrado)
            # Nivel de loudness objetivo en LUFS (estándar)
            loudness_objetivo = -23.0
            audio_filtrado = pyln.normalize.loudness(
                audio_filtrado, loudness_actual, loudness_objetivo)

            return audio_filtrado

        except Exception as e:
            print(f"Error al filtrar y normalizar audio: {e}")
            return None

    def procesar_audios(self):

        os.makedirs(self.directorio_prefiltrados, exist_ok=True)
        try:
            for root, dirs, files in os.walk(self.directorio_crudos):
                for file in files:
                    if file.endswith(".wav"):
                        ruta = os.path.join(root, file)
                        try:
                            # Use soundfile to read audio
                            audio, sr = sf.read(ruta)
                        except RuntimeError as e:
                            print(f"Advertencia: No se pudo leer el archivo {
                                  file}. Será ignorado. Error: {e}")
                            continue

                        # Usar el nuevo método de filtrado
                        audio_normalizado = self.filtrar_y_normalizar_v2(
                            audio, sr, file)
                        if audio_normalizado is None:
                            print(f"Advertencia: No se pudo procesar el audio {
                                  file}. Será ignorado.")
                            continue

                        # Guardar el audio preprocesado (filtrado y normalizado)
                        ruta_guardado = os.path.join(
                            self.directorio_prefiltrados, file)
                        sf.write(ruta_guardado, audio_normalizado, sr)
                        print(f"Guardado audio con nuevo filtrado en: {
                              ruta_guardado}")

        except Exception as e:
            print(f"Error al procesar audios: {e}")
            return None
