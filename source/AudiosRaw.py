import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import keyboard  # Add this import to detect key presses
from pydub import AudioSegment
import random


class AudiosRaw:
    def __init__(self,  audio_path, frecuencia_muestreo=44100):
        self.frecuencia_muestreo = frecuencia_muestreo
        self.audio = None
        self.start_time = None
        self.counter = 0
        self.folder_path = '../anexos/berenjena'
        self.path_raw = audio_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        sd.default.blocksize = 1024

    def grabar(self):
        print("Iniciando grabación...")
        self.start_time = time.time()
        self.audio = sd.rec(int(self.frecuencia_muestreo * 60),  # Allocate enough space for up to 60 seconds
                            samplerate=self.frecuencia_muestreo, channels=1, dtype='float32')

    def detener(self):
        sd.stop()
        duracion = time.time() - self.start_time
        self.audio = self.audio[:int(duracion * self.frecuencia_muestreo)]

        print("Grabación finalizada.")

    def guardar(self):
        verduras = ['berenjena', 'camote', 'papa', 'zanahoria']
        nombre_archivo = f"test_{
            random.randint(0, 10000)}.wav"

        # escalado = np.int16(self.audio * 32767 * 0.5)
        if np.max(np.abs(self.audio)) > 1.0:
            print(
                "Advertencia: El audio no está en el rango esperado (-1 a 1). Normalizando...")
            self.audio = self.audio / np.max(np.abs(self.audio))

        # Guardar el archivo en formato float64
        write(os.path.join(self.path_raw, nombre_archivo),
              self.frecuencia_muestreo, self.audio.astype('float32'))

        print(f"Audio guardado como {nombre_archivo}.")
        self.counter += 1

    def iniciar(self):
        print("Mantén presionada la tecla ESPACIO para grabar. Suelta la tecla para detener la grabación. Presiona ESC para salir.")
        self.counter = 0
        while True:
            if keyboard.is_pressed('space'):
                self.grabar()
                while keyboard.is_pressed('space'):
                    time.sleep(0.1)
                self.detener()
                self.guardar()
            elif keyboard.is_pressed('esc'):
                print("Saliendo...")
                break

    def renombrar_archivos(self, nuevo_prefijo="berenjena"):
        archivos = [f for f in os.listdir(
            self.folder_path) if f.endswith('.wav') or f.endswith('.m4a')]
        for idx, archivo in enumerate(archivos, start=7):
            if archivo.endswith(".wav"):
                nueva_nombre = f"{nuevo_prefijo}_{idx}.wav"
                os.rename(
                    os.path.join(self.folder_path, archivo),
                    os.path.join(self.folder_path, nueva_nombre)
                )
                print(f"Renombrado: {archivo} a {nueva_nombre}")
            elif archivo.endswith(".m4a"):
                nueva_nombre = f"{nuevo_prefijo}_{idx}.wav"
                ruta_archivo = os.path.join(self.folder_path, archivo)
                ruta_nueva = os.path.join(self.folder_path, nueva_nombre)
                audio = AudioSegment.from_file(ruta_archivo, format="m4a")
                audio.export(ruta_nueva, format="wav")
                os.remove(ruta_archivo)
                print(f"Convertido y renombrado: {archivo} a {nueva_nombre}")
