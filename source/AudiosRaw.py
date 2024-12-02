import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import keyboard  
from pydub import AudioSegment
import random


class AudiosRaw:
    def __init__(self,  audio_path, frecuencia_muestreo=44100):
        self.frecuencia_muestreo = frecuencia_muestreo
        self.audio = None
        self.start_time = None
        self.counter = 0
        self.path_raw = audio_path
        
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

   
