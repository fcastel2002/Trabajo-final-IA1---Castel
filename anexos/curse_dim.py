import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class CurseDimensionalityVisualizer:
    def __init__(self):
        # Configuración inicial
        self.n_points = 1000
        self.initial_dim = 2
        
        # Crear la figura y los ejes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)  # Hacer espacio para el slider
        
        # Configurar el slider
        ax_dim = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.dim_slider = Slider(
            ax=ax_dim,
            label='Dimensiones',
            valmin=2,
            valmax=100,
            valinit=self.initial_dim,
            valstep=1
        )
        
        # Configurar el botón de reset
        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        
        # Conectar eventos
        self.dim_slider.on_changed(self.update)
        self.reset_button.on_clicked(self.reset)
        
        # Dibujar el histograma inicial
        self.update(self.initial_dim)
        
        # Configurar títulos y etiquetas
        self.ax.set_title('Maldición de la Dimensionalidad')
        self.ax.set_xlabel('Distancia Euclidiana')
        self.ax.set_ylabel('Frecuencia')
        
    def calculate_distances(self, dim):
        """Calcula las distancias entre puntos en dim dimensiones"""
        points = np.random.random((self.n_points, dim))
        distances = np.sqrt(np.sum((points - points[0])**2, axis=1))
        return distances
    
    def update(self, dim):
        """Actualiza el histograma cuando cambia la dimensión"""
        self.ax.clear()
        distances = self.calculate_distances(int(dim))
        self.ax.hist(distances, bins=50, density=True)
        self.ax.set_title(f'Distribución de Distancias en {int(dim)} Dimensiones')
        self.ax.set_xlabel('Distancia Euclidiana')
        self.ax.set_ylabel('Frecuencia')
        plt.draw()
    
    def reset(self, event):
        """Resetea el slider a su valor inicial"""
        self.dim_slider.reset()
    
    def show(self):
        """Muestra la visualización"""
        plt.show()

# Crear y mostrar la visualización
visualizer = CurseDimensionalityVisualizer()
visualizer.show()