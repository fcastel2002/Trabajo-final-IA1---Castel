import csv

def calcular_eficiencia(archivo):
    with open(archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        total = 0
        correctos = 0
        for row in reader:
            total += 1
            nombre = row[0].split('_')[0]
            etiqueta = row[6]
            if nombre == etiqueta:
                correctos += 1
        eficiencia = (correctos / total) * 100
        print(f'Eficiencia: {eficiencia}%')

if __name__ == "__main__":
    archivo = '../runtime_files/predicciones.csv'
    calcular_eficiencia(archivo)