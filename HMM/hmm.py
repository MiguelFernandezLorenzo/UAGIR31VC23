import os
import pandas as pd
from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Acciones disponibles
actions = np.array(['a', 'adios', 'amarillo', 'apellido', 'aprender', 'azul', 'bien', 'buenasnoches',
                    'buenastardes', 'buenosdias', 'camisa', 'catorce', 'cinco', 'color', 'cuatro',
                    'dar', 'de', 'decir', 'denada', 'diecinueve', 'dieciocho', 'dieciseis', 'diecisiete',
                    'diez', 'doce', 'donde', 'dos', 'el', 'ella', 'ensenar', 'entender', 'espana', 'estudiar',
                    'euro', 'gracias', 'gris', 'gustar', 'haber', 'hastaluego', 'hermano', 'hola', 'madre',
                    'mal', 'marron', 'morado', 'mucho', 'muchos', 'nacer', 'negro', 'nino', 'no', 'nohaber',
                    'nombre', 'nosaber', 'nosotras', 'nosotros', 'nueve', 'ocho', 'ojos', 'olvidar', 'once',
                    'padre', 'perdon', 'poder', 'porfavor', 'preguntar', 'quetal', 'quince', 'regular',
                    'repetir', 'rojo', 'saber', 'seis', 'si', 'siete', 'signar', 'signo', 'su', 'trabajar',
                    'trece', 'tres', 'tu', 'universidad', 'uno', 'veinte', 'verbos', 'verde', 'vivir',
                    'vosotras', 'vosotros', 'yo'])

# Directorio que contiene los archivos de datos (un archivo por video)
data_directory = "Dataset/separated_gestures"

# Lista para almacenar características y etiquetas
X_list = []
y_list = []

# Paso 1: Preprocesamiento de datos
# Iterar sobre los archivos en el directorio de datos
for action in actions:
    for i in range(40):
        # Cargar los datos del archivo CSV
        filepath = os.path.join(data_directory, action, action + '{}.csv'.format(i + 1))
        df = pd.read_csv(filepath)

        # Agregar las características a la lista
        X_list.append(df.values)  # df.values es una matriz (None, 42)
        y_list.append(action)

# Encontrar la longitud máxima de las secuencias
max_length = max(len(seq) for seq in X_list)

# Rellenar las secuencias con ceros para que todas tengan la misma longitud
X_padded = np.zeros((len(X_list), max_length, X_list[0].shape[1]))
for i, seq in enumerate(X_list):
    X_padded[i, :len(seq), :] = seq

# Convertir las listas en matrices numpy
X = np.array(X_padded)
y = np.array(y_list)

# Flatten la matriz X a (3640, 4200) ya que hmmlearn espera que las secuencias sean bidimensionales
X_flatten = X.reshape((X.shape[0], -1))

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2, random_state=42)

# Crea y entrena el modelo HMM
n_states = 3  # Experimenta con diferentes valores
model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)  # Usamos "diag" en lugar de "full"
model.fit(X_train) 

log_likelihood = model.score(X_test)
print(f'Log-verosimilitud del modelo HMM en el conjunto de prueba: {log_likelihood:.2f}')