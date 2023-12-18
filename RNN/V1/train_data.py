import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Masking, InputLayer, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

label_encoder = LabelEncoder()
y_numerical = label_encoder.fit_transform(y)
# Dividir el conjunto de datos en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y_numerical, test_size=0.2, random_state=42)

# Modelo 1
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_length, X.shape[2])))  # Asumo que la última dimensión es el número de características
model.add(SimpleRNN(units=32, activation='relu'))
model.add(Dense(units=len(actions), activation='softmax'))  # Asumiendo clasificación multiclase

# Modelo 2
"""model = Sequential()
model.add(InputLayer(input_shape=(max_length, 42)))
model.add(Masking(mask_value=2.0, input_shape=(max_length, 42)))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(150))
model.add(Dense(len(actions), activation='softmax'))"""

# Compilar el modelo
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

print("Evaluando Modelo...")

# Evaluar la precisión en el conjunto de prueba
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Accuracy on test set: {accuracy}')

# Mostrar las curvas de entrenamiento y validación
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
