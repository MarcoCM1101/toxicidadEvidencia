# Importación de librerías necesarias
# - numpy: para manejar arreglos numéricos y operaciones numéricas.
# - load_model: para cargar el modelo guardado en formato `.keras`.
# - pad_sequences: para aplicar padding a las secuencias de texto.
# - pickle: para cargar el tokenizer previamente guardado.

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Configuración
# Definimos la longitud máxima de secuencia y las etiquetas de toxicidad que usaremos para interpretar los resultados.
MAX_LEN = 150  # Longitud máxima de la secuencia que usaste durante el entrenamiento
labels = ['toxic', 'severe_toxic', 'obscene',
          'threat', 'insult', 'identity_hate']

# Cargar el modelo guardado en formato .keras
# `load_model` carga el modelo completo, incluyendo la arquitectura y los pesos.
model = load_model('modelo_toxicidad.keras')

# Cargar el tokenizer guardado
# `pickle.load` carga el tokenizer desde el archivo pickle, asegurando que el preprocesamiento sea consistente con el entrenamiento.
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Función para preprocesar la frase de entrada y predecir
# `predecir_toxicidad` toma una frase, la tokeniza, aplica padding y luego utiliza el modelo para predecir la toxicidad.


def predecir_toxicidad(frase):
    # Tokenizar y hacer padding a la frase
    # Convierte la frase en una secuencia de tokens y aplica padding para que tenga la longitud `MAX_LEN`.
    sequence = tokenizer.texts_to_sequences([frase])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

    # Realizar la predicción
    # `model.predict` devuelve una lista de probabilidades de toxicidad para cada categoría.
    prediccion = model.predict(padded_sequence)[0]

    # Mostrar resultados
    # Recorre cada categoría de toxicidad y muestra la probabilidad correspondiente.
    print("\nPredicción de toxicidad para cada categoría:")
    for label, score in zip(labels, prediccion):
        print(f"{label}: {score:.4f}")
    print("\n" + "-"*40 + "\n")


# Ciclo de interacción con el usuario
# Este bloque permite que el usuario ingrese frases para analizar su toxicidad en un bucle.
# Si el usuario ingresa "salir", el programa termina.
if __name__ == "__main__":
    print("Bienvenido al analizador de toxicidad.")
    while True:
        frase_usuario = input(
            "Ingrese una frase para analizar su toxicidad en inglés OBLIGATORIAMENTE(o escriba 'salir' para terminar): ")
        if frase_usuario.lower() == 'salir':
            print("Saliendo del programa.")
            break
        predecir_toxicidad(frase_usuario)
