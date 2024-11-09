# Análisis de Toxicidad en Comentarios

Este proyecto utiliza un modelo de deep learning para analizar la toxicidad en comentarios de texto. El modelo clasifica los comentarios en seis categorías de toxicidad: **toxic**, **severe_toxic**, **obscene**, **threat**, **insult**, y **identity_hate**. Los usuarios pueden ingresar frases y el programa devolverá la probabilidad de que el comentario pertenezca a cada categoría.

## Requisitos

- Python 3.9 o superior
- TensorFlow
- pickle (incluido en la biblioteca estándar de Python)
- pandas (para manejar los datos, en caso de que desees entrenar o modificar el modelo)

Puedes instalar las bibliotecas necesarias con:

```bash
pip install tensorflow pandas matplotlib numpy
```

## Archivos del proyecto

- `modelo_toxicidad.keras`: El modelo de deep learning guardado en formato Keras.
- `tokenizer.pickle`: El Tokenizer de Keras guardado, necesario para procesar el texto de entrada de la misma manera que durante el entrenamiento.
- `Toxicidad.py`: Script de Python que carga el modelo y el `Tokenizer`, permite al usuario ingresar frases y muestra la predicción de toxicidad.

## Configuración

1. **Entrenar el modelo** (opcional): Si deseas entrenar el modelo desde cero, asegúrate de tener los datos de entrenamiento y un archivo `.csv` con los comentarios y etiquetas de toxicidad.
2. **Guardar el Tokenizer**: Después de entrenar el modelo, guarda el `Tokenizer` usando `pickle` para poder cargarlo después sin necesidad de entrenarlo de nuevo.
3. **Guardar el Modelo**: Guarda el modelo en formato `.keras` o `.h5` para su uso posterior.

## Uso

1. **Asegúrate de que los archivos** `modelo_toxicidad.keras` y `tokenizer.pickle` estén en el mismo directorio que `predict_toxicity.py`.
2. **Ejecuta el programa**:

   ```bash
   python3 Toxicidad.py
   ```

3. **Ingresa una frase en inglés** para analizar su toxicidad y presiona Enter. El programa mostrará la probabilidad de que el comentario pertenezca a cada una de las seis categorías de toxicidad.

4. **Escribe "salir"** para terminar el programa.

## Ejemplo de uso

Al ejecutar el programa, el usuario verá un mensaje en la terminal:

```plaintext
Bienvenido al analizador de toxicidad.
Ingrese una frase para analizar su toxicidad (o escriba 'salir' para terminar): Fuck you
Predicción de toxicidad para cada categoría:
toxic: 0.9998
severe_toxic: 0.2956
obscene: 0.9933
threat: 0.0017
insult: 0.7848
identity_hate: 0.0021
```

## Estructura del Código

El script `Toxicidad.py` tiene las siguientes secciones:

1. **Carga del modelo y el `Tokenizer`**: Se cargan el modelo de deep learning y el `Tokenizer` guardados previamente.
2. **Preprocesamiento de texto**: Se tokeniza y aplica padding al texto ingresado por el usuario.
3. **Predicción**: Se generan las probabilidades de cada categoría de toxicidad.
4. **Interacción con el usuario**: El usuario puede ingresar múltiples frases para análisis hasta que decida salir.

## Notas

- Asegúrate de que el `Tokenizer` utilizado para entrenar el modelo es el mismo que se carga para hacer predicciones.
- Los resultados dependen de la calidad de los datos de entrenamiento y del preprocesamiento realizado.
- Dentro de la carpeta datatset se encuentra un zip con los archivos csv.
- En cada carpeta se encuentra un archivo zip, deben ser descomprimidos para poder ocuparse.
