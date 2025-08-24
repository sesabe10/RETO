import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Descargar recursos de NLTK (solo la primera vez) ---
try:
    stopwords.words('english')
except LookupError:
    print("Descargando recursos de NLTK (stopwords)...")
    nltk.download('stopwords')

# --- 2. Cargar y Explorar los Datos ---
try:
    df = pd.read_csv('challenge_data-18-ago.csv', delimiter=';')
except FileNotFoundError:
    print("Error: El archivo 'challenge_data-18-ago.csv' no fue encontrado.")
    exit()

# Combinar título y resumen en una sola columna de texto
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# --- 3. Preprocesamiento del Texto ---
# Función para limpiar el texto
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Limpia el texto: lo convierte a minúsculas, elimina caracteres no alfabéticos,
    quita stopwords y aplica stemming.
    """
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-z\s]', '', text)  # Eliminar caracteres no alfabéticos
    words = text.split()
    # Eliminar stopwords y aplicar stemming
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Aplicar la limpieza a la columna de texto
print("Limpiando y preprocesando el texto...")
df['cleaned_text'] = df['text'].apply(clean_text)

# --- 4. Preparación de las Etiquetas (Target) ---
# La columna 'group' tiene etiquetas separadas por '|'
# Usamos MultiLabelBinarizer para convertir esto a un formato binario
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['group'].str.split('|'))

# Mostramos las clases identificadas
print(f"\nClases identificadas por el binarizador: {mlb.classes_}")

# --- 5. División de Datos en Entrenamiento y Prueba ---
X = df['cleaned_text']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDatos divididos: {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")

# --- 6. Creación y Entrenamiento del Pipeline de Machine Learning ---
# El Pipeline automatiza el flujo de trabajo:
# 1. TfidfVectorizer: Convierte el texto en vectores numéricos TF-IDF.
# 2. OneVsRestClassifier: Permite la clasificación multietiqueta.
#    - Usa LogisticRegression como el estimador base para cada clase.

print("\nEntrenando el modelo...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)), # Usamos las 5000 palabras más frecuentes
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=-1))
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)
print("Entrenamiento completado.")

# --- 7. Evaluación del Modelo ---
print("\nEvaluando el rendimiento del modelo en el conjunto de prueba...")
predictions = pipeline.predict(X_test)

# Métricas de evaluación
# Accuracy Score (Jaccard similarity score para multietiqueta)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy (Puntaje Jaccard): {accuracy:.4f}")

# Reporte de clasificación detallado por clase
print("\nReporte de Clasificación:")
# `zero_division=0` evita warnings si una clase no tiene predicciones
print(classification_report(y_test, predictions, target_names=mlb.classes_, zero_division=0))


# --- 8. Ejemplo de Predicción con un Nuevo Texto ---
print("\n--- Ejemplo de Predicción con un Nuevo Artículo ---")
new_article_abstract = """
A study on the effects of statins on cholesterol levels in patients with a history of myocardial infarction.
The research observed a significant reduction in cardiovascular events and mortality rates.
Further investigation is needed to confirm the neuroprotective benefits.
"""

# Predecir las etiquetas para el nuevo texto
predicted_labels_bin = pipeline.predict([new_article_abstract])

# Convertir las predicciones binarias de vuelta a los nombres de las etiquetas
predicted_labels = mlb.inverse_transform(predicted_labels_bin)

print(f"Texto de ejemplo:\n\"{new_article_abstract.strip()}\"")
print(f"\nPredicción del modelo: {predicted_labels[0] if predicted_labels[0] else 'Ninguna categoría detectada'}")
#-----------------------------------------------------------------------
import joblib

# Guardar el pipeline del modelo
joblib.dump(pipeline, 'medical_pipeline.joblib')
print("Pipeline del modelo guardado en 'medical_pipeline.joblib'")

# Guardar el binarizador de etiquetas
joblib.dump(mlb, 'mlb_binarizer.joblib')
print("Binarizador de etiquetas guardado en 'mlb_binarizer.joblib'")
