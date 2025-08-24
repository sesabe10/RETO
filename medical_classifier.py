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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import numpy as np

# --- 1. Descargar recursos de NLTK (solo la primera vez) ---
try:
    stopwords.words('english')
except LookupError:
    print("Descargando recursos de NLTK (stopwords)...")
    nltk.download('stopwords')

# --- 2. Cargar los Datos ---
try:
    df = pd.read_csv('challenge_data-18-ago.csv', delimiter=';')
except FileNotFoundError:
    print("Error: El archivo 'challenge_data-18-ago.csv' no fue encontrado.")
    exit()

# Combinar título y resumen
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# --- 3. Preprocesamiento ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([stemmer.stem(w) for w in words if w not in stop_words])

print("Limpiando y preprocesando...")
df['cleaned_text'] = df['text'].apply(clean_text)

# --- 4. Etiquetas ---
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['group'].str.split('|'))

print(f"Clases: {mlb.classes_}")

# --- 5. Split ---
X = df['cleaned_text']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Datos: {len(X_train)} train, {len(X_test)} test")

# --- 6. Entrenamiento ---
print("Entrenando modelo...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=-1))
])
pipeline.fit(X_train, y_train)
print("Modelo entrenado ✅")

# --- 7. Evaluación ---
print("Evaluando...")
predictions = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Classification report en dict
report = classification_report(
    y_test, predictions, target_names=mlb.classes_, zero_division=0, output_dict=True
)

# Guardar métricas
metrics = {
    "accuracy": accuracy,
    "macro_f1": report["macro avg"]["f1-score"],
    "micro_f1": report["micro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "per_class_f1": {label: report[label]["f1-score"] for label in mlb.classes_}
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Métricas guardadas en metrics.json ✅")

# --- 8. Matriz de Confusión ---
# Para multilabel se simplifica a clasificación *multiclase* con argmax
cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
confusion = {
    "labels": mlb.classes_.tolist(),
    "matrix": cm.tolist()
}
with open("confusion.json", "w") as f:
    json.dump(confusion, f, indent=2)
print("Matriz de confusión guardada en confusion.json ✅")

# --- 9. Guardar artefactos del modelo ---
joblib.dump(pipeline, 'medical_pipeline.joblib')
print("Pipeline guardado en medical_pipeline.joblib")
joblib.dump(mlb, 'mlb_binarizer.joblib')
print("Binarizador guardado en mlb_binarizer.joblib")