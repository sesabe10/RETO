import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse

# --- 1. Descargar recursos de NLTK ---
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
    
# Combinar título y resumen en una sola columna de texto
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# --- 2b. Análisis Exploratorio ---
print("\n===== Análisis Inicial del Dataset =====")
print(f"Total de artículos: {len(df)}")
print(f"Categorías únicas: {df['group'].nunique()}")

class_counts = df['group'].value_counts()
print("\nDistribución de clases:")
print(class_counts)

plt.figure(figsize=(10,5))
class_counts.plot(kind="bar")
plt.title("Distribución de clases en el dataset")
plt.ylabel("Número de artículos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()
print("Gráfico guardado en class_distribution.png")

df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
print("\nLongitud de los textos (en número de palabras):")
print(df['text_length'].describe())

plt.figure(figsize=(8,5))
df['text_length'].hist(bins=50)
plt.title("Distribución de longitudes de texto")
plt.xlabel("Número de palabras")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("text_length_distribution.png")
plt.close()
print("Histograma guardado en text_length_distribution.png")

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

# --- 4. Etiquetas (LabelEncoder en lugar de MultiLabelBinarizer) ---
le = LabelEncoder()
y = le.fit_transform(df['group'])   # y ahora es vector 1D
print(f"Clases: {list(le.classes_)}")

# --- 5. Split ---
X = df['cleaned_text']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Datos: {len(X_train)} train, {len(X_test)} test")

# --- 6. Vectorización ---
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 7. Balanceo con SMOTE ---
if issparse(X_train_vec):
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_vec.toarray(), y_train)
else:
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_vec, y_train)

print("\nDistribución después de SMOTE:")
print(pd.Series(y_train_res).value_counts())

# --- 8. Entrenamiento ---
print("Entrenando modelo...")
model = LogisticRegression(solver='liblinear', max_iter=200, class_weight="balanced")
model.fit(X_train_res, y_train_res)
print("Modelo entrenado ✅")

# --- 9. Evaluación ---
print("Evaluando...")
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test, predictions, target_names=le.classes_, zero_division=0, output_dict=True
)

# Guardar métricas
metrics = {
    "accuracy": accuracy,
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "per_class_f1": {label: report[label]["f1-score"] for label in le.classes_}
}

# Solo guarda micro_f1 si existe
if "micro avg" in report:
    metrics["micro_f1"] = report["micro avg"]["f1-score"]

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Métricas guardadas en metrics.json ✅")

# --- 10. Matriz de Confusión ---
cm = confusion_matrix(y_test, predictions)
confusion = {
    "labels": le.classes_.tolist(),
    "matrix": cm.tolist()
}
with open("confusion.json", "w") as f:
    json.dump(confusion, f, indent=2)
print("Matriz de confusión guardada en confusion.json ✅")

# --- 11. Guardar artefactos ---
joblib.dump(model, 'medical_pipeline.joblib')
print("Modelo guardado en medical_pipeline.joblib")
joblib.dump(vectorizer, 'vectorizer.joblib')
print("Vectorizador guardado en vectorizer.joblib")
joblib.dump(le, 'label_encoder.joblib')
print("LabelEncoder guardado en label_encoder.joblib")
