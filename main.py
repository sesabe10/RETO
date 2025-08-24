from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import os

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="Medical Text Classifier API",
    description="API para clasificar textos médicos y exponer métricas.",
    version="1.1.0"
)

# --- Verificar/descargar recursos NLTK ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# --- Cargar artefactos del modelo ---
try:
    pipeline = load("medical_pipeline.joblib")
    mlb = load("mlb_binarizer.joblib")
except FileNotFoundError:
    raise RuntimeError("Los artefactos del modelo no se encontraron. Ejecuta medical_classifier.py primero.")

# --- Preprocesamiento ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# --- Request body ---
class Article(BaseModel):
    text: str

# --- Endpoint raíz ---
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Textos Médicos"}

# --- Endpoint de predicción ---
@app.post("/predict/")
def predict(article: Article):
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    cleaned = clean_text(article.text)
    predicted_bin = pipeline.predict([cleaned])
    predicted_labels = mlb.inverse_transform(predicted_bin)
    labels = predicted_labels[0] if predicted_labels and predicted_labels[0] else []

    return {"input_text": article.text, "predicted_categories": labels}

# --- Endpoint de métricas ---
@app.get("/metrics/")
def get_metrics():
    if not os.path.exists("metrics.json"):
        raise HTTPException(status_code=404, detail="metrics.json no encontrado. Ejecuta medical_classifier.py primero.")
    with open("metrics.json") as f:
        return json.load(f)

# --- Endpoint de matriz de confusión ---
@app.get("/confusion/")
def get_confusion():
    if not os.path.exists("confusion.json"):
        raise HTTPException(status_code=404, detail="confusion.json no encontrado. Ejecuta medical_classifier.py primero.")
    with open("confusion.json") as f:
        return json.load(f)
