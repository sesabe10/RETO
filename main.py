# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- Inicialización de la Aplicación FastAPI ---
app = FastAPI(
    title="Medical Text Classifier API",
    description="Una API para clasificar textos médicos en categorías.",
    version="1.0.0"
)

# --- Verificar/descargar recursos NLTK necesarios ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# --- Cargar los artefactos del modelo al iniciar la API ---
try:
    pipeline = load("medical_pipeline.joblib")
    mlb = load("mlb_binarizer.joblib")
except FileNotFoundError:
    raise RuntimeError("Los artefactos del modelo no se encontraron. Ejecuta medical_classifier.py primero.")

# --- Preprocesamiento (reutilizado del script de entrenamiento) ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# --- Definir el modelo de datos para la solicitud (Request Body) ---
class Article(BaseModel):
    text: str

# --- Definir el endpoint de predicción ---
@app.post("/predict/")
def predict(article: Article):
    """
    Recibe un texto de un artículo médico y devuelve las categorías predichas.
    """
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    # 1. Limpiar el texto de entrada
    cleaned = clean_text(article.text)

    # 2. Realizar la predicción con el pipeline
    predicted_bin = pipeline.predict([cleaned])

    # 3. Convertir la predicción binaria a etiquetas de texto
    predicted_labels = mlb.inverse_transform(predicted_bin)

    # 4. Formatear la respuesta
    labels = predicted_labels[0] if predicted_labels and predicted_labels[0] else []

    return {"input_text": article.text, "predicted_categories": labels}

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Textos Médicos"}
