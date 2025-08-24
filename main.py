from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

# --- Cargar artefactos entrenados ---
model = joblib.load("medical_pipeline.joblib")
vectorizer = joblib.load("vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# --- Inicializar API ---
app = FastAPI(
    title="Medical Text Classifier API",
    description="Clasificador de artículos médicos entrenado con Logistic Regression + SMOTE",
    version="2.0"
)

# --- Modelos de entrada ---
class InputText(BaseModel):
    input_text: str

# --- Rutas ---
@app.get("/")
def home():
    return {"message": "API de clasificación médica funcionando 🚑"}

@app.post("/predict")
def predict(data: InputText):
    text = data.input_text
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    probas = model.predict_proba(vec)[0]

    return {
        "input_text": text,
        "predicted_class": label_encoder.inverse_transform([pred])[0],
        "probabilities": {
            label_encoder.classes_[i]: float(probas[i]) for i in range(len(probas))
        }
    }

@app.get("/metrics")
def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        return {"error": "metrics.json no encontrado"}

@app.get("/confusion")
def get_confusion():
    try:
        with open("confusion.json", "r") as f:
            confusion = json.load(f)
        return confusion
    except FileNotFoundError:
        return {"error": "confusion.json no encontrado"}
