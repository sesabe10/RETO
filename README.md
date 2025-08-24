# 🧬 Medical Text Classifier

API y modelo de **clasificación de textos médicos**.  
Permite predecir categorías médicas a partir de títulos y resúmenes de artículos.  
El proyecto incluye entrenamiento, evaluación y despliegue en **Render**.

---

## ⚙️ Instalación y configuración

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tuusuario/medical-text-classifier.git
   cd medical-text-classifier

2. Instalar dependencias:
   ```bash
      pip install -r requirements.txt


## 📊 Entrenamiento del modelo
Ejecuta el script de entrenamiento:
```
      python model/medical_classifier.py

Esto generará:
- medical_pipeline.joblib → pipeline del modelo.
- mlb_binarizer.joblib → binarizador de etiquetas.
- metrics.json → métricas de desempeño (accuracy, F1, etc.).
- confusion.json → matriz de confusión.
