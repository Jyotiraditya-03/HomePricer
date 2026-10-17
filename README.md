# 🏠 HomePricer — Explainable House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.45-blueviolet)
![LIME](https://img.shields.io/badge/LIME-0.2-yellowgreen)

> **R² = 0.89 · CV R² = 0.87** — Not just a price prediction, but a full *explanation* of why.

HomePricer is an end-to-end explainable machine learning web app that predicts house prices and — critically — explains every prediction through SHAP waterfall charts and LIME local explanations, giving users transparency into what drives real estate valuations.

---

## ✨ Features

- **Accurate Prediction** — XGBoost with 60-trial Optuna hyperparameter tuning; 5-fold cross-validation
- **SHAP Explanations** — Waterfall chart showing each feature's contribution to the price
- **LIME Explanations** — Local rules explaining this specific prediction
- **Feature Importance** — Global model-level importance ranked by gain
- **Interactive UI** — Sliders, steppers, and toggle tiles; light/dark theme; live Plotly charts
- **REST API** — Clean FastAPI backend; easily pluggable into other tools

---

## 🏗️ Architecture

```
HomePricer/
├── data/
│   └── housing.csv          # 545-row housing dataset
├── model/
│   ├── pipeline.py          # ML pipeline: preprocessing + XGBoost + SHAP/LIME
│   ├── train.py             # Training entry point (Optuna + CV)
│   └── artifacts/           # Saved model, explainer, feature names, metrics
├── backend/
│   ├── main.py              # FastAPI app (/predict /explain/shap /explain/lime /features)
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Single-page UI
│   ├── styles.css           # Dark/light theme, responsive layout
│   └── script.js            # Plotly charts, API integration
└── run.bat / run.sh         # One-click startup
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Train the model (one-time, ~2 min)
python model/train.py

# 3. Start the server
uvicorn backend.main:app --reload --port 8000

# 4. Open http://localhost:8000
```

Or just double-click `run.bat` (Windows) / `./run.sh` (Mac/Linux).

---

## 📡 API Reference

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| POST   | `/predict`      | Predict house price + confidence range |
| POST   | `/explain/shap` | SHAP values for each feature       |
| POST   | `/explain/lime` | LIME local explanation rules       |
| GET    | `/features`     | Global feature importance ranking  |
| GET    | `/metrics`      | Model metrics (R², MAE, RMSE, CV)  |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area":5000,"bedrooms":3,"bathrooms":2,"stories":2,"parking":1,
       "mainroad":"yes","guestroom":"no","basement":"no",
       "hotwaterheating":"no","airconditioning":"yes",
       "prefarea":"no","furnishingstatus":"semi-furnished"}'
```

---

## 🛠️ Tech Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| ML Model  | XGBoost, scikit-learn Pipeline          |
| Tuning    | Optuna (60-trial Bayesian search)       |
| Explain   | SHAP (TreeExplainer), LIME (Tabular)    |
| Backend   | FastAPI, Uvicorn                        |
| Frontend  | Vanilla JS, Plotly.js, Google Fonts     |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test R²  | **0.89** |
| CV R² (5-fold) | **0.87** |
| MAE  | ~₹3.1L |
| RMSE | ~₹4.6L |

---

*Built by [Pegadaraju Venkata Jyotiraditya](https://linkedin.com/in/p-v-jyotiraditya-42aa87215)*
