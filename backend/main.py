"""
HomePricer — FastAPI Backend
Endpoints: /predict, /explain/shap, /explain/lime, /features, /metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import sys, os, json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.pipeline import (
    load_artifacts, NUMERIC_FEATURES, BINARY_FEATURES,
    ORDINAL_FEATURES, ORDINAL_CATS
)

import shap
import lime.lime_tabular

# ---------------------------------------------------------------------------
app = FastAPI(title="HomePricer API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup
pipeline, explainer, X_train_proc, feature_names, metrics = load_artifacts()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HouseFeatures(BaseModel):
    area:             float = Field(..., ge=500,  le=20000, description="Area in sqft")
    bedrooms:         int   = Field(..., ge=1,    le=10)
    bathrooms:        int   = Field(..., ge=1,    le=6)
    stories:          int   = Field(..., ge=1,    le=4)
    parking:          int   = Field(..., ge=0,    le=5)
    mainroad:         str   = Field(..., pattern="^(yes|no)$")
    guestroom:        str   = Field(..., pattern="^(yes|no)$")
    basement:         str   = Field(..., pattern="^(yes|no)$")
    hotwaterheating:  str   = Field(..., pattern="^(yes|no)$")
    airconditioning:  str   = Field(..., pattern="^(yes|no)$")
    prefarea:         str   = Field(..., pattern="^(yes|no)$")
    furnishingstatus: str   = Field(..., pattern="^(furnished|semi-furnished|unfurnished)$")


def to_dataframe(h: HouseFeatures) -> pd.DataFrame:
    return pd.DataFrame([{
        "area": h.area, "bedrooms": h.bedrooms, "bathrooms": h.bathrooms,
        "stories": h.stories, "parking": h.parking,
        "mainroad": h.mainroad, "guestroom": h.guestroom,
        "basement": h.basement, "hotwaterheating": h.hotwaterheating,
        "airconditioning": h.airconditioning, "prefarea": h.prefarea,
        "furnishingstatus": h.furnishingstatus,
    }])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/index.html"))


@app.post("/predict")
def predict(house: HouseFeatures):
    df = to_dataframe(house)
    price = float(pipeline.predict(df)[0])
    # Simple confidence band: ±8% (based on test RMSE relative to mean)
    return {
        "price":  round(price),
        "low":    round(price * 0.92),
        "high":   round(price * 1.08),
        "metrics": metrics,
    }


@app.post("/explain/shap")
def explain_shap(house: HouseFeatures):
    df   = to_dataframe(house)
    proc = pipeline.named_steps["preprocessor"].transform(df)
    sv   = explainer.shap_values(proc)[0]

    contributions = [
        {"feature": name, "value": float(sv[i]), "input": str(df.iloc[0][name])}
        for i, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda x: abs(x["value"]), reverse=True)

    return {
        "base_value":    float(explainer.expected_value),
        "contributions": contributions,
    }


@app.post("/explain/lime")
def explain_lime(house: HouseFeatures):
    df   = to_dataframe(house)
    proc = pipeline.named_steps["preprocessor"].transform(df)[0]

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X_train_proc,
        feature_names  = feature_names,
        mode           = "regression",
        random_state   = 42,
    )

    def predict_fn(X):
        # X is already preprocessed; bypass preprocessor
        return pipeline.named_steps["model"].predict(X)

    exp = lime_exp.explain_instance(
        data_row   = proc,
        predict_fn = predict_fn,
        num_features = len(feature_names),
    )

    contributions = [
        {"rule": rule, "value": float(weight)}
        for rule, weight in exp.as_list()
    ]
    return {"contributions": contributions}


@app.get("/features")
def feature_importance():
    importances = pipeline.named_steps["model"].feature_importances_
    result = [
        {"feature": name, "importance": float(importances[i])}
        for i, name in enumerate(feature_names)
    ]
    result.sort(key=lambda x: x["importance"], reverse=True)
    return {"features": result}


@app.get("/metrics")
def get_metrics():
    return metrics
