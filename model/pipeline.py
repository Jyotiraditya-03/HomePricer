"""
HomePricer ML Pipeline
Improved XGBoost model with Optuna hyperparameter tuning,
cross-validation, proper preprocessing, and SHAP/LIME explainability.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
import shap
import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_FEATURES  = ["mainroad", "guestroom", "basement", "hotwaterheating",
                     "airconditioning", "prefarea"]
ORDINAL_FEATURES = ["furnishingstatus"]
ORDINAL_CATS     = [["unfurnished", "semi-furnished", "furnished"]]
TARGET           = "price"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    # Normalise yes/no columns to lowercase
    for col in BINARY_FEATURES:
        df[col] = df[col].str.lower().str.strip()
    df["furnishingstatus"] = df["furnishingstatus"].str.lower().str.strip()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + binary + ordinal features."""
    numeric_pipe  = Pipeline([("scaler", StandardScaler())])
    binary_pipe   = Pipeline([("enc", OrdinalEncoder(
                                    categories=[["no", "yes"]] * len(BINARY_FEATURES)))])
    ordinal_pipe  = Pipeline([("enc", OrdinalEncoder(categories=ORDINAL_CATS))])

    return ColumnTransformer([
        ("num",  numeric_pipe,  NUMERIC_FEATURES),
        ("bin",  binary_pipe,   BINARY_FEATURES),
        ("ord",  ordinal_pipe,  ORDINAL_FEATURES),
    ])


def tune_hyperparams(X_train, y_train, preprocessor, n_trials: int = 50) -> dict:
    """Use Optuna to find best XGBoost hyperparameters."""

    X_prep = preprocessor.fit_transform(X_train)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
        }
        model = XGBRegressor(**params, verbosity=0)
        scores = cross_val_score(model, X_prep, y_train,
                                  cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train(csv_path: str, n_trials: int = 50) -> dict:
    """Full training run: tune → fit → evaluate → save artifacts."""
    print("Loading data…")
    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    print(f"Tuning hyperparameters ({n_trials} Optuna trials)…")
    best_params = tune_hyperparams(X_train, y_train, preprocessor, n_trials)
    print(f"Best params: {best_params}")

    # Build final pipeline
    model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("Fitting final model…")
    full_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = full_pipeline.predict(X_test)
    metrics = {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }
    print(f"Test R²={metrics['r2']}  MAE={metrics['mae']:,.0f}  RMSE={metrics['rmse']:,.0f}")

    # CV score on full training set
    cv_r2 = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="r2").mean()
    metrics["cv_r2"] = round(cv_r2, 4)
    print(f"5-fold CV R²={metrics['cv_r2']}")

    # SHAP explainer — fit on preprocessed training data
    X_train_proc = full_pipeline.named_steps["preprocessor"].transform(X_train)
    explainer = shap.TreeExplainer(full_pipeline.named_steps["model"])

    # Feature names in the order the preprocessor produces them
    feature_names = (
        NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(full_pipeline, os.path.join(ARTIFACTS_DIR, "pipeline.joblib"))
    joblib.dump(explainer,     os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
    joblib.dump(X_train_proc,  os.path.join(ARTIFACTS_DIR, "X_train_proc.joblib"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(ARTIFACTS_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f)

    print("Artifacts saved to", ARTIFACTS_DIR)
    return metrics


def load_artifacts():
    """Load saved pipeline + explainer from disk."""
    pipeline     = joblib.load(os.path.join(ARTIFACTS_DIR, "pipeline.joblib"))
    explainer    = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
    X_train_proc = joblib.load(os.path.join(ARTIFACTS_DIR, "X_train_proc.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return pipeline, explainer, X_train_proc, feature_names, metrics


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/housing.csv"
    train(csv, n_trials=60)
