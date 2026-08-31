"""
Generate placeholder ML models for the PAIMANA intelligence platform.

The application (src/analytics/project_ml.py) expects a pickle file at
`data/project_intelligence_models.pkl` containing a dictionary with the
following keys:

    - cost_model:      sklearn regressor exposing .predict()
    - delay_model:      sklearn regressor exposing .predict()
    - risk_model:       sklearn classifier exposing .predict() / .predict_proba()
    - scaler:           sklearn transformer exposing .transform()
    - label_encoders:   dict of {column_name: sklearn LabelEncoder}

This script builds lightweight, *untrained-on-real-data* versions of these
objects so the application can start up and serve predictions without
crashing on import. They are fit on small synthetic datasets purely so the
underlying sklearn objects are valid, fitted estimators with the correct
method signatures and feature dimensions. They should be replaced with
models trained on real project data as soon as it is available.

Run this script any time the pickle file is missing, empty, or corrupted:

    python scripts/generate_models.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "data" / "project_intelligence_models.pkl"

# Feature columns used by src/analytics/project_ml.py, in order:
# Sector, State, Ministry, Original Cost (Rs. Crore), Approval_Year,
# Physical Progress (%), Expenditure_Ratio, Cost_Per_Progress,
# Is_Large_Project, Budget_Utilization
N_FEATURES = 10
N_SAMPLES = 50


def build_label_encoders():
    """Fit simple LabelEncoders on placeholder category values."""

    sectors = ["Infrastructure", "Power", "Roads", "Railways", "Urban Development"]
    states = ["Maharashtra", "Uttar Pradesh", "Karnataka", "Gujarat", "Tamil Nadu"]
    ministries = ["Ministry of Roads", "Ministry of Power", "Ministry of Railways"]

    encoders = {}

    sector_encoder = LabelEncoder()
    sector_encoder.fit(sectors)
    encoders["Sector"] = sector_encoder

    state_encoder = LabelEncoder()
    state_encoder.fit(states)
    encoders["State"] = state_encoder

    ministry_encoder = LabelEncoder()
    ministry_encoder.fit(ministries)
    encoders["Ministry"] = ministry_encoder

    return encoders


def build_synthetic_features(rng):
    """Generate a small synthetic feature matrix matching the expected shape."""

    X = rng.random((N_SAMPLES, N_FEATURES))
    # Scale some columns to look roughly like real-world magnitudes.
    X[:, 3] *= 5000   # Original Cost (Rs. Crore)
    X[:, 4] = rng.integers(2015, 2025, size=N_SAMPLES)  # Approval_Year
    X[:, 5] *= 100    # Physical Progress (%)
    X[:, 8] = (X[:, 3] >= 1000).astype(float)  # Is_Large_Project
    return X


def main():
    rng = np.random.default_rng(seed=42)

    X = build_synthetic_features(rng)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cost overrun ratio target (e.g. 0.0 - 1.0+)
    cost_target = rng.random(N_SAMPLES)
    cost_model = LinearRegression()
    cost_model.fit(X_scaled, cost_target)

    # Delay in months target
    delay_target = rng.random(N_SAMPLES) * 12
    delay_model = LinearRegression()
    delay_model.fit(X_scaled, delay_target)

    # Risk classification target (LOW / MEDIUM / HIGH)
    risk_labels = np.array(["LOW", "MEDIUM", "HIGH"])
    risk_target = risk_labels[rng.integers(0, 3, size=N_SAMPLES)]
    risk_model = LogisticRegression(max_iter=1000)
    risk_model.fit(X_scaled, risk_target)

    label_encoders = build_label_encoders()

    models = {
        "cost_model": cost_model,
        "delay_model": delay_model,
        "risk_model": risk_model,
        "scaler": scaler,
        "label_encoders": label_encoders,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

    print(f"Placeholder models written to {MODEL_PATH}")


if __name__ == "__main__":
    main()
