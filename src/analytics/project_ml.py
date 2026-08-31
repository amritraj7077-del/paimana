import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# LOAD TRAINED ML MODELS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "data" / "project_intelligence_models.pkl"

# Try to load trained ML models, fall back to None if not available
try:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100:
        with open(MODEL_PATH, "rb") as f:
            models = pickle.load(f)
        cost_model = models["cost_model"]
        delay_model = models["delay_model"]
        risk_model = models["risk_model"]
        scaler = models["scaler"]
        label_encoders = models["label_encoders"]
        MODELS_AVAILABLE = True
    else:
        MODELS_AVAILABLE = False
        cost_model = None
        delay_model = None
        risk_model = None
        scaler = None
        label_encoders = None
except Exception as e:
    MODELS_AVAILABLE = False
    cost_model = None
    delay_model = None
    risk_model = None
    scaler = None
    label_encoders = None


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_projects(df):
    """
    Run the trained PAIMANA ML models on project data.

    Models:
    - Cost overrun prediction
    - Delay prediction
    - Risk classification
    """
    
    # If models are not available, return dataframe with placeholder predictions
    if not MODELS_AVAILABLE:
        result = df.copy()
        result["ML_Predicted_Cost_Overrun_%"] = 0.0
        result["ML_Predicted_Delay_Months"] = 0.0
        result["ML_Predicted_Delay_Days"] = 0
        result["ML_Risk_Level"] = "LOW"
        result["ML_Risk_Confidence_%"] = 50.0
        return result

    result = df.copy()

    feature_rows = []

    for _, row in result.iterrows():

        # -----------------------------
        # Categorical encoding
        # -----------------------------

        sector = row.get("Sector", "")
        state = row.get("State", "")
        ministry = row.get("Ministry", "")

        sector_encoder = label_encoders["Sector"]
        state_encoder = label_encoders["State"]
        ministry_encoder = label_encoders["Ministry"]

        if sector in sector_encoder.classes_:
            sector_encoded = sector_encoder.transform([sector])[0]
        else:
            sector_encoded = 0

        if state in state_encoder.classes_:
            state_encoded = state_encoder.transform([state])[0]
        else:
            state_encoded = 0

        if ministry in ministry_encoder.classes_:
            ministry_encoded = ministry_encoder.transform([ministry])[0]
        else:
            ministry_encoded = 0

        # -----------------------------
        # Numerical features
        # -----------------------------

        original_cost = float(
            row.get("Original Cost (Rs. Crore)", 0) or 0
        )

        approval_year = int(
            row.get("Approval_Year", 2024) or 2024
        )

        physical_progress = float(
            row.get("Physical Progress (%)", 0) or 0
        )

        cumulative_expenditure = float(
            row.get("Cumulative Expenditure (Rs. Crore)", 0) or 0
        )

        # Avoid division by zero
        if original_cost > 0:
            expenditure_ratio = (
                cumulative_expenditure / original_cost
            )
        else:
            expenditure_ratio = 0

        if physical_progress > 0:
            cost_per_progress = (
                original_cost / physical_progress
            )
        else:
            cost_per_progress = 0

        # Large project flag
        is_large_project = (
            1 if original_cost >= 1000 else 0
        )

        budget_utilization = expenditure_ratio

        feature_rows.append([
            sector_encoded,
            state_encoded,
            ministry_encoded,
            original_cost,
            approval_year,
            physical_progress,
            expenditure_ratio,
            cost_per_progress,
            is_large_project,
            budget_utilization
        ])

    # ========================================================
    # CREATE FEATURE MATRIX
    # ========================================================

    X = pd.DataFrame(
        feature_rows,
        columns=[
            "Sector",
            "State",
            "Ministry",
            "Original Cost (Rs. Crore)",
            "Approval_Year",
            "Physical Progress (%)",
            "Expenditure_Ratio",
            "Cost_Per_Progress",
            "Is_Large_Project",
            "Budget_Utilization"
        ]
    )

    # Scale exactly as during training
    X_scaled = scaler.transform(X)

    # ========================================================
    # PREDICTIONS
    # ========================================================

    # Cost model predicts cost-overrun ratio
    cost_prediction = cost_model.predict(X_scaled)

    # Delay model predicts delay in months
    delay_prediction = delay_model.predict(X_scaled)

    # Risk model predicts LOW / MEDIUM / HIGH
    risk_prediction = risk_model.predict(X_scaled)

    # Risk probability
    try:
        risk_probability = risk_model.predict_proba(X_scaled).max(axis=1)
    except Exception:
        risk_probability = np.ones(len(result))

    # ========================================================
    # ADD RESULTS TO DATAFRAME
    # ========================================================

    result["ML_Predicted_Cost_Overrun_%"] = (
        np.maximum(cost_prediction, 0) * 100
    )

    result["ML_Predicted_Delay_Months"] = (
        np.maximum(delay_prediction, 0)
    )

    result["ML_Predicted_Delay_Days"] = (
        np.maximum(delay_prediction, 0) * 30.44
    ).round().astype(int)

    result["ML_Risk_Level"] = risk_prediction

    result["ML_Risk_Confidence_%"] = (
        risk_probability * 100
    ).round(1)

    return result
