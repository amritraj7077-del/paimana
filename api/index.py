"""
PAIMANA Project Intelligence API
Railway-compatible FastAPI backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import json
import os
from typing import List, Dict, Optional

app = FastAPI(
    title="PAIMANA Project Intelligence",
    description="AI-powered risk assessment for infrastructure projects",
    version="1.0.0"
)

# CORS - allow your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD MODELS ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(DATA_DIR, "project_intelligence_models.pkl"), "rb") as f:
    models = pickle.load(f)

with open(os.path.join(DATA_DIR, "api_config.json"), "r") as f:
    config = json.load(f)

with open(os.path.join(DATA_DIR, "insights_db.json"), "r") as f:
    insights_db = json.load(f)

with open(os.path.join(DATA_DIR, "sector_stats.json"), "r") as f:
    sector_stats = json.load(f)

cost_model = models['cost_model']
delay_model = models['delay_model']
risk_model = models['risk_model']
nn_model = models['nn_model']
scaler = models['scaler']
sim_scaler = models['sim_scaler']
label_encoders = models['label_encoders']
feature_cols = models['feature_cols']
similarity_features = models['similarity_features']
df_ref = models['df_reference']

# ========== REQUEST SCHEMAS ==========
class ProjectInput(BaseModel):
    project_name: str
    sector: str
    state: str
    ministry: str
    original_cost: float
    approval_year: int
    physical_progress: float
    cumulative_expenditure: float
    start_year: Optional[int] = None
    agency: Optional[str] = ""

# ========== HELPER FUNCTIONS ==========
def encode_features(sector, state, ministry, original_cost, approval_year, 
                    physical_progress, expenditure_ratio, cost_per_progress, 
                    is_large, budget_util):
    sector_enc = label_encoders["Sector"].transform([sector])[0] if sector in label_encoders["Sector"].classes_ else 0
    state_enc = label_encoders["State"].transform([state])[0] if state in label_encoders["State"].classes_ else 0
    ministry_enc = label_encoders["Ministry"].transform([ministry])[0] if ministry in label_encoders["Ministry"].classes_ else 0
    
    features = np.array([[sector_enc, state_enc, ministry_enc, 
                          original_cost, approval_year, physical_progress,
                          expenditure_ratio, cost_per_progress, is_large, budget_util]])
    return scaler.transform(features)

def calculate_risk_score(overrun_ratio, delay_months, risk_level, progress, budget_util):
    score = 0
    if overrun_ratio > 1.0: score += 35
    elif overrun_ratio > 0.5: score += 28
    elif overrun_ratio > 0.3: score += 20
    elif overrun_ratio > 0.1: score += 10
    else: score += 5
    if delay_months > 36: score += 30
    elif delay_months > 24: score += 24
    elif delay_months > 12: score += 18
    elif delay_months > 6: score += 10
    else: score += 3
    if progress < 20: score += 15
    elif progress < 40: score += 10
    elif progress < 60: score += 5
    if budget_util > 1.5: score += 20
    elif budget_util > 1.0: score += 15
    elif budget_util > 0.8: score += 8
    return min(100, max(0, int(score)))

def get_ai_insight(sector, overrun_ratio, delay_months, progress, budget_util, risk_level):
    insights = []
    sector_key = sector.lower().replace(" ", "_").replace(",", "")
    
    if "road" in sector_key or "highway" in sector_key:
        insights.append("Similar highway projects in comparable locations experienced higher-than-average delays and cost overruns. The proposed budget may be underestimated.")
    elif "rail" in sector_key:
        insights.append("Railway projects of this scale typically see 15-25% cost escalation due to land acquisition and terrain challenges.")
    elif "coal" in sector_key or "mine" in sector_key:
        insights.append("Mining projects in this region have historically faced regulatory delays. Factor in additional 6-12 months for clearance timelines.")
    elif "power" in sector_key or "electric" in sector_key:
        insights.append("Power sector projects show strong correlation between delay and cost overrun. Early equipment procurement is recommended.")
    
    if delay_months > 12:
        insights.append("Historical data indicates projects with similar characteristics face significant schedule slippage. Critical path items need close monitoring.")
    if progress < 30 and budget_util >
