"""
Simple Flask Dashboard for PAIMANA Intelligence Platform
Provides web interface for viewing project analytics and data
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import math
import pickle
from pathlib import Path
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analytics.delay_detector import DelayAnalyzer
from src.audit.quality_checker import DataQualityAuditor
from src.analytics.chatbot_engine import PAIMANAChatbotEngine
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)
chatbot_engine = PAIMANAChatbotEngine()

# Configure CORS - allow all origins for development, or specific frontend domain
# For Railway deployment with separate frontend, set FRONTEND_URL environment variable
frontend_url = os.environ.get('FRONTEND_URL', '*')
if frontend_url == '*':
    CORS(app)  # Enable CORS for all routes (development)
else:
    CORS(app, resources={r"/*": {"origins": frontend_url}})

# Global data cache
data_cache = {
    'projects': None,
    'analytics': None,
    'quality_report': None
}


def load_or_generate_data():
    """Load project data from CSV file (df_reference extracted from pickle)"""
    if data_cache['projects'] is None:
        # Load from df_reference.csv (extracted from project_intelligence_models.pkl)
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'df_reference.csv'
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded {len(df)} projects from df_reference.csv")
                
                # Verify required columns exist
                required_cols = ['Project Code', 'Project Name', 'State', 'Sector', 
                               'Original Cost (Rs. Crore)', 'Cumulative Expenditure (Rs. Crore)', 
                               'Physical Progress (%)']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Missing columns in df_reference: {missing_cols}")
                
            except Exception as e:
                print(f"Error loading CSV: {e}")
                raise Exception(f"Failed to load project data from {csv_path}: {e}")
        else:
            raise Exception(f"Project dataset file not found: {csv_path}")
        
        # Run analytics on df_reference
        analyzer = DelayAnalyzer()
        
        # Map df_reference columns to expected analytics columns
        df['delay_days'] = df.get('Actual_Delay_Months', 0) * 30  # Convert months to days
        df['sanctioned_cost'] = df.get('Original Cost (Rs. Crore)', 0) * 10000000  # Convert crore to rupees
        df['expenditure_to_date'] = df.get('Cumulative Expenditure (Rs. Crore)', 0) * 10000000  # Convert crore to rupees
        df['physical_progress_percent'] = df.get('Physical Progress (%)', 0)
        df['project_id'] = df.get('Project Code', '')
        df['project_name'] = df.get('Project Name', '')
        df['district'] = df.get('State', '')
        df['category'] = df.get('Sector', '')
        
        analytics_report = analyzer.generate_analytics_report(df)
        
        # Run quality audit
        auditor = DataQualityAuditor()
        quality_report = auditor.audit(df)
        quality_summary = auditor.generate_audit_summary(quality_report)
        
        # ── Run REAL ML model inference ──────────────────────────────────────
        predictions_df = df.copy()
        pkl_path = Path(__file__).parent.parent.parent / 'data' / 'project_intelligence_models.pkl'

        try:
            with open(pkl_path, 'rb') as f:
                models = pickle.load(f)

            ml_feature_cols = models['feature_cols']
            ml_label_encoders = models['label_encoders']
            ml_scaler = models['scaler']
            ml_delay_model = models['delay_model']
            ml_cost_model = models['cost_model']
            ml_risk_model = models['risk_model']

            # Prepare features with SAME preprocessing as training
            X = predictions_df[ml_feature_cols].copy()
            for col, le in ml_label_encoders.items():
                if col in X.columns:
                    X[col] = X[col].astype(str).map(
                        lambda s, _le=le: _le.transform([s])[0] if s in _le.classes_ else 0
                    )
            # Replace any NaN with 0 before scaling
            X = X.fillna(0)
            X_scaled = ml_scaler.transform(X)

            # Run trained models
            pred_delay_months = ml_delay_model.predict(X_scaled)
            pred_cost_ratio = ml_cost_model.predict(X_scaled)
            pred_risk = ml_risk_model.predict(X_scaled)

            # Clamp: negative predicted future delay → 0
            pred_delay_days = np.maximum(pred_delay_months * 30, 0).round().astype(int)

            predictions_df["ML_Predicted_Delay_Days"] = pred_delay_days
            predictions_df["ML_Predicted_Cost_Overrun_%"] = (pred_cost_ratio * 100).round(2)
            predictions_df["ML_Risk_Level"] = pred_risk

            # Confidence from risk model probability (if available)
            if hasattr(ml_risk_model, 'predict_proba'):
                probas = ml_risk_model.predict_proba(X_scaled)
                predictions_df["ML_Risk_Confidence_%"] = (probas.max(axis=1) * 100).round(1)
            else:
                predictions_df["ML_Risk_Confidence_%"] = 75.0

            unique_preds = len(set(pred_delay_days))
            print(f"[ML] Real model inference complete: {unique_preds} unique delay predictions across {len(df)} projects")
            print(f"[ML] Predicted delay range: {pred_delay_days.min()}–{pred_delay_days.max()} days (mean {pred_delay_days.mean():.0f})")

        except Exception as e:
            import traceback
            print(f"[ML] WARNING: Model inference failed, using dataset values as fallback: {e}")
            print(traceback.format_exc())
            predictions_df["ML_Predicted_Delay_Days"] = np.maximum(predictions_df.get("Actual_Delay_Months", 0) * 30, 0)
            predictions_df["ML_Predicted_Cost_Overrun_%"] = predictions_df.get("Cost_Overrun_Ratio", 0) * 100
            predictions_df["ML_Risk_Level"] = predictions_df.get("Risk_Level", "LOW")
            predictions_df["ML_Risk_Confidence_%"] = 0.0  # 0% confidence = fallback

        # Dashboard compatibility columns
        predictions_df["predicted_delay_days"] = predictions_df["ML_Predicted_Delay_Days"]
        predictions_df["predicted_completion_date"] = pd.NaT

        data_cache['projects'] = df
        data_cache['analytics'] = analytics_report
        data_cache['quality_report'] = quality_summary
        data_cache['predictor'] = None
        data_cache['predictions'] = predictions_df

    return data_cache


@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    return jsonify({'status': 'healthy', 'service': 'paimana-intelligence-platform'})


@app.route('/ai-insights/report/<project_id>')
def ai_insights_report_page(project_id):
    """AI Insights report page - dedicated project intelligence report for specific project"""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>PAIMANA AI Insights - Project Intelligence Report</title>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #F7F9F8;
                color: #17201B;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .header {{
                background: #0B5D3B;
                color: white;
                padding: 25px 40px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            
            .header-content {{
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 15px;
            }}
            
            .header-branding {{
                flex: 1;
                min-width: 300px;
            }}
            
            .header h1 {{
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 5px;
                letter-spacing: 0.5px;
            }}
            
            .header h2 {{
                font-size: 16px;
                font-weight: 400;
                opacity: 0.9;
                margin-bottom: 8px;
            }}
            
            .header-meta {{
                font-size: 13px;
                opacity: 0.85;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }}
            
            .header-meta span {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            
            .header-actions {{
                display: flex;
                gap: 10px;
                align-items: center;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 30px 40px;
            }}
            
            .back-button {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 20px;
                background: rgba(255,255,255,0.15);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 500;
                font-size: 14px;
                transition: background 0.2s ease;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            
            .back-button:hover {{
                background: rgba(255,255,255,0.25);
            }}
            
            .pdf-button {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 28px;
                background: #0B5D3B;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                transition: background 0.2s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .pdf-button:hover {{
                background: #06452C;
            }}
            
            .pdf-button:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            
            .report-container {{
                display: none;
            }}
            
            .report-container.active {{
                display: block;
            }}
            
            .section {{
                background: #FFFFFF;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                margin-bottom: 25px;
                border: 1px solid #E8EBE9;
                border-left: 4px solid #0B5D3B;
            }}
            
            .section-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 2px solid #E8EBE9;
            }}
            
            .section-header h3 {{
                color: #0B5D3B;
                font-size: 20px;
                font-weight: 700;
                margin: 0;
            }}
            
            .section-icon {{
                width: 36px;
                height: 36px;
                background: #F0F7F3;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #0B5D3B;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
            }}
            
            .info-item {{
                padding: 20px;
                background: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E8EBE9;
                transition: all 0.2s ease;
            }}
            
            .info-item:hover {{
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border-color: #0B5D3B;
            }}
            
            .info-item label {{
                display: block;
                font-weight: 600;
                color: #64706A;
                margin-bottom: 8px;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .info-item .value {{
                font-size: 20px;
                color: #17201B;
                font-weight: 700;
                line-height: 1.4;
            }}
            
            .status-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                color: white;
            }}
            
            .status-critical {{ background: #C62828; }}
            .status-delayed {{ background: #B7791F; color: #000; }}
            .status-ontime {{ background: #16803C; }}
            
            .risk-high {{ background: #C62828; }}
            .risk-medium {{ background: #B7791F; color: #000; }}
            .risk-low {{ background: #16803C; }}
            
            .risk-list {{
                list-style: none;
                display: grid;
                gap: 12px;
            }}
            
            .risk-list li {{
                padding: 16px 20px;
                background: #FFF5F5;
                border-left: 4px solid #C62828;
                border-radius: 8px;
                display: flex;
                align-items: center;
                gap: 12px;
                color: #17201B;
            }}
            
            .risk-list li::before {{
                content: '⚠';
                font-size: 18px;
                color: #C62828;
            }}
            
            .comparison-table {{
                width: 100%%;
                border-collapse: collapse;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            }}
            
            .comparison-table th,
            .comparison-table td {{
                padding: 16px;
                text-align: left;
                border-bottom: 1px solid #E8EBE9;
            }}
            
            .comparison-table th {{
                background: #F0F7F3;
                font-weight: 600;
                color: #0B5D3B;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .comparison-table tr:hover {{
                background: #F7F9F8;
            }}
            
            .comparison-table td {{
                color: #17201B;
            }}
            
            .assessment-box {{
                background: #F0F7F3;
                padding: 30px;
                border-radius: 12px;
                border-left: 4px solid #0B5D3B;
            }}
            
            .assessment-box h4 {{
                color: #0B5D3B;
                margin-bottom: 15px;
                font-size: 18px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .assessment-box p {{
                margin-bottom: 15px;
                line-height: 1.7;
                color: #17201B;
            }}
            
            .assessment-box p:last-child {{
                margin-bottom: 0;
            }}
            
            .assessment-box strong {{
                color: #0B5D3B;
            }}
            
            .loading {{
                text-align: center;
                padding: 60px 40px;
                color: #64706A;
                font-size: 18px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }}
            
            .loading::before {{
                content: '';
                width: 50px;
                height: 50px;
                border: 4px solid #E8EBE9;
                border-top-color: #0B5D3B;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
            
            .error {{
                background: #FFF5F5;
                color: #C62828;
                padding: 25px 30px;
                border-radius: 12px;
                border-left: 4px solid #C62828;
                box-shadow: 0 2px 8px rgba(198, 40, 40, 0.1);
                font-size: 16px;
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            
            .error::before {{
                content: '❌';
                font-size: 24px;
            }}
            
            .progress-bar-container {{
                background: #E8EBE9;
                border-radius: 10px;
                height: 24px;
                margin: 12px 0;
                overflow: hidden;
            }}
            
            .progress-bar {{
                height: 100%%;
                border-radius: 10px;
                transition: width 0.6s ease;
            }}
            
            .progress-physical {{ background: #0B5D3B; }}
            .progress-budget {{ background: #16803C; }}
            
            .metric-card {{
                background: #FFFFFF;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #E8EBE9;
                text-align: center;
                transition: all 0.2s ease;
            }}
            
            .metric-card:hover {{
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border-color: #0B5D3B;
            }}
            
            .metric-card .label {{
                font-size: 12px;
                font-weight: 600;
                color: #64706A;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 10px;
            }}
            
            .metric-card .value {{
                font-size: 32px;
                font-weight: 700;
                color: #0B5D3B;
                margin-bottom: 5px;
            }}
            
            .metric-card .unit {{
                font-size: 14px;
                color: #64706A;
                font-weight: 500;
            }}
            
            .comparison-card {{
                background: #FFFFFF;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #E8EBE9;
                margin-bottom: 15px;
                border-left: 3px solid #2563A6;
            }}
            
            .comparison-card .comparison-label {{
                font-size: 13px;
                font-weight: 600;
                color: #64706A;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .comparison-card .comparison-value {{
                font-size: 20px;
                font-weight: 700;
                color: #17201B;
            }}
            
            .comparison-card .comparison-diff {{
                font-size: 14px;
                margin-top: 5px;
                font-weight: 600;
            }}
            
            .comparison-diff.positive {{ color: #16803C; }}
            .comparison-diff.negative {{ color: #C62828; }}
            
            @media (max-width: 1024px) {{
                .header {{
                    padding: 20px 30px;
                }}
                
                .header-content {{
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 15px;
                }}
                
                .header-branding {{
                    width: 100%%;
                }}
                
                .header-meta {{
                    flex-wrap: wrap;
                    gap: 15px;
                }}
                
                .header-actions {{
                    width: 100%%;
                    justify-content: space-between;
                }}
                
                .container {{
                    padding: 25px 30px;
                }}
            }}
            
            @media (max-width: 768px) {{
                .header {{
                    padding: 15px 20px;
                }}
                
                .header h1 {{
                    font-size: 20px;
                }}
                
                .header h2 {{
                    font-size: 14px;
                }}
                
                .header-meta {{
                    font-size: 12px;
                    flex-direction: column;
                    gap: 8px;
                }}
                
                .container {{
                    padding: 20px;
                }}
                
                .section {{
                    padding: 20px;
                }}
                
                .section-header h3 {{
                    font-size: 18px;
                }}
                
                .info-grid {{
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
                
                .info-item .value {{
                    font-size: 18px;
                }}
                
                .metric-card .value {{
                    font-size: 28px;
                }}
                
                .status-badge {{
                    font-size: 12px;
                    padding: 8px 14px;
                }}
                
                .comparison-table {{
                    font-size: 13px;
                }}
                
                .comparison-table th,
                .comparison-table td {{
                    padding: 12px 8px;
                }}
                
                .header-actions {{
                    flex-direction: column;
                    gap: 10px;
                }}
                
                .back-button,
                .pdf-button {{
                    width: 100%%;
                    justify-content: center;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <div class="header-branding">
                    <h1>PAIMANA Intelligence Platform</h1>
                    <h2>Project Intelligence Report</h2>
                    <div class="header-meta">
                        <span><i data-lucide="file-text" style="width:16px;height:16px;"></i> Project ID: <span id="reportProjectId">Loading...</span></span>
                        <span><i data-lucide="calendar" style="width:16px;height:16px;"></i> Generated: <span id="reportDate">Loading...</span></span>
                    </div>
                </div>
                <div class="header-actions">
                    <a href="/" class="back-button">
                        <i data-lucide="arrow-left" style="width:16px;height:16px;"></i>
                        Back to Dashboard
                    </a>
                    <button class="pdf-button" id="pdfButton" onclick="downloadPDF()" disabled>
                        <i data-lucide="download" style="width:16px;height:16px;"></i>
                        Download PDF
                    </button>
                </div>
            </div>
        </div>
        
        <div class="container">
            <div id="reportContainer" class="report-container">
                <div id="reportContent"></div>
            </div>
        </div>
        
        <script>
            lucide.createIcons();
            
            // Auto-generate report on page load
            const projectId = "{project_id}";
            console.log('Report page loaded with Project ID:', projectId);
            
            // Show loading state immediately
            const reportContainer = document.getElementById('reportContainer');
            const reportContent = document.getElementById('reportContent');
            reportContainer.classList.add('active');
            reportContent.innerHTML = '<div class="loading">Generating Project Intelligence Report…</div>';
            
            generateReport(projectId);
            
            async function generateReport(projectId) {{
                const reportContainer = document.getElementById('reportContainer');
                const reportContent = document.getElementById('reportContent');
                
                console.log('Generating report for Project ID:', projectId);
                
                try {{
                    const response = await fetch(`/api/generate-insights/${{encodeURIComponent(projectId)}}`);
                    console.log('API response status:', response.status);
                    const data = await response.json();
                    console.log('API response data:', data);
                    
                    if (!data.success) {{
                        console.error('API returned error:', data.error);
                        reportContent.innerHTML = `<div class="error">${{data.error || 'Project not found. Please enter a valid Project ID.'}}</div>`;
                        document.getElementById('reportProjectId').textContent = 'Error';
                        document.getElementById('reportDate').textContent = '-';
                        return;
                    }}
                    
                    const insights = data.data;
                    console.log('Insights data received:', insights);
                    
                    // Update header
                    document.getElementById('reportProjectId').textContent = insights.project_info.project_id;
                    document.getElementById('reportDate').textContent = new Date().toLocaleDateString();
                    
                    // Enable PDF button
                    document.getElementById('pdfButton').disabled = false;
                    
                    // Build report HTML
                    let html = '';
                    
                    // A. Project Overview
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="info" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>A. Project Overview</h3>
                            </div>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Project ID</label>
                                    <div class="value">${{insights.project_info.project_id}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Project Name</label>
                                    <div class="value">${{insights.project_info.project_name}}</div>
                                </div>
                                <div class="info-item">
                                    <label>State</label>
                                    <div class="value">${{insights.project_info.district}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Category/Sector</label>
                                    <div class="value">${{insights.project_info.category}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Ministry</label>
                                    <div class="value">${{insights.project_info.ministry}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Physical Progress</label>
                                    <div class="value">${{insights.project_info.physical_progress_percent.toFixed(1)}}%</div>
                                </div>
                                <div class="info-item">
                                    <label>Current Status</label>
                                    <div class="value">
                                        <span class="status-badge ${{insights.project_info.current_status === 'Critical' ? 'status-critical' : insights.project_info.current_status === 'Delayed' ? 'status-delayed' : 'status-ontime'}}">${{insights.project_info.current_status}}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // B. ML Assessment
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="brain" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>B. ML Assessment</h3>
                            </div>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>ML Predicted Risk</label>
                                    <div class="value">
                                        <span class="status-badge ${{insights.ml_predictions.predicted_risk === 'HIGH' ? 'risk-high' : insights.ml_predictions.predicted_risk === 'MEDIUM-HIGH' || insights.ml_predictions.predicted_risk === 'MEDIUM' ? 'risk-medium' : 'risk-low'}}">${{insights.ml_predictions.predicted_risk}}</span>
                                    </div>
                                </div>
                                <div class="info-item">
                                    <label>ML Predicted Delay</label>
                                    <div class="value">${{insights.ml_predictions.predicted_delay_days}} days</div>
                                </div>
                                <div class="info-item">
                                    <label>ML Predicted Cost Overrun</label>
                                    <div class="value">${{insights.ml_predictions.predicted_cost_overrun_percent}}%</div>
                                </div>
                                <div class="info-item">
                                    <label>ML Confidence</label>
                                    <div class="value">${{insights.ml_predictions.risk_confidence_percent > 0 ? insights.ml_predictions.risk_confidence_percent + '%' : 'Data unavailable'}}</div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // C. Financial Assessment
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="dollar-sign" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>C. Financial Health</h3>
                            </div>
                            <div class="info-grid">
                                <div class="metric-card">
                                    <div class="label">Sanctioned Cost</div>
                                    <div class="value">₹${{insights.financial_health.sanctioned_cost_cr}}</div>
                                    <div class="unit">Crores</div>
                                </div>
                                <div class="metric-card">
                                    <div class="label">Expenditure to Date</div>
                                    <div class="value">₹${{insights.financial_health.expenditure_to_date_cr}}</div>
                                    <div class="unit">Crores</div>
                                </div>
                                <div class="metric-card">
                                    <div class="label">Budget Utilization</div>
                                    <div class="value">${{insights.financial_health.budget_utilization_percent}}</div>
                                    <div class="unit">%</div>
                                </div>
                                <div class="metric-card">
                                    <div class="label">Cost Overrun</div>
                                    <div class="value">${{insights.financial_health.cost_overrun_percent}}</div>
                                    <div class="unit">%</div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // D. Schedule & Delay Assessment
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="clock" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>D. Schedule & Delay Assessment</h3>
                            </div>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Current Delay</label>
                                    <div class="value">${{insights.delay_analysis.current_delay_days}} days</div>
                                </div>
                                <div class="info-item">
                                    <label>Delay Severity</label>
                                    <div class="value">
                                        <span class="status-badge ${{insights.delay_analysis.delay_category === 'Critical' ? 'status-critical' : insights.delay_analysis.delay_category === 'Delayed' ? 'status-delayed' : 'status-ontime'}}">${{insights.delay_analysis.delay_category}}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // E. Key Risk Factors
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="alert-triangle" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>E. Key Risk Factors</h3>
                            </div>
                            <ul class="risk-list">
                                ${{insights.risk_factors.map(factor => `<li>${{factor}}</li>`).join('')}}
                            </ul>
                        </div>
                    `;
                    
                    // F. Progress vs Expenditure
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="bar-chart-2" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>F. Progress vs Expenditure</h3>
                            </div>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Physical Progress</label>
                                    <div class="value">${{insights.project_info.physical_progress_percent.toFixed(1)}}%</div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar progress-physical" style="width: ${{insights.project_info.physical_progress_percent}}%"></div>
                                    </div>
                                </div>
                                <div class="info-item">
                                    <label>Budget Utilization</label>
                                    <div class="value">${{insights.financial_health.budget_utilization_percent}}%</div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar progress-budget" style="width: ${{insights.financial_health.budget_utilization_percent}}%"></div>
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top: 20px; padding: 20px; background: #F0F7F3; border-radius: 10px; border-left: 3px solid #0B5D3B;">
                                <strong style="color: #0B5D3B;">Analysis:</strong> ${{insights.progress_expenditure_analysis}}
                            </div>
                        </div>
                    `;
                    
                    // G. Category/District Comparison
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="scale" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>G. Category/District Comparison</h3>
                            </div>
                            <div style="display: grid; gap: 15px;">
                                ${{insights.category_comparison.map(comp => `
                                    <div class="comparison-card">
                                        <div class="comparison-label">Comparison Metric</div>
                                        <div class="comparison-value">${{comp}}</div>
                                    </div>
                                `).join('')}}
                            </div>
                        </div>
                    `;
                    
                    // H. Similar Projects
                    if (insights.similar_projects && insights.similar_projects.length > 0) {{
                        html += `
                            <div class="section">
                                <div class="section-header">
                                    <div class="section-icon">
                                        <i data-lucide="git-branch" style="width:18px;height:18px;"></i>
                                    </div>
                                    <h3>H. Similar Projects</h3>
                                </div>
                                <table class="comparison-table">
                                    <tr>
                                        <th>Project ID</th>
                                        <th>Project Name</th>
                                        <th>Category</th>
                                        <th>Similarity</th>
                                    </tr>
                                    ${{insights.similar_projects.map(p => `
                                        <tr>
                                            <td><strong>${{p.project_id}}</strong></td>
                                            <td>${{p.project_name}}</td>
                                            <td>${{p.sector}}</td>
                                            <td><span class="status-badge status-ontime">${{(p.similarity_score * 100).toFixed(1)}}%%</span></td>
                                        </tr>
                                    `).join('')}}
                                </table>
                            </div>
                        `;
                    }}
                    
                    // I. AI Project Assessment
                    html += `
                        <div class="section">
                            <div class="section-header">
                                <div class="section-icon">
                                    <i data-lucide="target" style="width:18px;height:18px;"></i>
                                </div>
                                <h3>I. AI Project Assessment</h3>
                            </div>
                            <div class="assessment-box">
                                <h4><i data-lucide="check-circle" style="width:20px;height:20px;"></i> Overall Assessment</h4>
                                <p>${{insights.risk_assessment}}</p>
                                <p style="margin-top: 20px;"><strong><i data-lucide="lightbulb" style="width:16px;height:16px;"></i> Recommended Action:</strong> ${{insights.recommendation}}</p>
                            </div>
                        </div>
                    `;
                    
                    reportContent.innerHTML = html;
                    lucide.createIcons();
                    
                }} catch (error) {{
                    console.error('Error generating report:', error);
                    reportContent.innerHTML = '<div class="error">Error communicating with server.</div>';
                }}
            }}
            
            function escapeHTML(str) {{
                if (!str) return '';
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }}
            
            async function downloadPDF() {{
                try {{
                    const response = await fetch(`/api/download-insights-pdf/${{encodeURIComponent(projectId)}}`);
                    if (response.ok) {{
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `PAIMANA_Project_Report_${{projectId}}.pdf`;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                    }} else {{
                        const error = await response.json();
                        alert(error.error || 'Error generating PDF');
                    }}
                }} catch (error) {{
                    console.error('Error downloading PDF:', error);
                    alert('Error downloading PDF');
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_template.format(project_id=project_id)


@app.route('/ai-insights')
def ai_insights_page():
    """AI Insights page - dedicated project intelligence report"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>PAIMANA AI Insights - Project Intelligence Report</title>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #F4F0E4;
                color: #183F32;
                line-height: 1.6;
            }}
            
            .header {{
                background: linear-gradient(135deg, #174D3B 0%, #1B6B3A 100%);
                color: white;
                padding: 30px 40px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .header h1 {{
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 8px;
            }}
            
            .header p {{
                font-size: 14px;
                opacity: 0.9;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px;
            }}
            
            .search-section {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 30px;
            }}
            
            .search-section h2 {{
                margin-bottom: 20px;
                color: #174D3B;
            }}
            
            .search-box {{
                display: flex;
                gap: 15px;
            }}
            
            .search-box input {{
                flex: 1;
                padding: 15px;
                border: 2px solid #E8E4D8;
                border-radius: 8px;
                font-size: 16px;
            }}
            
            .search-box button {{
                padding: 15px 30px;
                background: #174D3B;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 16px;
            }}
            
            .search-box button:hover {{
                background: #0F3A2C;
            }}
            
            .back-button {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 24px;
                background: #6c757d;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                margin-bottom: 20px;
                font-weight: 500;
            }}
            
            .back-button:hover {{
                background: #5a6268;
            }}
            
            .report-container {{
                display: none;
            }}
            
            .report-container.active {{
                display: block;
            }}
            
            .report-header {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 25px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .report-header h2 {{
                color: #174D3B;
                margin-bottom: 5px;
            }}
            
            .report-header .meta {{
                color: #6c757d;
                font-size: 14px;
            }}
            
            .pdf-button {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 24px;
                background: #1B6B3A;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 15px;
            }}
            
            .pdf-button:hover {{
                background: #174D3B;
            }}
            
            .section {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 25px;
            }}
            
            .section h3 {{
                color: #174D3B;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #E8E4D8;
                font-size: 18px;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }}
            
            .info-item {{
                padding: 15px;
                background: #F8F9FA;
                border-radius: 8px;
            }}
            
            .info-item label {{
                display: block;
                font-weight: 600;
                color: #6c757d;
                margin-bottom: 5px;
                font-size: 12px;
                text-transform: uppercase;
            }}
            
            .info-item .value {{
                font-size: 16px;
                color: #183F32;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                color: white;
            }}
            
            .status-critical {{ background: #dc3545; }}
            .status-delayed {{ background: #ffc107; color: #000; }}
            .status-ontime {{ background: #28a745; }}
            
            .risk-high {{ background: #dc3545; }}
            .risk-medium {{ background: #ffc107; color: #000; }}
            .risk-low {{ background: #28a745; }}
            
            .risk-list {{
                list-style: none;
            }}
            
            .risk-list li {{
                padding: 10px 15px;
                background: #FFF3CD;
                border-left: 4px solid #FFC107;
                margin-bottom: 10px;
                border-radius: 4px;
            }}
            
            .comparison-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            .comparison-table th,
            .comparison-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #E8E4D8;
            }}
            
            .comparison-table th {{
                background: #F8F9FA;
                font-weight: 600;
                color: #174D3B;
            }}
            
            .assessment-box {{
                background: #E8F5E9;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            
            .assessment-box h4 {{
                color: #28a745;
                margin-bottom: 10px;
            }}
            
            .loading {{
                text-align: center;
                padding: 40px;
                color: #6c757d;
            }}
            
            .error {{
                background: #F8D7DA;
                color: #721C24;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #DC3545;
            }}
            
            .progress-bar-container {{
                background: #E8E4D8;
                border-radius: 10px;
                height: 30px;
                margin: 10px 0;
                overflow: hidden;
            }}
            
            .progress-bar {{
                height: 100%;
                border-radius: 10px;
                transition: width 0.3s ease;
            }}
            
            .progress-physical {{ background: #174D3B; }}
            .progress-budget {{ background: #1B6B3A; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PAIMANA Intelligence Platform</h1>
            <p>Project Intelligence Report</p>
        </div>
        
        <div class="container">
            <a href="/" class="back-button">
                <i data-lucide="arrow-left"></i>
                Back to Dashboard
            </a>
            
            <div class="search-section">
                <h2>Generate Project Intelligence Report</h2>
                <div class="search-box">
                    <input type="text" id="projectIdInput" placeholder="Enter Project Code (e.g., PRJ001)">
                    <button onclick="generateReport()">Generate Report</button>
                </div>
            </div>
            
            <div id="reportContainer" class="report-container">
                <div class="report-header">
                    <div>
                        <h2>Project Intelligence Report</h2>
                        <div class="meta">
                            Project ID: <span id="reportProjectId">-</span> | 
                            Generated: <span id="reportDate">-</span>
                        </div>
                    </div>
                    <button class="pdf-button" onclick="downloadPDF()">
                        <i data-lucide="file-text"></i>
                        Download PDF
                    </button>
                </div>
                
                <div id="reportContent"></div>
            </div>
        </div>
        
        <script>
            lucide.createIcons();
            
            async function generateReport() {{
                const projectId = document.getElementById('projectIdInput').value.trim();
                const reportContainer = document.getElementById('reportContainer');
                const reportContent = document.getElementById('reportContent');
                
                if (!projectId) {{
                    reportContent.innerHTML = '<div class="error">Please enter a Project Code.</div>';
                    reportContainer.classList.add('active');
                    return;
                }}
                
                reportContent.innerHTML = '<div class="loading">Generating report...</div>';
                reportContainer.classList.add('active');
                
                try {{
                    const response = await fetch(`/api/generate-insights/${{encodeURIComponent(projectId)}}`);
                    const data = await response.json();
                    
                    if (!data.success) {{
                        reportContent.innerHTML = `<div class="error">${{data.error || 'Error generating report.'}}</div>`;
                        return;
                    }}
                    
                    const insights = data.data;
                    
                    // Update header
                    document.getElementById('reportProjectId').textContent = insights.project_info.project_id;
                    document.getElementById('reportDate').textContent = new Date().toLocaleDateString();
                    
                    // Build report HTML
                    let html = '';
                    
                    // A. Project Overview
                    html += `
                        <div class="section">
                            <h3>A. Project Overview</h3>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Project ID</label>
                                    <div class="value">${{insights.project_info.project_id}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Project Name</label>
                                    <div class="value">${{insights.project_info.project_name}}</div>
                                </div>
                                <div class="info-item">
                                    <label>State</label>
                                    <div class="value">${{insights.project_info.district}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Category/Sector</label>
                                    <div class="value">${{insights.project_info.category}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Ministry</label>
                                    <div class="value">${{insights.project_info.ministry}}</div>
                                </div>
                                <div class="info-item">
                                    <label>Physical Progress</label>
                                    <div class="value">${{insights.project_info.physical_progress_percent.toFixed(1)}}%</div>
                                </div>
                                <div class="info-item">
                                    <label>Current Status</label>
                                    <div class="value">
                                        <span class="status-badge ${{
                                            insights.project_info.current_status === 'Critical' ? 'status-critical' : 
                                            insights.project_info.current_status === 'Delayed' ? 'status-delayed' : 'status-ontime'
                                        }}">{{{insights.project_info.current_status}}}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // B. ML Assessment
                    html += `
                        <div class="section">
                            <h3>B. ML Assessment</h3>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Predicted Risk</label>
                                    <div class="value">
                                        <span class="status-badge ${{
                                            insights.ml_predictions.predicted_risk === 'HIGH' ? 'risk-high' : 
                                            insights.ml_predictions.predicted_risk === 'MEDIUM-HIGH' || insights.ml_predictions.predicted_risk === 'MEDIUM' ? 'risk-medium' : 'risk-low'
                                        }}">{{{insights.ml_predictions.predicted_risk}}}</span>
                                    </div>
                                </div>
                                <div class="info-item">
                                    <label>Predicted Delay</label>
                                    <div class="value">${{insights.ml_predictions.predicted_delay_days}} days</div>
                                </div>
                                <div class="info-item">
                                    <label>Predicted Cost Overrun</label>
                                    <div class="value">${{insights.ml_predictions.predicted_cost_overrun_percent}}%</div>
                                </div>
                                <div class="info-item">
                                    <label>Confidence</label>
                                    <div class="value">${{insights.ml_predictions.risk_confidence_percent > 0 ? insights.ml_predictions.risk_confidence_percent + '%' : 'Data unavailable'}}</div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // C. Financial Assessment
                    html += `
                        <div class="section">
                            <h3>C. Financial Assessment</h3>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Sanctioned Cost</label>
                                    <div class="value">₹${{insights.financial_health.sanctioned_cost_cr}} Cr</div>
                                </div>
                                <div class="info-item">
                                    <label>Expenditure to Date</label>
                                    <div class="value">₹${{insights.financial_health.expenditure_to_date_cr}} Cr</div>
                                </div>
                                <div class="info-item">
                                    <label>Budget Utilization</label>
                                    <div class="value">${{insights.financial_health.budget_utilization_percent}}%</div>
                                </div>
                                <div class="info-item">
                                    <label>Cost Overrun</label>
                                    <div class="value">${{insights.financial_health.cost_overrun_percent}}%</div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // D. Schedule & Delay Assessment
                    html += `
                        <div class="section">
                            <h3>D. Schedule & Delay Assessment</h3>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Current Delay</label>
                                    <div class="value">${{insights.delay_analysis.current_delay_days}} days</div>
                                </div>
                                <div class="info-item">
                                    <label>Delay Severity</label>
                                    <div class="value">
                                        <span class="status-badge ${{
                                            insights.delay_analysis.delay_category === 'Critical' ? 'status-critical' : 
                                            insights.delay_analysis.delay_category === 'Delayed' ? 'status-delayed' : 'status-ontime'
                                        }}">{{{insights.delay_analysis.delay_category}}}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // E. Key Risk Factors
                    html += `
                        <div class="section">
                            <h3>E. Key Risk Factors</h3>
                            <ul class="risk-list">
                                ${{insights.risk_factors.map(factor => `<li>${{factor}}</li>`).join('')}}
                            </ul>
                        </div>
                    `;
                    
                    // F. Progress vs Expenditure
                    html += `
                        <div class="section">
                            <h3>F. Progress vs Expenditure</h3>
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Physical Progress</label>
                                    <div class="value">${{insights.project_info.physical_progress_percent.toFixed(1)}}%</div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar progress-physical" style="width: ${{insights.project_info.physical_progress_percent}}%"></div>
                                    </div>
                                </div>
                                <div class="info-item">
                                    <label>Budget Utilization</label>
                                    <div class="value">${{insights.financial_health.budget_utilization_percent}}%</div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar progress-budget" style="width: ${{insights.financial_health.budget_utilization_percent}}%"></div>
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top: 15px; padding: 15px; background: #F8F9FA; border-radius: 8px;">
                                <strong>Analysis:</strong> ${{insights.progress_expenditure_analysis}}
                            </div>
                        </div>
                    `;
                    
                    // G. Category/District Comparison
                    html += `
                        <div class="section">
                            <h3>G. Category/District Comparison</h3>
                            <ul class="risk-list" style="background: #E2E3E5; border-left-color: #6c757d;">
                                ${{insights.category_comparison.map(comp => `<li>${{comp}}</li>`).join('')}}
                            </ul>
                        </div>
                    `;
                    
                    // H. Similar Projects
                    if (insights.similar_projects && insights.similar_projects.length > 0) {{
                        html += `
                            <div class="section">
                                <h3>H. Similar Projects</h3>
                                <table class="comparison-table">
                                    <tr>
                                        <th>Project ID</th>
                                        <th>Project Name</th>
                                        <th>Category</th>
                                        <th>Similarity</th>
                                    </tr>
                                    ${{insights.similar_projects.map(p => `
                                        <tr>
                                            <td><strong>${{p.project_id}}</strong></td>
                                            <td>${{p.project_name}}</td>
                                            <td>${{p.sector}}</td>
                                            <td>${{(p.similarity_score * 100).toFixed(1)}}%</td>
                                        </tr>
                                    `).join('')}}
                                </table>
                            </div>
                        `;
                    }}
                    
                    // I. AI Project Assessment
                    html += `
                        <div class="section">
                            <h3>I. AI Project Assessment</h3>
                            <div class="assessment-box">
                                <h4>Overall Assessment</h4>
                                <p>${{insights.risk_assessment}}</p>
                                <p style="margin-top: 15px;"><strong>Recommended Action:</strong> ${{insights.recommendation}}</p>
                            </div>
                        </div>
                    `;
                    
                    reportContent.innerHTML = html;
                    
                }} catch (error) {{
                    console.error('Error generating report:', error);
                    reportContent.innerHTML = '<div class="error">Error communicating with server.</div>';
                }}
            }}
            
            function escapeHTML(str) {{
                if (!str) return '';
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }}
            
            async function downloadPDF() {{
                const projectId = document.getElementById('projectIdInput').value.trim();
                if (!projectId) {{
                    alert('Please enter a Project Code first.');
                    return;
                }}
                
                try {{
                    const response = await fetch(`/api/download-insights-pdf/${{encodeURIComponent(projectId)}}`);
                    if (response.ok) {{
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `PAIMANA_Project_Report_${{projectId}}.pdf`;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                    }} else {{
                        const error = await response.json();
                        alert(error.error || 'Error generating PDF');
                    }}
                }} catch (error) {{
                    console.error('Error downloading PDF:', error);
                    alert('Error downloading PDF');
                }}
            }}
        </script>
    </body>
    </html>
    """


@app.route('/')
def index():
    """Home page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>PAIMANA Intelligence Platform</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            :root {
                /* Premium Earthy Color Palette */
                --bg-primary: #F4F0E4;
                --bg-card: #FAF8F0;
                --bg-sidebar: #FAF8F0;
                --text-primary: #183F32;
                --text-secondary: #4A5568;
                --text-muted: #718096;
                --border-color: #E8E4D8;
                --border-dark: #D4D0C4;
                --primary: #174D3B;
                --primary-light: #E8F0EC;
                --primary-hover: #0F3A2C;
                --accent-sage: #8FA58A;
                --accent-cream: #F4F0E4;
                --shadow-soft: 0 2px 12px rgba(24, 63, 50, 0.06);
                --shadow-subtle: 0 1px 4px rgba(24, 63, 50, 0.04);
                --radius-card: 20px;
                --radius-sm: 12px;
                --radius-pill: 50px;
            }

            [data-theme="dark"] {
                /* Premium Dark Mode */
                --bg-primary: #1A1A1A;
                --bg-card: #242424;
                --bg-sidebar: #242424;
                --text-primary: #E8E8E8;
                --text-secondary: #A0A0A0;
                --text-muted: #707070;
                --border-color: #333333;
                --border-dark: #2A2A2A;
                --primary: #8FA58A;
                --primary-light: #1E3A2E;
                --primary-hover: #A8BFA3;
                --accent-sage: #6B8A66;
                --accent-cream: #2A2A2A;
                --shadow-soft: 0 2px 12px rgba(0, 0, 0, 0.3);
                --shadow-subtle: 0 1px 4px rgba(0, 0, 0, 0.2);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: var(--bg-primary);
                min-height: 100vh;
                display: flex;
                color: var(--text-primary);
            }
            
            /* Sidebar Styles */
            .sidebar {
                position: fixed;
                left: 0;
                top: 0;
                width: 260px;
                height: 100vh;
                background: var(--bg-sidebar);
                border-right: 1px solid var(--border-color);
                display: flex;
                flex-direction: column;
                z-index: 1000;
                box-shadow: var(--shadow-subtle);
            }
            
            .sidebar-header {
                padding: 32px 24px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .sidebar-title {
                font-size: 28px;
                font-weight: 700;
                color: var(--primary);
                margin-bottom: 6px;
                letter-spacing: 0.5px;
            }
            
            .sidebar-subtitle {
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 2.5px;
                font-weight: 600;
            }
            
            .sidebar-nav {
                flex: 1;
                padding: 24px 16px;
                overflow-y: auto;
            }
            
            .nav-item {
                display: flex;
                align-items: center;
                padding: 14px 20px;
                color: var(--text-secondary);
                text-decoration: none;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.25s ease;
                border-radius: var(--radius-sm);
                cursor: pointer;
                margin-bottom: 4px;
            }
            
            .nav-item:hover {
                background: var(--primary-light);
                color: var(--primary);
            }
            
            .nav-item.active {
                background: var(--primary);
                color: #FFFFFF;
            }
            
            .nav-item i {
                margin-right: 14px;
                width: 20px;
                height: 20px;
            }
            
            .sidebar-footer {
                padding: 20px 24px;
                border-top: 1px solid var(--border-color);
                font-size: 12px;
                color: var(--text-muted);
                text-align: center;
            }
            
            /* Mobile menu button */
            .mobile-menu-btn {
                display: none;
                position: fixed;
                top: 20px;
                left: 20px;
                z-index: 1001;
                background: var(--primary);
                color: white;
                border: none;
                padding: 12px;
                border-radius: var(--radius-sm);
                cursor: pointer;
                box-shadow: var(--shadow-soft);
            }
            
            /* Main content area */
            .main-content {
                margin-left: 260px;
                flex: 1;
                padding: 32px;
                background: var(--bg-primary);
                min-height: 100vh;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                background: var(--bg-card);
                padding: 32px;
                border-radius: var(--radius-card);
                box-shadow: var(--shadow-soft);
                margin-bottom: 32px;
                border: 1px solid var(--border-color);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 24px;
            }
            .header-content {
                flex: 1;
            }
            h1 {
                color: var(--text-primary);
                font-size: 2.25em;
                margin-bottom: 12px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            .subtitle {
                color: var(--text-secondary);
                font-size: 1.1em;
                line-height: 1.6;
            }
            .header-controls {
                display: flex;
                gap: 12px;
                align-items: center;
            }
            .control-btn {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px 20px;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-pill);
                color: var(--text-primary);
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.25s ease;
                box-shadow: var(--shadow-subtle);
            }
            .control-btn:hover {
                background: var(--primary-light);
                border-color: var(--primary);
                color: var(--primary);
                transform: translateY(-1px);
            }
            .control-btn i {
                width: 18px;
                height: 18px;
            }
            .nav {
                display: none;
            }
            .alert {
                background: var(--primary-light);
                padding: 20px 24px;
                border-radius: var(--radius-sm);
                color: var(--primary);
                margin-bottom: 32px;
                border: 1px solid var(--accent-sage);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 32px;
            }
            .stat-card {
                background: var(--bg-card);
                padding: 28px;
                border-radius: var(--radius-card);
                box-shadow: var(--shadow-soft);
                border: 1px solid var(--border-color);
                transition: all 0.3s ease;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }
            .stat-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 4px 20px rgba(24, 63, 50, 0.1);
            }
            .stat-value {
                font-size: 2.5em;
                font-weight: 700;
                color: var(--primary);
                margin-bottom: 8px;
                letter-spacing: -1px;
            }
            .stat-label {
                color: var(--text-muted);
                font-size: 0.8em;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-weight: 600;
            }
            .card {
                background: var(--bg-card);
                padding: 32px;
                border-radius: var(--radius-card);
                box-shadow: var(--shadow-soft);
                margin-bottom: 32px;
                border: 1px solid var(--border-color);
            }
            h2 {
                color: var(--text-primary);
                margin-bottom: 20px;
                font-size: 1.5em;
                font-weight: 700;
                letter-spacing: -0.3px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th {
                background: var(--primary);
                color: white;
                padding: 16px;
                text-align: left;
                font-weight: 600;
                font-size: 0.9em;
                letter-spacing: 0.3px;
            }
            td {
                padding: 16px;
                border-bottom: 1px solid var(--border-color);
                font-size: 0.95em;
                color: var(--text-secondary);
            }
            tr:hover {
                background: var(--primary-light);
            }
            #map {
                height: 500px;
                margin: 20px 0;
                border-radius: var(--radius-card);
                overflow: hidden;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-subtle);
            }
            .loading {
                text-align: center;
                padding: 48px;
                color: var(--text-muted);
                font-size: 1.1em;
            }
            .loading::after {
                content: '...';
                animation: dots 1.5s steps(4, end) infinite;
            }
            @keyframes dots {
                0%, 20% { content: '.'; }
                40% { content: '..'; }
                60%, 100% { content: '...'; }
            }
            .filter-bar {
                background: var(--bg-card);
                padding: 20px;
                border-radius: var(--radius-card);
                margin-bottom: 24px;
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
                align-items: center;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-subtle);
            }
            .filter-bar select, .filter-bar input {
                padding: 12px 16px;
                border: 1px solid var(--border-color);
                border-radius: var(--radius-sm);
                font-size: 14px;
                min-width: 160px;
                background: var(--bg-card);
                color: var(--text-primary);
                transition: all 0.2s ease;
            }
            .filter-bar select:focus, .filter-bar input:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(23, 77, 59, 0.1);
            }
            .filter-bar button {
                padding: 12px 24px;
                background: var(--primary);
                color: white;
                border: none;
                border-radius: var(--radius-pill);
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.25s ease;
            }
            .filter-bar button:hover {
                background: var(--primary-hover);
                transform: translateY(-1px);
            }
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(24, 63, 50, 0.6);
                backdrop-filter: blur(4px);
                z-index: 1000;
                justify-content: center;
                align-items: center;
            }
            .modal.active {
                display: flex;
            }
            .modal-content {
                background: var(--bg-card);
                padding: 32px;
                border-radius: var(--radius-card);
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-soft);
            }
            .modal-close {
                position: absolute;
                top: 20px;
                right: 20px;
                font-size: 24px;
                cursor: pointer;
                color: var(--text-muted);
                transition: color 0.2s ease;
            }
            .modal-close:hover {
                color: var(--primary);
            }
            .project-detail {
                margin: 16px 0;
            }
            .project-detail label {
                font-weight: 600;
                color: var(--text-primary);
                display: block;
                margin-bottom: 6px;
                font-size: 0.9em;
            }
            .project-detail span {
                color: var(--text-secondary);
                font-size: 0.95em;
            }
            .status-badge {
                display: inline-block;
                padding: 6px 16px;
                border-radius: var(--radius-pill);
                font-size: 12px;
                font-weight: 600;
            }
            .status-on-time {
                background: var(--primary-light);
                color: var(--primary);
            }
            .status-delayed {
                background: #FED7D7;
                color: #9B2C2C;
            }
            .status-critical {
                background: #FEB2B2;
                color: #9B2C2C;
            }
            tr.clickable {
                cursor: pointer;
            }
            tr.clickable:hover {
                background: var(--primary-light) !important;
            }

            /* Workflow Section */
            .workflow-section {
                background: var(--bg-card);
                padding: 40px;
                border-radius: var(--radius-card);
                margin-bottom: 32px;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-soft);
            }

            .workflow-title {
                color: var(--text-primary);
                font-size: 1.75em;
                font-weight: 700;
                margin-bottom: 32px;
                text-align: center;
                letter-spacing: -0.3px;
            }

            .workflow-steps {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 24px;
            }

            .workflow-step {
                flex: 1;
                text-align: center;
                position: relative;
            }

            .workflow-step-number {
                display: inline-block;
                width: 48px;
                height: 48px;
                background: var(--primary);
                color: white;
                border-radius: 50%;
                line-height: 48px;
                font-weight: 700;
                font-size: 16px;
                margin-bottom: 16px;
                box-shadow: var(--shadow-subtle);
            }

            .workflow-step-title {
                color: var(--primary);
                font-weight: 700;
                font-size: 14px;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .workflow-step-desc {
                color: var(--text-secondary);
                font-size: 13px;
                line-height: 1.6;
                white-space: pre-line;
            }

            .workflow-arrow {
                color: var(--accent-sage);
                font-size: 28px;
                margin-top: 24px;
            }

            @media (max-width: 768px) {
                .workflow-steps {
                    flex-direction: column;
                }

                .workflow-arrow {
                    transform: rotate(90deg);
                    margin: 16px auto;
                }
            }
            
            /* Responsive styles */
            @media (max-width: 1024px) {
                .sidebar {
                    transform: translateX(-100%);
                    transition: transform 0.3s ease;
                }
                
                .sidebar.open {
                    transform: translateX(0);
                }
                
                .mobile-menu-btn {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .main-content {
                    margin-left: 0;
                    padding: 24px;
                }
                
                .main-content.shifted {
                    margin-left: 0;
                }
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 0 16px;
                }
                
                .header {
                    padding: 24px;
                }
                
                .grid {
                    grid-template-columns: 1fr;
                }
                
                .filter-bar {
                    flex-direction: column;
                    align-items: stretch;
                }
                
                .filter-bar select, .filter-bar input, .filter-bar button {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <!-- Mobile menu button -->
        <button class="mobile-menu-btn" onclick="toggleSidebar()">
            <i data-lucide="menu"></i>
        </button>
        
        <!-- Sidebar -->
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">PAIMANA</div>
                <div class="sidebar-subtitle">PROJECT INTELLIGENCE</div>
            </div>
            
            <div class="sidebar-nav">
                <div class="nav-item active" onclick="navigateTo('dashboard')" id="nav-dashboard">
                    <i data-lucide="layout-dashboard"></i>
                    <span>Dashboard</span>
                </div>
                <div class="nav-item" onclick="navigateTo('map')" id="nav-map">
                    <i data-lucide="map"></i>
                    <span>Project Location Map</span>
                </div>
                <div class="nav-item" onclick="navigateTo('analysis')" id="nav-analysis">
                    <i data-lucide="bar-chart-3"></i>
                    <span>Project Analysis</span>
                </div>
                <div class="nav-item" onclick="navigateTo('delay')" id="nav-delay">
                    <i data-lucide="clock"></i>
                    <span>Delay Analysis</span>
                </div>
                <div class="nav-item" onclick="navigateTo('cost')" id="nav-cost">
                    <i data-lucide="indian-rupee"></i>
                    <span>Cost Analysis</span>
                </div>
                <div class="nav-item" onclick="navigateTo('risk')" id="nav-risk">
                    <i data-lucide="shield-alert"></i>
                    <span>Risk Prediction</span>
                </div>
                <div class="nav-item" onclick="navigateTo('similar')" id="nav-similar">
                    <i data-lucide="search"></i>
                    <span>Similar Projects</span>
                </div>
                <div class="nav-item" onclick="navigateTo('chatbot')" id="nav-chatbot">
    <i data-lucide="message-circle"></i>
    <span>AI Assistant</span>
</div>
                <div class="nav-item" onclick="window.location.href='/ai-insights'" id="nav-insights">
                    <i data-lucide="sparkles"></i>
                    <span>AI Insights</span>
                </div>
                <div class="nav-item" onclick="navigateTo('reports')" id="nav-reports">
                    <i data-lucide="file-down"></i>
                    <span>Reports & Export</span>
                </div>
            </div>
            
            <div class="sidebar-footer">
                © 2026 PAIMANA Platform
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content" id="mainContent">
            <div class="container">
            <div class="header">
                <div class="header-content">
                    <h1>PAIMANA Intelligence Platform</h1>
                    <p class="subtitle">Infrastructure Transparency Through AI</p>
                </div>
                <div class="header-controls">
                    <button class="control-btn" id="themeToggle">
                        <i data-lucide="sun" id="themeIcon"></i>
                        <span id="themeText">Light</span>
                    </button>
                </div>
            </div>
            
            <div class="alert">
                <strong>Live Demo Active</strong><br>
                This dashboard demonstrates real-time infrastructure project monitoring with AI-powered analytics and geo-visualization.
            </div>

            <!-- How PAIMANA Works Section -->
            <div class="workflow-section">
                <h2 class="workflow-title">How PAIMANA Works</h2>
                <div class="workflow-steps">
                    <div class="workflow-step">
                        <div class="workflow-step-number">01</div>
                        <div class="workflow-step-title">DATA COLLECTION</div>
                        <div class="workflow-step-desc">Project data
Cost • Progress • Delay • Location</div>
                    </div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">
                        <div class="workflow-step-number">02</div>
                        <div class="workflow-step-title">AI ANALYSIS</div>
                        <div class="workflow-step-desc">Analyze project performance,
patterns and anomalies</div>
                    </div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">
                        <div class="workflow-step-number">03</div>
                        <div class="workflow-step-title">ML PREDICTION</div>
                        <div class="workflow-step-desc">Predict potential delay
and project risk</div>
                    </div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">
                        <div class="workflow-step-number">04</div>
                        <div class="workflow-step-title">ACTIONABLE INSIGHTS</div>
                        <div class="workflow-step-desc">Help decision-makers
take better actions</div>
                    </div>
                </div>
            </div>

            <div class="filter-bar">
                <input type="text" id="searchInput" placeholder="Search projects...">
                <select id="districtFilter">
                    <option value="">All Districts</option>
                </select>
                <select id="categoryFilter">
                    <option value="">All Categories</option>
                </select>
                <select id="statusFilter">
                    <option value="">All Status</option>
                    <option value="on-time">On Time</option>
                    <option value="delayed">Delayed</option>
                    <option value="critical">Critical</option>
                </select>
                <button onclick="applyFilters()">Apply Filters</button>
                <button onclick="resetFilters()" style="background: #95a5a6;">Reset</button>
            </div>
            
            <h2 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Quick Statistics</h2>
            <div class="grid" id="stats">
                <div class="loading">Loading stats</div>
            </div>
            
            <div class="card" id="section-map">
                <h2>Project Location Map</h2>
                <div id="map"></div>
            </div>
            
            <div class="card" id="section-analysis">
                <h2>Category Performance Comparison</h2>
                <div id="category-chart"></div>
            </div>
            
            <div class="card" id="section-delay">
                <h2>Top Delayed Projects</h2>
                <div id="delayed">
                    <div class="loading">Loading delayed projects</div>
                </div>
            </div>
            
            <div class="card" id="section-cost">
                <h2>Cost Overruns</h2>
                <div id="overruns">
                    <div class="loading">Loading cost overruns</div>
                </div>
            </div>
            
            <div class="card" id="section-risk">
                <h2>AI Delay Predictions (ML Model)</h2>
                <p style="color: #7f8c8d; margin-bottom: 20px;">Machine learning predictions of project completion delays based on current progress and expenditure patterns.</p>
                <div id="ml-predictions">
                    <div class="loading">Loading ML predictions</div>
                </div>
            </div>
            
            <div class="card" id="section-similar">
                <h2>Similar Projects Analysis</h2>
                <p style="color: #7f8c8d; margin-bottom: 20px;">Compare projects with similar characteristics to identify patterns and insights.</p>
                <div id="similar-projects">
                    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                        <input type="text" id="similarProjectInput" placeholder="Enter Project Code (e.g. PRJ-001)" style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 8px;">
                        <button onclick="searchSimilarProjects()" style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 8px; cursor: pointer;">Find Similar</button>
                    </div>
                    <div id="similar-projects-results">
                        <div class="loading" style="background: transparent; border: none; padding: 0; color: #6c757d;">Enter a project code to find similar projects.</div>
                    </div>
                </div>
            </div>
            <div class="card" id="section-chatbot">

    <h2>🤖 PAIMANA Intelligence Assistant</h2>

    <p style="color:#7f8c8d;margin-bottom:20px;">
        Ask questions about infrastructure projects,
        delays, cost overruns, risk and PAIMANA.
    </p>

    <div id="chat-messages"
         style="
         height:360px;
         overflow-y:auto;
         background:#f8f9fa;
         border:1px solid #dee2e6;
         border-radius:10px;
         padding:15px;
         margin-bottom:15px;
         ">

        <div style="
             background:#e8f5e9;
             padding:12px;
             border-radius:10px;
             margin-bottom:10px;
             max-width:80%;
             ">
            <strong>🤖 PAIMANA Assistant</strong><br>
            Hello! I can answer questions about PAIMANA,
            project delays, cost overruns, risk prediction,
            machine learning and the project methodology.
        </div>

    </div>

    <div style="display:flex;gap:10px;">

        <input
            id="chat-input"
            type="text"
            placeholder="Ask something..."
            style="
                flex:1;
                padding:13px;
                border:1px solid #ced4da;
                border-radius:8px;
                font-size:15px;
            "
            onkeydown="if(event.key==='Enter') sendMessage()"
        >

        <button
            id="chat-send-btn"
            onclick="sendMessage()"
            style="
                padding:13px 22px;
                background:#1B6B3A;
                color:white;
                border:none;
                border-radius:8px;
                cursor:pointer;
                font-weight:600;
            "
        >
            Send
        </button>

    </div>

</div>
            <div class="card" id="section-insights">
                <h2>AI-Generated Insights</h2>
                <p style="color: #7f8c8d; margin-bottom: 20px;">Generate a detailed project intelligence report using project data and existing ML analysis.</p>
                
                <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                    <input
                        id="insightProjectId"
                        type="text"
                        placeholder="Enter Project Code (e.g., PRJ001)"
                        style="flex: 1; padding: 12px; border: 1px solid #ced4da; border-radius: 8px; font-size: 14px;"
                    >
                    <button
                        onclick="generateInsightsReport()"
                        style="padding: 12px 24px; background: #1B6B3A; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;"
                    >
                        Generate Report
                    </button>
                </div>
                
                <div id="ai-insights-message"></div>
            </div>
            
            <div class="card" id="section-reports">
                <h2>Data Quality Assessment</h2>
                <div id="quality">
                    <div class="loading">Loading quality report</div>
                </div>
                <div style="margin-top: 20px;">
                    <a href="/api/projects" style="display: inline-block; padding: 10px 20px; background: #1B6B3A; color: white; text-decoration: none; border-radius: 8px; margin-right: 10px;">Download CSV</a>
                    <a href="/api/projects-excel" style="display: inline-block; padding: 10px 20px; background: #1B6B3A; color: white; text-decoration: none; border-radius: 8px;">Download Excel</a>
                </div>
            </div>
            </div>
        </div>
        
        <!-- Project Detail Modal -->
        <div class="modal" id="projectModal">
            <div class="modal-content">
                <span class="modal-close" onclick="closeModal()">&times;</span>
                <h2 id="modalTitle">Project Details</h2>
                <div id="modalBody">
                    <div class="loading">Loading project details...</div>
                </div>
            </div>
        </div>
        
        <script>
            // Initialize Lucide icons
            lucide.createIcons();
            
            let allProjects = [];
            let filteredProjects = [];
            
            // Sidebar navigation functions
            function toggleSidebar() {
                const sidebar = document.getElementById('sidebar');
                const mainContent = document.getElementById('mainContent');
                sidebar.classList.toggle('open');
                mainContent.classList.toggle('shifted');
            }
            
            function navigateTo(section) {
                // Update active state
                document.querySelectorAll('.nav-item').forEach(item => {
                    item.classList.remove('active');
                });
                document.getElementById('nav-' + section).classList.add('active');
                
                // Scroll to section
                if (section === 'dashboard') {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                } else {
                    const sectionElement = document.getElementById('section-' + section);
                    if (sectionElement) {
                        sectionElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }
                
                // Close sidebar on mobile after navigation
                if (window.innerWidth <= 1024) {
                    toggleSidebar();
                }
            }
            
            // Close sidebar when clicking outside on mobile
            document.addEventListener('click', function(event) {
                const sidebar = document.getElementById('sidebar');
                const menuBtn = document.querySelector('.mobile-menu-btn');
                
                if (window.innerWidth <= 1024 && 
                    !sidebar.contains(event.target) && 
                    !menuBtn.contains(event.target) &&
                    sidebar.classList.contains('open')) {
                    toggleSidebar();
                }
            });
            
            // ── helpers ──────────────────────────────────────────────────────────
            function fetchWithTimeout(url, timeoutMs = 30000) {
                console.log(`[Dashboard] fetching ${url}`);
                const ctrl = new AbortController();
                const id = setTimeout(() => ctrl.abort(), timeoutMs);
                return fetch(url, { signal: ctrl.signal })
                    .then(res => {
                        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
                        return res.json();
                    })
                    .finally(() => clearTimeout(id));
            }

            // Fetch and display Quick Statistics + Delayed Projects + Cost Overruns
            fetchWithTimeout('/api/analytics', 30000)
                .then(res => {
                    console.log('[Dashboard] analytics response:', res);
                    const data = (res && res.data) ? res.data : res;
                    if (!data || !data.summary_statistics) throw new Error('Invalid analytics data structure');

                    const stats = data.summary_statistics;
                    document.getElementById('stats').innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${stats.total_projects || 0}</div>
                            <div class="stat-label">Total Projects</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.delayed_projects || 0}</div>
                            <div class="stat-label">Delayed</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.on_time_projects || 0}</div>
                            <div class="stat-label">On Time</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${(stats.average_delay_days || 0).toFixed(0)}</div>
                            <div class="stat-label">Avg Delay (days)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${(stats.average_progress_percent || 0).toFixed(1)}%</div>
                            <div class="stat-label">Avg Progress</div>
                        </div>
                    `;
                    
                    // Delayed projects table
                    allProjects = data.top_delayed_projects || [];
                    filteredProjects = [...allProjects];
                    populateFilters();
                    
                    if (data.top_delayed_projects && data.top_delayed_projects.length > 0) {
                        renderDelayedProjects(data.top_delayed_projects);
                    } else {
                        document.getElementById('delayed').innerHTML = '<p style="color:#6c757d;padding:20px;">No delayed projects found.</p>';
                    }
                    
                    // Cost overruns
                    if (data.top_cost_overruns && data.top_cost_overruns.length > 0) {
                        renderCostOverruns(data.top_cost_overruns);
                    } else {
                        document.getElementById('overruns').innerHTML = '<p style="color:#6c757d;padding:20px;">No significant cost overruns detected.</p>';
                    }
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('[Dashboard] Error loading analytics:', err);
                    document.getElementById('stats').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load statistics: ${msg}</p>`;
                    document.getElementById('delayed').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load delayed projects: ${msg}</p>`;
                    document.getElementById('overruns').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load cost overruns: ${msg}</p>`;
                });

            
            // Fetch ML predictions
            fetchWithTimeout('/api/ml-predictions', 30000)
                .then(res => {
                    console.log('[Dashboard] ML predictions response:', res);
                    const predictions = (res && res.data) ? res.data : (Array.isArray(res) ? res : []);
                    
                    if (predictions && predictions.length > 0) {
                        renderMLPredictions(predictions);
                    } else {
                        document.getElementById('ml-predictions').innerHTML = '<p style="color: #6c757d; padding: 20px;">No ML predictions available</p>';
                    }
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('[Dashboard] Error loading ML predictions:', err);
                    document.getElementById('ml-predictions').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load ML predictions: ${msg}</p>`;
                });

            
            // Fetch and render category chart
            fetchWithTimeout('/api/category-chart', 30000)
                .then(figJson => {
                    console.log('Category chart data received');
                    // figJson may be a pre-serialised string or already a plain object
                    const fig = (typeof figJson === 'string') ? JSON.parse(figJson) : figJson;
                    Plotly.newPlot('category-chart', fig.data, fig.layout, {responsive: true});
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('Error loading category chart:', err);
                    document.getElementById('category-chart').innerHTML = `<p style="text-align:center;padding:40px;color:#dc3545;">⚠ Failed to load category chart: ${msg}</p>`;
                });

            
            // Fetch quality data
            fetchWithTimeout('/api/quality', 30000)
                .then(res => {
                    console.log('[Dashboard] quality response:', res);
                    const data = (res && res.data) ? res.data : res;
                    console.log('[Dashboard] quality data:', data);
                    document.getElementById('quality').innerHTML = `
                        <div class="grid">
                            <div class="stat-card">
                                <div class="stat-value">${data.overall_grade || 'N/A'}</div>
                                <div class="stat-label">Quality Grade</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.reliability_score || 'N/A'}</div>
                                <div class="stat-label">Reliability</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.valid_records || 0}/${data.total_records || 0}</div>
                                <div class="stat-label">Valid Records</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.anomalies?.total || 0}</div>
                                <div class="stat-label">Anomalies</div>
                            </div>
                        </div>
                        <p style="padding: 12px; background: #e8f5e9; border-left: 4px solid #27ae60; border-radius: 4px; margin-top: 16px; font-size: 0.9em;">
                            <strong>${data.recommendation || 'No recommendation available'}</strong>
                        </p>
                    `;
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('Error loading quality data:', err);
                    document.getElementById('quality').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load quality data: ${msg}</p>`;
                });

            // Fetch map data
            fetchWithTimeout('/api/map-data', 30000)
                .then(data => {
                    console.log('Map data received:', data);
                    
                    if (data.error) {
                        document.getElementById('map').innerHTML = `<p style="text-align: center; padding: 40px; color: #dc3545;">${data.message}</p>`;
                        return;
                    }
                    
                    if (!data.latitudes || data.latitudes.length === 0) {
                        document.getElementById('map').innerHTML = '<p style="text-align: center; padding: 40px; color: #6c757d;">No location data available</p>';
                        return;
                    }
                    const trace = {
                        type: 'scattermapbox',
                        lat: data.latitudes,
                        lon: data.longitudes,
                        mode: 'markers',
                        marker: {
                            size: 12,
                            color: data.colors,
                            colorscale: [[0, '#27ae60'], [0.5, '#f39c12'], [1, '#e74c3c']],
                            showscale: true,
                            colorbar: {
                                title: 'Delay Status',
                                ticktext: ['On Time', 'Moderate', 'Critical'],
                                tickvals: [0, 50, 100]
                            }
                        },
                        text: data.labels,
                        hoverinfo: 'text'
                    };
                    
                    const layout = {
                        mapbox: {
                            style: 'open-street-map',
                            center: { lat: 19.5, lon: 75.5 },
                            zoom: 5.5
                        },
                        margin: { t: 0, b: 0, l: 0, r: 0 },
                        showlegend: false
                    };
                    
                    Plotly.newPlot('map', [trace], layout, {responsive: true});
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('Error loading map data:', err);
                    document.getElementById('map').innerHTML = `<p style="text-align:center;padding:40px;color:#dc3545;">⚠ Failed to load map data: ${msg}</p>`;
                });

            
            // Filter functions
            function populateFilters() {
                const districts = [...new Set(allProjects.map(p => p.district))].sort();
                const categories = [...new Set(allProjects.map(p => p.category || 'N/A'))].sort();
                
                const districtSelect = document.getElementById('districtFilter');
                const categorySelect = document.getElementById('categoryFilter');
                
                districts.forEach(d => {
                    const option = document.createElement('option');
                    option.value = d;
                    option.textContent = d;
                    districtSelect.appendChild(option);
                });
                
                categories.forEach(c => {
                    const option = document.createElement('option');
                    option.value = c;
                    option.textContent = c;
                    categorySelect.appendChild(option);
                });
            }
            
            function renderDelayedProjects(projects) {
                if (!projects || !Array.isArray(projects) || projects.length === 0) {
                    document.getElementById('delayed').innerHTML = '<p style="color: #6c757d; padding: 20px;">No delayed projects found.</p>';
                    return;
                }
                let table = '<table><tr><th>Project ID</th><th>Name</th><th>District</th><th>Delay (days)</th><th>Progress</th></tr>';
                projects.forEach(p => {
                    const delayDays = typeof p.delay_days === 'number' ? p.delay_days : 0;
                    const progress = typeof p.physical_progress_percent === 'number' ? p.physical_progress_percent : 0;
                    const statusClass = delayDays > 100 ? 'status-critical' : (delayDays > 0 ? 'status-delayed' : 'status-on-time');
                    const statusText = delayDays > 100 ? 'Critical' : (delayDays > 0 ? 'Delayed' : 'On Time');
                    const pid = escapeHTML(String(p.project_id || 'N/A'));
                    const pname = escapeHTML(String(p.project_name || 'Unknown'));
                    const dist = escapeHTML(String(p.district || 'N/A'));

                    table += `<tr class="clickable" onclick="showProjectDetails('${pid}')">
                        <td><strong>${pid}</strong></td>
                        <td>${pname}</td>
                        <td>${dist}</td>
                        <td><span class="status-badge ${statusClass}">${statusText}</span> ${delayDays}</td>
                        <td><div style="background: #ecf0f1; border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #00b4db, #0083b0); width: ${Math.min(100, Math.max(0, progress))}%; height: 100%;"></div>
                        </div>${progress.toFixed(1)}%</td>
                    </tr>`;
                });
                table += '</table>';
                document.getElementById('delayed').innerHTML = table;
            }

            function renderCostOverruns(overruns) {
                if (!overruns || !Array.isArray(overruns) || overruns.length === 0) {
                    document.getElementById('overruns').innerHTML = '<p style="color: #6c757d; padding: 20px;">No significant cost overruns detected.</p>';
                    return;
                }
                let table = '<table><tr><th>Project ID</th><th>Name</th><th>Overrun %</th><th>Sanctioned (₹ Cr)</th><th>Spent (₹ Cr)</th></tr>';
                overruns.slice(0, 10).forEach(p => {
                    const overrun = typeof p.cost_overrun_percent === 'number' ? p.cost_overrun_percent : 0;
                    const sanctioned = typeof p.sanctioned_cost === 'number' ? (p.sanctioned_cost > 1e6 ? p.sanctioned_cost / 1e7 : p.sanctioned_cost) : 0;
                    const spent = typeof p.expenditure_to_date === 'number' ? (p.expenditure_to_date > 1e6 ? p.expenditure_to_date / 1e7 : p.expenditure_to_date) : 0;
                    const pid = escapeHTML(String(p.project_id || 'N/A'));
                    const pname = escapeHTML(String(p.project_name || 'Unknown'));

                    table += `<tr>
                        <td><strong>${pid}</strong></td>
                        <td>${pname}</td>
                        <td style="color: #dc3545; font-weight: 600;">+${overrun.toFixed(1)}%</td>
                        <td>₹${sanctioned.toFixed(2)} Cr</td>
                        <td>₹${spent.toFixed(2)} Cr</td>
                    </tr>`;
                });
                table += '</table>';
                document.getElementById('overruns').innerHTML = table;
            }

            function renderMLPredictions(predictions) {
                if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
                    document.getElementById('ml-predictions').innerHTML = '<p style="color: #6c757d; padding: 20px;">No ML predictions available.</p>';
                    return;
                }
                let table = '<table><tr><th>Project ID</th><th>Name</th><th>District</th><th>Progress</th><th>Predicted Delay</th><th>Risk Level</th></tr>';
                predictions.forEach(p => {
                    const progress = typeof p.physical_progress_percent === 'number' ? p.physical_progress_percent : 0;
                    const delayDays = typeof p.predicted_delay_days === 'number' ? p.predicted_delay_days : Math.round((p.predicted_delay_months || 0) * 30);
                    const risk = String(p.risk_level || p.ML_Risk_Level || 'LOW');
                    const pid = escapeHTML(String(p.project_id || 'N/A'));
                    const pname = escapeHTML(String(p.project_name || 'Unknown'));
                    const dist = escapeHTML(String(p.district || 'N/A'));

                    let riskColor = '#27ae60';
                    if (risk === 'HIGH') riskColor = '#dc3545';
                    else if (risk === 'MEDIUM-HIGH' || risk === 'MEDIUM') riskColor = '#f39c12';

                    table += `<tr>
                        <td><strong>${pid}</strong></td>
                        <td>${pname}</td>
                        <td>${dist}</td>
                        <td>${progress.toFixed(1)}%</td>
                        <td style="color: #dc3545; font-weight: 600;">${delayDays} days</td>
                        <td><span class="status-badge" style="background:${riskColor};color:white;">${risk}</span></td>
                    </tr>`;
                });
                table += '</table>';
                document.getElementById('ml-predictions').innerHTML = table;
            }
            
            async function searchSimilarProjects() {
                const projectId = document.getElementById('similarProjectInput').value.trim();
                const resultsContainer = document.getElementById('similar-projects-results');
                
                if (!projectId) {
                    resultsContainer.innerHTML = '<p style="color: #e74c3c; padding: 20px;">Please enter a Project Code.</p>';
                    return;
                }
                
                resultsContainer.innerHTML = '<div class="loading" style="background: transparent; border: none;">Finding similar projects...</div>';
                
                try {
                    const response = await fetch(`/api/similar-projects/${encodeURIComponent(projectId)}`);
                    const data = await response.json();
                    
                    if (!data.success) {
                        resultsContainer.innerHTML = `<p style="color: #e74c3c; padding: 20px;">${data.error || 'Error finding similar projects.'}</p>`;
                        return;
                    }
                    
                    if (!data.data || data.data.length === 0) {
                        resultsContainer.innerHTML = '<p style="color: #6c757d; padding: 20px;">No similar projects found or invalid Project Code.</p>';
                        return;
                    }
                    
                    let table = '<table><tr><th>Project ID</th><th>Name</th><th>Similarity</th><th>Sector</th><th>Cost</th><th>Progress</th></tr>';
                    data.data.forEach(p => {
                        const pid = escapeHTML(String(p.project_id || 'N/A'));
                        const pname = escapeHTML(String(p.project_name || 'Unknown'));
                        const sim = (typeof p.similarity_score === 'number' ? p.similarity_score * 100 : 0).toFixed(1);
                        const sector = escapeHTML(String(p.sector || 'N/A'));
                        const cost = typeof p.cost === 'number' ? p.cost : 0;
                        const prog = typeof p.progress === 'number' ? p.progress : 0;
                        
                        table += `<tr>
                            <td><strong>${pid}</strong></td>
                            <td>${pname}</td>
                            <td><span class="status-badge" style="background:#3498db;color:white;">${sim}%</span></td>
                            <td>${sector}</td>
                            <td>₹${cost.toFixed(2)} Cr</td>
                            <td>${prog.toFixed(1)}%</td>
                        </tr>`;
                    });
                    table += '</table>';
                    resultsContainer.innerHTML = table;
                } catch (error) {
                    console.error('Error fetching similar projects:', error);
                    resultsContainer.innerHTML = '<p style="color: #e74c3c; padding: 20px;">Error communicating with server.</p>';
                }
            }
            
            async function generateInsights() {
                const projectId = document.getElementById('insightProjectId').value.trim();
                const resultsContainer = document.getElementById('ai-insights');
                
                if (!projectId) {
                    resultsContainer.innerHTML = '<p style="color: #e74c3c; padding: 20px;">Please enter a Project Code.</p>';
                    return;
                }
                
                resultsContainer.innerHTML = '<div class="loading">Generating AI insights...</div>';
                
                try {
                    const response = await fetch(`/api/generate-insights/${encodeURIComponent(projectId)}`);
                    const data = await response.json();
                    
                    if (!data.success) {
                        resultsContainer.innerHTML = `<p style="color: #e74c3c; padding: 20px;">${data.error || 'Error generating insights.'}</p>`;
                        return;
                    }
                    
                    const insights = data.data;
                    
                    // Build insights HTML
                    let html = '<div style="display: grid; gap: 20px;">';
                    
                    // Project Info Section
                    html += `
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #1B6B3A;">
                            <h3 style="margin-top: 0; color: #1B6B3A;">📋 Project Information</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                <div><strong>Project ID:</strong> ${escapeHTML(insights.project_info.project_id)}</div>
                                <div><strong>Name:</strong> ${escapeHTML(insights.project_info.project_name)}</div>
                                <div><strong>District:</strong> ${escapeHTML(insights.project_info.district)}</div>
                                <div><strong>Category:</strong> ${escapeHTML(insights.project_info.category)}</div>
                                <div><strong>Ministry:</strong> ${escapeHTML(insights.project_info.ministry)}</div>
                                <div><strong>Status:</strong> <span class="status-badge" style="background: ${
                                    insights.project_info.current_status === 'Critical' ? '#e74c3c' : 
                                    insights.project_info.current_status === 'Delayed' ? '#f39c12' : '#27ae60'
                                }; color: white;">${escapeHTML(insights.project_info.current_status)}</span></div>
                                <div><strong>Progress:</strong> ${insights.project_info.physical_progress_percent.toFixed(1)}%</div>
                            </div>
                        </div>
                    `;
                    
                    // ML Predictions Section
                    html += `
                        <div style="background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 4px solid #27ae60;">
                            <h3 style="margin-top: 0; color: #27ae60;">🤖 ML Model Predictions</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                <div><strong>Predicted Risk:</strong> <span class="status-badge" style="background: ${
                                    insights.ml_predictions.predicted_risk === 'HIGH' ? '#e74c3c' : 
                                    insights.ml_predictions.predicted_risk === 'MEDIUM-HIGH' || insights.ml_predictions.predicted_risk === 'MEDIUM' ? '#f39c12' : '#27ae60'
                                }; color: white;">${escapeHTML(insights.ml_predictions.predicted_risk)}</span></div>
                                <div><strong>Predicted Delay:</strong> ${insights.ml_predictions.predicted_delay_days} days</div>
                                <div><strong>Predicted Cost Overrun:</strong> ${insights.ml_predictions.predicted_cost_overrun_percent}%</div>
                                <div><strong>Risk Confidence:</strong> ${insights.ml_predictions.risk_confidence_percent}%</div>
                            </div>
                        </div>
                    `;
                    
                    // Financial Health Section
                    html += `
                        <div style="background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 4px solid #ffc107;">
                            <h3 style="margin-top: 0; color: #856404;">💰 Financial Health</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                <div><strong>Sanctioned Cost:</strong> ₹${insights.financial_health.sanctioned_cost_cr} Cr</div>
                                <div><strong>Expenditure to Date:</strong> ₹${insights.financial_health.expenditure_to_date_cr} Cr</div>
                                <div><strong>Budget Utilization:</strong> ${insights.financial_health.budget_utilization_percent}%</div>
                                <div><strong>Cost Overrun:</strong> ${insights.financial_health.cost_overrun_percent}%</div>
                            </div>
                        </div>
                    `;
                    
                    // Delay Analysis Section
                    html += `
                        <div style="background: #f8d7da; padding: 20px; border-radius: 10px; border-left: 4px solid #dc3545;">
                            <h3 style="margin-top: 0; color: #721c24;">⏱️ Delay Analysis</h3>
                            <div><strong>Current Delay:</strong> ${insights.delay_analysis.current_delay_days} days (${escapeHTML(insights.delay_analysis.delay_category)})</div>
                        </div>
                    `;
                    
                    // Risk Factors Section
                    html += `
                        <div style="background: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 4px solid #17a2b8;">
                            <h3 style="margin-top: 0; color: #0c5460;">⚠️ Risk Factors</h3>
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                ${insights.risk_factors.map(factor => `<li>${escapeHTML(factor)}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                    
                    // Progress vs Expenditure Analysis
                    html += `
                        <div style="background: #d4edda; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745;">
                            <h3 style="margin-top: 0; color: #155724;">📊 Progress vs Expenditure Analysis</h3>
                            <p>${escapeHTML(insights.progress_expenditure_analysis)}</p>
                        </div>
                    `;
                    
                    // Category Comparison
                    html += `
                        <div style="background: #e2e3e5; padding: 20px; border-radius: 10px; border-left: 4px solid #6c757d;">
                            <h3 style="margin-top: 0; color: #383d41;">📈 Category Comparison</h3>
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                ${insights.category_comparison.map(comp => `<li>${escapeHTML(comp)}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                    
                    // Similar Projects Section
                    if (insights.similar_projects && insights.similar_projects.length > 0) {
                        html += `
                            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #6f42c1;">
                                <h3 style="margin-top: 0; color: #6f42c1;">🔍 Similar Projects</h3>
                                <table style="width: 100%; border-collapse: collapse;">
                                    <tr style="background: #e9ecef;">
                                        <th style="padding: 8px; text-align: left;">Project ID</th>
                                        <th style="padding: 8px; text-align: left;">Name</th>
                                        <th style="padding: 8px; text-align: left;">Similarity</th>
                                        <th style="padding: 8px; text-align: left;">Sector</th>
                                    </tr>
                                    ${insights.similar_projects.map(p => `
                                        <tr style="border-bottom: 1px solid #dee2e6;">
                                            <td style="padding: 8px;"><strong>${escapeHTML(p.project_id)}</strong></td>
                                            <td style="padding: 8px;">${escapeHTML(p.project_name)}</td>
                                            <td style="padding: 8px;"><span class="status-badge" style="background: #6f42c1; color: white;">${(p.similarity_score * 100).toFixed(1)}%</span></td>
                                            <td style="padding: 8px;">${escapeHTML(p.sector)}</td>
                                        </tr>
                                    `).join('')}
                                </table>
                            </div>
                        `;
                    }
                    
                    // Risk Assessment & Recommendation
                    html += `
                        <div style="background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 4px solid #ffc107;">
                            <h3 style="margin-top: 0; color: #856404;">🎯 Risk Assessment & Recommendation</h3>
                            <p><strong>Assessment:</strong> ${escapeHTML(insights.risk_assessment)}</p>
                            <p><strong>Recommendation:</strong> ${escapeHTML(insights.recommendation)}</p>
                        </div>
                    `;
                    
                    html += '</div>';
                    resultsContainer.innerHTML = html;
                    
                } catch (error) {
                    console.error('Error generating insights:', error);
                    resultsContainer.innerHTML = '<p style="color: #e74c3c; padding: 20px;">Error communicating with server.</p>';
                }
            }
            
            async function generateInsightsReport() {
                const projectId = document.getElementById('insightProjectId').value.trim();
                const messageDiv = document.getElementById('ai-insights-message');
                
                if (!projectId) {
                    messageDiv.innerHTML = '<p style="color: #e74c3c; padding: 10px;">Please enter a Project Code.</p>';
                    return;
                }
                
                messageDiv.innerHTML = '<p style="color: #6c757d; padding: 10px;">Validating Project ID...</p>';
                
                try {
                    // Validate project ID by checking if it exists in the dataset
                    const response = await fetch(`/api/generate-insights/${encodeURIComponent(projectId)}`);
                    const data = await response.json();
                    
                    if (!data.success) {
                        messageDiv.innerHTML = `<p style="color: #e74c3c; padding: 10px;">${data.error || 'Project not found. Please check the Project ID.'}</p>`;
                        return;
                    }
                    
                    // Project is valid, navigate to report page
                    messageDiv.innerHTML = '<p style="color: #28a745; padding: 10px;">Project ID validated. Generating report...</p>';
                    setTimeout(() => {
                        window.location.href = `/ai-insights/report/${encodeURIComponent(projectId)}`;
                    }, 500);
                } catch (error) {
                    console.error('Error validating project ID:', error);
                    messageDiv.innerHTML = '<p style="color: #e74c3c; padding: 10px;">Error validating Project ID. Please try again.</p>';
                }
            }
            
            function applyFilters() {
                const searchTerm = document.getElementById('searchInput').value.toLowerCase();
                const district = document.getElementById('districtFilter').value;
                const category = document.getElementById('categoryFilter').value;
                const status = document.getElementById('statusFilter').value;
                
                filteredProjects = allProjects.filter(p => {
                    const matchesSearch = p.project_name.toLowerCase().includes(searchTerm) || 
                                          p.project_id.toLowerCase().includes(searchTerm);
                    const matchesDistrict = !district || p.district === district;
                    const matchesCategory = !category || (p.category || 'N/A') === category;
                    
                    let matchesStatus = true;
                    if (status === 'on-time') matchesStatus = p.delay_days <= 0;
                    else if (status === 'delayed') matchesStatus = p.delay_days > 0 && p.delay_days <= 100;
                    else if (status === 'critical') matchesStatus = p.delay_days > 100;
                    
                    return matchesSearch && matchesDistrict && matchesCategory && matchesStatus;
                });
                
                renderDelayedProjects(filteredProjects);
            }
            
            function resetFilters() {
                document.getElementById('searchInput').value = '';
                document.getElementById('districtFilter').value = '';
                document.getElementById('categoryFilter').value = '';
                document.getElementById('statusFilter').value = '';
                filteredProjects = [...allProjects];
                renderDelayedProjects(filteredProjects);
            }
            
            function showProjectDetails(projectId) {
                const project = allProjects.find(p => p.project_id === projectId);
                if (!project) return;
                
                const statusClass = project.delay_days > 100 ? 'status-critical' : (project.delay_days > 0 ? 'status-delayed' : 'status-on-time');
                const statusText = project.delay_days > 100 ? 'Critical' : (project.delay_days > 0 ? 'Delayed' : 'On Time');
                
                document.getElementById('modalTitle').textContent = project.project_name;
                document.getElementById('modalBody').innerHTML = `
                    <div class="project-detail">
                        <label>Project ID</label>
                        <span>${project.project_id}</span>
                    </div>
                    <div class="project-detail">
                        <label>District</label>
                        <span>${project.district}</span>
                    </div>
                    <div class="project-detail">
                        <label>Category</label>
                        <span>${project.category || 'N/A'}</span>
                    </div>
                    <div class="project-detail">
                        <label>Status</label>
                        <span class="status-badge ${statusClass}">${statusText}</span>
                    </div>
                    <div class="project-detail">
                        <label>Delay</label>
                        <span>${project.delay_days} days</span>
                    </div>
                    <div class="project-detail">
                        <label>Physical Progress</label>
                        <span>${project.physical_progress_percent.toFixed(1)}%</span>
                    </div>
                `;
                
                document.getElementById('projectModal').classList.add('active');
            }
            
            function closeModal() {
                document.getElementById('projectModal').classList.remove('active');
            }
            
            // Close modal when clicking outside
            window.onclick = function(event) {
                const modal = document.getElementById('projectModal');
                if (event.target === modal) {
                    closeModal();
                }
            }
            
            // Search on Enter key
            document.getElementById('searchInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    applyFilters();
                }
            });  
                        // ============================================================
// PAIMANA DATASET-AWARE AI ASSISTANT (Connected to /api/chat)
// Fully Lazy-Loaded & Independent of Dashboard Load
// ============================================================

let chatHistory = [];
let isChatProcessing = false;

function loadMarkedLibrary() {
    if (typeof marked !== 'undefined') return Promise.resolve();
    return new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
        script.onload = () => resolve();
        script.onerror = () => resolve(); // fallback gracefully to simpleMarkdownParse if network blocks CDN
        document.head.appendChild(script);
    });
}

async function sendMessage() {
    if (isChatProcessing) return;

    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send-btn");
    const messages = document.getElementById("chat-messages");

    const question = input.value.trim();
    if (!question) return;

    // Lock UI during processing
    isChatProcessing = true;
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    // Display user message safely
    const escapedQuestion = escapeHTML(question);
    messages.innerHTML += `
        <div style="
            background:#1B6B3A;
            color:white;
            padding:12px 16px;
            border-radius:12px;
            margin-bottom:12px;
            margin-left:20%;
            word-wrap:break-word;
            box-shadow:0 1px 3px rgba(0,0,0,0.1);
        ">
            <strong>You</strong><br>
            ${escapedQuestion}
        </div>
    `;

    chatHistory.push({ role: "user", content: question });

    // Show loading indicator
    const loadingId = "chat-loading-" + Date.now();
    messages.innerHTML += `
        <div id="${loadingId}" style="
            background:#f0f4f1;
            color:#2e7d32;
            padding:12px 16px;
            border-radius:12px;
            margin-bottom:12px;
            margin-right:20%;
            display:flex;
            align-items:center;
            gap:8px;
            font-style:italic;
        ">
            <span>🤖 PAIMANA Assistant is thinking...</span>
        </div>
    `;

    input.value = "";
    messages.scrollTop = messages.scrollHeight;

    // Set 15-second timeout so chatbot never freezes dashboard UI
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    try {
        // Attempt lazy loading of marked.js in background if not loaded
        loadMarkedLibrary().catch(() => {});

        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: question,
                history: chatHistory
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) loadingElem.remove();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const answer = data.answer || "I couldn't find that information in the PAIMANA dataset.";

        chatHistory.push({ role: "assistant", content: answer });

        let formattedAnswer;
        if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
            formattedAnswer = marked.parse(answer);
        } else {
            formattedAnswer = simpleMarkdownParse(answer);
        }

        messages.innerHTML += `
            <div style="
                background:#e8f5e9;
                color:#1b5e20;
                padding:14px 18px;
                border-radius:12px;
                margin-bottom:12px;
                margin-right:15%;
                word-wrap:break-word;
                border:1px solid #c8e6c9;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);
            ">
                <strong style="color:#1B6B3A;">🤖 PAIMANA Assistant</strong><br>
                <div class="chat-response-content" style="margin-top:6px; line-height:1.5;">
                    ${formattedAnswer}
                </div>
            </div>
        `;

    } catch (error) {
        console.error("Chat error:", error);
        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) loadingElem.remove();

        const errorMsg = error.name === 'AbortError' 
            ? '⚠️ The request timed out (15s). Please try asking your question again.'
            : '⚠️ Sorry, I encountered an error connecting to the PAIMANA backend. Please try again.';

        messages.innerHTML += `
            <div style="
                background:#ffebee;
                color:#c62828;
                padding:12px 16px;
                border-radius:12px;
                margin-bottom:12px;
                margin-right:20%;
                border:1px solid #ffcdd2;
            ">
                <strong>🤖 PAIMANA Assistant</strong><br>
                ${errorMsg}
            </div>
        `;
    } finally {
        isChatProcessing = false;
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
    }
}

function escapeHTML(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function simpleMarkdownParse(text) {
    if (!text) return "";
    let html = escapeHTML(text);
    html = html.replace(/^### (.*$)/gim, '<h4 style="margin:10px 0 5px 0;color:#1B6B3A;">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 style="margin:12px 0 6px 0;color:#1B6B3A;">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 style="margin:14px 0 8px 0;color:#1B6B3A;">$1</h2>');
    html = html.replace(/[*][*](.*?)[*][*]/g, '<strong>$1</strong>');
    html = html.replace(/[*](.*?)[*]/g, '<em>$1</em>');
    html = html.replace(/`(.*?)`/g, '<code style="background:#e0e0e0;padding:2px 5px;border-radius:4px;">$1</code>');
    html = html.replace(/\\n/g, '<br>');
    return html;
}

// Theme Toggle - Deferred Initialization
(function initThemeToggle() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupThemeToggle);
    } else {
        setupThemeToggle();
    }

    function setupThemeToggle() {
        let currentTheme = localStorage.getItem('paimana_theme') || 'light';
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        const themeText = document.getElementById('themeText');

        if (themeToggle && themeIcon && themeText) {
            // Initialize theme
            document.documentElement.setAttribute('data-theme', currentTheme);
            updateThemeUI();

            themeToggle.addEventListener('click', function() {
                currentTheme = currentTheme === 'light' ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', currentTheme);
                localStorage.setItem('paimana_theme', currentTheme);
                updateThemeUI();
            });

            function updateThemeUI() {
                if (currentTheme === 'dark') {
                    themeIcon.setAttribute('data-lucide', 'moon');
                    themeText.textContent = 'Dark';
                } else {
                    themeIcon.setAttribute('data-lucide', 'sun');
                    themeText.textContent = 'Light';
                }
                lucide.createIcons();
            }
        }
    }
})();
        </script>
    </body>
    </html>
    """


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    PAIMANA Intelligence Chatbot Endpoint
    Request: {"message": "...", "history": [...]}
    Response: {"answer": "...", "sources": [...]}
    """
    try:
        req_data = request.get_json(silent=True) or {}
        message = req_data.get('message', '').strip()
        history = req_data.get('history', [])

        if not message:
            return jsonify({'error': 'Message parameter is required'}), 400

        data = load_or_generate_data()
        projects_df = data['projects']
        analytics_report = data['analytics']

        result = chatbot_engine.process_chat(
            message=message,
            history=history,
            df=projects_df,
            analytics=analytics_report
        )
        return jsonify(result), 200

    except Exception as e:
        import traceback
        print(f"Error in /api/chat: {e}\n{traceback.format_exc()}")
        return jsonify({
            'answer': "I encountered an error processing your request. Please try again.",
            'sources': []
        }), 500


@app.route('/api/projects')
def api_projects():
    """Get all projects as CSV"""
    data = load_or_generate_data()
    csv = data['projects'].to_csv(index=False)
    return csv, 200, {'Content-Type': 'text/csv', 
                      'Content-Disposition': 'attachment; filename=projects.csv'}


@app.route('/api/analytics')
def api_analytics():
    """Get analytics report"""
    try:
        print("[API] delayed projects & cost overruns: generating analytics report")
        data = load_or_generate_data()
        report = data['analytics']
        
        # Ensure values in report are sanitized and JSON serializable
        delayed_list = report.get('top_delayed_projects', [])
        overruns_list = report.get('top_cost_overruns', [])

        print(f"[API] analytics: dataset processed successfully ({len(delayed_list)} delayed, {len(overruns_list)} cost overruns)")

        payload = {
            "success": True,
            "data": {
                "summary_statistics": report.get("summary_statistics", {}),
                "top_delayed_projects": delayed_list,
                "top_cost_overruns": overruns_list,
                "category_analysis": report.get("category_analysis", {}),
                "generated_at": report.get("generated_at", "")
            },
            # Also keep top-level fields for direct backwards compatibility
            "summary_statistics": report.get("summary_statistics", {}),
            "top_delayed_projects": delayed_list,
            "top_cost_overruns": overruns_list,
            "category_analysis": report.get("category_analysis", {}),
            "generated_at": report.get("generated_at", "")
        }
        return json.dumps(payload, default=str), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        import traceback
        print(f"[API] analytics ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Failed to generate analytics: {e}"}), 500


@app.route('/api/quality')
def api_quality():
    """Get quality report"""
    try:
        print("[API] quality check: fetching report")
        data = load_or_generate_data()
        report = data['quality_report']
        return jsonify({"success": True, "data": report, **report})
    except Exception as e:
        print(f"[API] quality ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/delayed')
def api_delayed():
    """Get delayed projects list"""
    try:
        print("[API] delayed projects: fetching list")
        data = load_or_generate_data()
        df = data['projects'].copy()
        delayed = df[df['Actual_Delay_Months'] > 0].copy()
        
        result = []
        for _, row in delayed.iterrows():
            delay_days = row.get('delay_days', row.get('Actual_Delay_Months', 0) * 30)
            if pd.isna(delay_days) or math.isnan(float(delay_days)):
                delay_days = 0
            progress = row.get('Physical Progress (%)', 0)
            if pd.isna(progress) or math.isnan(float(progress)):
                progress = 0.0

            result.append({
                'project_id': str(row.get('Project Code', 'N/A')),
                'project_name': str(row.get('Project Name', 'Unknown')),
                'district': str(row.get('State', 'N/A')),
                'delay_days': float(delay_days),
                'physical_progress_percent': float(progress),
                'category': str(row.get('Sector', 'N/A')),
                'risk_level': str(row.get('Risk_Level', 'N/A')),
            })
        print(f"[API] delayed projects: returning {len(result)} records")
        payload = {"success": True, "data": result, "count": len(result)}
        return json.dumps(payload, default=str), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        import traceback
        print(f"[API] delayed projects ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/ml-predictions')
def api_ml_predictions():
    """Get predictions from trained PAIMANA ML models"""
    try:
        print("[API] AI delay predictions: starting ML inference pipeline")
        data = load_or_generate_data()
        predictions_df = data['predictions'].copy()

        predictions_df = predictions_df.sort_values(
            'predicted_delay_days',
            ascending=False
        ).head(10)

        result = []
        for _, row in predictions_df.iterrows():
            delay_days = row.get('predicted_delay_days', row.get('ML_Predicted_Delay_Days', 0))
            if pd.isna(delay_days) or math.isnan(float(delay_days)):
                delay_days = 0

            cost_overrun = row.get('ML_Predicted_Cost_Overrun_%', row.get('Cost_Overrun_Ratio', 0) * 100)
            if pd.isna(cost_overrun) or math.isnan(float(cost_overrun)):
                cost_overrun = 0.0

            progress = row.get('Physical Progress (%)', 0)
            if pd.isna(progress) or math.isnan(float(progress)):
                progress = 0.0

            risk = str(row.get('ML_Risk_Level', row.get('Risk_Level', 'LOW')))

            result.append({
                'project_id': str(row.get('Project Code', 'N/A')),
                'project_name': str(row.get('Project Name', 'Unknown')),
                'district': str(row.get('State', 'N/A')),
                'physical_progress_percent': float(progress),
                'predicted_delay_days': int(float(delay_days)),
                'cost_overrun_percent': round(float(cost_overrun), 2),
                'risk_level': risk,
                'risk_confidence': float(row.get('ML_Risk_Confidence_%', 75.0))
            })

        print(f"[API] AI delay predictions: returning {len(result)} ML prediction records")
        payload = {"success": True, "data": result}
        return json.dumps(payload, default=str), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        import traceback
        print(f"[API] ml-predictions ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": f"ML model prediction error: {e}"}), 500


@app.route('/api/category-chart')
def api_category_chart():
    """Generate category comparison chart data"""
    data = load_or_generate_data()
    analytics = data['analytics']
    
    category_stats = analytics.get('category_analysis', {})
    
    categories = list(category_stats.keys())
    delayed_counts = [stats['delayed_projects'] for stats in category_stats.values()]
    total_counts = [stats['total_projects'] for stats in category_stats.values()]
    avg_delays = [stats['average_delay'] for stats in category_stats.values()]
    
    # Create Plotly bar chart with earthy color palette
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Delayed Projects',
        x=categories,
        y=delayed_counts,
        marker_color='#8FA58A'  # Sage green for delayed
    ))
    
    fig.add_trace(go.Bar(
        name='On-Time Projects',
        x=categories,
        y=[total - delayed for total, delayed in zip(total_counts, delayed_counts)],
        marker_color='#174D3B'  # Forest green for on-time
    ))
    
    fig.update_layout(
        title='Project Status by Category',
        xaxis_title='Category',
        yaxis_title='Number of Projects',
        barmode='stack',
        template='plotly_white',
        height=400,
        font=dict(
            family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
            size=12,
            color='#183F32'
        ),
        plot_bgcolor='#FAF8F0',
        paper_bgcolor='#FAF8F0',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return jsonify(plotly.utils.PlotlyJSONEncoder().encode(fig))


@app.route('/api/projects-excel')
def api_projects_excel():
    """Export projects to Excel format"""
    data = load_or_generate_data()
    df = data['projects']
    
    # Create Excel file in memory
    from io import BytesIO
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Projects', index=False)
        data['predictions'].to_excel(writer, sheet_name='ML Predictions', index=False)
    
    output.seek(0)
    
    return output.getvalue(), 200, {
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'Content-Disposition': 'attachment; filename=paimana_projects.xlsx'
    }


@app.route('/api/similar-projects/<project_id>')
def api_similar_projects(project_id):
    """Get top 5 similar projects using real data heuristics"""
    try:
        data = load_or_generate_data()
        df = data['projects']
        
        if project_id not in df['Project Code'].astype(str).values:
            return jsonify({"success": False, "error": f"Project Code '{project_id}' not found."}), 404
            
        target = df[df['Project Code'].astype(str) == project_id].iloc[0]
        
        scores = []
        for _, row in df.iterrows():
            curr_id = str(row.get('Project Code', ''))
            if curr_id == project_id:
                continue
                
            score = 0
            
            # Sector match (30%)
            if str(row.get('Sector', '')) == str(target.get('Sector', '')):
                score += 0.3
                
            # State match (20%)
            if str(row.get('State', '')) == str(target.get('State', '')):
                score += 0.2
                
            # Cost similarity (25%)
            cost1 = float(target.get('Original Cost (Rs. Crore)', 0) or 0)
            cost2 = float(row.get('Original Cost (Rs. Crore)', 0) or 0)
            if pd.isna(cost1): cost1 = 0
            if pd.isna(cost2): cost2 = 0
            max_cost = max(cost1, cost2, 1)
            score += 0.25 * (1 - abs(cost1 - cost2) / max_cost)
            
            # Progress similarity (25%)
            prog1 = float(target.get('Physical Progress (%)', 0) or 0)
            prog2 = float(row.get('Physical Progress (%)', 0) or 0)
            if pd.isna(prog1): prog1 = 0
            if pd.isna(prog2): prog2 = 0
            score += 0.25 * (1 - abs(prog1 - prog2) / 100)
            
            scores.append({
                'project_id': curr_id,
                'project_name': str(row.get('Project Name', 'Unknown')),
                'similarity_score': float(score),
                'sector': str(row.get('Sector', 'N/A')),
                'cost': float(cost2),
                'progress': float(prog2)
            })
            
        scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_5 = scores[:5]
        
        return jsonify({"success": True, "data": top_5}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        import traceback
        print(f"[API] similar-projects ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Similar projects error: {e}"}), 500


@app.route('/api/generate-insights/<project_id>')
def api_generate_insights(project_id):
    """Generate detailed project intelligence report using existing data and ML predictions"""
    try:
        print(f"[API] generate-insights: Starting analysis for project {project_id}")
        data = load_or_generate_data()
        df = data['projects']
        predictions_df = data['predictions']
        analytics = data['analytics']
        
        # Check if project exists
        if project_id not in df['Project Code'].astype(str).values:
            return jsonify({"success": False, "error": f"Project Code '{project_id}' not found."}), 404
        
        # Get project data
        project = df[df['Project Code'].astype(str) == project_id].iloc[0]
        
        # Get ML prediction for this project
        project_prediction = predictions_df[predictions_df['Project Code'].astype(str) == project_id]
        if len(project_prediction) > 0:
            pred_row = project_prediction.iloc[0]
            predicted_delay = pred_row.get('predicted_delay_days', pred_row.get('ML_Predicted_Delay_Days', 0))
            predicted_cost_overrun = pred_row.get('ML_Predicted_Cost_Overrun_%', pred_row.get('Cost_Overrun_Ratio', 0) * 100)
            predicted_risk = str(pred_row.get('ML_Risk_Level', pred_row.get('Risk_Level', 'LOW')))
            risk_confidence = float(pred_row.get('ML_Risk_Confidence_%', 75.0))
        else:
            predicted_delay = 0
            predicted_cost_overrun = 0
            predicted_risk = 'LOW'
            risk_confidence = 0
        
        # Extract project details
        project_name = str(project.get('Project Name', 'Unknown'))
        district = str(project.get('State', 'N/A'))
        category = str(project.get('Sector', 'N/A'))
        ministry = str(project.get('Ministry', 'Data unavailable'))
        progress = float(project.get('Physical Progress (%)', 0) or 0)
        sanctioned_cost = float(project.get('Original Cost (Rs. Crore)', 0) or 0)
        expenditure = float(project.get('Cumulative Expenditure (Rs. Crore)', 0) or 0)
        
        # Use actual delay from dataset (in months), convert to days
        actual_delay_months = float(project.get('Actual_Delay_Months', 0) or 0)
        delay_days = actual_delay_months * 30 if actual_delay_months > 0 else 0
        
        # Calculate financial metrics from actual dataset values
        budget_utilization = (expenditure / sanctioned_cost * 100) if sanctioned_cost > 0 else 0
        cost_overrun_percent = ((expenditure - sanctioned_cost) / sanctioned_cost * 100) if sanctioned_cost > 0 else 0
        
        # Determine current status
        if delay_days <= 0:
            current_status = "On Time"
        elif delay_days <= 100:
            current_status = "Delayed"
        else:
            current_status = "Critical"
        
        # Calculate category averages
        category_stats = analytics.get('category_analysis', {}).get(category, {})
        category_avg_delay = category_stats.get('average_delay', 0)
        category_avg_progress = category_stats.get('average_progress', 0)
        
        # Risk factors analysis
        risk_factors = []
        if delay_days > 100:
            risk_factors.append("Significant delay (>100 days)")
        if budget_utilization > 120:
            risk_factors.append("High budget utilization (>120%)")
        if progress < 30 and delay_days > 50:
            risk_factors.append("Low progress with significant delay")
        if cost_overrun_percent > 20:
            risk_factors.append("High cost overrun (>20%)")
        if predicted_risk in ['HIGH', 'MEDIUM-HIGH']:
            risk_factors.append(f"ML model predicts {predicted_risk} risk")
        
        if not risk_factors:
            risk_factors.append("No significant risk factors identified")
        
        # Progress vs expenditure analysis
        progress_expenditure_gap = progress - budget_utilization
        if abs(progress_expenditure_gap) > 20:
            if progress_expenditure_gap > 0:
                expenditure_analysis = f"Progress ({progress:.1f}%) exceeds budget utilization ({budget_utilization:.1f}%) by {progress_expenditure_gap:.1f}%. This may indicate efficient spending."
            else:
                expenditure_analysis = f"Budget utilization ({budget_utilization:.1f}%) exceeds progress ({progress:.1f}%) by {abs(progress_expenditure_gap):.1f}%. This may indicate cost overruns relative to work completed."
        else:
            expenditure_analysis = f"Progress ({progress:.1f}%) and budget utilization ({budget_utilization:.1f}%) are well aligned."
        
        # Comparison with category averages
        comparison_insights = []
        if category_avg_delay > 0:
            delay_comparison = delay_days - category_avg_delay
            if delay_comparison > 30:
                comparison_insights.append(f"Project delay ({delay_days:.0f} days) is {delay_comparison:.0f} days higher than category average ({category_avg_delay:.0f} days)")
            elif delay_comparison < -30:
                comparison_insights.append(f"Project delay ({delay_days:.0f} days) is {abs(delay_comparison):.0f} days better than category average ({category_avg_delay:.0f} days)")
            else:
                comparison_insights.append(f"Project delay ({delay_days:.0f} days) is close to category average ({category_avg_delay:.0f} days)")
        
        if category_avg_progress > 0:
            progress_comparison = progress - category_avg_progress
            if progress_comparison > 15:
                comparison_insights.append(f"Progress ({progress:.1f}%) is {progress_comparison:.1f}% higher than category average ({category_avg_progress:.1f}%)")
            elif progress_comparison < -15:
                comparison_insights.append(f"Progress ({progress:.1f}%) is {abs(progress_comparison):.1f}% lower than category average ({category_avg_progress:.1f}%)")
            else:
                comparison_insights.append(f"Progress ({progress:.1f}%) is close to category average ({category_avg_progress:.1f}%)")
        
        # Get similar projects
        similar_projects = []
        try:
            similar_response = api_similar_projects(project_id)
            if similar_response[1] == 200:
                # Extract JSON from Response object
                response_data = similar_response[0].get_json() if hasattr(similar_response[0], 'get_json') else json.loads(similar_response[0].get_data(as_text=True))
                if response_data and response_data.get('success'):
                    similar_projects = response_data.get('data', [])[:5]
        except Exception as e:
            print(f"[API] generate-insights: Error fetching similar projects: {e}")
            similar_projects = []
        
        # Generate recommendation
        if predicted_risk == 'HIGH' or delay_days > 100:
            recommendation = "URGENT: Project requires immediate attention. Review resource allocation, identify bottlenecks, and consider intervention strategies."
        elif predicted_risk == 'MEDIUM-HIGH' or delay_days > 50:
            recommendation = "Project shows concerning indicators. Monitor closely and implement corrective measures to prevent further delays."
        elif delay_days > 0:
            recommendation = "Project is experiencing minor delays. Regular monitoring and proactive management recommended."
        else:
            recommendation = "Project is performing well. Continue current management practices and maintain regular monitoring."
        
        # Build insights report
        insights = {
            'project_info': {
                'project_id': project_id,
                'project_name': project_name,
                'district': district,
                'category': category,
                'ministry': ministry,
                'current_status': current_status,
                'physical_progress_percent': progress
            },
            'ml_predictions': {
                'predicted_risk': predicted_risk,
                'predicted_delay_days': int(predicted_delay) if not pd.isna(predicted_delay) else 0,
                'predicted_cost_overrun_percent': round(float(predicted_cost_overrun), 2) if not pd.isna(predicted_cost_overrun) else 0,
                'risk_confidence_percent': round(risk_confidence, 1)
            },
            'financial_health': {
                'sanctioned_cost_cr': round(sanctioned_cost, 2) if sanctioned_cost > 0 else 0,
                'expenditure_to_date_cr': round(expenditure, 2) if expenditure > 0 else 0,
                'budget_utilization_percent': round(budget_utilization, 1) if budget_utilization > 0 else 0,
                'cost_overrun_percent': round(cost_overrun_percent, 1) if cost_overrun_percent != 0 else 0
            },
            'delay_analysis': {
                'current_delay_days': int(delay_days) if delay_days > 0 else 0,
                'delay_category': current_status
            },
            'risk_factors': risk_factors,
            'progress_expenditure_analysis': expenditure_analysis,
            'category_comparison': comparison_insights if comparison_insights else ['Data unavailable for comparison'],
            'similar_projects': similar_projects,
            'risk_assessment': f"Project is considered {predicted_risk} risk based on ML model predictions and current performance metrics.",
            'recommendation': recommendation
        }
        
        print(f"[API] generate-insights: Successfully generated insights for {project_id}")
        return jsonify({"success": True, "data": insights}), 200, {'Content-Type': 'application/json'}
        
    except Exception as e:
        import traceback
        print(f"[API] generate-insights ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Failed to generate insights: {e}"}), 500


@app.route('/api/download-insights-pdf/<project_id>')
def download_insights_pdf(project_id):
    """Generate and download PDF report for project insights"""
    try:
        print(f"[API] download-insights-pdf: Generating PDF for project {project_id}")
        
        # Get insights data
        insights_response = api_generate_insights(project_id)
        if insights_response[1] != 200:
            return jsonify({"success": False, "error": "Failed to generate insights data"}), 400
        
        insights_data = json.loads(insights_response[0].get_data(as_text=True))
        if not insights_data.get('success'):
            return jsonify({"success": False, "error": insights_data.get('error', 'Failed to generate insights')}), 400
        
        insights = insights_data['data']
        
        # Create PDF
        from io import BytesIO
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#174D3B'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#174D3B'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 11
        
        # Header
        story.append(Paragraph("PAIMANA Intelligence Platform", title_style))
        story.append(Paragraph("Project Intelligence Report", ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=18, alignment=TA_CENTER, spaceAfter=10)))
        story.append(Spacer(1, 12))
        
        # Project ID and Date
        meta_data = [
            ['Project ID:', insights['project_info']['project_id']],
            ['Generated:', datetime.now().strftime('%d %b %Y')],
        ]
        meta_table = Table(meta_data, colWidths=[2*inch, 3*inch], hAlign=TA_CENTER)
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#6c757d')),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 30))
        
        # A. Project Overview
        story.append(Paragraph("A. Project Overview", heading_style))
        overview_data = [
            ['Project ID:', insights['project_info']['project_id']],
            ['Project Name:', insights['project_info']['project_name']],
            ['State:', insights['project_info']['district']],
            ['Category/Sector:', insights['project_info']['category']],
            ['Ministry:', insights['project_info']['ministry']],
            ['Physical Progress:', f"{insights['project_info']['physical_progress_percent']:.1f}%"],
            ['Current Status:', insights['project_info']['current_status']],
        ]
        overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
        overview_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6c757d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#183F32')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8E4D8')),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # B. ML Assessment
        story.append(Paragraph("B. ML Assessment", heading_style))
        ml_data = [
            ['Predicted Risk:', insights['ml_predictions']['predicted_risk']],
            ['Predicted Delay:', f"{insights['ml_predictions']['predicted_delay_days']} days"],
            ['Predicted Cost Overrun:', f"{insights['ml_predictions']['predicted_cost_overrun_percent']}%"],
            ['Confidence:', f"{insights['ml_predictions']['risk_confidence_percent']}%" if insights['ml_predictions']['risk_confidence_percent'] > 0 else 'Data unavailable'],
        ]
        ml_table = Table(ml_data, colWidths=[2*inch, 4*inch])
        ml_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6c757d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#183F32')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8E4D8')),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 20))
        
        # C. Financial Assessment
        story.append(Paragraph("C. Financial Assessment", heading_style))
        financial_data = [
            ['Sanctioned Cost:', f"₹{insights['financial_health']['sanctioned_cost_cr']} Cr"],
            ['Expenditure to Date:', f"₹{insights['financial_health']['expenditure_to_date_cr']} Cr"],
            ['Budget Utilization:', f"{insights['financial_health']['budget_utilization_percent']}%"],
            ['Cost Overrun:', f"{insights['financial_health']['cost_overrun_percent']}%"],
        ]
        financial_table = Table(financial_data, colWidths=[2*inch, 4*inch])
        financial_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6c757d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#183F32')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8E4D8')),
        ]))
        story.append(financial_table)
        story.append(Spacer(1, 20))
        
        # D. Schedule & Delay Assessment
        story.append(Paragraph("D. Schedule & Delay Assessment", heading_style))
        delay_data = [
            ['Current Delay:', f"{insights['delay_analysis']['current_delay_days']} days"],
            ['Delay Severity:', insights['delay_analysis']['delay_category']],
        ]
        delay_table = Table(delay_data, colWidths=[2*inch, 4*inch])
        delay_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6c757d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#183F32')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8E4D8')),
        ]))
        story.append(delay_table)
        story.append(Spacer(1, 20))
        
        # E. Key Risk Factors
        story.append(Paragraph("E. Key Risk Factors", heading_style))
        for factor in insights['risk_factors']:
            story.append(Paragraph(f"• {factor}", normal_style))
        story.append(Spacer(1, 20))
        
        # F. Progress vs Expenditure
        story.append(Paragraph("F. Progress vs Expenditure", heading_style))
        progress_data = [
            ['Physical Progress:', f"{insights['project_info']['physical_progress_percent']:.1f}%"],
            ['Budget Utilization:', f"{insights['financial_health']['budget_utilization_percent']}%"],
        ]
        progress_table = Table(progress_data, colWidths=[2*inch, 4*inch])
        progress_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6c757d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#183F32')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8E4D8')),
        ]))
        story.append(progress_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Analysis:</b> {insights['progress_expenditure_analysis']}", normal_style))
        story.append(Spacer(1, 20))
        
        # G. Category/District Comparison
        story.append(Paragraph("G. Category/District Comparison", heading_style))
        for comp in insights['category_comparison']:
            story.append(Paragraph(f"• {comp}", normal_style))
        story.append(Spacer(1, 20))
        
        # H. Similar Projects
        if insights['similar_projects'] and len(insights['similar_projects']) > 0:
            story.append(Paragraph("H. Similar Projects", heading_style))
            similar_data = [['Project ID', 'Project Name', 'Category', 'Similarity']]
            for p in insights['similar_projects']:
                similar_data.append([
                    p['project_id'],
                    p['project_name'],
                    p['sector'],
                    f"{p['similarity_score']*100:.1f}%"
                ])
            similar_table = Table(similar_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1*inch])
            similar_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8F9FA')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E8E4D8')),
            ]))
            story.append(similar_table)
            story.append(Spacer(1, 20))
        
        # I. AI Project Assessment
        story.append(Paragraph("I. AI Project Assessment", heading_style))
        story.append(Paragraph(f"<b>Overall Assessment:</b> {insights['risk_assessment']}", normal_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Recommended Action:</b> {insights['recommendation']}", normal_style))
        
        # Build PDF
        doc.build(story)
        
        # Return PDF
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'PAIMANA_Project_Report_{project_id}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        import traceback
        print(f"[API] download-insights-pdf ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Failed to generate PDF: {e}"}), 500


@app.route('/api/map-data')
def api_map_data():
    """Get geo-location data for map visualization derived from the State column.

    Uses a hardcoded Indian-state centroid lookup — no external geocoding,
    no extra data loading, no repeated expensive work.
    Projects are aggregated per state so we get at most ~35 markers.
    """
    # ── Hardcoded Indian state / UT centroids (lat, lon) ─────────────────────
    STATE_COORDS = {
        'andhra pradesh':       (15.9129,  79.7400),
        'arunachal pradesh':    (28.2180,  94.7278),
        'assam':                (26.2006,  92.9376),
        'bihar':                (25.0961,  85.3131),
        'chhattisgarh':         (21.2787,  81.8661),
        'goa':                  (15.2993,  74.1240),
        'gujarat':              (22.2587,  71.1924),
        'haryana':              (29.0588,  76.0856),
        'himachal pradesh':     (31.1048,  77.1734),
        'jharkhand':            (23.6102,  85.2799),
        'jammu and kashmir':    (33.7782,  76.5762),
        'jammu & kashmir':      (33.7782,  76.5762),
        'karnataka':            (15.3173,  75.7139),
        'kerala':               (10.8505,  76.2711),
        'ladakh':               (34.1526,  77.5770),
        'madhya pradesh':       (22.9734,  78.6569),
        'maharashtra':          (19.7515,  75.7139),
        'manipur':              (24.6637,  93.9063),
        'meghalaya':            (25.4670,  91.3662),
        'mizoram':              (23.1645,  92.9376),
        'nagaland':             (26.1584,  94.5624),
        'odisha':               (20.9517,  85.0985),
        'punjab':               (31.1471,  75.3412),
        'rajasthan':            (27.0238,  74.2179),
        'sikkim':               (27.5330,  88.5122),
        'tamil nadu':           (11.1271,  78.6569),
        'telangana':            (18.1124,  79.0193),
        'tripura':              (23.9408,  91.9882),
        'uttar pradesh':        (26.8467,  80.9462),
        'uttarakhand':          (30.0668,  79.0193),
        'west bengal':          (22.9868,  87.8550),
        # Union territories
        'andaman & nicobar':    (11.7401,  92.6586),
        'andaman and nicobar':  (11.7401,  92.6586),
        'chandigarh':           (30.7333,  76.7794),
        'dadra & nagar haveli and daman & diu': (20.1809, 73.0169),
        'dadra and nagar haveli and daman and diu': (20.1809, 73.0169),
        'delhi':                (28.6139,  77.2090),
        'lakshadweep':          (10.5667,  72.6417),
        'puducherry':           (11.9416,  79.8083),
    }
    # Fallback: geographic centre of India
    INDIA_CENTRE = (20.5937, 78.9629)

    data = load_or_generate_data()
    df = data['projects'].copy()

    # ── Check whether the dataset already has GPS columns ────────────────────
    EXACT_GEO = {'latitude', 'longitude', 'lat', 'lon', 'lng'}
    geo_cols = [
        col for col in df.columns
        if col.strip().lower() in EXACT_GEO
        or col.strip().lower().startswith('latitude')
        or col.strip().lower().startswith('longitude')
    ]
    lat_col = next((c for c in geo_cols if c.strip().lower().startswith('lat')), None)
    lon_col = next((c for c in geo_cols
                    if c.strip().lower() in {'longitude', 'lon', 'lng'}
                    or c.strip().lower().startswith('longitude')), None)

    if lat_col and lon_col:
        # Dataset has real GPS — use it directly (original code path)
        latitudes  = df[lat_col].tolist()
        longitudes = df[lon_col].tolist()
        labels, colors = [], []
        for _, row in df.iterrows():
            dm = float(row.get('Actual_Delay_Months', 0) or 0)
            pr = float(row.get('Physical Progress (%)', 0) or 0)
            lbl = (f"<b>{row.get('Project Name','Unknown')}</b><br>"
                   f"State: {row.get('State','N/A')}<br>"
                   f"Progress: {pr:.1f}%<br>Delay: {dm:.0f} months")
            labels.append(lbl)
            colors.append(0 if dm <= 0 else (50 if dm <= 6 else 100))
        return jsonify({'latitudes': latitudes, 'longitudes': longitudes,
                        'labels': labels, 'colors': colors})

    # ── Derive coordinates from the State column ──────────────────────────────
    state_col = next((c for c in df.columns if c.strip().lower() == 'state'), None)
    if not state_col:
        return jsonify({
            'error': 'No usable geographic field found',
            'latitudes': [], 'longitudes': [], 'labels': [], 'colors': [],
            'message': 'No State, latitude or longitude column found in the dataset.'
        })

    def _resolve(raw_state):
        """Return (lat, lon) for a state string, or None if unresolvable."""
        if not raw_state or str(raw_state).strip().lower() in ('nan', 'offshore', 'pan india'):
            return None
        cleaned = ' '.join(str(raw_state).replace('\n', ' ').split()).lower()
        # Direct match
        if cleaned in STATE_COORDS:
            return STATE_COORDS[cleaned]
        # Try each word segment for multi-state entries
        # e.g. "multi-states\n(Bihar, Uttar\nPradesh)" → use first named state
        import re as _re
        # Strip the "Multi-States (...)" wrapper and try states inside
        inner = _re.sub(r'multi-?states?\s*\(', '', cleaned, flags=_re.I)
        inner = inner.replace(')', '').strip()
        for part in _re.split(r'[,\n&]+', inner):
            part = ' '.join(part.split()).lower()
            if part in STATE_COORDS:
                return STATE_COORDS[part]
        # Partial match as last resort
        for key, coord in STATE_COORDS.items():
            if key in cleaned or cleaned in key:
                return coord
        return None

    # Aggregate: one marker per unique (state → coord) with project counts
    from collections import defaultdict
    agg = defaultdict(lambda: {'lat': None, 'lon': None, 'count': 0,
                                'delayed': 0, 'total_delay': 0.0, 'total_progress': 0.0})

    for _, row in df.iterrows():
        raw = str(row.get(state_col, '') or '')
        coord = _resolve(raw)
        if coord is None:
            continue
        key = raw.replace('\n', ' ').strip()[:60]   # short stable key
        entry = agg[key]
        entry['lat']    = coord[0]
        entry['lon']    = coord[1]
        entry['count'] += 1
        dm = float(row.get('Actual_Delay_Months', 0) or 0)
        entry['total_delay']    += dm
        entry['total_progress'] += float(row.get('Physical Progress (%)', 0) or 0)
        if dm > 0:
            entry['delayed'] += 1

    if not agg:
        return jsonify({
            'error': 'No mappable state data',
            'latitudes': [], 'longitudes': [], 'labels': [], 'colors': [],
            'message': 'State column exists but no entries could be mapped to coordinates.'
        })

    latitudes, longitudes, labels, colors = [], [], [], []
    for state_key, e in agg.items():
        n       = e['count']
        avg_dm  = e['total_delay']    / n
        avg_pr  = e['total_progress'] / n
        delayed = e['delayed']

        latitudes.append(e['lat'])
        longitudes.append(e['lon'])

        lbl = (f"<b>{state_key}</b><br>"
               f"Projects: {n}<br>"
               f"Delayed: {delayed} ({100*delayed//n}%)<br>"
               f"Avg Progress: {avg_pr:.1f}%<br>"
               f"Avg Delay: {avg_dm:.0f} months")
        labels.append(lbl)

        # Colour by fraction delayed
        frac = delayed / n if n else 0
        colors.append(0 if frac < 0.25 else (50 if frac < 0.60 else 100))

    return jsonify({
        'latitudes':  latitudes,
        'longitudes': longitudes,
        'labels':     labels,
        'colors':     colors
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 PAIMANA Intelligence Platform Dashboard")
    print("="*60)
    print("\nStarting server at http://localhost:5001")
    print("\nAvailable endpoints:")
    print("  - http://localhost:5001/          (Dashboard)")
    print("  - http://localhost:5001/api/projects    (CSV)")
    print("  - http://localhost:5001/api/analytics   (JSON)")
    print("  - http://localhost:5001/api/quality     (JSON)")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5001)
