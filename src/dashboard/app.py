"""
Simple Flask Dashboard for PAIMANA Intelligence Platform
Provides web interface for viewing project analytics and data
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import json
import pickle
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analytics.delay_detector import DelayAnalyzer
from src.audit.quality_checker import DataQualityAuditor
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)

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
        
        # ML predictions are already in df_reference (Risk_Level, Delay_Months, etc.)
        # Create predictions DataFrame from df_reference
        predictions_df = df.copy()
        
        # Map df_reference columns to expected prediction columns
        predictions_df["ML_Predicted_Delay_Days"] = predictions_df.get("Actual_Delay_Months", 0) * 30
        predictions_df["ML_Predicted_Cost_Overrun_%"] = predictions_df.get("Cost_Overrun_Ratio", 0) * 100
        predictions_df["ML_Risk_Level"] = predictions_df.get("Risk_Level", "LOW")
        predictions_df["ML_Risk_Confidence_%"] = 75.0  # Default confidence
        
        # Keep old column names for dashboard compatibility
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


@app.route('/')
def index():
    """Home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PAIMANA Intelligence Platform</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f5f5;
                min-height: 100vh;
                display: flex;
            }
            
            /* Sidebar Styles */
            .sidebar {
                position: fixed;
                left: 0;
                top: 0;
                width: 240px;
                height: 100vh;
                background: #ffffff;
                border-right: 1px solid #e9ecef;
                display: flex;
                flex-direction: column;
                z-index: 1000;
                box-shadow: 2px 0 8px rgba(0,0,0,0.05);
            }
            
            .sidebar-header {
                padding: 24px;
                border-bottom: 1px solid #e9ecef;
            }
            
            .sidebar-title {
                font-size: 24px;
                font-weight: 700;
                color: #1a365d;
                margin-bottom: 4px;
                letter-spacing: 1px;
            }
            
            .sidebar-subtitle {
                font-size: 11px;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 2px;
                font-weight: 600;
            }
            
            .sidebar-nav {
                flex: 1;
                padding: 16px 0;
                overflow-y: auto;
            }
            
            .nav-item {
                display: flex;
                align-items: center;
                padding: 12px 24px;
                color: #4a5568;
                text-decoration: none;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s ease;
                border-left: 3px solid transparent;
                cursor: pointer;
            }
            
            .nav-item:hover {
                background: #f7fafc;
                color: #1a365d;
            }
            
            .nav-item.active {
                background: #f0fdf4;
                color: #1B6B3A;
                border-left-color: #1B6B3A;
            }
            
            .nav-item i {
                margin-right: 12px;
                width: 20px;
                height: 20px;
            }
            
            .sidebar-footer {
                padding: 16px 24px;
                border-top: 1px solid #e9ecef;
                font-size: 12px;
                color: #a0aec0;
                text-align: center;
            }
            
            /* Mobile menu button */
            .mobile-menu-btn {
                display: none;
                position: fixed;
                top: 16px;
                left: 16px;
                z-index: 1001;
                background: #1B6B3A;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
            
            /* Main content area */
            .main-content {
                margin-left: 240px;
                flex: 1;
                padding: 20px;
                background: #e9ecef;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                background: #ffffff;
                padding: 24px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 24px;
                border: 1px solid #dee2e6;
            }
            h1 {
                color: #1a365d;
                font-size: 1.75em;
                margin-bottom: 8px;
                font-weight: 600;
            }
            .subtitle {
                color: #6c757d;
                font-size: 1em;
            }
            .nav {
                display: none;
            }
            .alert {
                background: #d4edda;
                padding: 16px;
                border-radius: 6px;
                color: #155724;
                margin-bottom: 24px;
                border: 1px solid #c3e6cb;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .stat-card {
                background: #ffffff;
                padding: 20px;
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #dee2e6;
                transition: transform 0.2s;
            }
            .stat-card:hover {
                transform: translateY(-2px);
            }
            .stat-value {
                font-size: 2em;
                font-weight: 600;
                color: #1B6B3A;
                margin-bottom: 4px;
            }
            .stat-label {
                color: #6c757d;
                font-size: 0.85em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .card {
                background: #ffffff;
                padding: 24px;
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 24px;
                border: 1px solid #dee2e6;
            }
            h2 {
                color: #1a365d;
                margin-bottom: 16px;
                font-size: 1.5em;
                font-weight: 600;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 16px 0;
            }
            th {
                background: #1B6B3A;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                font-size: 0.9em;
            }
            td {
                padding: 12px;
                border-bottom: 1px solid #dee2e6;
                font-size: 0.9em;
            }
            tr:hover {
                background: #f8f9fa;
            }
            #map {
                height: 500px;
                margin: 16px 0;
                border-radius: 6px;
                overflow: hidden;
                border: 1px solid #dee2e6;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #7f8c8d;
                font-size: 1.2em;
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
                background: #ffffff;
                padding: 16px;
                border-radius: 6px;
                margin-bottom: 20px;
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                align-items: center;
                border: 1px solid #dee2e6;
            }
            .filter-bar select, .filter-bar input {
                padding: 8px 12px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                min-width: 150px;
            }
            .filter-bar select:focus, .filter-bar input:focus {
                outline: none;
                border-color: #1B6B3A;
                box-shadow: 0 0 0 2px rgba(27, 107, 58, 0.1);
            }
            .filter-bar button {
                padding: 8px 16px;
                background: #1B6B3A;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                font-size: 14px;
                transition: background-color 0.2s;
            }
            .filter-bar button:hover {
                background: #145a32;
            }
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 1000;
                justify-content: center;
                align-items: center;
            }
            .modal.active {
                display: flex;
            }
            .modal-content {
                background: white;
                padding: 24px;
                border-radius: 6px;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
                border: 1px solid #dee2e6;
            }
            .modal-close {
                position: absolute;
                top: 12px;
                right: 16px;
                font-size: 20px;
                cursor: pointer;
                color: #6c757d;
            }
            .modal-close:hover {
                color: #dc3545;
            }
            .project-detail {
                margin: 12px 0;
            }
            .project-detail label {
                font-weight: 600;
                color: #1a365d;
                display: block;
                margin-bottom: 4px;
                font-size: 0.9em;
            }
            .project-detail span {
                color: #6c757d;
                font-size: 0.95em;
            }
            .status-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }
            .status-on-time {
                background: #d4edda;
                color: #155724;
            }
            .status-delayed {
                background: #f8d7da;
                color: #721c24;
            }
            .status-critical {
                background: #f5c6cb;
                color: #721c24;
            }
            tr.clickable {
                cursor: pointer;
            }
            tr.clickable:hover {
                background: #e3f2fd !important;
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
                    padding: 20px;
                }
                
                .main-content.shifted {
                    margin-left: 0;
                }
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 0 10px;
                }
                
                .header {
                    padding: 20px;
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
                <div class="nav-item" onclick="navigateTo('insights')" id="nav-insights">
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
                <h1>PAIMANA Intelligence Platform</h1>
                <p class="subtitle">Infrastructure Transparency Through AI</p>
                
                <div class="nav">
                    <a href="/">Dashboard</a>
                    <a href="/api/projects">Download CSV</a>
                    <a href="/api/projects-excel">Download Excel</a>
                    <a href="/api/analytics">Analytics API</a>
                    <a href="/api/quality">Quality Report</a>
                </div>
            </div>
            
            <div class="alert">
                <strong>Live Demo Active</strong><br>
                This dashboard demonstrates real-time infrastructure project monitoring with AI-powered analytics and geo-visualization.
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
                    <div class="loading">Similar projects feature coming soon</div>
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
                <p style="color: #7f8c8d; margin-bottom: 20px;">AI-powered insights and recommendations based on project data analysis.</p>
                <div id="ai-insights">
                    <div class="loading">AI insights feature coming soon</div>
                </div>
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
            function fetchWithTimeout(url, timeoutMs) {
                const ctrl = new AbortController();
                const id = setTimeout(() => ctrl.abort(), timeoutMs);
                return fetch(url, { signal: ctrl.signal })
                    .finally(() => clearTimeout(id));
            }

            // Fetch and display Quick Statistics + Delayed Projects + Cost Overruns
            fetchWithTimeout('/api/analytics', 30000)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                    return r.json();
                })
                .then(data => {
                    console.log('Analytics data received:', data);
                    if (!data || !data.summary_statistics) throw new Error('Invalid analytics data structure');

                    const stats = data.summary_statistics;
                    document.getElementById('stats').innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${stats.total_projects}</div>
                            <div class="stat-label">Total Projects</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.delayed_projects}</div>
                            <div class="stat-label">Delayed</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.on_time_projects}</div>
                            <div class="stat-label">On Time</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.average_delay_days.toFixed(0)}</div>
                            <div class="stat-label">Avg Delay (days)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.average_progress_percent.toFixed(1)}%</div>
                            <div class="stat-label">Avg Progress</div>
                        </div>
                    `;
                    
                    // Delayed projects table
                    allProjects = data.top_delayed_projects || [];
                    filteredProjects = [...allProjects];
                    populateFilters();
                    
                    if (data.top_delayed_projects && data.top_delayed_projects.length > 0) {
                        renderDelayedProjects(data.top_delayed_projects);
                    }
                    
                    // Cost overruns — costs already in Crore from CSV; display directly
                    if (data.top_cost_overruns && data.top_cost_overruns.length > 0) {
                        let table = '<table><tr><th>Project ID</th><th>Name</th><th>Overrun %</th><th>Sanctioned (₹ Cr)</th><th>Spent (₹ Cr)</th></tr>';
                        data.top_cost_overruns.slice(0, 10).forEach(p => {
                            const sanctioned = (p.sanctioned_cost / 10000000).toFixed(2);
                            const spent = (p.expenditure_to_date / 10000000).toFixed(2);
                            table += `<tr>
                                <td><strong>${p.project_id}</strong></td>
                                <td>${p.project_name}</td>
                                <td style="color: #dc3545; font-weight: 600;">${p.cost_overrun_percent.toFixed(1)}%</td>
                                <td>₹${sanctioned} Cr</td>
                                <td>₹${spent} Cr</td>
                            </tr>`;
                        });
                        table += '</table>';
                        document.getElementById('overruns').innerHTML = table;
                    } else {
                        document.getElementById('overruns').innerHTML = '<p style="color:#6c757d;padding:20px;">No significant cost overruns detected.</p>';
                    }
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('Error loading analytics:', err);
                    document.getElementById('stats').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load statistics: ${msg}. Please refresh the page.</p>`;
                    document.getElementById('delayed').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load delayed projects: ${msg}</p>`;
                    document.getElementById('overruns').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load cost overruns: ${msg}</p>`;
                });

            
            // Fetch ML predictions
            fetchWithTimeout('/api/ml-predictions', 30000)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                    return r.json();
                })
                .then(predictions => {
                    console.log('ML predictions received:', predictions);
                    
                    if (predictions && predictions.length > 0) {
                        let table = '<table><tr><th>Project ID</th><th>Name</th><th>District</th><th>Progress</th><th>Predicted Delay</th><th>Est. Completion</th></tr>';
                        predictions.forEach(p => {
                            const completion = p.predicted_completion_date ? p.predicted_completion_date.split('T')[0] : 'N/A';
                            table += `<tr>
                                <td><strong>${p.project_id}</strong></td>
                                <td>${p.project_name}</td>
                                <td>${p.district}</td>
                                <td>${p.physical_progress_percent.toFixed(1)}%</td>
                                <td style="color: #dc3545; font-weight: 600;">${p.predicted_delay_days} days</td>
                                <td>${completion}</td>
                            </tr>`;
                        });
                        table += '</table>';
                        document.getElementById('ml-predictions').innerHTML = table;
                    } else {
                        document.getElementById('ml-predictions').innerHTML = '<p style="color: #6c757d; padding: 20px;">No ML predictions available</p>';
                    }
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('Error loading ML predictions:', err);
                    document.getElementById('ml-predictions').innerHTML = `<p style="color:#dc3545;padding:20px;">⚠ Failed to load ML predictions: ${msg}</p>`;
                });

            
            // Fetch and render category chart
            fetchWithTimeout('/api/category-chart', 30000)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                    return r.json();
                })
                .then(figJson => {
                    console.log('Category chart data received');
                    const fig = JSON.parse(figJson);
                    Plotly.newPlot('category-chart', fig.data, fig.layout, {responsive: true});
                })
                .catch(err => {
                    const msg = err.name === 'AbortError' ? 'Request timed out (30 s)' : err.message;
                    console.error('Error loading category chart:', err);
                    document.getElementById('category-chart').innerHTML = `<p style="text-align:center;padding:40px;color:#dc3545;">⚠ Failed to load category chart: ${msg}</p>`;
                });

            
            // Fetch quality data
            fetchWithTimeout('/api/quality', 30000)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                    return r.json();
                })
                .then(data => {
                    console.log('Quality data received:', data);
                    document.getElementById('quality').innerHTML = `
                        <div class="grid">
                            <div class="stat-card">
                                <div class="stat-value">${data.overall_grade}</div>
                                <div class="stat-label">Quality Grade</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.reliability_score}</div>
                                <div class="stat-label">Reliability</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.valid_records}/${data.total_records}</div>
                                <div class="stat-label">Valid Records</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.anomalies.total}</div>
                                <div class="stat-label">Anomalies</div>
                            </div>
                        </div>
                        <p style="padding: 12px; background: #e8f5e9; border-left: 4px solid #27ae60; border-radius: 4px; margin-top: 16px; font-size: 0.9em;">
                            <strong>${data.recommendation}</strong>
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
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                    return r.json();
                })
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
                let table = '<table><tr><th>Project ID</th><th>Name</th><th>District</th><th>Delay (days)</th><th>Progress</th></tr>';
                projects.forEach(p => {
                    const statusClass = p.delay_days > 100 ? 'status-critical' : (p.delay_days > 0 ? 'status-delayed' : 'status-on-time');
                    const statusText = p.delay_days > 100 ? 'Critical' : (p.delay_days > 0 ? 'Delayed' : 'On Time');
                    table += `<tr class="clickable" onclick="showProjectDetails('${p.project_id}')">
                        <td><strong>${p.project_id}</strong></td>
                        <td>${p.project_name}</td>
                        <td>${p.district}</td>
                        <td><span class="status-badge ${statusClass}">${statusText}</span> ${p.delay_days}</td>
                        <td><div style="background: #ecf0f1; border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #00b4db, #0083b0); width: ${p.physical_progress_percent}%; height: 100%;"></div>
                        </div>${p.physical_progress_percent.toFixed(1)}%</td>
                    </tr>`;
                });
                table += '</table>';
                document.getElementById('delayed').innerHTML = table;
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
// PAIMANA OFFLINE INTELLIGENCE ASSISTANT
// No external API / No API key
// ============================================================

function sendMessage() {

    const input = document.getElementById("chat-input");
    const messages = document.getElementById("chat-messages");

    const question = input.value.trim();

    if (!question) return;

    // Show user message
    messages.innerHTML += `
        <div style="
            background:#1B6B3A;
            color:white;
            padding:12px;
            border-radius:10px;
            margin-bottom:10px;
            margin-left:20%;
        ">
            <strong>You</strong><br>
            ${escapeHTML(question)}
        </div>
    `;

    input.value = "";

    // Generate local answer
    const answer = getPAIMANAAnswer(question);

    // Show assistant answer
    messages.innerHTML += `
        <div style="
            background:#e8f5e9;
            padding:12px;
            border-radius:10px;
            margin-bottom:10px;
            margin-right:20%;
            white-space:pre-line;
        ">
            <strong>🤖 PAIMANA Assistant</strong><br>
            ${answer}
        </div>
    `;

    messages.scrollTop = messages.scrollHeight;
}


function escapeHTML(text) {

    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function getPAIMANAAnswer(question) {

    const q = question.toLowerCase();

    // Greeting
    if (
        q.includes("hello") ||
        q.includes("hi") ||
        q.includes("hey") ||
        q.includes("namaste")
    ) {
        return "Hello! 👋 I am the PAIMANA Intelligence Assistant. How can I help you?";
    }


    // What is PAIMANA
    if (
        q.includes("what is paimana") ||
        q.includes("about paimana") ||
        q.includes("paimana platform")
    ) {
        return "PAIMANA is an infrastructure intelligence platform that combines historical project data, analytics, GIS and machine learning to predict potential cost overruns, delays and project risks.";
    }


    // Problem
    if (
        q.includes("problem") ||
        q.includes("why paimana") ||
        q.includes("why is paimana needed")
    ) {
        return "Infrastructure projects commonly face cost overruns, delays, unrealistic budgets and location-specific risks. PAIMANA aims to identify these risks earlier using historical data and predictive analytics.";
    }


    // Methodology
    if (
        q.includes("methodology") ||
        q.includes("how does it work") ||
        q.includes("how it works") ||
        q.includes("workflow")
    ) {
        return "PAIMANA follows this workflow:\\n\\n1. Historical project data\\n2. Data cleaning and quality audit\\n3. GIS and location intelligence\\n4. Similar project analysis\\n5. Machine learning prediction\\n6. Cost, delay and risk assessment\\n7. Actionable insights";
    }


    // Delay
    if (
        q.includes("delay") ||
        q.includes("delayed") ||
        q.includes("late project")
    ) {
        return "PAIMANA analyses physical progress and expenditure patterns to identify projects with potential schedule problems. The trained ML system also estimates potential delay duration.";
    }


    // Cost
    if (
        q.includes("cost") ||
        q.includes("cost overrun") ||
        q.includes("budget") ||
        q.includes("expenditure")
    ) {
        return "The platform analyses project cost, expenditure and progress to identify potential cost overruns. Its trained ML model predicts cost-overrun behaviour using project characteristics and financial indicators.";
    }


    // Risk
    if (
        q.includes("risk") ||
        q.includes("high risk") ||
        q.includes("risk prediction")
    ) {
        return "PAIMANA's risk model evaluates project characteristics, progress and expenditure-related features and classifies the project's predicted risk level. This helps decision-makers prioritise projects requiring attention.";
    }


    // Machine learning
    if (
        q.includes("machine learning") ||
        q.includes("ml model") ||
        q === "ml" ||
        q.includes("algorithm")
    ) {
        return "PAIMANA uses trained machine learning models for cost-overrun prediction, delay prediction and project-risk classification. The prediction pipeline also uses categorical encoding and feature scaling.";
    }


    // Features
    if (
        q.includes("features") ||
        q.includes("input") ||
        q.includes("data used") ||
        q.includes("what data")
    ) {
        return "The predictive models use features such as sector, state, ministry, original project cost, approval year, physical progress, expenditure ratio, cost per progress, project size and budget utilisation.";
    }


    // GIS
    if (
        q.includes("gis") ||
        q.includes("location") ||
        q.includes("geographical")
    ) {
        return "GIS provides location intelligence. PAIMANA can use regional patterns, historical project outcomes and geographical characteristics to understand how location can influence cost, delay and project risk.";
    }


    // Similar projects
    if (
        q.includes("similar project") ||
        q.includes("historical project") ||
        q.includes("past project")
    ) {
        return "PAIMANA can compare a project with historical projects using factors such as category, location, budget, scale and historical performance. These comparisons provide evidence for future project predictions.";
    }


    // Benefits
    if (
        q.includes("benefit") ||
        q.includes("advantage") ||
        q.includes("useful")
    ) {
        return "PAIMANA can help stakeholders identify delayed projects, monitor expenditure, detect potential cost overruns, understand regional patterns and prioritise high-risk infrastructure projects.";
    }


    // Technology
    if (
        q.includes("technology") ||
        q.includes("tech stack") ||
        q.includes("built with")
    ) {
        return "The platform is built using Python, Flask, Pandas, Scikit-learn and interactive data-visualisation technologies, with trained ML models integrated directly into the application.";
    }


    // SIH
    if (
        q.includes("sih") ||
        q.includes("hackathon")
    ) {
        return "For the SIH prototype, PAIMANA demonstrates an end-to-end infrastructure intelligence workflow: project monitoring, data analytics, machine learning prediction, risk assessment, GIS visualisation and decision support.";
    }


    // No API question
    if (
        q.includes("api") ||
        q.includes("external")
    ) {
        return "The PAIMANA Intelligence Assistant operates locally in the application and does not depend on an external generative-AI API.";
    }


    // Help
    if (
        q.includes("help") ||
        q.includes("what can you do")
    ) {
        return "You can ask me:\\n\\n• What is PAIMANA?\\n• How does risk prediction work?\\n• How are delays predicted?\\n• How is cost overrun predicted?\\n• What data is used?\\n• What is the methodology?\\n• How does GIS help?\\n• What ML models are used?";
    }


    return "I can help with PAIMANA, project delays, cost overruns, risk prediction, GIS, machine learning, methodology and project data. Try asking: 'How does risk prediction work?'";
}
        </script>
    </body>
    </html>
    """


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
    data = load_or_generate_data()
    # Convert to JSON string and back to handle numpy/pandas types
    import json
    analytics_json = json.dumps(data['analytics'], default=str)
    return analytics_json, 200, {'Content-Type': 'application/json'}


@app.route('/api/quality')
def api_quality():
    """Get quality report"""
    data = load_or_generate_data()
    return jsonify(data['quality_report'])


@app.route('/api/delayed')
def api_delayed():
    """Get delayed projects only — returns the columns the frontend table expects"""
    data = load_or_generate_data()
    df = data['projects'].copy()
    # Filter to actually delayed rows
    delayed = df[df['Actual_Delay_Months'] > 0].copy()
    # Shape to what the frontend renders
    result = []
    for _, row in delayed.iterrows():
        result.append({
            'project_id': str(row.get('Project Code', 'N/A')),
            'project_name': str(row.get('Project Name', 'Unknown')),
            'district': str(row.get('State', 'N/A')),
            'delay_days': float(row.get('delay_days', row.get('Actual_Delay_Months', 0) * 30)),
            'physical_progress_percent': float(row.get('Physical Progress (%)', 0)),
            'category': str(row.get('Sector', 'N/A')),
            'risk_level': str(row.get('Risk_Level', 'N/A')),
        })
    import json as _json
    return _json.dumps(result, default=str), 200, {'Content-Type': 'application/json'}


@app.route('/api/ml-predictions')
def api_ml_predictions():
    """Get predictions from the trained PAIMANA ML models."""

    data = load_or_generate_data()

    # Get predictions from df_reference (already contains ML predictions)
    predictions_df = data['predictions'].copy()

    # Sort by predicted delay
    predictions_df = predictions_df.sort_values(
        'predicted_delay_days',
        ascending=False
    ).head(10)

    # Return data required by the website
    result = []

    for _, row in predictions_df.iterrows():
        result.append({
            'project_id': row.get('Project Code', 'N/A'),
            'project_name': row.get('Project Name', 'Unknown'),
            'district': row.get('State', 'N/A'),
            'physical_progress_percent': float(
                row.get('Physical Progress (%)', 0)
            ),
            'predicted_delay_days': int(
                row.get('predicted_delay_days', 0)
            ),
            'cost_overrun_percent': round(
                float(row.get('ML_Predicted_Cost_Overrun_%', 0)),
                2
            ),
            'risk_level': str(
                row.get('ML_Risk_Level', 'UNKNOWN')
            ),
            'risk_confidence': float(
                row.get('ML_Risk_Confidence_%', 0)
            )
        })

    return jsonify(result)


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
    
    # Create Plotly bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Delayed Projects',
        x=categories,
        y=delayed_counts,
        marker_color='#e74c3c'
    ))
    
    fig.add_trace(go.Bar(
        name='On-Time Projects',
        x=categories,
        y=[total - delayed for total, delayed in zip(total_counts, delayed_counts)],
        marker_color='#27ae60'
    ))
    
    fig.update_layout(
        title='Project Status by Category',
        xaxis_title='Category',
        yaxis_title='Number of Projects',
        barmode='stack',
        template='plotly_white',
        height=400
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


@app.route('/api/map-data')
def api_map_data():
    """Get geo-location data for map visualization"""
    data = load_or_generate_data()
    df = data['projects'].copy()

    # Strict geo-column detection: match ONLY columns whose name IS (or starts with)
    # 'lat' / 'latitude' / 'lon' / 'longitude'.  This avoids false positives like
    # 'Cumulative Expenditure (Rs. Crore)' which contains 'lon' as a substring.
    EXACT_GEO = {'latitude', 'longitude', 'lat', 'lon', 'lng'}
    geo_cols = [
        col for col in df.columns
        if col.strip().lower() in EXACT_GEO
        or col.strip().lower().startswith('latitude')
        or col.strip().lower().startswith('longitude')
    ]

    lat_col = next((col for col in geo_cols if col.strip().lower().startswith('lat')), None)
    lon_col = next((col for col in geo_cols if col.strip().lower() in {'longitude', 'lon', 'lng'}
                    or col.strip().lower().startswith('longitude')), None)

    if not lat_col or not lon_col:
        # Dataset has no GPS coordinates — return a clean informational response
        return jsonify({
            'error': 'Geographic coordinates not available in dataset',
            'latitudes': [],
            'longitudes': [],
            'labels': [],
            'colors': [],
            'message': 'The project dataset does not contain latitude/longitude coordinates. '
                       'Map visualization requires geographic data.'
        })

    latitudes = df[lat_col].tolist()
    longitudes = df[lon_col].tolist()

    labels = []
    colors = []

    for _, project in df.iterrows():
        delay_months = project.get('Actual_Delay_Months', 0)
        progress = project.get('Physical Progress (%)', 0)
        project_name = project.get('Project Name', 'Unknown')
        state = project.get('State', 'N/A')

        label = f"<b>{project_name}</b><br>"
        label += f"State: {state}<br>"
        label += f"Progress: {progress:.1f}%<br>"
        label += f"Delay: {delay_months} months"
        labels.append(label)

        if delay_months <= 0:
            colors.append(0)   # Green - on time
        elif delay_months <= 6:
            colors.append(50)  # Orange - moderate delay
        else:
            colors.append(100) # Red - critical delay

    return jsonify({
        'latitudes': latitudes,
        'longitudes': longitudes,
        'labels': labels,
        'colors': colors
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
