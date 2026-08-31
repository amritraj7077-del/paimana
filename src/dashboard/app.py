"""
Simple Flask Dashboard for PAIMANA Intelligence Platform
Provides web interface for viewing project analytics and data
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.scrapers.paimana_scraper import PAIMANAScraper
from src.analytics.delay_detector import DelayAnalyzer
from src.audit.quality_checker import DataQualityAuditor
from src.analytics.ml_predictor import DelayPredictor
from src.analytics.project_ml import predict_projects
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)

# Global data cache
data_cache = {
    'projects': None,
    'analytics': None,
    'quality_report': None
}


def load_or_generate_data():
    """Load existing data or generate sample data"""
    if data_cache['projects'] is None:
        # For MVP, generate sample data
        scraper = PAIMANAScraper(state='maharashtra')
        df = scraper.extract_projects()
        
        # Run analytics
        analyzer = DelayAnalyzer()
        df = analyzer.calculate_delay_days(df)
        df = analyzer.calculate_cost_overrun(df)
        analytics_report = analyzer.generate_analytics_report(df)
        
        # Run quality audit
        auditor = DataQualityAuditor()
        quality_report = auditor.audit(df)
        quality_summary = auditor.generate_audit_summary(quality_report)
        
        # Run trained ML models
        predictions_df = predict_projects(df)

        # Keep old column names for dashboard compatibility
        predictions_df["predicted_delay_days"] = (
            predictions_df["ML_Predicted_Delay_Days"]
        )

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
            
            // Fetch and display data
            fetch('/api/analytics')
                .then(r => r.json())
                .then(data => {
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
                    allProjects = data.top_delayed_projects;
                    filteredProjects = [...allProjects];
                    populateFilters();
                    
                    if (data.top_delayed_projects.length > 0) {
                        renderDelayedProjects(data.top_delayed_projects);
                    }
                    
                    // Cost overruns
                    if (data.top_cost_overruns.length > 0) {
                        let table = '<table><tr><th>Project ID</th><th>Name</th><th>Overrun %</th><th>Sanctioned</th><th>Spent</th></tr>';
                        data.top_cost_overruns.slice(0, 10).forEach(p => {
                            table += `<tr>
                                <td><strong>${p.project_id}</strong></td>
                                <td>${p.project_name}</td>
                                <td style="color: #dc3545; font-weight: 600;">${p.cost_overrun_percent.toFixed(1)}%</td>
                                <td>₹${(p.sanctioned_cost/10000000).toFixed(2)} Cr</td>
                                <td>₹${(p.expenditure_to_date/10000000).toFixed(2)} Cr</td>
                            </tr>`;
                        });
                        table += '</table>';
                        document.getElementById('overruns').innerHTML = table;
                    }
                })

;
            
            // Fetch ML predictions
            fetch('/api/ml-predictions')
                .then(r => r.json())
                .then(predictions => {
                    if (predictions.length > 0) {
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
                    }
                })
                .catch(err => {
                    document.getElementById('ml-predictions').innerHTML = '<p style="color: #6c757d;">ML predictions temporarily unavailable</p>';
                });
            
            // Fetch and render category chart
            fetch('/api/category-chart')
                .then(r => r.json())
                .then(figJson => {
                    const fig = JSON.parse(figJson);
                    Plotly.newPlot('category-chart', fig.data, fig.layout, {responsive: true});
                })
                .catch(err => {
                    document.getElementById('category-chart').innerHTML = '<p style="text-align: center; padding: 40px; color: #6c757d;">Category chart temporarily unavailable</p>';
                });
            
            // Fetch quality data
            fetch('/api/quality')
                .then(r => r.json())
                .then(data => {
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
                });
            
            // Fetch map data
            fetch('/api/map-data')
                .then(r => r.json())
                .then(data => {
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
                    document.getElementById('map').innerHTML = '<p style="text-align: center; padding: 40px; color: #6c757d;">Map visualization temporarily unavailable</p>';
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
    """Get delayed projects only"""
    data = load_or_generate_data()
    delayed = data['projects'][data['projects']['delay_days'] > 0]
    return delayed.to_json(orient='records')


@app.route('/api/ml-predictions')
def api_ml_predictions():
    """Get predictions from the trained PAIMANA ML models."""

    data = load_or_generate_data()

    # Get the projects currently displayed by the dashboard
    df = data['projects'].copy()

    # Convert dashboard data into the format expected by the trained model
    df['Sector'] = df.get('category', 'Construction')
    df['State'] = df.get('state', 'Maharashtra').astype(str).str.title()
    df['Ministry'] = 'Ministry of Housing & Urban Affairs'

    # Convert rupees to crore
    df['Original Cost (Rs. Crore)'] = (
        pd.to_numeric(df.get('sanctioned_cost', 0), errors='coerce')
        .fillna(0) / 10000000
    )

    df['Cumulative Expenditure (Rs. Crore)'] = (
        pd.to_numeric(df.get('expenditure_to_date', 0), errors='coerce')
        .fillna(0) / 10000000
    )

    df['Physical Progress (%)'] = (
        pd.to_numeric(df.get('physical_progress_percent', 0), errors='coerce')
        .fillna(0)
    )

    # Demo projects do not have approval years,
    # so use the current project-era year.
    df['Approval_Year'] = 2024

    # Run the ACTUAL trained ML models
    predictions = predict_projects(df)

    # Sort by predicted delay
    predictions = predictions.sort_values(
        'ML_Predicted_Delay_Days',
        ascending=False
    ).head(10)

    # Return data required by the website
    result = []

    for _, row in predictions.iterrows():

        result.append({
            'project_id': row.get('project_id', 'N/A'),
            'project_name': row.get('project_name', 'Unknown'),
            'district': row.get('district', 'N/A'),
            'physical_progress_percent': float(
                row.get('physical_progress_percent', 0)
            ),
            'predicted_delay_days': int(
                row.get('ML_Predicted_Delay_Days', 0)
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
    
    # For demo: use approximate coordinates for Maharashtra districts
    district_coords = {
        'Mumbai': (19.0760, 72.8777),
        'Pune': (18.5204, 73.8567),
        'Nagpur': (21.1458, 79.0882),
        'Nashik': (19.9975, 73.7898),
        'Aurangabad': (19.8762, 75.3433),
        'Solapur': (17.6599, 75.9064),
        'Kolhapur': (16.7050, 74.2433)
    }
    
    latitudes = []
    longitudes = []
    labels = []
    colors = []
    
    for _, project in df.iterrows():
        district = project.get('district', 'Mumbai')
        coords = district_coords.get(district, (19.5, 75.5))  # Default to Maharashtra center
        
        # Add small random variation to avoid overlapping markers
        import random
        lat = coords[0] + random.uniform(-0.1, 0.1)
        lon = coords[1] + random.uniform(-0.1, 0.1)
        
        latitudes.append(lat)
        longitudes.append(lon)
        
        # Create hover label
        delay_days = project.get('delay_days', 0)
        progress = project.get('physical_progress_percent', 0)
        label = f"<b>{project.get('project_name', 'Unknown')}</b><br>"
        label += f"District: {district}<br>"
        label += f"Progress: {progress:.1f}%<br>"
        label += f"Delay: {delay_days} days"
        labels.append(label)
        
        # Color based on delay status
        if delay_days <= 0:
            colors.append(0)  # Green - on time
        elif delay_days <= 100:
            colors.append(50)  # Orange - moderate delay
        else:
            colors.append(100)  # Red - critical delay
    
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
