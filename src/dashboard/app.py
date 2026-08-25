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
        
        # Train ML predictor
        predictor = DelayPredictor()
        predictor.train(df)
        predictions_df = predictor.predict_completion_date(df)
        
        data_cache['projects'] = df
        data_cache['analytics'] = analytics_report
        data_cache['quality_report'] = quality_summary
        data_cache['predictor'] = predictor
        data_cache['predictions'] = predictions_df
    
    return data_cache


@app.route('/')
def index():
    """Home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PAIMANA Intelligence Platform</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
            }
            h1 {
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .subtitle {
                color: #7f8c8d;
                font-size: 1.1em;
            }
            .nav {
                display: flex;
                gap: 15px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .nav a {
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 25px;
                transition: transform 0.2s, box-shadow 0.2s;
                font-weight: 600;
            }
            .nav a:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .alert {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin-bottom: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: rgba(255, 255, 255, 0.95);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                transition: transform 0.3s;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 5px;
            }
            .stat-label {
                color: #7f8c8d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .card {
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
            }
            h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.8em;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }
            td {
                padding: 15px;
                border-bottom: 1px solid #ecf0f1;
            }
            tr:hover {
                background: #f8f9fa;
            }
            #map {
                height: 600px;
                margin: 20px 0;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                align-items: center;
            }
            .filter-bar select, .filter-bar input {
                padding: 10px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                min-width: 150px;
            }
            .filter-bar select:focus, .filter-bar input:focus {
                outline: none;
                border-color: #00b4db;
            }
            .filter-bar button {
                padding: 10px 20px;
                background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.2s;
            }
            .filter-bar button:hover {
                transform: translateY(-2px);
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
                padding: 30px;
                border-radius: 15px;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
            }
            .modal-close {
                position: absolute;
                top: 15px;
                right: 20px;
                font-size: 24px;
                cursor: pointer;
                color: #7f8c8d;
            }
            .modal-close:hover {
                color: #e74c3c;
            }
            .project-detail {
                margin: 15px 0;
            }
            .project-detail label {
                font-weight: 600;
                color: #2c3e50;
                display: block;
                margin-bottom: 5px;
            }
            .project-detail span {
                color: #7f8c8d;
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏗️ PAIMANA Intelligence Platform</h1>
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
                <strong>✅ Live Demo Active</strong><br>
                This dashboard demonstrates real-time infrastructure project monitoring with AI-powered analytics and geo-visualization.
            </div>
            
            <div class="filter-bar">
                <input type="text" id="searchInput" placeholder="🔍 Search projects...">
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
            
            <h2 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📊 Quick Statistics</h2>
            <div class="grid" id="stats">
                <div class="loading">Loading stats</div>
            </div>
            
            <div class="card">
                <h2>🗺️ Project Location Map</h2>
                <div id="map"></div>
            </div>
            
            <div class="card">
                <h2>🤖 AI Delay Predictions (ML Model)</h2>
                <p style="color: #7f8c8d; margin-bottom: 20px;">Machine learning predictions of project completion delays based on current progress and expenditure patterns.</p>
                <div id="ml-predictions">
                    <div class="loading">Loading ML predictions</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Category Performance Comparison</h2>
                <div id="category-chart"></div>
            </div>
            
            <div class="card">
                <h2>⚠️ Top Delayed Projects</h2>
                <div id="delayed">
                    <div class="loading">Loading delayed projects</div>
                </div>
            </div>
            
            <div class="card">
                <h2>💰 Cost Overruns</h2>
                <div id="overruns">
                    <div class="loading">Loading cost overruns</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📋 Data Quality Assessment</h2>
                <div id="quality">
                    <div class="loading">Loading quality report</div>
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
            let allProjects = [];
            let filteredProjects = [];
            
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
                                <td style="color: #e67e22; font-weight: bold;">${p.cost_overrun_percent.toFixed(1)}%</td>
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
                                <td style="color: #e74c3c; font-weight: bold;">${p.predicted_delay_days} days</td>
                                <td>${completion}</td>
                            </tr>`;
                        });
                        table += '</table>';
                        document.getElementById('ml-predictions').innerHTML = table;
                    }
                })
                .catch(err => {
                    document.getElementById('ml-predictions').innerHTML = '<p style="color: #7f8c8d;">ML predictions temporarily unavailable</p>';
                });
            
            // Fetch and render category chart
            fetch('/api/category-chart')
                .then(r => r.json())
                .then(figJson => {
                    const fig = JSON.parse(figJson);
                    Plotly.newPlot('category-chart', fig.data, fig.layout, {responsive: true});
                })
                .catch(err => {
                    document.getElementById('category-chart').innerHTML = '<p style="text-align: center; padding: 40px; color: #7f8c8d;">Category chart temporarily unavailable</p>';
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
                        <p style="padding: 15px; background: #e8f5e9; border-left: 4px solid #27ae60; border-radius: 5px; margin-top: 20px;">
                            <strong>💡 ${data.recommendation}</strong>
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
                    document.getElementById('map').innerHTML = '<p style="text-align: center; padding: 40px; color: #7f8c8d;">Map visualization temporarily unavailable</p>';
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
    """Get ML-based delay predictions"""
    data = load_or_generate_data()
    predictions_df = data['predictions']
    
    # Return top 10 predictions with highest predicted delays
    top_predictions = predictions_df.nlargest(10, 'predicted_delay_days')[
        ['project_id', 'project_name', 'district', 'physical_progress_percent', 
         'predicted_delay_days', 'predicted_completion_date']
    ]
    
    # Convert datetime to string for JSON serialization
    result = top_predictions.copy()
    if 'predicted_completion_date' in result.columns:
        result['predicted_completion_date'] = result['predicted_completion_date'].astype(str)
    
    import json
    return json.dumps(result.to_dict('records'), default=str), 200, {'Content-Type': 'application/json'}


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
