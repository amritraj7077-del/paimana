"""
Netlify Serverless Function for PAIMANA Intelligence Platform
Wraps the Flask application for deployment on Netlify Functions
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the Flask app
from src.dashboard.app import app

# Use serverless-http to adapt Flask for Netlify Functions
from serverless_http import handler

# Export the handler for Netlify
lambda_handler = handler(app)
