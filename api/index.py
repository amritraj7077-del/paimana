"""
Vercel Serverless Function for PAIMANA Intelligence Platform
Wraps the Flask application for deployment on Vercel
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the Flask app
from src.dashboard.app import app

# Vercel entry point - ASGI/WSGI handler
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware

def handler(environ, start_response):
    """Vercel serverless function handler using WSGI interface"""
    return app(environ, start_response)
