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

def handler(event, context):
    """Netlify function handler that wraps Flask app"""
    # Convert Netlify event to WSGI environment
    path = event.get('path', '/')
    http_method = event.get('httpMethod', 'GET')
    headers = event.get('headers', {})
    query_string = event.get('queryStringParameters', {}) or {}
    body = event.get('body', '')
    
    # Create WSGI environment
    environ = {
        'REQUEST_METHOD': http_method,
        'PATH_INFO': path,
        'QUERY_STRING': '&'.join(f"{k}={v}" for k, v in query_string.items()),
        'CONTENT_TYPE': headers.get('content-type', ''),
        'CONTENT_LENGTH': str(len(body)) if body else '0',
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '5000',
        'wsgi.input': None,
        'wsgi.url_scheme': 'https',
        'wsgi.errors': sys.stderr,
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
    }
    
    # Add headers to environ
    for key, value in headers.items():
        environ_key = f'HTTP_{key.upper().replace("-", "_")}'
        environ[environ_key] = value
    
    # Call Flask app
    from io import BytesIO
    if body:
        environ['wsgi.input'] = BytesIO(body.encode())
    
    def start_response(status, response_headers):
        pass
    
    response = app(environ, start_response)
    
    # Convert response to Netlify format
    response_body = b''.join(response)
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': response_body.decode('utf-8')
    }
