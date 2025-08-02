#!/usr/bin/env python3
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Import your Flask app
    from api import app
    
    # This is what gunicorn will call
    application = app
    
    if __name__ == "__main__":
        app.run()
        
except ImportError as e:
    print(f"Import error: {e}")
    # Create a simple fallback app for debugging
    from flask import Flask, jsonify
    application = Flask(__name__)
    
    @application.route('/')
    def debug():
        return jsonify({
            'error': 'Import failed', 
            'details': str(e),
            'python_path': sys.path,
            'current_dir': os.getcwd(),
            'files': os.listdir('.')
        })
        
except Exception as e:
    print(f"Startup error: {e}")
    from flask import Flask, jsonify
    application = Flask(__name__)
    
    @application.route('/')
    def error():
        return jsonify({'error': 'Startup failed', 'details': str(e)})