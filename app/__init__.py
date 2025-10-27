import os
import logging
import sys
import json
from flask import Flask, g, session, request
from supabase import create_client, Client
import redis
from rq import Queue
from config import Config

# ===================================================================
# === 1. JSON LOGGING SETUP (FOR AUDIT TRAIL)
# ===================================================================

class JsonFormatter(logging.Formatter):
    """
    Custom log formatter that outputs log records as a single JSON object.
    This is ideal for containerized environments and log aggregation systems.
    """
    def format(self, record):
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.name,
        }
        # Add extra contextual data if it has been injected
        if hasattr(record, 'extra_data'):
            log_object.update(record.extra_data)
        # Add exception information if the log record has it
        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)
        
        return json.dumps(log_object)

class ContextualFilter(logging.Filter):
    """
    A logging filter that injects B2B-critical contextual information 
    (like user_id and org_id) into each log record.
    """
    def filter(self, record):
        try:
            # This runs within a Flask request context
            record.extra_data = {
                'url': request.path,
                'method': request.method,
                'ip': request.remote_addr,
                'user_id': session.get('user', {}).get('id', 'anonymous'),
                'org_id': session.get('org_id', 'none')
            }
        except RuntimeError:
            # This happens for logs generated outside of a request context (e.g., app startup)
            record.extra_data = {
                'user_id': 'system',
                'org_id': 'system'
            }
        return True

# ===================================================================
# === 2. EXTENSIONS INITIALIZATION
# ===================================================================

# Load Supabase credentials from environment variables
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")

# Create the Supabase clients
supabase_anon_client = create_client(url, key)
supabase_admin: Client = create_client(url, service_key)

def get_supabase() -> Client:
    """
    Gets or creates a Supabase client for the current request,
    authenticating it if a user is logged in.
    """
    if 'supabase' not in g:
        g.supabase = create_client(url, key)

    if 'access_token' in session:
        g.supabase.auth.set_session(session['access_token'], session.get('refresh_token'))

    return g.supabase

# ===================================================================
# === 3. FLASK APP FACTORY
# ===================================================================

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    
    # Load all configuration from the Config object
    app.config.from_object(Config)
    
    # --- CONFIGURE STRUCTURED JSON LOGGING ---
    # Clear any default handlers Flask might have set up.
    app.logger.handlers.clear()
    
    # Create a handler that streams logs to standard output (stdout),
    # the standard for Docker and cloud-native environments.
    handler = logging.StreamHandler(sys.stdout)
    
    # Set our custom JSON formatter and contextual filter on the handler.
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextualFilter())
    
    # Add the fully configured handler to the Flask app's logger.
    app.logger.addHandler(handler)
    
    # Set the desired log level. INFO is a good default for production.
    app.logger.setLevel(logging.INFO)

    # Also configure the root logger to use our handler. This is crucial
    # for capturing logs from libraries like RQ, Supabase-py, etc.
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # --- END LOGGING CONFIGURATION ---
    
    # Configure the shared data folder based on config.py
    shared_folder = app.config['SHARED_DATA_FOLDER']
    app.config['UPLOAD_FOLDER'] = shared_folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize Redis connection and the RQ task queue
    app.redis = redis.from_url(app.config['REDIS_URL'])
    app.task_queue = Queue('default', connection=app.redis)

    # Import and register the routes
    with app.app_context():
        from . import routes

    # Log that the application has started successfully
    app.logger.info("Application factory startup complete.")
    
    return app