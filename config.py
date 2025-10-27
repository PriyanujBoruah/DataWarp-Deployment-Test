# config.py
import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'a-default-secret-key')
    
    # --- NEW: Define the mount point for shared data inside the container ---
    SHARED_DATA_FOLDER = '/home/appuser/shared_data'
    
    # --- CENTRALIZED REDIS CONFIGURATION ---
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Define the queues the application will use.
    QUEUES = ['high', 'default', 'low']