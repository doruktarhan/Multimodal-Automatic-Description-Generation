import os
from dotenv import load_dotenv
from pathlib import Path

# Load the .env file (only once)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def get_env_var(key, default=None):
    """Retrieve environment variables safely."""
    return os.getenv(key, default)
