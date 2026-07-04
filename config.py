import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Read the Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")