from .gemini_service import GeminiService
import os
from dotenv import load_dotenv
from utils import logger

logger.setLevel("DEBUG")
load_dotenv()

gemini_client = None
try:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    gemini = GeminiService(api_key=key)
    if gemini.client:
        gemini_client = gemini
        
except Exception as e:
    logger.warning(f"Error initializing GeminiService: {e}")
