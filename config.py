# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    HF_TOKEN: str = os.getenv("HF_TOKEN", "").strip()
    PROJECT_NAME: str = "IntelliDocs Engine"
    DEBUG: bool = False
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017").strip()
    MONGO_DB: str = os.getenv("MONGO_DB", "intellidocs").strip()
    
    # Provider Secrets
    GROQ_API_KEY: str  
    OPENAI_API_KEY: Optional[str] = None  
    
    # Model Configurations
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # ──> FIX: Overwrite using an active live production model ID
    LLM_MODEL: str = "openai/gpt-oss-120b"  
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

settings = Settings()
