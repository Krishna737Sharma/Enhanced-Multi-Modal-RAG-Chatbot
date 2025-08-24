import os
import torch
from typing import Dict, Any
from pathlib import Path

class Settings:
    """Configuration settings for the RAG chatbot"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    VECTORSTORE_DIR = DATA_DIR / "vectorstore"
    SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
    
    # Model configurations
    EMBEDDING_MODELS = {
        "sentence-transformers": "all-MiniLM-L6-v2",
        "openai": "text-embedding-3-small",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    LLM_MODELS = {
        "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        "google": ["gemini-pro", "gemini-1.5-pro"],
        "huggingface": ["microsoft/DialoGPT-medium"]
    }
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval parameters
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # GPU settings
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if USE_GPU else "cpu"
    
    @classmethod
    def get_api_key(cls, provider: str) -> str:
        """Get API key for specified provider"""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        return os.getenv(key_map.get(provider, ""))
