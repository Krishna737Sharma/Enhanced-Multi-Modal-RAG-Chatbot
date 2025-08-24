from typing import List, Optional, Dict
import torch
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

class EnhancedEmbeddingGenerator:
    """Advanced embedding generation with GPU support"""
    
    def __init__(self, provider: str = "sentence-transformers", model_name: Optional[str] = None):
        self.provider = provider
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = self._initialize_model(model_name)
    
    def _initialize_model(self, model_name: Optional[str]):
        """Initialize embedding model based on provider"""
        try:
            if self.provider == "openai":
                return OpenAIEmbeddings(
                    model=model_name or "text-embedding-3-small"
                )
            
            elif self.provider == "google":
                return GoogleGenerativeAIEmbeddings(
                    model=model_name or "models/embedding-001"
                )
            
            elif self.provider == "huggingface":
                return HuggingFaceEmbeddings(
                    model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": self.device}
                )
            
            else:  # sentence-transformers
                model = SentenceTransformer(
                    model_name or "all-MiniLM-L6-v2",
                    device=self.device
                )
                return model
                
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            # Fallback to sentence-transformers
            return SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        try:
            if self.provider == "sentence-transformers":
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                return embeddings.tolist()
            else:
                return self.embedding_model.embed_documents(texts)
        except Exception as e:
            print(f"Error generating document embeddings: {e}")
            # Return dummy embeddings
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for query"""
        try:
            if self.provider == "sentence-transformers":
                embedding = self.embedding_model.encode([text])
                return embedding[0].tolist()
            else:
                return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            # Return dummy embedding
            return [0.0] * 384
