from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from .vector_store import EnhancedVectorStore

class EnhancedRetriever(BaseRetriever):
    """Enhanced retriever with multiple search strategies"""
    
    def __init__(self, vector_store: EnhancedVectorStore, k: int = 5):
        super().__init__()
        self.vector_store = vector_store
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        return self.vector_store.hybrid_search(query, k=self.k)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query"""
        return self._get_relevant_documents(query)
