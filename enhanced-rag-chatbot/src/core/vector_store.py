from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb
from pathlib import Path

class EnhancedVectorStore:
    """Enhanced vector store with advanced retrieval capabilities"""
    
    def __init__(self, persist_directory: str, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize ChromaDB vector store"""
        try:
            # Ensure directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Fallback: create in-memory store
            self.vectorstore = Chroma(embedding_function=self.embedding_function)
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to vector store in batches"""
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Basic similarity search"""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Similarity search with relevance scores"""
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Error in similarity search with score: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Document]:
        """Hybrid search combining dense and sparse retrieval"""
        try:
            # Get initial results
            dense_results = self.similarity_search_with_score(query, k=k*2)
            
            if not dense_results:
                return []
            
            # Apply re-ranking
            reranked_results = []
            for doc, score in dense_results:
                # Boost score based on query term overlap
                query_terms = set(query.lower().split())
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
                
                # Combine dense score with term overlap
                hybrid_score = alpha * (1 - score) + (1 - alpha) * overlap
                reranked_results.append((doc, hybrid_score))
            
            # Sort by hybrid score and return top k
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in reranked_results[:k]]
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return self.similarity_search(query, k)
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self._initialize_vectorstore()
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if hasattr(self.vectorstore, '_collection'):
                collection = self.vectorstore._collection
                return {
                    "name": getattr(collection, 'name', "default"),
                    "count": getattr(collection, 'count', lambda: 0)(),
                    "metadata": {}
                }
            else:
                return {"name": "default", "count": 0, "metadata": {}}
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"name": "default", "count": 0, "metadata": {}}
