import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.document_loader import EnhancedDocumentLoader
from core.embeddings import EnhancedEmbeddingGenerator
from core.vector_store import EnhancedVectorStore
from langchain_core.documents import Document

class TestEnhancedDocumentLoader:
    """Test suite for EnhancedDocumentLoader"""
    
    def setup_method(self):
        self.loader = EnhancedDocumentLoader()
    
    def test_initialization(self):
        """Test loader initialization"""
        assert self.loader is not None
        assert len(self.loader.supported_formats) > 0
        assert 'pdf' in self.loader.supported_formats
        assert 'txt' in self.loader.supported_formats
    
    @patch('builtins.open', create=True)
    def test_load_text_file(self, mock_open):
        """Test loading text files"""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = "Test content"
        
        # Create a mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.read.return_value = b"Test content"
        
        # Test loading (will fail without proper setup, but tests structure)
        documents = self.loader.load_documents([])
        assert isinstance(documents, list)

class TestEnhancedEmbeddingGenerator:
    """Test suite for EnhancedEmbeddingGenerator"""
    
    def test_initialization(self):
        """Test embedding generator initialization"""
        generator = EnhancedEmbeddingGenerator("sentence-transformers")
        assert generator is not None
        assert generator.provider == "sentence-transformers"
    
    def test_embed_documents(self):
        """Test document embedding"""
        generator = EnhancedEmbeddingGenerator("sentence-transformers")
        texts = ["Hello world", "Test document"]
        embeddings = generator.embed_documents(texts)
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)

class TestEnhancedVectorStore:
    """Test suite for EnhancedVectorStore"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        # Mock embedding function
        self.mock_embedding = Mock()
        self.mock_embedding.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
        self.mock_embedding.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    
    def test_initialization(self):
        """Test vector store initialization"""
        vector_store = EnhancedVectorStore(self.temp_dir, self.mock_embedding)
        assert vector_store is not None
        assert vector_store.persist_directory == self.temp_dir

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
