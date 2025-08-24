import pytest
from langchain_core.documents import Document
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from processors.text_processor import AdvancedTextProcessor
from processors.multimodal_processor import MultiModalProcessor

class TestAdvancedTextProcessor:
    """Test suite for AdvancedTextProcessor"""
    
    def setup_method(self):
        self.processor = AdvancedTextProcessor()
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor is not None
        assert self.processor.chunk_size == 1000
        assert self.processor.chunk_overlap == 200
    
    def test_process_documents(self):
        """Test document processing"""
        doc = Document(
            page_content="This is a test document with some content that should be processed.",
            metadata={"source": "test.txt"}
        )
        
        processed_docs = self.processor.process_documents([doc])
        assert len(processed_docs) >= 1
        assert all(isinstance(doc, Document) for doc in processed_docs)
    
    def test_clean_text(self):
        """Test text cleaning"""
        dirty_text = "This has extra spaces\n\nand\n\nlines"
        clean_text = self.processor._clean_text(dirty_text)
        assert "  " not in clean_text
        assert clean_text.strip() == clean_text

class TestMultiModalProcessor:
    """Test suite for MultiModalProcessor"""
    
    def setup_method(self):
        self.processor = MultiModalProcessor()
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor is not None
        assert self.processor.text_processor is not None
    
    def test_process_mixed_documents(self):
        """Test processing mixed document types"""
        text_doc = Document(
            page_content="Text content",
            metadata={"source": "test.txt", "file_type": "text"}
        )
        
        image_doc = Document(
            page_content="Image description",
            metadata={"source": "test.png", "file_type": "image"}
        )
        
        processed_docs = self.processor.process_documents([text_doc, image_doc])
        assert len(processed_docs) >= 2

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
