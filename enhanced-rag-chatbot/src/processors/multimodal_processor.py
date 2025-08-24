from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .text_processor import AdvancedTextProcessor
from .image_processor import ImageProcessor

class MultiModalProcessor:
    """Unified processor for multi-modal documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_processor = AdvancedTextProcessor(chunk_size, chunk_overlap)
        self.image_processor = ImageProcessor()
    
    def process_documents(self, documents: List[Document], strategy: str = "adaptive") -> List[Document]:
        """Process documents based on their type"""
        processed_docs = []
        text_docs = []
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "text")
            
            if file_type == "image":
                # Images are already processed during loading
                processed_docs.append(doc)
            else:
                # Collect text documents for batch processing
                text_docs.append(doc)
        
        # Process text documents
        if text_docs:
            if strategy == "adaptive":
                processed_text_docs = self._adaptive_processing(text_docs)
            elif strategy == "hierarchical":
                processed_text_docs = self.text_processor.hierarchical_chunking(text_docs)
            else:
                processed_text_docs = self.text_processor.process_documents(text_docs, strategy)
            
            processed_docs.extend(processed_text_docs)
        
        return processed_docs
    
    def _adaptive_processing(self, documents: List[Document]) -> List[Document]:
        """Adaptively choose processing strategy based on document characteristics"""
        processed_docs = []
        
        for doc in documents:
            content_length = len(doc.page_content)
            
            # Choose strategy based on document characteristics
            if content_length > 10000:
                # Large documents: use hierarchical chunking
                chunks = self.text_processor.hierarchical_chunking([doc])
            elif doc.metadata.get("file_type") == "pdf":
                # PDFs: use recursive chunking
                chunks = self.text_processor.process_documents([doc], "recursive")
            else:
                # Default: recursive chunking
                chunks = self.text_processor.process_documents([doc], "recursive")
            
            processed_docs.extend(chunks)
        
        return processed_docs
    
    def create_cross_modal_connections(self, documents: List[Document]) -> List[Document]:
        """Create connections between text and image content"""
        # Group documents by source
        source_groups = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        enhanced_docs = []
        
        for source, docs in source_groups.items():
            # Find text and image documents from same source
            text_docs = [d for d in docs if d.metadata.get("file_type") != "image"]
            image_docs = [d for d in docs if d.metadata.get("file_type") == "image"]
            
            # Add cross-references
            for text_doc in text_docs:
                if image_docs:
                    text_doc.metadata["related_images"] = [
                        img.metadata.get("source") for img in image_docs
                    ]
            
            for image_doc in image_docs:
                if text_docs:
                    image_doc.metadata["related_text"] = [
                        txt.metadata.get("chunk_id", txt.metadata.get("source"))
                        for txt in text_docs
                    ]
            
            enhanced_docs.extend(docs)
        
        return enhanced_docs
