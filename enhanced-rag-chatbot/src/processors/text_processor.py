from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document
import re

class AdvancedTextProcessor:
    """Advanced text processing with multiple chunking strategies"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitters = self._initialize_splitters()
    
    def _initialize_splitters(self):
        """Initialize different text splitters"""
        return {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            "token": TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        }
    
    def process_documents(self, documents: List[Document], strategy: str = "recursive") -> List[Document]:
        """Process documents with specified chunking strategy"""
        if strategy not in self.splitters:
            strategy = "recursive"
        
        splitter = self.splitters[strategy]
        processed_docs = []
        
        for doc in documents:
            # Clean text
            cleaned_text = self._clean_text(doc.page_content)
            doc.page_content = cleaned_text
            
            # Split document
            chunks = splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_strategy": strategy
                })
            
            processed_docs.extend(chunks)
        
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def hierarchical_chunking(self, documents: List[Document]) -> List[Document]:
        """Hierarchical chunking for better context preservation"""
        processed_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Split by sections (assuming markdown-style headers)
            sections = re.split(r'\n#{1,6}\s+', content)
            
            for i, section in enumerate(sections):
                if section.strip():
                    # Further split large sections
                    if len(section) > self.chunk_size:
                        splitter = self.splitters["recursive"]
                        temp_doc = Document(page_content=section, metadata=doc.metadata.copy())
                        subsections = splitter.split_documents([temp_doc])
                        
                        for j, subsection in enumerate(subsections):
                            subsection.metadata.update({
                                "section_id": i,
                                "subsection_id": j,
                                "hierarchy_level": "subsection"
                            })
                        
                        processed_docs.extend(subsections)
                    else:
                        section_doc = Document(
                            page_content=section.strip(),
                            metadata={
                                **doc.metadata,
                                "section_id": i,
                                "hierarchy_level": "section"
                            }
                        )
                        processed_docs.append(section_doc)
        
        return processed_docs
