import streamlit as st
from typing import Dict, Any

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar configuration"""
    config = {}
    
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Provider selection
        config["embedding_provider"] = st.selectbox(
            "Embedding Provider",
            ["sentence-transformers", "openai", "huggingface"],
            help="Choose the embedding model provider"
        )
        
        config["llm_provider"] = st.selectbox(
            "LLM Provider",
            ["openai", "google", "huggingface"],
            help="Choose the language model provider"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            config["chunk_size"] = st.slider(
                "Chunk Size", 500, 2000, 1000,
                help="Size of text chunks for processing"
            )
            
            config["chunk_overlap"] = st.slider(
                "Chunk Overlap", 50, 500, 200,
                help="Overlap between consecutive chunks"
            )
            
            config["top_k"] = st.slider(
                "Top K Results", 3, 10, 5,
                help="Number of documents to retrieve"
            )
            
            config["use_gpu"] = st.checkbox(
                "Use GPU (if available)", True,
                help="Enable GPU acceleration for embeddings"
            )
        
        # Performance monitoring
        st.subheader("📈 Performance")
        if st.button("View Analytics"):
            st.session_state.show_analytics = True
    
    return config
