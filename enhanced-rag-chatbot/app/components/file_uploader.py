import streamlit as st
from typing import List, Any, Tuple

def render_file_uploader() -> Tuple[List[Any], Dict[str, Any]]:
    """Render file upload interface"""
    st.header("📎 Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx', 'csv', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload documents to build your knowledge base"
    )
    
    # Display upload statistics
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} files")
        
        # Show file details
        with st.expander("📄 File Details"):
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
    
    # Processing options
    with st.expander("🔧 Processing Options"):
        processing_strategy = st.selectbox(
            "Processing Strategy",
            ["adaptive", "recursive", "hierarchical"],
            help="Choose how documents should be processed"
        )
        
        enable_ocr = st.checkbox(
            "Enable OCR for images", True,
            help="Extract text from images using OCR"
        )
        
        enable_vision = st.checkbox(
            "Enable image description", True,
            help="Generate descriptions for images"
        )
    
    return uploaded_files, {
        "strategy": processing_strategy,
        "enable_ocr": enable_ocr,
        "enable_vision": enable_vision
    }
