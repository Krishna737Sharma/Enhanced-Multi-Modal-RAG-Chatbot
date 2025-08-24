import streamlit as st
from typing import List, Dict, Any
import time

def render_chat_interface(chatbot, messages: List[Dict[str, str]]):
    """Render chat interface"""
    st.header("💬 Chat with your Documents")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = chatbot.query(prompt)
                response_time = time.time() - start_time
            
            st.markdown(response)
            
            # Show response time
            st.caption(f"Response time: {response_time:.2f}s")
            
            # Get sources
            retrieved_docs = chatbot.get_last_retrieved_docs()
            sources = [doc.metadata.get("source", "Unknown") for doc in retrieved_docs]
            
            if sources:
                with st.expander("📚 Sources"):
                    for source in set(sources):  # Remove duplicates
                        st.write(f"• {source}")
        
        # Add assistant message with sources
        messages.append({
            "role": "assistant",
            "content": response,
            "sources": list(set(sources)) if sources else []
        })
    
    return messages
