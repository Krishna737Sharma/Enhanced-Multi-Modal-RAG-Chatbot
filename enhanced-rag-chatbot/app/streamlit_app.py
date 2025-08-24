import streamlit as st
import sys
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Add BOTH project root AND src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Add project root first (for config), then src
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import with error handling - CORRECTED SYNTAX
try:
    from core.document_loader import EnhancedDocumentLoader
    from core.embeddings import EnhancedEmbeddingGenerator
    from core.vector_store import EnhancedVectorStore
    from processors.multimodal_processor import MultiModalProcessor
    from llm.openai_handler import OpenAIHandler
    from llm.google_handler import GoogleHandler
    from llm.huggingface_handler import HuggingFaceHandler
    from evaluation.evaluator import BatchEvaluator
    from config.settings import Settings  # This will now work
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required packages are installed and files are correctly placed.")
    st.error(f"Current working directory: {Path.cwd()}")
    st.error(f"Project root: {project_root}")
    st.error(f"Python path: {sys.path[:5]}")
    st.stop()

# Rest of your code remains exactly the same...
class RAGChatbot:
    """Main RAG Chatbot class"""
    
    def __init__(self):
        self.document_loader = EnhancedDocumentLoader()
        self.processor = MultiModalProcessor()
        self.embedding_generator = None
        self.vector_store = None
        self.llm_handler = None
        self.evaluator = None
        self.last_retrieved_docs = []
    
    def initialize_components(self, embedding_provider: str, llm_provider: str, llm_model: str):
        """Initialize RAG components"""
        try:
            # Initialize embeddings
            self.embedding_generator = EnhancedEmbeddingGenerator(
                provider=embedding_provider,
                model_name=Settings.EMBEDDING_MODELS.get(embedding_provider)
            )
            
            # Initialize vector store
            self.vector_store = EnhancedVectorStore(
                persist_directory=str(Settings.VECTORSTORE_DIR),
                embedding_function=self.embedding_generator
            )
            
            # Initialize LLM
            api_key = Settings.get_api_key(llm_provider)
            if llm_provider == "openai":
                self.llm_handler = OpenAIHandler(llm_model, api_key)
            elif llm_provider == "google":
                self.llm_handler = GoogleHandler(llm_model, api_key)
            elif llm_provider == "huggingface":
                self.llm_handler = HuggingFaceHandler(llm_model)
            
            # Initialize evaluator
            self.evaluator = BatchEvaluator(self)
            
        except Exception as e:
            raise Exception(f"Failed to initialize components: {str(e)}")
    
    def add_documents(self, uploaded_files):
        """Add documents to the knowledge base"""
        if not uploaded_files:
            return
        
        try:
            # Load documents
            documents = self.document_loader.load_documents(uploaded_files)
            
            if not documents:
                st.error("No documents could be loaded")
                return
            
            # Process documents
            processed_docs = self.processor.process_documents(documents)
            
            # Add to vector store
            with st.spinner("Adding documents to knowledge base..."):
                self.vector_store.add_documents(processed_docs)
            
            st.success(f"Added {len(processed_docs)} document chunks to knowledge base")
            
            # Display collection info
            info = self.vector_store.get_collection_info()
            st.info(f"Total documents in knowledge base: {info['count']}")
            
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.vector_store or not self.llm_handler:
            return "Please initialize the system and add documents first."
        
        try:
            start_time = time.time()
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.hybrid_search(question, k=Settings.TOP_K)
            self.last_retrieved_docs = retrieved_docs
            
            if not retrieved_docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Generate response
            def retriever(q):
                return retrieved_docs
            
            chain = self.llm_handler.create_rag_chain(retriever)
            
            if chain and hasattr(chain, 'invoke'):
                response = chain.invoke(question)
            elif chain:
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                response = chain({"question": question, "context": context})
            else:
                # Fallback response
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                response = f"Based on the provided context, here's what I found about '{question}':\n\n{context[:500]}..."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_last_retrieved_docs(self):
        """Get last retrieved documents for evaluation"""
        return self.last_retrieved_docs

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Enhanced RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Enhanced RAG Chatbot with Multi-Modal Capabilities")
    st.markdown("---")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Embedding configuration
        st.subheader("Embeddings")
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["sentence-transformers", "openai", "huggingface"],
            index=0
        )
        
        # LLM configuration
        st.subheader("Language Model")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "google", "huggingface"],
            index=0
        )
        
        # Fallback model configuration if Settings not available
        try:
            llm_models = Settings.LLM_MODELS[llm_provider]
        except (NameError, AttributeError):
            llm_models = {
                "openai": ["gpt-3.5-turbo", "gpt-4"],
                "google": ["gemini-pro"],
                "huggingface": ["microsoft/DialoGPT-medium"]
            }[llm_provider]
        
        llm_model = st.selectbox(
            "Model",
            llm_models,
            index=0
        )
        
        # API Keys
        st.subheader("API Keys")
        if llm_provider in ["openai", "google"]:
            api_key = st.text_input(
                f"{llm_provider.title()} API Key",
                type="password",
                help="Enter your API key for the selected provider"
            )
            
            if api_key:
                key_name = f"{llm_provider.upper()}_API_KEY"
                os.environ[key_name] = api_key
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                try:
                    st.session_state.chatbot.initialize_components(
                        embedding_provider, llm_provider, llm_model
                    )
                    st.session_state.system_initialized = True
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {e}")
        
        st.markdown("---")
        
        # Vector store management
        st.subheader("📚 Knowledge Base")
        if st.session_state.system_initialized:
            try:
                info = st.session_state.chatbot.vector_store.get_collection_info()
                st.metric("Documents", info.get('count', 0))
            except:
                st.metric("Documents", 0)
            
            if st.button("🗑️ Clear Knowledge Base"):
                try:
                    st.session_state.chatbot.vector_store.delete_collection()
                    st.success("Knowledge base cleared!")
                except:
                    st.error("Could not clear knowledge base")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            if not st.session_state.system_initialized:
                st.error("Please initialize the system first!")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.query(prompt)
                    st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📎 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'docx', 'csv', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, CSV, PNG, JPG, JPEG"
        )
        
        if st.button("📥 Process Documents") and uploaded_files:
            if not st.session_state.system_initialized:
                st.error("Please initialize the system first!")
            else:
                st.session_state.chatbot.add_documents(uploaded_files)
        
        st.markdown("---")
        
        # Evaluation section
        st.header("📊 Evaluation")
        if st.session_state.system_initialized and st.session_state.messages:
            if st.button("🔍 Evaluate Last Response"):
                # Get last user question and assistant response
                user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
                assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
                
                if user_messages and assistant_messages:
                    last_question = user_messages[-1]["content"]
                    last_answer = assistant_messages[-1]["content"]
                    
                    # Get retrieved contexts
                    retrieved_docs = st.session_state.chatbot.get_last_retrieved_docs()
                    contexts = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
                    
                    # Evaluate
                    try:
                        from evaluation.metrics import RAGEvaluator
                        evaluator = RAGEvaluator()
                        
                        with st.spinner("Evaluating response..."):
                            metrics = evaluator.evaluate_response(
                                question=last_question,
                                answer=last_answer,
                                retrieved_contexts=contexts
                            )
                        
                        # Display metrics
                        st.subheader("Evaluation Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Faithfulness", f"{metrics.faithfulness:.2f}")
                        with col2:
                            st.metric("Answer Relevance", f"{metrics.answer_relevance:.2f}")
                        with col3:
                            st.metric("Context Precision", f"{metrics.context_precision:.2f}")
                        
                        # Create bar chart
                        metrics_data = {
                            "Metric": ["Faithfulness", "Answer Relevance", "Context Precision"],
                            "Score": [metrics.faithfulness, metrics.answer_relevance, metrics.context_precision]
                        }
                        
                        fig = px.bar(
                            x=metrics_data["Metric"],
                            y=metrics_data["Score"],
                            title="RAG Performance Metrics",
                            color=metrics_data["Score"],
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
