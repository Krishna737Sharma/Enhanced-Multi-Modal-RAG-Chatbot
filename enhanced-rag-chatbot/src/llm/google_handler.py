from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class GoogleHandler:
    """Handle Google AI LLM interactions"""
    
    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = self._initialize_llm()
        self.chain = None
    
    def _initialize_llm(self):
        """Initialize Google AI LLM"""
        try:
            kwargs = {"model": self.model_name}
            if self.api_key:
                kwargs["google_api_key"] = self.api_key
            return ChatGoogleGenerativeAI(**kwargs)
        except Exception as e:
            print(f"Error initializing Google LLM: {e}")
            return None
    
    def create_rag_chain(self, retriever):
        """Create RAG chain with retriever"""
        if not self.llm:
            return None
            
        template = """You are a knowledgeable AI assistant. Based on the provided context, answer the user's question accurately and comprehensively.

Context Information:
{context}

User Question: {question}

Please provide a detailed and helpful response based on the context. If the context doesn't contain sufficient information, acknowledge this limitation."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                formatted.append(f"Source {i} ({source}):\n{doc.page_content}")
            return "\n\n".join(formatted)
        
        try:
            self.chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            return self.chain
        except Exception as e:
            print(f"Error creating Google RAG chain: {e}")
            return None
    
    def generate_response(self, query: str, context_docs: List[Any] = None) -> str:
        """Generate response using the LLM"""
        if not self.llm:
            return "Google LLM not available. Please check your API key."
            
        try:
            if self.chain:
                return self.chain.invoke(query)
            else:
                if context_docs:
                    context = "\n\n".join([
                        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                        for doc in context_docs
                    ])
                    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                else:
                    prompt = query
                
                response = self.llm.invoke(prompt)
                return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
