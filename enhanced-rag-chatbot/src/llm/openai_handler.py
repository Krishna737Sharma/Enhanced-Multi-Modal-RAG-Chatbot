from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class OpenAIHandler:
    """Handle OpenAI LLM interactions"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = self._initialize_llm()
        self.chain = None
    
    def _initialize_llm(self):
        """Initialize OpenAI LLM"""
        try:
            kwargs = {"model": self.model_name}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            return ChatOpenAI(**kwargs)
        except Exception as e:
            print(f"Error initializing OpenAI LLM: {e}")
            return None
    
    def create_rag_chain(self, retriever):
        """Create RAG chain with retriever"""
        if not self.llm:
            return None
            
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        try:
            self.chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            return self.chain
        except Exception as e:
            print(f"Error creating RAG chain: {e}")
            return None
    
    def generate_response(self, query: str, context_docs: List[Any] = None) -> str:
        """Generate response using the LLM"""
        if not self.llm:
            return "OpenAI LLM not available. Please check your API key."
            
        try:
            if self.chain:
                return self.chain.invoke(query)
            else:
                # Fallback direct generation
                if context_docs:
                    context = "\n\n".join([doc.page_content for doc in context_docs])
                    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                else:
                    prompt = query
                
                response = self.llm.invoke(prompt)
                return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
