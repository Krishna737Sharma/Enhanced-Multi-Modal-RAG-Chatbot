# Enhanced Multi-Modal RAG Chatbot

A fully modular, research-oriented chatbot framework that supports multi-modal document retrieval, advanced chunking and pre-processing, dense and hybrid semantic search, and seamless integration with multiple LLM providers. Built for extensibility, testability, and high performance in computer vision and NLP research settings.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Directory Details](#directory-details)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-modal Input:** Supports PDF, DOCX, TXT, CSV, and image (PNG, JPG, JPEG) file ingestion.
- **Adaptive Chunking:** Hierarchical and adaptive chunking strategies for efficient document splitting.
- **Hybrid Retrieval:** Dense (vector, transformer-based) and lexical (overlap-based) searching, with optional re-ranking.
- **LLM Integration:** Easy switching among OpenAI, Google Gemini, and HuggingFace chat/completion models.
- **Image Understanding:** OCR (Tesseract), image description, and syntactic feature extraction.
- **Faithful RAG Evaluation:** Automated faithfulness, answer relevance, and context precision scoring.
- **GPU Utilization:** Smart device selection and GPU info, with code for memory optimization and cache management.
- **Extensive Logging:** Logging facilities configurable for both console and persistent log files.
- **Thorough Testing:** Pytest-based test-suite with mocking and workflow tests.

## Project Structure

```
enhanced-rag-chatbot/
├── app/
│   ├── components/
│   │   ├── chat_interface.py
│   │   ├── file_uploader.py
│   │   └── sidebar.py
│   └── streamlit_app.py
├── config/
│   └── settings.py
├── data/
│   ├── sample_documents/
│   └── vectorstore/
├── logs/
├── models/
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── llm/
│   │   ├── google_handler.py
│   │   ├── huggingface_handler.py
│   │   └── openai_handler.py
│   ├── processors/
│   │   ├── image_processor.py
│   │   ├── multimodal_processor.py
│   │   └── text_processor.py
│   └── utils/
│       ├── gpu_utils.py
│       └── helpers.py
├── tests/
│   ├── test_core.py
│   ├── test_evaluation.py
│   └── test_processors.py
└── .env.example
```

## Installation

### Prerequisites

- Python >= 3.8
- (Optional) CUDA-enabled GPU for acceleration

### Steps

```bash
git clone <your-repo-url>
cd enhanced-rag-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Then set your API keys as needed
```

## Quick Start

1. **Edit your `.env` file** with any required API keys (OpenAI, Google, HuggingFace).

2. **Start the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Navigate** to the Streamlit web page (usually at http://localhost:8501).

4. **Configure system, upload documents, and start chatting!**

## Usage

- **Upload single or multiple documents/images** using the sidebar file uploader.
- **Select embedding and LLM providers** using configurable dropdowns in the sidebar.
- **Process and index your documents** for querying.
- **Ask questions** in the chat interface and get context-grounded answers.
- **See retrieved sources** (with metadata) and run evaluation on generated responses.
- **Inspect RAG metrics:** Faithfulness, Answer Relevance, Context Precision, and visualize with plots.

## Configuration

All global and system configuration is handled in `config/settings.py`.  
Some important parameters:

- **Providers:** Set `EMBEDDING_MODELS` and `LLM_MODELS` dicts as needed.
- **Chunk sizes:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, etc.
- **VectorStore path:** `VECTORSTORE_DIR`
- **Device:** Auto-selected via `torch.cuda.is_available()`, override via environment variables.

To customize API keys and sensitive parameters, use the `.env` file format as shown in `.env.example`:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxx
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
USE_GPU=true
LOG_LEVEL=INFO
```

## Testing

Test coverage for all critical components is provided using `pytest`:

- Core functional tests: `tests/test_core.py`
- Evaluation logic: `tests/test_evaluation.py`
- Processor and pipeline validation: `tests/test_processors.py`

Run the full suite:
```bash
pytest tests/
```

Or run a specific test module:
```bash
pytest tests/test_core.py::TestEnhancedDocumentLoader
```

## Directory Details

- `data/sample_documents/`: Place demo files here for testing. This folder is empty by default.
- `data/vectorstore/`: ChromaDB vector index files (auto-created).
- `logs/`: Logging output; empty until the app writes logs.
- `models/`: Downloaded model weights; managed by HuggingFace/transformers libraries (empty until first use).
- `tests/`: Pytest-based tests validating all major modules.
- `app/components/`: Streamlit UI components for modular, maintainable frontend.
- `src/core/`: Backend core logic (loading, vectorization, database).
- `src/processors/`: Modular document processing for both text and image modalities.
- `src/llm/`: Adapters for different language model providers.
- `src/evaluation/`: Metrics and batch performance evaluators.
- `src/utils/`: General-purpose utilities (GPU handling, helpers).

## Contributing

Contributions are welcome!  
- Create an issue or pull request.
- Add tests for your new features.
- Ensure coding style and docstrings match the codebase.
- See `requirements.txt` for development dependencies.

## License

This project is open source under the MIT License (see `LICENSE`).

## Acknowledgments

- Built using LangChain, Streamlit, ChromaDB, Sentence Transformers, and open-source ML libraries.
- Inspired by cutting-edge research in Retrieval-Augmented Generation and multi-modal AI.
