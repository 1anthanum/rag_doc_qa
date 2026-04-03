# RAG Document Q&A

A production-grade Retrieval-Augmented Generation system for document question answering. Upload PDFs, text files, or Word documents, then ask natural-language questions grounded in your data.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   FastAPI     │────▶│   RAGChain   │
│  Frontend    │     │   REST API   │     │  Orchestrator│
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
                  ┌─────────────┐       ┌──────────────┐       ┌──────────────┐
                  │  Ingestion   │       │  Retrieval    │       │  Generation   │
                  │  Pipeline    │       │  Pipeline     │       │  Pipeline     │
                  ├─────────────┤       ├──────────────┤       ├──────────────┤
                  │ Loader       │       │ FAISS / Chroma│       │ OpenAI / Ollama│
                  │ Chunker      │       │ Retriever     │       │ Prompt Eng.   │
                  │ Embedder     │       │ Reranker      │       │ RAG Chain     │
                  └─────────────┘       └──────────────┘       └──────────────┘
```

## Key Features

- **Multi-format ingestion** — PDF, TXT, Markdown, DOCX
- **3 chunking strategies** — fixed-size, sentence-boundary, recursive splitting
- **Dual embedding support** — local (sentence-transformers) or OpenAI API
- **FAISS & ChromaDB** vector stores with cosine similarity search
- **Optional cross-encoder reranking** (ms-marco-MiniLM)
- **4 query modes** — QA, summarize, compare, conversational
- **OpenAI + Ollama** LLM backends (cloud or fully local)
- **Streamlit chat UI** with document upload and source citations
- **FastAPI REST API** with OpenAPI docs
- **Docker + GitHub Actions CI/CD**

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<you>/rag-doc-qa.git
cd rag-doc-qa
pip install -r requirements.txt

# 2. Set your API key (or use Ollama for fully local)
export OPENAI_API_KEY="sk-..."

# 3. Run the Streamlit app
streamlit run app.py

# Or run the FastAPI server
uvicorn src.api.endpoints:create_app --factory --reload
```

### Fully Local Mode (No API Key Needed)

```bash
# Install and start Ollama
# https://ollama.ai
ollama pull llama3.1

# Run with local models
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=local streamlit run app.py
```

## CLI Indexing

```bash
# Index a directory of documents
python scripts/index_documents.py ./docs/ --save-index ./index

# Index a single file with OpenAI embeddings
python scripts/index_documents.py paper.pdf --provider openai
```

## Project Structure

```
rag-doc-qa/
├── app.py                    # Streamlit frontend
├── src/
│   ├── ingestion/
│   │   ├── loader.py         # Multi-format document loading
│   │   ├── chunker.py        # Text chunking strategies
│   │   └── embedder.py       # Embedding engine (local + OpenAI)
│   ├── retrieval/
│   │   ├── vector_store.py   # FAISS & ChromaDB implementations
│   │   └── retriever.py      # Query → Embed → Search → Rerank
│   ├── generation/
│   │   ├── llm_client.py     # OpenAI & Ollama LLM clients
│   │   ├── prompt_templates.py # RAG prompt engineering
│   │   └── chain.py          # End-to-end RAG orchestration
│   └── api/
│       └── endpoints.py      # FastAPI REST API
├── tests/                    # Pytest test suite
├── scripts/
│   └── index_documents.py    # CLI indexing tool
├── configs/
│   └── default.yaml          # Default configuration
├── Dockerfile
├── .github/workflows/ci.yml  # GitHub Actions CI
└── requirements.txt
```

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Tech Stack

| Layer       | Technology                              |
|-------------|------------------------------------------|
| Frontend    | Streamlit                                |
| API         | FastAPI + Uvicorn                        |
| Embeddings  | sentence-transformers / OpenAI API       |
| Vector DB   | FAISS (default) / ChromaDB              |
| Reranking   | cross-encoder/ms-marco-MiniLM           |
| LLM         | OpenAI GPT-4o-mini / Ollama (local)     |
| CI/CD       | GitHub Actions + Docker                  |

## License

MIT
