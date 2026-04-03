"""
Streamlit frontend for RAG Document Q&A.
Provides a chat-like interface for uploading documents and asking questions.
"""

import os
import tempfile
import logging
from pathlib import Path

import streamlit as st

from src.config import get as cfg
from src.security import sanitize_filename, validate_file_magic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
)

# ── Session State Initialization ────────────────────────────────


@st.cache_resource
def init_pipeline():
    """Initialize RAG pipeline components (cached across reruns)."""
    from src.ingestion import DocumentLoader, TextChunker, ChunkingStrategy
    from src.ingestion.embedder import EmbeddingEngine
    from src.retrieval import FAISSVectorStore, Retriever
    from src.generation import RAGChain
    from src.generation.llm_client import OpenAIClient, OllamaClient

    # Config — YAML defaults with environment variable overrides
    embedding_provider = os.getenv(
        "EMBEDDING_PROVIDER", cfg("ingestion.embedding.provider", "local")
    )
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", cfg("ingestion.embedding.model", "all-MiniLM-L6-v2")
    )
    llm_provider = os.getenv(
        "LLM_PROVIDER", cfg("generation.provider", "openai")
    )
    llm_model = os.getenv(
        "LLM_MODEL", cfg("generation.model", "gpt-4o-mini")
    )

    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=512,
        overlap=64,
        strategy=ChunkingStrategy.RECURSIVE,
    )
    engine = EmbeddingEngine(
        provider=embedding_provider, model_name=embedding_model
    )
    store = FAISSVectorStore(dimension=engine.dimension)
    retriever = Retriever(
        embedding_engine=engine,
        vector_store=store,
        top_k=5,
    )

    if llm_provider == "ollama":
        llm = OllamaClient(model=llm_model)
    else:
        llm = OpenAIClient(model=llm_model)

    chain = RAGChain(retriever=retriever, llm=llm)

    return {
        "loader": loader,
        "chunker": chunker,
        "engine": engine,
        "store": store,
        "retriever": retriever,
        "chain": chain,
    }


def get_pipeline():
    """Get or create the pipeline."""
    return init_pipeline()


# ── Sidebar: Document Upload ────────────────────────────────────


def render_sidebar():
    """Render the sidebar with document upload and settings."""
    with st.sidebar:
        st.header("📄 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, MD, DOCX)",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("🔄 Index Documents", type="primary"):
            index_documents(uploaded_files)

        st.divider()

        # Index stats
        pipeline = get_pipeline()
        store = pipeline["store"]
        st.metric("Indexed Chunks", store.size)

        st.divider()

        # Query settings
        st.header("⚙️ Settings")
        mode = st.selectbox(
            "Query Mode",
            ["qa", "summarize", "compare", "conversational"],
            index=0,
        )

        top_k = st.slider("Top-K Results", 1, 20, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

        return mode, top_k, temperature


def index_documents(uploaded_files):
    """Process and index uploaded documents."""
    pipeline = get_pipeline()
    loader = pipeline["loader"]
    chunker = pipeline["chunker"]
    engine = pipeline["engine"]
    store = pipeline["store"]

    progress = st.sidebar.progress(0, text="Loading documents...")
    documents = []

    max_file_size = cfg("app.max_file_size_mb", 50) * 1024 * 1024

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.size > max_file_size:
                st.sidebar.warning(
                    f"Skipped {uploaded_file.name}: exceeds "
                    f"{cfg('app.max_file_size_mb', 50)} MB limit"
                )
                continue

            # Sanitize filename to prevent path traversal
            safe_name = sanitize_filename(uploaded_file.name)
            file_bytes = uploaded_file.getbuffer()

            # Validate file content matches its extension
            ext = Path(safe_name).suffix.lower()
            if not validate_file_magic(bytes(file_bytes), ext):
                st.sidebar.warning(
                    f"Skipped {safe_name}: file content does not match "
                    f"expected format for {ext}"
                )
                continue

            tmp_path = Path(tmp_dir) / safe_name
            tmp_path.write_bytes(file_bytes)

            try:
                doc = loader.load_file(str(tmp_path))
                documents.append(doc)
            except Exception as e:
                st.sidebar.warning(f"Failed to load {safe_name}: {e}")

            progress.progress(
                (i + 1) / len(uploaded_files) / 3,
                text=f"Loaded {i + 1}/{len(uploaded_files)} files",
            )

    if not documents:
        st.sidebar.error("No documents could be loaded.")
        return

    # Chunk
    progress.progress(0.4, text="Chunking documents...")
    chunks = chunker.chunk_documents(documents)

    # Embed
    progress.progress(0.6, text=f"Embedding {len(chunks)} chunks...")
    embedded = engine.embed_chunks(chunks)

    # Store
    progress.progress(0.8, text="Adding to vector store...")
    store.add(embedded)

    progress.progress(1.0, text="Done!")
    st.sidebar.success(
        f"Indexed {len(documents)} documents → {len(chunks)} chunks"
    )


# ── Main Chat Interface ─────────────────────────────────────────


def render_chat(mode: str, top_k: int, temperature: float):
    """Render the main chat interface."""
    st.title("📚 RAG Document Q&A")
    st.caption("Upload documents in the sidebar, then ask questions below.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📎 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- `{src}`")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        pipeline = get_pipeline()
        chain = pipeline["chain"]
        store = pipeline["store"]

        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if documents are indexed
        if store.size == 0:
            msg = "Please upload and index documents first using the sidebar."
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )
            with st.chat_message("assistant"):
                st.warning(msg)
            return

        # Generate response (pass settings per-request, don't mutate shared state)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.query(
                        question=prompt,
                        mode=mode,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    st.markdown(response.answer)

                    sources = response.sources
                    if sources:
                        with st.expander("📎 Sources"):
                            for src in sources:
                                st.markdown(f"- `{src}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


# ── Main ────────────────────────────────────────────────────────

def main():
    mode, top_k, temperature = render_sidebar()
    render_chat(mode, top_k, temperature)


if __name__ == "__main__":
    main()
