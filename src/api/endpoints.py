"""
FastAPI endpoints for the RAG Document Q&A service.
Provides REST API for document ingestion and querying.
"""

import asyncio
import os
import logging
import tempfile
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get as cfg
from ..security import sanitize_filename, validate_file_magic

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = cfg("app.max_file_size_mb", 50) * 1024 * 1024

# ── Request/Response Models ──────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    mode: str = "qa"  # qa, summarize, compare, conversational
    top_k: int = Field(default=5, ge=1, le=100)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    num_chunks: int
    mode: str
    model: str


class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    index_size: int
    embedding_model: str


# ── App Factory ──────────────────────────────────────────────────


def create_app(
    rag_chain=None,
    embedding_engine=None,
    vector_store=None,
    retriever=None,
    document_loader=None,
    chunker=None,
) -> FastAPI:
    """
    Create the FastAPI application with injected dependencies.

    In production, pass fully initialized components.
    For development, components are lazily initialized.
    """

    app = FastAPI(
        title="RAG Document Q&A API",
        description="Ask questions about your documents using RAG.",
        version="0.1.0",
    )

    # ── CORS (restricted to specific origins and methods) ─────
    cors_origins = os.getenv(
        "CORS_ORIGINS",
        ",".join(cfg("api.cors_origins", ["http://localhost:8501"])),
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # ── Rate Limiting ──────────────────────────────────────────
    # Simple in-memory rate limiter for per-IP request throttling.
    # For production, consider Redis-backed solutions (e.g., slowapi).
    from collections import defaultdict
    import time

    _rate_limit_store: dict = defaultdict(list)
    _RATE_LIMIT_WINDOW = 60   # seconds
    _RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX", "30"))

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Simple per-IP rate limiting middleware."""
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip]
            if now - t < _RATE_LIMIT_WINDOW
        ]

        if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX_REQUESTS:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. "
                    f"Limit: {_RATE_LIMIT_MAX_REQUESTS} per {_RATE_LIMIT_WINDOW}s"
                },
            )

        _rate_limit_store[client_ip].append(now)
        return await call_next(request)

    # Store dependencies on app state
    app.state.rag_chain = rag_chain
    app.state.embedding_engine = embedding_engine
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.document_loader = document_loader
    app.state.chunker = chunker

    # ── Endpoints ────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API health and index status."""
        store = app.state.vector_store
        engine = app.state.embedding_engine
        return HealthResponse(
            status="healthy",
            index_size=store.size if store else 0,
            embedding_model=(
                engine.provider if engine else "not initialized"
            ),
        )

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Query the document index with a question."""
        chain = app.state.rag_chain
        if not chain:
            raise HTTPException(
                status_code=503,
                detail="RAG chain not initialized. Index documents first.",
            )

        try:
            # Pass per-request settings as arguments (thread-safe)
            response = await asyncio.to_thread(
                chain.query,
                question=request.question,
                mode=request.mode,
                temperature=request.temperature,
                top_k=request.top_k,
            )
            return QueryResponse(
                answer=response.answer,
                sources=response.sources,
                num_chunks=len(response.retrieval.results),
                mode=response.mode,
                model=response.generation.model,
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/index", response_model=IndexResponse)
    async def index_documents(files: List[UploadFile] = File(...)):
        """Upload and index documents."""
        loader = app.state.document_loader
        chunker_inst = app.state.chunker
        engine = app.state.embedding_engine
        store = app.state.vector_store

        if not all([loader, chunker_inst, engine, store]):
            raise HTTPException(
                status_code=503,
                detail="Pipeline components not initialized.",
            )

        documents = []
        skipped = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for upload_file in files:
                content = await upload_file.read()

                # Sanitize filename and validate content
                safe_name, error = validate_upload(
                    filename=upload_file.filename or "unnamed",
                    content=content,
                    max_size=MAX_FILE_SIZE,
                )
                if error:
                    logger.warning(f"Skipped {safe_name}: {error}")
                    skipped.append(f"{safe_name}: {error}")
                    continue

                tmp_path = Path(tmp_dir) / safe_name
                tmp_path.write_bytes(content)

                try:
                    doc = loader.load_file(str(tmp_path))
                    documents.append(doc)
                except Exception as e:
                    logger.warning(
                        f"Failed to load {safe_name}: {e}"
                    )

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents could be loaded.",
            )

        # Chunk and embed (offload to thread to avoid blocking event loop)
        chunks = await asyncio.to_thread(chunker_inst.chunk_documents, documents)
        embedded = await asyncio.to_thread(engine.embed_chunks, chunks)
        await asyncio.to_thread(store.add, embedded)

        return IndexResponse(
            message="Documents indexed successfully.",
            documents_indexed=len(documents),
            total_chunks=len(chunks),
        )

    @app.delete("/index")
    async def clear_index():
        """Clear the document index."""
        store = app.state.vector_store
        if not store:
            raise HTTPException(
                status_code=503, detail="Vector store not initialized.",
            )
        store.clear()
        return {"message": "Index cleared."}

    return app
