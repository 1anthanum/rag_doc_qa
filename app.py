"""
RAG Document Q&A — Streamlit 前端界面
支持文档上传、混合检索、HyDE 查询优化、CRAG 自纠错，实时配置切换。
"""

import time
import tempfile
import logging
from pathlib import Path

import streamlit as st

from src.config import get as cfg
from src.security import sanitize_filename, validate_file_magic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 页面配置 ─────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG 文档问答系统",
    page_icon="📚",
    layout="wide",
)

# ── 自定义样式 ────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* 主标题 */
    .main-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-title  { color: #888; font-size: 0.95rem; margin-bottom: 1.2rem; }

    /* 指标卡片 */
    [data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
    }

    /* Pipeline badge */
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; margin-right: 4px;
    }
    .badge-blue   { background: #dbeafe; color: #1e40af; }
    .badge-purple { background: #ede9fe; color: #6d28d9; }
    .badge-green  { background: #dcfce7; color: #166534; }
    .badge-amber  { background: #fef3c7; color: #92400e; }

    /* 元数据行 */
    .meta-row {
        display: flex; gap: 16px; padding: 8px 0;
        font-size: 0.82rem; color: #666;
    }
    .meta-row span { white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Pipeline 构建 ─────────────────────────────────────────────


def build_pipeline(
    use_hybrid: bool = False,
    use_hyde: bool = False,
    use_crag: bool = False,
    llm_provider: str = "anthropic",
):
    """
    按给定配置构建 RAG pipeline，由 Streamlit 缓存管理。

    与旧版 init_pipeline() 的区别：接受显式参数而非只读 config，
    这样侧边栏开关变化时可以重建不同配置的 pipeline。
    """
    from src.ingestion import DocumentLoader, TextChunker, ChunkingStrategy
    from src.ingestion.embedder import EmbeddingEngine
    from src.retrieval import FAISSVectorStore, Retriever, HybridRetriever
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import RAGChain, AgenticRAGChain
    from src.generation.llm_client import (
        OpenAIClient,
        OllamaClient,
        AnthropicClient,
    )

    # Embedding
    emb_provider = cfg("ingestion.embedding.provider", "local")
    emb_model = cfg("ingestion.embedding.model", "BAAI/bge-small-zh-v1.5")
    engine = EmbeddingEngine(provider=emb_provider, model_name=emb_model)

    # Chunking
    chunk_strategy_str = cfg("ingestion.chunking.strategy", "recursive")
    strategy_map = {
        "fixed_size": ChunkingStrategy.FIXED_SIZE,
        "sentence": ChunkingStrategy.SENTENCE,
        "recursive": ChunkingStrategy.RECURSIVE,
        "semantic": ChunkingStrategy.SEMANTIC,
    }
    strategy = strategy_map.get(chunk_strategy_str, ChunkingStrategy.RECURSIVE)

    chunker_kwargs = {
        "chunk_size": cfg("ingestion.chunking.chunk_size", 512),
        "overlap": cfg("ingestion.chunking.overlap", 64),
        "strategy": strategy,
    }
    if strategy == ChunkingStrategy.SEMANTIC:
        chunker_kwargs["embedder"] = engine.embedder
        chunker_kwargs["semantic_threshold"] = cfg(
            "ingestion.chunking.semantic_threshold", 0.5
        )

    loader = DocumentLoader()
    chunker = TextChunker(**chunker_kwargs)
    store = FAISSVectorStore(dimension=engine.dimension)

    # Retriever
    top_k = cfg("retrieval.top_k", 5)
    rerank = cfg("retrieval.rerank", False)
    rerank_model = cfg("retrieval.rerank_model", None)

    if use_hybrid:
        retriever = HybridRetriever(
            embedding_engine=engine,
            vector_store=store,
            top_k=top_k,
            dense_weight=cfg("retrieval.hybrid_search.dense_weight", 0.7),
            sparse_weight=cfg("retrieval.hybrid_search.sparse_weight", 0.3),
            rrf_k=cfg("retrieval.hybrid_search.rrf_k", 60),
            rerank=rerank,
            rerank_model=rerank_model,
        )
    else:
        retriever = Retriever(
            embedding_engine=engine,
            vector_store=store,
            top_k=top_k,
            rerank=rerank,
            rerank_model=rerank_model,
        )

    # LLM
    llm_model = cfg("generation.model", "claude-sonnet-4-20250514")
    if llm_provider == "anthropic":
        llm = AnthropicClient(model=llm_model)
    elif llm_provider == "ollama":
        llm = OllamaClient(model=cfg("generation.ollama.model", "llama3.1"))
    else:
        llm = OpenAIClient(model=llm_model)

    # Query processor
    query_processor = None
    if use_hyde:
        query_processor = QueryProcessor(llm=llm, strategy="hyde")

    # Chain
    if use_crag:
        chain = AgenticRAGChain(
            retriever=retriever,
            llm=llm,
            query_processor=query_processor,
            max_correction_rounds=cfg("agentic.max_correction_rounds", 2),
            adaptive_retrieval=cfg("agentic.adaptive_retrieval", True),
        )
    else:
        chain = RAGChain(
            retriever=retriever,
            llm=llm,
            query_processor=query_processor,
        )

    return {
        "loader": loader,
        "chunker": chunker,
        "engine": engine,
        "store": store,
        "retriever": retriever,
        "chain": chain,
        "llm_provider": llm_provider,
        "is_hybrid": use_hybrid,
        "is_hyde": use_hyde,
        "is_crag": use_crag,
    }


def get_pipeline():
    """获取或创建 pipeline（从 session_state 读取配置）。"""
    s = st.session_state
    key = (
        s.get("use_hybrid", False),
        s.get("use_hyde", False),
        s.get("use_crag", False),
        s.get("llm_provider", "anthropic"),
    )

    # 缓存：配置不变时复用 pipeline
    if "pipeline" not in s or s.get("pipeline_key") != key:
        s["pipeline"] = build_pipeline(*key)
        s["pipeline_key"] = key

    return s["pipeline"]


# ── 侧边栏 ───────────────────────────────────────────────────


def render_sidebar():
    """渲染侧边栏：文档上传 + Pipeline 配置 + 查询参数。"""
    with st.sidebar:
        # ── 文档上传 ──
        st.markdown("### 📄 文档上传")

        uploaded_files = st.file_uploader(
            "支持 PDF、TXT、Markdown、DOCX",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button(
            "索引文档", type="primary", use_container_width=True
        ):
            index_documents(uploaded_files)

        st.divider()

        # ── Pipeline 配置 ──
        st.markdown("### ⚡ Pipeline 配置")

        st.toggle(
            "Hybrid Search (BM25 + 向量 + RRF)",
            value=False,
            key="use_hybrid",
            help="启用混合检索：关键词匹配 + 语义检索，召回率提升 15-25%",
        )
        st.toggle(
            "HyDE 查询优化",
            value=False,
            key="use_hyde",
            help="生成假设性回答作为检索 query，提升检索精度",
        )
        st.toggle(
            "CRAG 自纠错",
            value=False,
            key="use_crag",
            help="Corrective RAG：自动评估检索质量，不相关时重写并重新检索",
        )

        # LLM 选择
        llm_options = ["anthropic", "openai", "ollama"]
        llm_labels = {
            "anthropic": "Claude (Anthropic)",
            "openai": "GPT (OpenAI)",
            "ollama": "Ollama (本地)",
        }
        st.selectbox(
            "LLM 提供商",
            options=llm_options,
            format_func=lambda x: llm_labels[x],
            key="llm_provider",
        )

        st.divider()

        # ── 索引统计 ──
        pipeline = get_pipeline()
        store = pipeline["store"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("索引 Chunks", store.size)
        with col2:
            doc_count = st.session_state.get("indexed_doc_count", 0)
            st.metric("文档数", doc_count)

        # Active badges
        badges_html = ""
        if pipeline["is_hybrid"]:
            badges_html += '<span class="badge badge-blue">Hybrid</span>'
        if pipeline["is_hyde"]:
            badges_html += '<span class="badge badge-purple">HyDE</span>'
        if pipeline["is_crag"]:
            badges_html += '<span class="badge badge-green">CRAG</span>'
        provider = pipeline["llm_provider"]
        badges_html += f'<span class="badge badge-amber">{provider.title()}</span>'

        if badges_html:
            st.markdown(badges_html, unsafe_allow_html=True)

        st.divider()

        # ── 查询参数 ──
        st.markdown("### ⚙️ 查询参数")
        mode = st.selectbox(
            "查询模式",
            ["qa", "summarize", "compare", "conversational"],
            format_func=lambda x: {
                "qa": "问答 (QA)",
                "summarize": "总结 (Summarize)",
                "compare": "对比 (Compare)",
                "conversational": "对话 (Conversational)",
            }[x],
        )
        top_k = st.slider("Top-K 检索数量", 1, 20, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

        return mode, top_k, temperature


# ── 文档索引 ──────────────────────────────────────────────────


def index_documents(uploaded_files):
    """处理并索引上传的文档。"""
    pipeline = get_pipeline()
    loader = pipeline["loader"]
    chunker = pipeline["chunker"]
    engine = pipeline["engine"]
    store = pipeline["store"]
    retriever = pipeline["retriever"]

    progress = st.sidebar.progress(0, text="加载文档中...")
    documents = []
    max_file_size = cfg("app.max_file_size_mb", 50) * 1024 * 1024

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.size > max_file_size:
                st.sidebar.warning(
                    f"跳过 {uploaded_file.name}：超过 "
                    f"{cfg('app.max_file_size_mb', 50)} MB 限制"
                )
                continue

            safe_name = sanitize_filename(uploaded_file.name)
            file_bytes = uploaded_file.getbuffer()

            ext = Path(safe_name).suffix.lower()
            if not validate_file_magic(bytes(file_bytes), ext):
                st.sidebar.warning(f"跳过 {safe_name}：文件内容与扩展名 {ext} 不匹配")
                continue

            tmp_path = Path(tmp_dir) / safe_name
            tmp_path.write_bytes(file_bytes)

            try:
                doc = loader.load_file(str(tmp_path))
                documents.append(doc)
            except Exception as e:
                st.sidebar.warning(f"加载失败 {safe_name}: {e}")

            progress.progress(
                (i + 1) / len(uploaded_files) / 3,
                text=f"已加载 {i + 1}/{len(uploaded_files)} 个文件",
            )

    if not documents:
        st.sidebar.error("没有文档可以加载。")
        return

    # 分块
    progress.progress(0.4, text="文档分块中...")
    chunks = chunker.chunk_documents(documents)

    # 嵌入
    progress.progress(0.6, text=f"向量化 {len(chunks)} 个 chunks...")
    embedded = engine.embed_chunks(chunks)

    # 存储
    progress.progress(0.8, text="写入向量存储...")
    store.add(embedded)

    # Hybrid: BM25 索引
    from src.retrieval.hybrid_retriever import HybridRetriever

    if isinstance(retriever, HybridRetriever):
        progress.progress(0.9, text="构建 BM25 索引...")
        retriever.index_sparse(chunks)

    progress.progress(1.0, text="完成！")
    st.session_state["indexed_doc_count"] = st.session_state.get(
        "indexed_doc_count", 0
    ) + len(documents)
    st.sidebar.success(f"已索引 {len(documents)} 个文档 → {len(chunks)} 个 chunks")


# ── 主聊天界面 ────────────────────────────────────────────────


def render_chat(mode: str, top_k: int, temperature: float):
    """渲染主聊天区域。"""
    st.markdown(
        '<div class="main-title">📚 RAG 文档问答系统</div>'
        '<div class="sub-title">'
        "上传文档后，用自然语言提问。支持 Hybrid Search · HyDE · CRAG 全链路。"
        "</div>",
        unsafe_allow_html=True,
    )

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 元数据行
            if "meta" in message:
                meta = message["meta"]
                parts = []
                if "latency" in meta:
                    parts.append(f"⏱ {meta['latency']}")
                if "chunks" in meta:
                    parts.append(f"📄 {meta['chunks']} chunks")
                if "tokens" in meta:
                    parts.append(f"🔤 {meta['tokens']} tokens")
                if parts:
                    st.markdown(
                        f'<div class="meta-row">{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(f"<span>{p}</span>" for p in parts)}</div>',
                        unsafe_allow_html=True,
                    )

            # 来源展开
            if "sources" in message and message["sources"]:
                with st.expander("📎 检索来源"):
                    for src_name in message["sources"]:
                        st.markdown(f"- `{src_name}`")

    # 输入框
    if prompt := st.chat_input("请输入问题..."):
        pipeline = get_pipeline()
        chain = pipeline["chain"]
        store = pipeline["store"]

        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 检查是否有索引
        if store.size == 0:
            msg = "请先在侧边栏上传并索引文档。"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.warning(msg)
            return

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                t0 = time.time()
                try:
                    response = chain.query(
                        question=prompt,
                        mode=mode,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    dt = time.time() - t0

                    st.markdown(response.answer)

                    # 构建元数据
                    meta = {"latency": f"{dt:.1f}s"}

                    if (
                        hasattr(response, "retrieval_result")
                        and response.retrieval_result
                    ):
                        rr = response.retrieval_result
                        if hasattr(rr, "results"):
                            meta["chunks"] = len(rr.results)

                    if (
                        hasattr(response, "generation_result")
                        and response.generation_result
                    ):
                        gr = response.generation_result
                        if hasattr(gr, "usage") and gr.usage:
                            meta["tokens"] = gr.usage.get("total_tokens", "?")

                    # 元数据行
                    parts = []
                    if "latency" in meta:
                        parts.append(f"⏱ {meta['latency']}")
                    if "chunks" in meta:
                        parts.append(f"📄 {meta['chunks']} chunks")
                    if "tokens" in meta:
                        parts.append(f"🔤 {meta['tokens']} tokens")
                    if parts:
                        st.markdown(
                            f'<div class="meta-row">{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(f"<span>{p}</span>" for p in parts)}</div>',
                            unsafe_allow_html=True,
                        )

                    # 来源
                    sources = response.sources
                    if sources:
                        with st.expander("📎 检索来源"):
                            for src_name in sources:
                                st.markdown(f"- `{src_name}`")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "sources": sources,
                            "meta": meta,
                        }
                    )

                except Exception as e:
                    error_msg = f"错误: {str(e)}"
                    st.error(error_msg)
                    logger.exception("Query failed")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


# ── 入口 ─────────────────────────────────────────────────────


def main():
    mode, top_k, temperature = render_sidebar()
    render_chat(mode, top_k, temperature)


if __name__ == "__main__":
    main()
