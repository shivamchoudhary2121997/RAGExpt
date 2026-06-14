"""
app.py
──────
Streamlit UI for the Multimodal RAG demo.
All RAG logic lives in src/rag_pipeline.py — this file is UI-only.

Screens
  1. Setup  — upload PDFs, tune parameters, press Submit
  2. Indexing — live log of parsing / chunking / indexing progress
  3. Chat   — conversational UI with source rendering
"""

import base64
import logging
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from src.rag_pipeline import (
    CHROMA_DB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_MEM_TURNS,
    DEFAULT_MIN_TABLES,
    DEFAULT_TOP_K,
    build_vector_store,
    cleanup_vector_store,
    parse_and_chunk_pdfs,
    run_rag_query,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
)

# ─── Session-state defaults ───────────────────────────────────────────────────
_defaults = {
    "screen":        "setup",   # "setup" | "indexing" | "chat"
    "vector_store":  None,
    "chat_history":  [],
    "image_store":   {},
    "table_store":   {},
    "tmp_dir":       None,      # temp directory holding uploaded PDFs
    "params":        {},
    "index_logs":    [],
    "total_docs":    0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Utility ──────────────────────────────────────────────────────────────────

def reset_session() -> None:
    """Full cleanup: remove PDFs, vector DB, clear all state."""
    logger.info("Resetting session.")
    if st.session_state.tmp_dir and Path(st.session_state.tmp_dir).exists():
        shutil.rmtree(st.session_state.tmp_dir, ignore_errors=True)
        logger.info("Removed temp dir: %s", st.session_state.tmp_dir)
    cleanup_vector_store(CHROMA_DB_DIR)
    for k, v in _defaults.items():
        st.session_state[k] = v


def save_uploads(uploaded_files) -> list[Path]:
    """Save Streamlit UploadedFile objects to a temp directory; return paths."""
    tmp = tempfile.mkdtemp()
    st.session_state.tmp_dir = tmp
    paths = []
    for f in uploaded_files:
        dest = Path(tmp) / f.name
        dest.write_bytes(f.read())
        paths.append(dest)
        logger.info("Saved upload: %s (%d bytes)", dest.name, dest.stat().st_size)
    return paths


# ─── Screens ──────────────────────────────────────────────────────────────────

def render_setup() -> None:
    st.title("🔍 Multimodal RAG — Setup")
    st.markdown(
        "Upload up to **3 PDF documents**, tune the pipeline parameters, "
        "then press **Submit** to parse, chunk, and index them."
    )

    # ── PDF uploads ───────────────────────────────────────────────────────────
    st.subheader("📄 Upload PDFs")
    uploaded = st.file_uploader(
        "Choose up to 3 PDF files (Each PDF should be smaller than 2 MB)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 3 PDFs; each is parsed with Docling (tables + figures extracted).",
    )
    if uploaded and len(uploaded) > 3:
        st.warning("Please upload at most 3 PDFs. Only the first 3 will be used.")
        uploaded = uploaded[:3]

    # ── Parameter panel ───────────────────────────────────────────────────────
    st.subheader("⚙️ Pipeline Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Chunking**")
        chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=200, max_value=2000,
            value=DEFAULT_CHUNK_SIZE, step=50,
            help=(
                "Number of characters per text chunk. "
                "Smaller → more precise retrieval; larger → more context per chunk."
            ),
        )
        chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            min_value=0, max_value=500,
            value=DEFAULT_CHUNK_OVERLAP, step=20,
            help="Characters shared between consecutive chunks to preserve context at boundaries.",
        )

    with col2:
        st.markdown("**Retrieval**")
        top_k = st.slider(
            "Top-K retrieval",
            min_value=1, max_value=20,
            value=DEFAULT_TOP_K, step=1,
            help="Number of chunks retrieved per query. Higher → more context but slower and noisier.",
        )
        min_tables = st.slider(
            "Min table chunks guaranteed",
            min_value=0, max_value=5,
            value=DEFAULT_MIN_TABLES, step=1,
            help=(
                "If fewer than this many table chunks appear in the top-K results, "
                "extra table chunks are injected from a metadata-filtered search."
            ),
        )

    st.markdown("**Memory**")
    max_memory_turns = st.slider(
        "Conversation memory turns",
        min_value=0, max_value=10,
        value=DEFAULT_MAX_MEM_TURNS, step=1,
        help=(
            "How many past Q&A turns to include in the LLM context. "
            "0 = no memory (each query is independent). "
            "Higher values improve multi-turn coherence but consume more tokens."
        ),
    )

    st.markdown(
        """
        <details>
        <summary>ℹ️ Full parameter reference</summary>

        | Parameter | What it controls |
        |---|---|
        | **Chunk size** | Characters per text chunk fed into the vector store. |
        | **Chunk overlap** | Characters repeated between adjacent chunks (boundary context). |
        | **Top-K retrieval** | Chunks returned by similarity search per query. |
        | **Min table chunks** | Floor on table chunks included in LLM context (table top-up heuristic). |
        | **Memory turns** | Past conversation turns injected into each LLM call. |

        Table and image extraction options (OCR, scale, structure detection) are
        currently fixed at sensible defaults inside `src/rag_pipeline.py`.
        </details>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    submit_disabled = not uploaded
    if st.button("🚀 Submit — Parse & Index", disabled=submit_disabled, type="primary"):
        st.session_state.params = {
            "chunk_size":       chunk_size,
            "chunk_overlap":    chunk_overlap,
            "top_k":            top_k,
            "min_tables":       min_tables,
            "max_memory_turns": max_memory_turns,
        }
        st.session_state.pdf_paths    = save_uploads(uploaded)
        st.session_state.screen       = "indexing"
        st.session_state.index_logs   = []
        st.session_state.image_store  = {}
        st.session_state.table_store  = {}
        st.session_state.chat_history = []
        st.rerun()


def render_indexing() -> None:
    st.title("⚙️ Parsing, Chunking & Indexing…")
    st.markdown(
        "The pipeline is running. Logs appear below as each stage completes."
    )

    # Guard: if indexing already finished, just show the button — don't re-parse
    if st.session_state.vector_store is not None:
        st.success(
            f"Indexed **{st.session_state.total_docs}** chunks. "
            "Click below to start chatting!"
        )
        if st.button("💬 Go to Chat", type="primary"):
            st.session_state.screen = "chat"
        return

    log_area  = st.empty()
    prog_bar  = st.progress(0, text="Starting…")

    pdf_paths = st.session_state.pdf_paths
    params    = st.session_state.params
    n_pdfs    = len(pdf_paths)

    all_docs: list     = []
    image_store: dict  = {}
    table_store: dict  = {}
    logs: list[str]    = []

    def flush_logs():
        log_area.markdown(
            "<div style='font-family:monospace;font-size:0.85em;'>"
            + "<br>".join(logs[-60:])
            + "</div>",
            unsafe_allow_html=True,
        )

    total_steps = n_pdfs * 3
    step        = 0

    for pdf_idx, pdf_path in enumerate(pdf_paths):
        for msg, new_docs, img_delta, tbl_delta in parse_and_chunk_pdfs(
            [pdf_path],
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
        ):
            logs.append(msg)
            all_docs.extend(new_docs)
            image_store.update(img_delta)
            table_store.update(tbl_delta)
            step += 1
            pct  = min(int(step / max(total_steps, 1) * 80), 80)
            prog_bar.progress(pct, text=msg.replace("**", "").replace("*", ""))
            flush_logs()

    if not all_docs:
        st.error(
            "❌ No content was extracted from your PDFs — all files failed to parse. "
            "Check the logs above for details (common cause: a missing system library). "
            "If you see a `libGL` error, make sure `packages.txt` contains `libgl1`."
        )
        if st.button("⬅️ Back to Setup"):
            reset_session()
            st.rerun()
        st.stop()

    logs.append(f"🗄️ Indexing **{len(all_docs)}** document chunks into vector store…")
    flush_logs()
    prog_bar.progress(85, text="Indexing into Chroma…")

    vs = build_vector_store(all_docs, db_dir=CHROMA_DB_DIR)

    logs.append(f"✅ Indexing complete! **{vs._collection.count()}** vectors stored.")
    logs.append("🎉 Ready to chat!")
    flush_logs()
    prog_bar.progress(100, text="Done!")

    st.session_state.vector_store = vs
    st.session_state.image_store  = image_store
    st.session_state.table_store  = table_store
    st.session_state.total_docs   = len(all_docs)

    st.success(
        f"Indexed **{len(all_docs)}** chunks from {n_pdfs} PDF(s). "
        "Click below to start chatting!"
    )
    if st.button("💬 Go to Chat", type="primary"):
        st.session_state.screen = "chat"


def render_source(doc) -> None:
    """Render a single retrieved source chunk (text / table / image)."""
    dtype = doc.metadata.get("type", "text")
    paper = doc.metadata.get("paper", "?").upper()
    page  = doc.metadata.get("page", "?")

    with st.expander(f"**[{paper}]** Page {page} — *{dtype}*", expanded=False):
        if dtype == "image":
            img_id = doc.metadata.get("image_id")
            if img_id and img_id in st.session_state.image_store:
                img_bytes = base64.b64decode(st.session_state.image_store[img_id])
                st.image(img_bytes, caption=f"Figure from {paper}, p{page}")
            else:
                st.markdown(f"> _{doc.page_content}_")

        elif dtype == "table":
            table_id = doc.metadata.get("table_id")
            if table_id and table_id in st.session_state.table_store:
                st.markdown(st.session_state.table_store[table_id])
            else:
                st.markdown(doc.page_content)

        else:
            preview = doc.page_content[:600]
            if len(doc.page_content) > 600:
                preview += "…"
            st.markdown(f"> {preview}")


def render_chat() -> None:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Multimodal RAG")
        vs    = st.session_state.vector_store
        n_vec = vs._collection.count() if vs else 0
        st.markdown(f"**Vectors indexed:** {n_vec}")
        st.markdown(f"**Docs processed:** {st.session_state.total_docs}")
        st.divider()
        params = st.session_state.params
        st.markdown("**Active params**")
        for k, v in params.items():
            st.markdown(f"- `{k}`: **{v}**")
        st.divider()
        if st.button("🗑️ Clear conversation memory"):
            st.session_state.chat_history = []
            st.rerun()
        st.divider()
        if st.button("⬅️ Back to Setup (clears everything)"):
            reset_session()
            st.rerun()

    # ── Main chat area ────────────────────────────────────────────────────────
    st.title("💬 Chat with your PDFs")

    # Render history
    for turn in st.session_state.chat_history:
        role = turn["role"]
        with st.chat_message(role):
            st.markdown(turn["content"])
            if role == "assistant" and "sources" in turn:
                st.markdown("---")
                st.markdown("**📋 Retrieved Sources**")
                for doc in turn["sources"]:
                    render_source(doc)

    # Input
    query = st.chat_input("Ask a question about your documents…")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                answer, retrieved_docs = run_rag_query(
                    query=query,
                    vector_store=st.session_state.vector_store,
                    chat_history=st.session_state.chat_history,
                    image_store=st.session_state.image_store,
                    table_store=st.session_state.table_store,
                    top_k=st.session_state.params["top_k"],
                    max_memory_turns=st.session_state.params["max_memory_turns"],
                    min_tables=st.session_state.params["min_tables"],
                )
            st.markdown(answer)
            st.markdown("---")
            st.markdown("**📋 Retrieved Sources**")
            for doc in retrieved_docs:
                render_source(doc)

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": answer,
            "sources": retrieved_docs,
        })


# ─── Router ───────────────────────────────────────────────────────────────────

screen = st.session_state.screen
if screen == "setup":
    render_setup()
elif screen == "indexing":
    render_indexing()
elif screen == "chat":
    render_chat()