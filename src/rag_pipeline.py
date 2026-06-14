"""
rag_pipeline.py
───────────────
Core RAG pipeline: parsing, chunking, indexing, retrieval, and generation.
All heavy lifting lives here; app.py handles only Streamlit UI.
"""

import gc
import io
import base64
import logging
import shutil
from pathlib import Path
from typing import Generator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 600
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K         = 10
DEFAULT_MAX_MEM_TURNS = 2
DEFAULT_MIN_TABLES    = 1
CHROMA_DB_DIR         = "./chroma_db_streamlit_rag"
EMBEDDING_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_LLM_MODEL        = "openai/gpt-oss-120b"
VISION_LLM_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"


# ─── Singleton caches (avoid reloading embeddings on every run) ───────────────
_embedding_model: HuggingFaceEmbeddings | None = None
_text_llm: ChatGroq | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_text_llm() -> ChatGroq:
    global _text_llm
    if _text_llm is None:
        logger.info("Initialising text LLM: %s", TEXT_LLM_MODEL)
        _text_llm = ChatGroq(model_name=TEXT_LLM_MODEL, temperature=0.0)
    return _text_llm


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_metadata(meta: dict) -> dict:
    """Ensure all metadata values are strings (Chroma requirement)."""
    return {k: (str(v) if v is not None else "") for k, v in meta.items()}


def is_real_data_table(table) -> bool:
    """
    Heuristic: skip TOCs, author blocks, etc.
    A 'real' table has ≥ 3 rows and at least one numeric cell.
    """
    if not table.data or not table.data.grid:
        return False
    rows = table.data.grid
    if len(rows) < 3:
        return False
    has_numeric = any(
        any(
            any(ch.isdigit() for ch in (cell.text or ""))
            for cell in row
            if cell is not None
        )
        for row in rows
    )
    return has_numeric


def summarize_image_with_llm(b64_str: str, label: str, page_num: int) -> str:
    """Send a figure to the Groq vision LLM and return a descriptive summary."""
    vision_llm = ChatGroq(
        model_name=VISION_LLM_MODEL,
        temperature=0.0,
    )
    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_str}"},
        },
        {
            "type": "text",
            "text": (
                f"This figure is from the '{label}' document, page {page_num}. "
                "Describe what this figure shows in detail: include any architecture "
                "components, flow of data, labels, axes, values, or structural relationships "
                "visible. Be specific and technical. Keep it under 150 words."
            ),
        },
    ])
    try:
        response = vision_llm.invoke([message])
        summary = response.content.strip()
        logger.debug("Vision summary for img %s p%d: %s…", label, page_num, summary[:80])
        return summary
    except Exception as exc:
        logger.warning("Vision LLM failed for img %s p%d: %s", label, page_num, exc)
        return (
            f"FIGURE FROM {label.upper()} PAGE {page_num}. "
            "Visual content could not be summarized."
        )


# ─── Parsing + Chunking ───────────────────────────────────────────────────────

def build_docling_converter() -> DocumentConverter:
    """Construct a Docling converter with table + picture extraction enabled."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure      = True
    pipeline_options.do_ocr                  = False
    pipeline_options.generate_page_images    = False
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale            = 1.0
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_and_chunk_pdfs(
    pdf_paths: list[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Generator[tuple[str, list[Document], dict[str, str], dict[str, str]], None, None]:
    """
    Generator that processes each PDF one at a time.
    Yields (log_message, new_docs, new_image_store_delta, new_table_store_delta)
    so callers can surface progress messages incrementally.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for pdf_path in pdf_paths:
        label = pdf_path.stem
        logger.info("Starting parse: [%s] -> %s", label, pdf_path.name)
        yield f"📄 Parsing **{pdf_path.name}**…", [], {}, {}

        # Fresh converter per PDF — releasing it after each doc prevents
        # cumulative memory growth that OOM-kills the process on large papers.
        converter = build_docling_converter()
        try:
            result = converter.convert(str(pdf_path))
        except Exception as exc:
            msg = f"❌ Failed to parse {pdf_path.name}: {exc}"
            logger.error(msg)
            yield msg, [], {}, {}
            del converter
            gc.collect()
            continue

        doc = result.document
        del result  # free raw conversion result immediately
        new_docs: list[Document] = []
        img_delta: dict[str, str] = {}
        tbl_delta: dict[str, str] = {}

        # ── Tables ────────────────────────────────────────────────────────────
        logger.info("[%s] Found %d tables", label, len(doc.tables))
        kept_tables = 0
        for t_idx, table in enumerate(doc.tables):
            try:
                if not is_real_data_table(table):
                    logger.debug("[%s] Skipping table %d (TOC/author block)", label, t_idx)
                    continue
                markdown_table = table.export_to_markdown(doc)
                page_num       = table.prov[0].page_no if table.prov else 0
                table_id       = f"table_{label}_p{page_num}_{t_idx}"
                tbl_delta[table_id] = markdown_table
                new_docs.append(Document(
                    page_content=(
                        f"RESULTS TABLE FROM {label.upper()} PAGE {page_num}. "
                        f"Contains structured benchmark scores, model comparisons, "
                        f"or experimental metrics:\n{markdown_table}"
                    ),
                    metadata=clean_metadata({
                        "page": page_num, "type": "table",
                        "table_id": table_id, "paper": label,
                        "source": str(pdf_path),
                    }),
                ))
                kept_tables += 1
            except Exception as exc:
                logger.warning("[%s] Table %d failed: %s", label, t_idx, exc)

        logger.info("[%s] Tables kept: %d / %d", label, kept_tables, len(doc.tables))
        yield (
            f"  ✅ Tables extracted: **{kept_tables}** of {len(doc.tables)}",
            new_docs, img_delta, tbl_delta,
        )
        new_docs, img_delta, tbl_delta = [], {}, {}

        # ── Images ────────────────────────────────────────────────────────────
        logger.info("[%s] Found %d figures", label, len(doc.pictures))
        kept_imgs = 0
        for img_idx, figure in enumerate(doc.pictures):
            try:
                page_num = figure.prov[0].page_no if figure.prov else 0
                image_id = f"img_{label}_p{page_num}_{img_idx}"
                pil_image = figure.get_image(doc)
                if pil_image is None:
                    logger.debug("[%s] No image data for figure %d p%d", label, img_idx, page_num)
                    continue
                # Resize large figures before encoding: full-res PNG can be
                # several MB per image; 800px JPEG is plenty for the vision
                # LLM and in-UI display, and saves ~10x memory.
                max_w = 800
                if pil_image.width > max_w:
                    ratio = max_w / pil_image.width
                    pil_image = pil_image.resize(
                        (max_w, int(pil_image.height * ratio))
                    )
                buf = io.BytesIO()
                pil_image.save(buf, format="JPEG", quality=85)
                b64_str = base64.b64encode(buf.getvalue()).decode()
                buf.close()
                del pil_image
                img_delta[image_id] = b64_str

                yield (
                    f"  🔍 Summarising figure {img_idx + 1} on page {page_num}…",
                    [], img_delta, {},
                )
                img_delta = {}

                summary = summarize_image_with_llm(b64_str, label, page_num)
                new_docs.append(Document(
                    page_content=summary,
                    metadata=clean_metadata({
                        "page": page_num, "type": "image",
                        "image_id": image_id, "paper": label,
                        "source": str(pdf_path),
                    }),
                ))
                # Re-emit the stored b64 so the caller can accumulate it
                img_delta[image_id] = b64_str
                kept_imgs += 1
            except Exception as exc:
                logger.warning("[%s] Figure %d failed: %s", label, img_idx, exc)

        logger.info("[%s] Images kept: %d / %d", label, kept_imgs, len(doc.pictures))
        yield (
            f"  🖼️  Figures summarised: **{kept_imgs}** of {len(doc.pictures)}",
            new_docs, img_delta, {},
        )
        new_docs, img_delta = [], {}

        # ── Text chunks ───────────────────────────────────────────────────────
        text_only = doc.export_to_markdown(strict_text=True)
        temp_doc  = Document(
            page_content=text_only,
            metadata={"page": "0", "type": "text", "paper": label, "source": str(pdf_path)},
        )
        chunks = text_splitter.split_documents([temp_doc])
        for c in chunks:
            c.metadata = clean_metadata(c.metadata)
        new_docs.extend(chunks)

        logger.info("[%s] Text chunks: %d", label, len(chunks))
        yield (
            f"  📝 Text chunks: **{len(chunks)}**",
            new_docs, {}, {},
        )

        # Explicitly free the Docling document tree and converter before the
        # next PDF — these hold 100s of MB for large papers.
        del doc, converter
        gc.collect()


# ─── Indexing ─────────────────────────────────────────────────────────────────

def build_vector_store(
    docs: list[Document],
    db_dir: str = CHROMA_DB_DIR,
) -> Chroma:
    """Create (or replace) a Chroma vector store from the given documents."""
    logger.info("Clearing existing DB at %s", db_dir)
    shutil.rmtree(db_dir, ignore_errors=True)

    logger.info("Building vector store with %d documents…", len(docs))
    vs = Chroma.from_documents(
        documents=docs,
        embedding=get_embedding_model(),
        collection_name="streamlit_multimodal_rag",
        persist_directory=db_dir,
    )
    count = vs._collection.count()
    logger.info("Vector store built. Vectors indexed: %d", count)
    return vs


def cleanup_vector_store(db_dir: str = CHROMA_DB_DIR) -> None:
    """Remove the persisted Chroma DB from disk."""
    shutil.rmtree(db_dir, ignore_errors=True)
    logger.info("Vector store at %s removed.", db_dir)


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_with_table_guarantee(
    query: str,
    vector_store: Chroma,
    top_k: int = DEFAULT_TOP_K,
    min_tables: int = DEFAULT_MIN_TABLES,
) -> list[Document]:
    """
    Standard similarity retrieval with a table top-up guarantee:
    if fewer than `min_tables` table chunks are returned, extra table
    chunks are injected from a metadata-filtered search.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    logger.debug("Retrieved %d docs for query: %s", len(docs), query[:80])

    table_hits = [d for d in docs if d.metadata.get("type") == "table"]
    if len(table_hits) < min_tables:
        logger.debug("Table top-up triggered (%d < %d)", len(table_hits), min_tables)
        extra_tables = vector_store.similarity_search(
            query, k=5, filter={"type": "table"}
        )
        existing = {d.page_content for d in docs}
        extras   = [d for d in extra_tables if d.page_content not in existing]
        docs     = extras[:min_tables] + docs

    return docs


# ─── Generation ───────────────────────────────────────────────────────────────

def build_messages(
    query: str,
    retrieved_docs: list[Document],
    history: list[dict],
    max_memory_turns: int = DEFAULT_MAX_MEM_TURNS,
) -> list:
    """Assemble the full message list (system + history + context + query)."""
    system_prompt = (
        "You are an elite research assistant. Answer using only the provided context blocks.\n"
        "You have access to text chunks, markdown tables, and figure descriptions.\n"
        "Cite every factual claim as [PaperLabel, Page X] inline in your answer.\n"
        "When quoting numbers from tables, reproduce them exactly as they appear."
    )
    context_blocks = []
    for doc in retrieved_docs:
        paper = doc.metadata.get("paper", "?")
        page  = doc.metadata.get("page",  "?")
        dtype = doc.metadata.get("type",  "?")
        context_blocks.append(
            f"[{paper.upper()} | Page {page} | Type: {dtype}]:\n{doc.page_content}"
        )
    compiled_context = "--- CONTEXT ---\n" + "\n\n".join(context_blocks)

    messages = [SystemMessage(content=system_prompt)]
    slice_size = max_memory_turns * 2
    for turn in (history[-slice_size:] if slice_size > 0 else []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=f"{compiled_context}\n\nUser Question: {query}"))
    return messages


def run_rag_query(
    query: str,
    vector_store: Chroma,
    chat_history: list[dict],
    image_store: dict[str, str],
    table_store: dict[str, str],
    top_k: int = DEFAULT_TOP_K,
    max_memory_turns: int = DEFAULT_MAX_MEM_TURNS,
    min_tables: int = DEFAULT_MIN_TABLES,
) -> tuple[str, list[Document]]:
    """
    Run a full RAG query: retrieve → build prompt → generate.
    Returns (answer_text, retrieved_docs).
    The caller is responsible for updating chat_history.
    """
    logger.info("RAG query: %s", query[:100])

    retrieved_chunks = retrieve_with_table_guarantee(
        query=query,
        vector_store=vector_store,
        top_k=top_k,
        min_tables=min_tables,
    )
    logger.info("Retrieved %d chunks for generation.", len(retrieved_chunks))

    messages = build_messages(
        query=query,
        retrieved_docs=retrieved_chunks,
        history=chat_history,
        max_memory_turns=max_memory_turns,
    )

    llm      = get_text_llm()
    response = llm.invoke(messages)
    answer   = response.content
    logger.info("LLM response generated (%d chars).", len(answer))

    return answer, retrieved_chunks