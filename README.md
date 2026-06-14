# RAGExpt: Multimodal RAG with Sources, Citations & Memory

A Streamlit demo for **Retrieval-Augmented Generation** over PDF documents.  
Powered by [Docling](https://github.com/DS4SD/docling) for multimodal parsing (text + tables + figures), [Chroma](https://www.trychroma.com/) for vector storage, [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and [Groq](https://groq.com/) for LLM inference.

---

## Features

- Upload up to **3 PDFs** and have them parsed, chunked, and indexed in one click
- **Multimodal extraction** — prose text, structured tables, and figures all go into the vector store
- **Figure summarisation** via a Groq vision LLM (Llama 4 Scout)
- **Table top-up heuristic** — guarantees at least one table chunk appears in every LLM context
- **Conversation memory** — configurable number of past turns injected into each query
- **Inline source rendering** — every answer shows the retrieved text chunks, markdown tables, and figures that grounded it
- Fully tunable pipeline parameters via the UI (chunk size, overlap, top-K, memory turns, etc.)

---

## Project Structure

```
RAGExpt/
├── app.py                  # Streamlit UI (setup → indexing → chat)
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py     # All RAG logic: parsing, chunking, indexing, retrieval, generation
├── packages.txt            # System-level apt dependencies (for Streamlit Cloud)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/RAGExpt.git
cd RAGExpt
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and Docling model weights. Subsequent runs use the local cache.

### 4. Set your Groq API key

```bash
export GROQ_API_KEY="gsk_your_key_here"   # Linux / macOS
set GROQ_API_KEY=gsk_your_key_here        # Windows CMD
```

Or create a `.streamlit/secrets.toml` file (gitignored):

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. **Push the repo to GitHub** — make sure `app.py`, `packages.txt`, `requirements.txt`, and `src/` are all at the repo root.

2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.

3. Select your repo, branch, and set **Main file path** to `app.py`.

4. Before deploying, go to **App settings → Secrets** and add your Groq API key:

   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```

   Streamlit Cloud automatically exposes secrets as environment variables — no code changes needed.

5. Click **Deploy**. Streamlit Cloud will:
   - Install system packages from `packages.txt` via `apt-get`
   - Install Python packages from `requirements.txt` via `pip`
   - Launch the app

> **First cold start** takes 5–10 minutes on the free tier due to model weight downloads. Subsequent restarts are faster once the cache is warm.

---

## Pipeline Parameters

All parameters are tunable from the Setup screen in the UI.

| Parameter | Default | Description |
|---|---|---|
| Chunk size | 600 | Characters per text chunk fed into the vector store |
| Chunk overlap | 120 | Characters shared between adjacent chunks (preserves boundary context) |
| Top-K retrieval | 10 | Number of chunks returned by similarity search per query |
| Min table chunks | 1 | Floor on table chunks in LLM context; triggers a metadata-filtered top-up if not met |
| Memory turns | 2 | Number of past Q&A turns included in each LLM call (0 = stateless) |

---

## Models Used

| Role | Model |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Text LLM | `openai/gpt-oss-120b` via Groq |
| Vision LLM | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq |

---

## Notes

- The Chroma vector DB is created at runtime in `./chroma_db_streamlit_rag/` and is wiped on every new session (when the user returns to the Setup screen).
- Uploaded PDFs are saved to a temporary directory and deleted on session reset.
- Table extraction uses a heuristic filter (`is_real_data_table`) to skip TOCs, author blocks, and other false-positive tables — only tables with ≥ 3 rows and at least one numeric cell are kept.
- OCR is disabled by default; Docling's native text layer extraction is used instead.
