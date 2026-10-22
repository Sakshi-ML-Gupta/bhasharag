# BhashaRAG — Multilingual Grounded RAG

A retrieval-augmented generation (RAG) system that answers questions about any PDF
in **22 Indic languages + English**, with page-level citations.

## Demo

![Dashboard screenshot](docs/screenshot.png)
*(Optional — add a screenshot of your running app here)*

## Features

- **Hybrid OCR** — Chandra for image-heavy pages, PyMuPDF for text-layer pages
- **List-aware semantic chunking** — preserves numbered lists and tables as single chunks
- **Hybrid retrieval** — Dense (BGE-small) + BM25 + query expansion + section scoping
- **Multilingual Q&A** — English + 22 scheduled Indian languages via AI4Bharat IndicTrans2
- **Grounded answers** — every claim cited as `[Page X]`
- **Zero hallucination** on genuinely-absent topics
- **MongoDB persistence** — chat sessions and upload history
- **React frontend** — modern dashboard with auto language detection

## Architecture
PDF upload
↓
Chandra OCR (image pages) + PyMuPDF (text pages)
↓
List-aware semantic chunking (~70-90 chunks per 42-page PDF)
↓
BGE-small embeddings → Qdrant (local, on-disk)
↓
Query → BM25 + dense hybrid search + query expansion + section scoping
↓
Groq LLM (openai/gpt-oss-120b) with strict grounding prompt
↓
AI4Bharat IndicTrans2 translation → response in user's language


## Accuracy (39-Question Audit)

Tested on a 42-page social media project report, queried in **English, Hindi,
Marathi, Bengali, Assamese, Gujarati, Punjabi, Odia, Tamil, Telugu, Kannada,
Malayalam, Urdu, Nepali, Sanskrit, Maithili, Bodo, Dogri, Konkani, and Kashmiri**.

| Metric | Score |
|--------|-------|
| Weighted accuracy | ~78–82% |
| Strict pass rate (fully correct) | ~60–68% |
| Citation accuracy | ~85% |
| Language accuracy | 93%+ |
| Hallucination rate | <5% |

## Prerequisites

- **Python 3.11** (3.12+ has compatibility issues with IndicTransToolkit)
- **MongoDB Community Server** running locally on port 27017
- **Node.js 18+** for the frontend
- **Datalab API key** (for Chandra OCR — free tier at https://datalab.to)
- **Groq API key** (free at https://console.groq.com/keys)

## Quick Start

### Backend

```bash
# 1. Clone
git clone https://github.com/Sakshi-ML-Gupta/bhasharag.git
cd bhasharag

# 2. Create venv
python -m venv inteliidocs
inteliidocs\Scripts\activate      # Windows
# source inteliidocs/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env — add your Groq and Datalab API keys

# 5. Download IndicLID language ID model (280 MB, once)
mkdir -p IndicLID/models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ai4bharat/IndicLID-FTN', filename='model_baseline_native.bin', local_dir='IndicLID/models')"

# 6. Start server
python main.py
# → http://127.0.0.1:8000

FRONTEND
cd intellidocs-frontend
npm install
npm run dev
# → http://localhost:5173
First Use
Open http://localhost:5173

Upload a PDF via the Document Ingestion panel

Wait for [index] upserted N chunks in the server console

Ask questions in any language — the response comes back in the same language

Configuration
Variable	Purpose	Default
GROQ_API_KEY	Groq LLM API key	required
LLM_MODEL	Groq model name	openai/gpt-oss-120b
MONGO_URI	MongoDB connection string	mongodb://localhost:27017
MONGO_DB	MongoDB database name	intellidocs
DATALAB_API_KEY	Datalab Chandra OCR key	required
API Endpoints
Method	Endpoint	Purpose
POST	/api/v1/upload	Upload PDF, index chunks
POST	/api/v1/chat	Query the RAG
GET	/api/v1/uploads	List all uploads
POST	/api/v1/sessions	Create chat session
GET	/api/v1/sessions/{id}/messages	Get chat history
DELETE	/api/v1/sessions/{id}/messages	Clear chat
GET	/api/v1/debug/index-stats	Vector DB stats
POST	/api/v1/debug/reindex	Re-index default PDF
Tech Stack
Backend: FastAPI, Qdrant, sentence-transformers (BGE-small)

OCR: Datalab Chandra (image pages), PyMuPDF (text pages)

Retrieval: Dense (BGE) + BM25 + query expansion + section boost

LLM: Groq openai/gpt-oss-120b

Translation: AI4Bharat IndicTrans2 (200M distilled)

Language ID: AI4Bharat IndicLID (fasttext)

Persistence: MongoDB (sessions, uploads) + Qdrant (vectors)

Frontend: React (Vite)

Known Limitations
Image-based section headings — Chandra OCR occasionally drops section headings rendered as bold styled text. The KNOWN_PAGE_SECTIONS fallback in vector_db.py mitigates this for known pages.

Devanagari inside dark code blocks — Garbled in ~3 queries due to OCR limitations on styled text.

Chandra API dependency — Requires DATALAB_API_KEY for image-heavy pages.

Test scope — Tested on a 42-page PDF. May need re-tuning for 500+ page documents.

Concurrent access — Uses local Qdrant (single writer). For multi-user deployments, switch to Qdrant server mode.

License
MIT

text

---

## Then Update `requirements.txt`

Add these pinned versions to your `requirements.txt`. This is the exact list of dependencies your code uses:
fastapi>=0.115,<1.0
uvicorn>=0.30,<1.0
qdrant-client==1.9.2
sentence-transformers>=3.3,<4.0
transformers>=4.51,<4.57
torch==2.5.1
groq>=0.11,<1.0
motor>=3.5,<4.0
pymongo>=4.8,<5.0
pymupdf>=1.24,<2.0
python-multipart>=0.0.9
python-dotenv>=1.0
rank-bm25>=0.2.2
datalab-python-sdk
fasttext-wheel
numpy>=1.26,<2.0
pydantic>=2.0,<3.0
pydantic-settings>=2.0
huggingface-hub>=0.36

**Why this matters:**
- `qdrant-client==1.9.2` — newer versions removed the `.search()` method your code uses
- `transformers>=4.51,<4.57` — the AI4Bharat code breaks on 4.57+
- `torch==2.5.1` — PyTorch 2.6+ has a KV-cache incompatibility with IndicTrans2
- `numpy<2.0` — NumPy 2.x breaks `fasttext`

Pin all of these or your README instructions will fail for anyone who clones the repo.

---

## Final Push

```powershell
cd D:\intellidocs-backend

# Overwrite README with the clean version
# (save the markdown above to README.md)

# Overwrite requirements.txt with the pinned versions above

# Stage and commit
git add README.md requirements.txt
git commit -m "Clean up README and pin dependency versions"
git push origin main