# IntelliDocs — Multilingual Grounded RAG

A retrieval-augmented generation (RAG) system that answers questions about a PDF
in any of 22 Indic languages, with page-level citations. Combines **Chandra OCR**
(image-heavy pages), **pymupdf** (text-layer pages), **BM25 + dense retrieval**,
**AI4Bharat IndicTrans2**, and **Groq LLMs**.

## Features

- **Hybrid OCR** — Chandra for scanned/image pages, pymupdf for text-layer pages
- **List-aware semantic chunking** — preserves numbered lists and tables
- **Hybrid retrieval** — dense vectors + BM25 + query expansion + section boost
- **Multilingual Q&A** — English + 22 scheduled Indian languages
- **Grounded answers** — every claim cited as `[Page X]`
- **Zero hallucination** — refuses only when content is genuinely absent
- **MongoDB persistence** — chat sessions and upload history
- **React frontend** — modern dashboard with language selector and Clear Chat

## Architecture
PDF upload
↓
Chandra OCR (image-heavy pages) + pymupdf (text pages)
↓
List-aware semantic chunking (~90 chunks per 42-page PDF)
↓
BGE-small embeddings → Qdrant (local, on-disk)
↓
Query → BM25 + dense hybrid search + query expansion
↓
Groq LLM (gpt-oss-120b) with strict grounding prompt
↓
IndicTrans2 translation → response in user's language


## Quick Start

### Prerequisites

- **Python 3.11** (3.12+ has compatibility issues with IndicTransToolkit)
- **MongoDB Community Server** running locally on port 27017
- **Node.js 18+** for the frontend
- **Tesseract** — not required if using Chandra API; optional fallback

### Backend Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/intellidocs.git
cd intellidocs

# 2. Create venv
python -m venv inteliidocs
inteliidocs\Scripts\activate      # Windows
# source inteliidocs/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env — add your Groq and Datalab API keys

# 5. Download IndicLID model (280 MB, once)
mkdir -p IndicLID/models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ai4bharat/IndicLID-FTN', filename='model_baseline_native.bin', local_dir='IndicLID/models')"

# 6. Start server
python main.py
# → http://127.0.0.1:8000

#Frontend Setup
cd intellidocs-frontend
npm install
npm run dev
# → http://localhost:5173

First Use
Open http://localhost:5173

Upload a PDF via the Document Ingestion panel

Wait for [index] upserted N chunks in the server console

Select your language from the dropdown

Ask questions — answers come back in your chosen language

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
Accuracy
Tested on a 42-page multilingual social media project report:

Metric	Score
Content accuracy (strict)	~62%
Content accuracy (weighted)	~67%
Hallucination rate	<5%
Language accuracy	93%+
Known limitations: heavy refusals for confirmed-negatives, retrieval misses on
some short sections, image-only content needs Chandra API (not local).

Tech Stack
Backend: FastAPI, Qdrant, sentence-transformers (BGE-small)

OCR: Datalab Chandra (image pages), pymupdf (text pages)

Retrieval: Dense (BGE) + BM25 + query expansion + section boost

LLM: Groq openai/gpt-oss-120b

Translation: AI4Bharat IndicTrans2 (200M distilled)

Persistence: MongoDB (sessions, uploads)

Frontend: React (Vite)

License
MIT (or whatever you choose)

text

---

## Step 5: Create `requirements.txt`

If you don't have one, generate it from your working environment:

```powershell
cd D:\intellidocs-backend
pip freeze > requirements.txt
Then edit the file to remove packages you don't actually need. Common bloat: transitive deps, dev tools, GPU packages you never use.

Also pin critical versions by hand to avoid future breakage:

text
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
Pinning transformers and torch matters — the AI4Bharat code breaks on newer versions.

Step 6: Initialize Git Locally
powershell
cd D:\intellidocs-backend

# Initialize the repo
git init

# Set main as the default branch
git branch -M main

# See what will be committed
git status
Look at the output carefully. If you see inteliidocs/, qdrant_local_db/, .env, or IndicLID/models/ in the "Untracked files" list — stop and fix .gitignore before proceeding.

Verify the ignore rules are working
powershell
git check-ignore -v inteliidocs/ .env qdrant_local_db/ IndicLID/models/
Each of those should return a line showing which .gitignore rule matched. If any doesn't print anything → that path is NOT ignored.