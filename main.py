# main.py
from contextlib import asynccontextmanager
import logging
import traceback
import uuid
import uvicorn

from fastapi import (
    FastAPI, Request, UploadFile, File, Form, HTTPException, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from services.vector_db import VectorStoreManager
from services.rag_pipeline import GroundedRAGEngine
from services.ocr_engine import DocumentProcessor
from services.indic_translator import IndicTranslator
from services.database import MongoStore

# ---------------- Logging ----------------
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("pymongo.topology").setLevel(logging.WARNING)
logging.getLogger("motor").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("intellidocs")

# ---------------- Global state ----------------
vector_manager: VectorStoreManager = None
rag_engine: GroundedRAGEngine = None
translator: IndicTranslator = None
mongo: MongoStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_manager, rag_engine, translator, mongo
    logger.info("[IntelliDocs Core] Starting up...")

    vector_manager = VectorStoreManager()
    rag_engine = GroundedRAGEngine()
    translator = IndicTranslator()

    mongo = MongoStore(settings.MONGO_URI, settings.MONGO_DB)
    await mongo.ensure_indexes()

    logger.info(f"[IntelliDocs Core] LLM target: {settings.LLM_MODEL}")
    yield
    logger.info("[IntelliDocs Core] Shutdown complete.")


app = FastAPI(title="BhashaRAG Engine Core", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"UNHANDLED on {request.method} {request.url.path}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": type(e).__name__,
                "detail": str(e),
                "trace": traceback.format_exc().splitlines()[-10:],
            },
        )


ENUMERATIVE_MARKERS = [
    "how", "where", "which", "list", "all", "enumerate",
    "explain", "describe", "what are", "name the",
]


def _is_enumerative(query: str) -> bool:
    q = query.lower()
    return any(marker in q for marker in ENUMERATIVE_MARKERS)


# ---------------- Endpoints ----------------

ALLOWED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")
ALLOWED_CONTENT_TYPES = (
    "application/pdf",
    "image/png",
    "image/jpeg",
    "application/octet-stream",  # some browsers send this
)


@app.post("/api/v1/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"[/upload] filename={file.filename!r} content_type={file.content_type!r}")

    fname = (file.filename or "").strip().lower()
    ext_ok = fname.endswith(ALLOWED_EXTENSIONS)
    type_ok = file.content_type in ALLOWED_CONTENT_TYPES

    if not (ext_ok or type_ok):
        logger.warning(f"[/upload] rejected: filename={file.filename!r} type={file.content_type!r}")
        raise HTTPException(
            400,
            f"Invalid file type. filename={file.filename!r}, "
            f"content_type={file.content_type!r}",
        )

    try:
        file_bytes = await file.read()
        logger.info(f"[/upload] read {len(file_bytes)} bytes")
        doc_id = str(uuid.uuid4())

        parsed_data = await DocumentProcessor.extract_layout_and_text(
            file_bytes, file.filename
        )
        logger.info(f"[/upload] parsed {len(parsed_data)} pages")

        await vector_manager.index_document(doc_id, parsed_data)
        logger.info(f"[/upload] indexed doc {doc_id}")

        await mongo.save_upload(
            document_id=doc_id,
            filename=file.filename,
            pages_processed=len(parsed_data),
            size_bytes=len(file_bytes),
        )

        return {
            "status": "success",
            "document_id": doc_id,
            "pages_processed": len(parsed_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[UPLOAD FAILED]")
        raise HTTPException(500, f"Upload failed: {e}")


@app.get("/api/v1/uploads")
async def list_uploads():
    return {"uploads": await mongo.list_uploads()}


@app.post("/api/v1/sessions")
async def create_session():
    session_id = str(uuid.uuid4())
    await mongo.create_session(session_id)
    return {"session_id": session_id}


@app.get("/api/v1/sessions")
async def list_sessions():
    return {"sessions": await mongo.list_sessions()}


@app.get("/api/v1/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return {"messages": await mongo.get_history(session_id)}


@app.delete("/api/v1/sessions/{session_id}/messages")
async def clear_session_messages(session_id: str):
    deleted = await mongo.clear_history(session_id)
    return {"status": "cleared", "deleted_count": deleted}


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    await mongo.delete_session(session_id)
    return {"status": "deleted"}


@app.post("/api/v1/chat")
async def process_grounded_query(
    query: str = Form(...),
    lang: str = Form("auto"),  # Default changed to "auto"
    session_id: str = Form(None),
):
    if vector_manager is None or rag_engine is None or translator is None:
        raise HTTPException(503, "Engine not initialized.")

    if not session_id or session_id in ("null", "undefined", ""):
        session_id = str(uuid.uuid4())
        await mongo.create_session(session_id)
        logger.info(f"[/chat] auto-created session {session_id}")

    # ----- STEP 0: AUTOMATIC LANGUAGE DETECTION -----
    if lang == "auto":
        lang = translator.detect_language(query)
        logger.info(f"[/chat] Auto-detected language: {lang}")

    logger.info(f"[/chat] session={session_id} final_lang={lang} query={query!r}")

    try:
        await mongo.save_message(session_id, "user", query)
    except Exception as e:
        logger.warning(f"[/chat] failed to save user message: {e}")

    # Step 1: translate query to English (if not already English)
    english_query = query
    if lang != "en":
        try:
            english_query = translator.translate(query, src_lang=lang, tgt_lang="en")
            logger.info(f"[/chat] translated query to EN: {english_query!r}")
        except Exception as e:
            logger.exception("[/chat] query translation failed")
            return {
                "answer": f"Translation error: {e}",
                "citations": [],
                "retrieved_sources": [],
                "session_id": session_id,
            }

    # Step 2: retrieval
    search_limit = 14 if _is_enumerative(english_query) else 10
    try:
        matched_contexts = await vector_manager.hybrid_search(
            english_query, limit=search_limit
        )
    except Exception as e:
        logger.exception("[/chat] hybrid_search failed")
        raise HTTPException(500, f"Retrieval failed: {e}")

    if not matched_contexts:
        answer = "Information not located in document archives."
        if lang != "en":
            try:
                answer = translator.translate(answer, src_lang="en", tgt_lang=lang)
            except Exception:
                pass
        try:
            await mongo.save_message(session_id, "assistant", answer)
        except Exception:
            pass
        return {
            "answer": answer,
            "citations": [],
            "retrieved_sources": [],
            "session_id": session_id,
        }

    # Step 3: generate answer (English)
    try:
        execution_payload = await rag_engine.generate_answer(
            english_query, matched_contexts
        )
    except Exception as e:
        logger.exception("[/chat] generate_answer failed")
        raise HTTPException(500, f"Generation failed: {e}")

    # None guard
    if execution_payload is None:
        logger.error("[/chat] rag_engine.generate_answer returned None")
        english_answer = "Information not located in document archives."
        citations = []
    else:
        english_answer = execution_payload.get("answer", "")
        citations = execution_payload.get("citations", [])

    # Step 4: translate answer back to user's language (if not English)
    final_answer = english_answer
    if lang != "en" and english_answer:
        try:
            final_answer = translator.translate(
                english_answer,
                src_lang="en",
                tgt_lang=lang,
                protect_citations=True,
            )
            logger.info(f"[/chat] translated answer back to {lang}")
        except Exception as e:
            logger.exception(f"[/chat] answer translation FAILED: {e}")
    else:
        logger.info(f"[/chat] skipping answer translation (lang={lang})")

    # Step 5: persist assistant response
    try:
        await mongo.save_message(
            session_id, "assistant", final_answer,
            citations=citations, sources=matched_contexts,
        )
    except Exception as e:
        logger.warning(f"[/chat] failed to save assistant response: {e}")

    return {
        "answer": final_answer,
        "citations": citations,
        "retrieved_sources": matched_contexts,
        "session_id": session_id,
    }
# ---------------- Debug endpoints ----------------

@app.get("/api/v1/debug/index-stats")
async def index_stats():
    """Return Qdrant collection stats — points count, config."""
    try:
        info = vector_manager.client.get_collection(
            collection_name=vector_manager.collection_name
        )
        return {
            "collection": vector_manager.collection_name,
            "points_count": info.points_count,
            "config": str(info.config.params.vectors),
        }
    except Exception as e:
        return {"error": str(e), "collection": vector_manager.collection_name}


@app.get("/api/v1/debug/search")
async def debug_search(q: str, limit: int = 5):
    """Run a raw hybrid_search and return top results."""
    try:
        results = await vector_manager.hybrid_search(q, limit=limit)
        return {
            "query": q,
            "count": len(results),
            "results": [
                {"page": r["page"], "score": r["score"],
                 "section": r.get("section", ""),
                 "text": r["text"][:500]}
                for r in results
            ],
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/debug/section-map")
async def section_map():
    """Debug: inspect the in-memory section map."""
    if not hasattr(vector_manager, "_section_map"):
        return {"exists": False, "reason": "Attribute not set — index_document never ran"}
    sm = vector_manager._section_map
    return {
        "exists": True,
        "num_keys": len(sm),
        "keys_sample": sorted(sm.keys())[:50],
    } 

@app.get("/api/v1/debug/list-sections")
async def list_sections():
    """List all unique section values in the index."""
    try:
        sections = set()
        offset = None
        while True:
            points, offset = vector_manager.client.scroll(
                collection_name=vector_manager.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                s = p.payload.get("section", "")
                if s:
                    sections.add(s)
            if offset is None:
                break
        return {"sections": sorted(sections), "count": len(sections)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v1/debug/reindex")
async def debug_reindex():
    import os
    pdf_path = r"D:\intellidocs-backend\Social Media Project Report.pdf"
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    # FORCE delete — no silent fallback
    try:
        vector_manager.client.delete_collection(
            collection_name=vector_manager.collection_name
        )
        print(f"[reindex] DELETED collection {vector_manager.collection_name}")
    except Exception as e:
        print(f"[reindex] delete_collection failed: {e}")
        return {"error": f"Could not delete collection: {e}"}

    # Verify it's actually gone
    try:
        vector_manager.client.get_collection(vector_manager.collection_name)
        return {"error": "Collection still exists after delete — DB is locked"}
    except Exception:
        print(f"[reindex] confirmed collection is gone")

    # Recreate
    vector_manager._ensure_collection()
    print(f"[reindex] recreated collection {vector_manager.collection_name}")

    # Reset section map and BM25 cache
    if hasattr(vector_manager, "_section_map"):
        vector_manager._section_map = {}
    if hasattr(vector_manager, "_bm25"):
        vector_manager._bm25 = None
        vector_manager._bm25_ids = []
        vector_manager._bm25_payloads = []

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    parsed = await DocumentProcessor.extract_layout_and_text(
        file_bytes, os.path.basename(pdf_path)
    )
    await vector_manager.index_document("debug-reindex", parsed)
    return {"status": "ok", "pages": len(parsed)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)