"""
api/main_local_rag.py
VERSION 1 — Full Local Control
- INVENTORY intent → Text-to-SQL → DuckDB (local)
- RAG intent       → FAISS + LangChain + Azure OpenAI (local)

Run with: python api/main_local_rag.py
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="COE Analytics Chatbot API — Local RAG",
    description="Inventory via DuckDB + RAG via local FAISS/LangChain",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load local FAISS RAG chain once at startup
# ---------------------------------------------------------------------------

retriever = None
chain = None

@app.on_event("startup")
def load_rag():
    global retriever, chain
    try:
        from rag.retriever import load_retriever
        from rag.chain import build_chain
        retriever = load_retriever()
        chain = build_chain(retriever)
        logger.info("✅ Local FAISS RAG chain loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"⚠️ RAG chain not loaded: {e}. Run ingest/ingest.py first.")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    session_id: str
    conversation_history: list[ConversationMessage] = []
    confirm_topic_switch: bool = False


class QueryResponse(BaseModel):
    intent: str
    answer: Optional[str] = None
    data: Optional[list] = None
    sources: Optional[list] = None
    citations: Optional[list] = None        # document + page citations
    session_id: str
    handoff: bool = False
    topic_shift: bool = False
    topic_shift_message: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

session_store: dict = {}


def get_session(session_id: str) -> dict:
    if session_id not in session_store:
        session_store[session_id] = {
            "accumulated_filters": {},
            "turn_history": [],
            "last_intent": "",
            "chat_history_tuples": [],
        }
    return session_store[session_id]


def update_session(session_id: str, accumulated_filters: dict,
                   turn_history: list, last_intent: str,
                   chat_history_tuples: list):
    session_store[session_id] = {
        "accumulated_filters": accumulated_filters,
        "turn_history": turn_history[-10:],
        "last_intent": last_intent,
        "chat_history_tuples": chat_history_tuples[-10:],
    }


def clear_session(session_id: str):
    if session_id in session_store:
        del session_store[session_id]


# ---------------------------------------------------------------------------
# Main query endpoint
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    VERSION 1 — Local RAG + Local Inventory

    INVENTORY → DuckDB Text-to-SQL (local)
    RAG       → FAISS + LangChain + Azure OpenAI (local)
    """
    try:
        session = get_session(request.session_id)
        last_intent = session["last_intent"]

        if request.confirm_topic_switch:
            clear_session(request.session_id)
            session = get_session(request.session_id)

        accumulated_filters = session["accumulated_filters"]
        turn_history = session["turn_history"]
        chat_history_tuples = session["chat_history_tuples"]

        # ── Step 1: Intent classification ────────────────────────────────
        from rag.router import classify_intent
        decision = classify_intent(request.question, last_intent=last_intent)
        intent = decision.intent.value
        logger.info(f"[{request.session_id}] Intent: {intent} | Q: {request.question}")

        # ── INVENTORY PATH — DuckDB (local) ───────────────────────────────
        if intent == "INVENTORY":
            from rag.inventory import run_inventory_query

            answer, result_df, updated_filters, topic_shift_detected, parsed = run_inventory_query(
                question=request.question,
                accumulated_filters=accumulated_filters,
                turn_history=turn_history,
            )

            if topic_shift_detected:
                new_lob = parsed.get("new_filters", {}).get("LOB", "a new topic")
                old_lob = accumulated_filters.get("LOB", "previous context")
                return QueryResponse(
                    intent="INVENTORY",
                    session_id=request.session_id,
                    topic_shift=True,
                    topic_shift_message=(
                        f"You were exploring {old_lob} models. "
                        f"Did you want to switch to {new_lob}? "
                        f"Resend with confirm_topic_switch=true to proceed."
                    ),
                )

            data_json = None
            if result_df is not None and not result_df.empty:
                data_json = result_df.to_dict(orient="records")

            turn_history_updated = turn_history + [{
                "question": request.question,
                "filters": updated_filters,
                "result_count": len(result_df) if result_df is not None else 0,
                "query_type": parsed.get("query_type", "list"),
                "is_followup": parsed.get("is_followup", False),
            }]
            update_session(request.session_id, updated_filters,
                           turn_history_updated, intent, chat_history_tuples)

            return QueryResponse(
                intent="INVENTORY",
                answer=answer,
                data=data_json,
                sources=None,
                citations=None,
                session_id=request.session_id,
                topic_shift=False,
            )

        # ── RAG PATH — Local FAISS + LangChain ───────────────────────────
        else:
            if chain is None or retriever is None:
                raise HTTPException(
                    status_code=503,
                    detail="RAG chain not available. Run ingest/ingest.py first."
                )

            from rag.chain import run_rag

            answer, source_docs = run_rag(
                chain=chain,
                retriever=retriever,
                question=request.question,
                history=chat_history_tuples,
            )

            # Build citations from FAISS document metadata
            citations = [
                {
                    "title": doc.metadata.get("source", "").split("/")[-1],
                    "pages": str(doc.metadata.get("page", "")),
                    "url": doc.metadata.get("source", "")
                }
                for doc in source_docs
            ]

            # Deduplicated source file names
            sources = list({c["title"] for c in citations})

            chat_history_updated = chat_history_tuples + [
                (request.question, answer)
            ]
            update_session(request.session_id, accumulated_filters,
                           turn_history, intent, chat_history_updated)

            return QueryResponse(
                intent="RAG",
                answer=answer,
                data=None,
                sources=sources,
                citations=citations,
                session_id=request.session_id,
                topic_shift=False,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "LOCAL RAG",
        "rag_engine": "FAISS + LangChain (local)",
        "inventory_engine": "DuckDB (local)",
        "model": settings.llm_model,
        "inventory": settings.inventory_path,
        "rag_ready": chain is not None,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main_local_rag:app", host="0.0.0.0", port=8000, reload=False)
