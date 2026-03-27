"""
api/main.py
COE Analytics Chatbot API — Final Version
- INVENTORY intent → Text-to-SQL → DuckDB (local) → HTML table in assistantmsg
- RAG intent       → FAISS + LangChain + Azure OpenAI (local) → citations

Run with: python api/main.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = FastAPI(title="COE Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Load RAG at startup
# ------------------------------------------------------------------------------

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
        logger.info("✅ RAG loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ RAG not loaded: {e}")

# ------------------------------------------------------------------------------
# Request Model
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    chat_id: str                        # Unfyd session key — stays same
    msg_id: Optional[str] = None        # changes when user confirms switch
    confirm_topic_switch: bool = False

# ------------------------------------------------------------------------------
# Session Store — keyed by chat_id from Unfyd Backend
# ------------------------------------------------------------------------------

session_store = {}

def get_session(chat_id: str) -> dict:
    if chat_id not in session_store:
        session_store[chat_id] = {
            "filters": {},              # inventory filters (LOB, Status etc)
            "history": [],              # inventory turn history
            "last_intent": "",          # last intent for router context
            "chat": [],                 # RAG conversation history
            "last_msg_id": None,        # track msg_id for switch detection
            "pending_switch": False,    # waiting for user confirmation
            "pending_question": None,   # question that triggered switch
            "pending_intent": None,     # new intent after switch
        }
    return session_store[chat_id]


def update_session(chat_id: str, filters: dict, history: list,
                   intent: str, chat: list,
                   last_msg_id: str = None,
                   pending_switch: bool = False,
                   pending_question: str = None,
                   pending_intent: str = None):
    current = session_store.get(chat_id, {})
    session_store[chat_id] = {
        "filters": filters,
        "history": history[-10:],
        "last_intent": intent,
        "chat": chat[-10:],
        "last_msg_id": last_msg_id or current.get("last_msg_id"),
        "pending_switch": pending_switch,
        "pending_question": pending_question,
        "pending_intent": pending_intent,
    }

# ------------------------------------------------------------------------------
# HTML Table Formatter
# ------------------------------------------------------------------------------

def df_to_html_table(df) -> str:
    """
    Convert DataFrame to HTML pipe separated table string
    for frontend rendering.
    """
    if df is None or df.empty:
        return ""

    # Preferred display columns for inventory
    display_cols = [
        c for c in [
            "Name_of_the_Model",
            "LOB",
            "Function",
            "Status",
            "Owner",
            "Time_Line",
            "ML_Non_ML",
            "Document_Availability",
        ]
        if c in df.columns
    ]

    # Fall back to all columns if none matched
    if not display_cols:
        display_cols = list(df.columns)

    df_display = df[display_cols]

    # Build header
    header = "<tr>" + "".join(
        f"<th>{col.replace('_', ' ')}</th>"
        for col in display_cols
    ) + "</tr>"

    # Build data rows
    rows = []
    for _, row in df_display.iterrows():
        data_row = "<tr>" + "".join(
            f"<td>{row[col]}</td>"
            for col in display_cols
        ) + "</tr>"
        rows.append(data_row)

    html_table = (
        "<table border='1' cellpadding='5' cellspacing='0'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    return html_table

# ------------------------------------------------------------------------------
# Response Formatter
# ------------------------------------------------------------------------------

def format_response(
    chat_id: str,
    msg_id: str,
    question: str,
    intent: str,
    answer: str,
    citations: list = None,
    is_intent_switch: bool = False
):
    return {
        "success": True,
        "error": False,
        "message": "success",
        "code": 200,
        "data": [
            {
                "chatid": chat_id,
                "msgid": msg_id,
                "intent": intent,
                "usermsg": question,
                "assistantmsg": answer,         # HTML table embedded here
                "citation": json.dumps(citations or []),
            }
        ],
        "isIntentSwitch": is_intent_switch,
        "isOpeanAIResponse": True
    }

# ------------------------------------------------------------------------------
# Main Endpoint
# ------------------------------------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):
    try:
        # ── Step 1: Handle topic switch confirmation ───────────────────────
        if request.confirm_topic_switch:
            logger.info(
                f"[{request.chat_id}] Topic switch confirmed | "
                f"old_msg_id: {session_store.get(request.chat_id, {}).get('last_msg_id')} | "
                f"new_msg_id: {request.msg_id}"
            )
            # Clear only inventory context — RAG chat preserved
            current = session_store.get(request.chat_id, {})
            update_session(
                chat_id=request.chat_id,
                filters={},                         # reset inventory filters
                history=[],                         # reset inventory history
                intent=current.get("last_intent", ""),
                chat=current.get("chat", []),       # RAG history preserved
                last_msg_id=request.msg_id,         # save new msg_id
                pending_switch=False,
                pending_question=None,
                pending_intent=None,
            )

        session = get_session(request.chat_id)

        # ── Step 2: Intent classification ─────────────────────────────────
        from rag.router import classify_intent
        decision = classify_intent(
            request.question,
            last_intent=session["last_intent"]
        )
        intent = decision.intent.value
        logger.info(
            f"[{request.chat_id}] "
            f"msg_id: {request.msg_id} | "
            f"Intent: {intent} | "
            f"Q: {request.question}"
        )

        # ── INVENTORY FLOW ─────────────────────────────────────────────────
        if intent == "INVENTORY":
            from rag.inventory import run_inventory_query

            answer, df, filters, topic_shift, parsed = run_inventory_query(
                question=request.question,
                accumulated_filters=session["filters"],
                turn_history=session["history"],
            )

            # Topic shift detected — tell frontend to show confirmation popup
            if topic_shift:
                update_session(
                    chat_id=request.chat_id,
                    filters=session["filters"],     # keep old filters for now
                    history=session["history"],
                    intent=session["last_intent"],  # keep old intent for now
                    chat=session["chat"],
                    last_msg_id=request.msg_id,
                    pending_switch=True,
                    pending_question=request.question,
                    pending_intent=intent,
                )
                return format_response(
                    chat_id=request.chat_id,
                    msg_id=request.msg_id,
                    question=request.question,
                    intent="INVENTORY",
                    answer=(
                        "It looks like you are switching to a different topic. "
                        "Your previous context will be cleared. Do you want to continue?"
                    ),
                    is_intent_switch=True
                )

            # Convert df to HTML table and append to answer
            if df is not None and not df.empty:
                html_table = df_to_html_table(df)
                full_answer = f"{answer}<br><br>{html_table}"
            else:
                full_answer = answer

            # Update inventory context
            update_session(
                chat_id=request.chat_id,
                filters=filters,
                history=session["history"] + [{
                    "question": request.question,
                    "filters": filters,
                    "result_count": len(df) if df is not None else 0,
                    "query_type": parsed.get("query_type", "list"),
                    "is_followup": parsed.get("is_followup", False),
                }],
                intent=intent,
                chat=session["chat"],               # RAG chat unchanged
                last_msg_id=request.msg_id,
                pending_switch=False,
            )

            return format_response(
                chat_id=request.chat_id,
                msg_id=request.msg_id,
                question=request.question,
                intent="INVENTORY",
                answer=full_answer,                 # HTML table inside answer
                citations=[],
            )

        # ── RAG FLOW ───────────────────────────────────────────────────────
        else:
            if chain is None:
                raise HTTPException(
                    status_code=503,
                    detail="RAG not available. Run ingest first."
                )

            from rag.chain import run_rag

            answer, docs = run_rag(
                chain=chain,
                retriever=retriever,
                question=request.question,
                history=session["chat"]
            )

            citations = [
                {
                    "title": doc.metadata.get("source", "").split("/")[-1],
                    "pages": str(doc.metadata.get("page", "")),
                    "url": doc.metadata.get("source", "")
                }
                for doc in docs
            ]

            # Update RAG chat history
            update_session(
                chat_id=request.chat_id,
                filters=session["filters"],         # inventory filters unchanged
                history=session["history"],         # inventory history unchanged
                intent=intent,
                chat=session["chat"] + [(request.question, answer)],
                last_msg_id=request.msg_id,
                pending_switch=False,
            )

            return format_response(
                chat_id=request.chat_id,
                msg_id=request.msg_id,
                question=request.question,
                intent="RAG",
                answer=answer,
                citations=citations,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_ready": chain is not None,
        "rag_engine": "FAISS + LangChain (local)",
        "inventory_engine": "DuckDB (local)",
        "model": settings.llm_model
    }

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
