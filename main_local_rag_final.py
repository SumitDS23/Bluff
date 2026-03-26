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

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = FastAPI(title="COE Chatbot API - UNFYD Compatible")

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

        logger.info("RAG loaded")
    except Exception as e:
        logger.warning(f"RAG not loaded: {e}")

# ------------------------------------------------------------------------------
# Request Model
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str
    confirm_topic_switch: bool = False

# ------------------------------------------------------------------------------
# Session Store
# ------------------------------------------------------------------------------

session_store = {}

def get_session(session_id):
    if session_id not in session_store:
        session_store[session_id] = {
            "filters": {},
            "history": [],
            "last_intent": "",
            "chat": []
        }
    return session_store[session_id]

def update_session(session_id, filters, history, intent, chat):
    session_store[session_id] = {
        "filters": filters,
        "history": history[-10:],
        "last_intent": intent,
        "chat": chat[-10:]
    }

# ------------------------------------------------------------------------------
# UNFYD Response Formatter
# ------------------------------------------------------------------------------

def format_unfyd_response(
    request,
    intent,
    answer,
    citations=None,
    is_intent_switch=False
):
    return {
        "success": True,
        "error": False,
        "message": "success",
        "code": 200,
        "data": [
            {
                "userid": 0,
                "chatid": request.session_id,
                "msgid": str(hash(request.question))[:8],
                "intent": intent,
                "usermsg": request.question,
                "systemmsg": "",
                "assistantmsg": answer,
                "browsertime": datetime.utcnow().isoformat(),
                "likedislikestatus": None,
                "citation": json.dumps(citations or []),  # IMPORTANT
                "reference": None
            }
        ],
        "TokenIndex": {"token": ""},
        "isIntentSwitch": is_intent_switch,
        "isOpeanAIResponse": True
    }

# ------------------------------------------------------------------------------
# Main Endpoint
# ------------------------------------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):
    try:
        session = get_session(request.session_id)

        # Step 1: Intent classification
        from rag.router import classify_intent
        decision = classify_intent(request.question, last_intent=session["last_intent"])
        intent = decision.intent.value

        logger.info(f"[{request.session_id}] Intent: {intent}")

        # ----------------------------------------------------------------------
        # INVENTORY FLOW
        # ----------------------------------------------------------------------
        if intent == "INVENTORY":
            from rag.inventory import run_inventory_query

            answer, df, filters, topic_shift, parsed = run_inventory_query(
                question=request.question,
                accumulated_filters=session["filters"],
                turn_history=session["history"],
            )

            if topic_shift:
                return format_unfyd_response(
                    request,
                    intent="INVENTORY",
                    answer="It seems like you are switching context. Please confirm.",
                    is_intent_switch=True
                )

            update_session(
                request.session_id,
                filters,
                session["history"] + [request.question],
                intent,
                session["chat"]
            )

            return format_unfyd_response(
                request,
                intent="INVENTORY",
                answer=answer,
                citations=[]
            )

        # ----------------------------------------------------------------------
        # RAG FLOW
        # ----------------------------------------------------------------------
        else:
            if chain is None:
                raise HTTPException(status_code=503, detail="RAG not available")

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

            update_session(
                request.session_id,
                session["filters"],
                session["history"],
                intent,
                session["chat"] + [(request.question, answer)]
            )

            return format_unfyd_response(
                request,
                intent="RAG",
                answer=answer,
                citations=citations
            )

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
        "model": settings.llm_model
    }


# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)