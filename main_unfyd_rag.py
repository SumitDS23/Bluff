"""
api/main_unfyd_rag.py
VERSION 2 — Local Inventory + Vendor Unfyd RAG
- INVENTORY intent → Text-to-SQL → DuckDB (local)
- RAG intent       → Unfyd Enlighten API (vendor)

Run with: python api/main_unfyd_rag.py

Required .env additions:
    UNFYD_BASE_URL=https://enlighten_uat.insideabc.com/unfydseek-api/api/v2/openai
    UNFYD_USERNAME=pratikp
    UNFYD_AES_KEY=<32-byte key from vendor>
    UNFYD_AES_IV=<16-byte IV from vendor>
    UNFYD_PROCESS_ID=1
    UNFYD_LOB_ID=1
    UNFYD_DEPARTMENT_ID=1
    UNFYD_USER_ID=11
    UNFYD_USER_DEPARTMENT_TYPE=operations
"""

import sys
import json
import logging
import requests
import base64
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
    title="COE Analytics Chatbot API — Unfyd RAG",
    description="Inventory via DuckDB (local) + RAG via Unfyd Enlighten API",
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
# Unfyd API config — loaded from .env
# ---------------------------------------------------------------------------

UNFYD_BASE_URL          = settings.unfyd_base_url
UNFYD_USERNAME          = settings.unfyd_username
UNFYD_AES_KEY           = settings.unfyd_aes_key.encode()      # 32 bytes
UNFYD_AES_IV            = settings.unfyd_aes_iv.encode()       # 16 bytes
UNFYD_PROCESS_ID        = settings.unfyd_process_id
UNFYD_LOB_ID            = settings.unfyd_lob_id
UNFYD_DEPARTMENT_ID     = settings.unfyd_department_id
UNFYD_USER_ID           = settings.unfyd_user_id
UNFYD_USER_DEPT_TYPE    = settings.unfyd_user_department_type

# ---------------------------------------------------------------------------
# AES-256-CBC Encryption / Decryption helpers
# ---------------------------------------------------------------------------

def encrypt_payload(data: dict) -> str:
    """Encrypt a dict to AES-256-CBC Base64 string."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad

        cipher = AES.new(UNFYD_AES_KEY, AES.MODE_CBC, UNFYD_AES_IV)
        json_bytes = json.dumps(data).encode("utf-8")
        encrypted = cipher.encrypt(pad(json_bytes, AES.block_size))
        return base64.b64encode(encrypted).decode("utf-8")
    except ImportError:
        raise RuntimeError("pycryptodome not installed. Run: pip install pycryptodome")


def decrypt_payload(encrypted_str: str) -> dict:
    """Decrypt AES-256-CBC Base64 string to dict."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad

        cipher = AES.new(UNFYD_AES_KEY, AES.MODE_CBC, UNFYD_AES_IV)
        decoded = base64.b64decode(encrypted_str)
        decrypted = unpad(cipher.decrypt(decoded), AES.block_size)
        return json.loads(decrypted.decode("utf-8"))
    except ImportError:
        raise RuntimeError("pycryptodome not installed. Run: pip install pycryptodome")


# ---------------------------------------------------------------------------
# Unfyd token management
# ---------------------------------------------------------------------------

def get_bearer_token() -> str:
    """Call Unfyd /generatetoken and return bearer token."""
    try:
        payload = encrypt_payload({"username": UNFYD_USERNAME})
        response = requests.post(
            f"{UNFYD_BASE_URL}/generatetoken",
            json={"data": payload},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        decrypted = decrypt_payload(response.json()["data"])
        token = decrypted["data"]["token"]
        logger.info("✅ Unfyd bearer token obtained")
        return token
    except Exception as e:
        logger.error(f"❌ Failed to get Unfyd token: {e}")
        raise


# ---------------------------------------------------------------------------
# Unfyd RAG query
# ---------------------------------------------------------------------------

def call_unfyd_rag(question: str, token: str,
                   chat_id: Optional[str] = None,
                   is_followup: bool = False) -> tuple:
    """
    Call Unfyd /azureBlobAPIResponse.
    Returns: (answer, citations, new_chat_id, refreshed_token)
    """
    try:
        payload = encrypt_payload({
            "query": question,
            "userID": str(UNFYD_USER_ID),
            "chatID": chat_id,
            "isFollowUp": is_followup,
            "isChatIntentSwitch": False,
            "processID": UNFYD_PROCESS_ID,
            "LOB": UNFYD_LOB_ID,
            "departmentID": UNFYD_DEPARTMENT_ID,
            "browserTime": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            "isfaq": True,
            "userDepartmentType": UNFYD_USER_DEPT_TYPE,
            "isThemeFirstChat": False,
            "themeID": None,
            "theme": None,
        })

        response = requests.post(
            f"{UNFYD_BASE_URL}/azureBlobAPIResponse",
            json={"data": payload, "isencryptedres": True},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=60,     # LLM responses can be slow
        )
        response.raise_for_status()

        decrypted = decrypt_payload(response.json()["data"])

        # Extract fields from Unfyd response
        data = decrypted.get("data", [{}])[0]
        answer          = data.get("assistantmsg", "No answer returned.")
        new_chat_id     = data.get("chatid")
        raw_citations   = data.get("citation", "[]")
        refreshed_token = decrypted.get("TokenIndex", {}).get("token", token)

        # Parse citations (Unfyd returns them as JSON string)
        try:
            citations = json.loads(raw_citations) if raw_citations else []
        except Exception:
            citations = []

        # Normalise citation format to match our standard
        normalised_citations = [
            {
                "title": c.get("title", ""),
                "pages": c.get("pages", ""),
                "url":   c.get("url", ""),
            }
            for c in citations
        ]

        logger.info(f"✅ Unfyd RAG response received | chatid: {new_chat_id}")
        return answer, normalised_citations, new_chat_id, refreshed_token

    except Exception as e:
        logger.error(f"❌ Unfyd RAG call failed: {e}")
        raise


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
    citations: Optional[list] = None
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
            # Inventory context
            "accumulated_filters": {},
            "turn_history": [],
            "last_intent": "",
            "chat_history_tuples": [],
            # Unfyd RAG context
            "bearer_token": None,       # refreshed after every RAG call
            "unfyd_chat_id": None,      # links multi-turn RAG conversation
        }
    return session_store[session_id]


def update_session(session_id: str, accumulated_filters: dict,
                   turn_history: list, last_intent: str,
                   chat_history_tuples: list,
                   bearer_token: Optional[str] = None,
                   unfyd_chat_id: Optional[str] = None):
    current = session_store.get(session_id, {})
    session_store[session_id] = {
        "accumulated_filters": accumulated_filters,
        "turn_history": turn_history[-10:],
        "last_intent": last_intent,
        "chat_history_tuples": chat_history_tuples[-10:],
        "bearer_token": bearer_token or current.get("bearer_token"),
        "unfyd_chat_id": unfyd_chat_id or current.get("unfyd_chat_id"),
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
    VERSION 2 — Local Inventory + Unfyd RAG

    INVENTORY → DuckDB Text-to-SQL (local)
    RAG       → Unfyd Enlighten API (vendor)

    Both return the same unified QueryResponse to the frontend.
    Frontend does not know or care which engine answered.
    """
    try:
        session = get_session(request.session_id)
        last_intent = session["last_intent"]

        if request.confirm_topic_switch:
            clear_session(request.session_id)
            session = get_session(request.session_id)

        accumulated_filters     = session["accumulated_filters"]
        turn_history            = session["turn_history"]
        chat_history_tuples     = session["chat_history_tuples"]
        bearer_token            = session["bearer_token"]
        unfyd_chat_id           = session["unfyd_chat_id"]

        # ── Step 1: Intent classification (unchanged) ─────────────────────
        from rag.router import classify_intent
        decision = classify_intent(request.question, last_intent=last_intent)
        intent = decision.intent.value
        logger.info(f"[{request.session_id}] Intent: {intent} | Q: {request.question}")

        # ── INVENTORY PATH — DuckDB local (unchanged) ─────────────────────
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
            update_session(
                request.session_id,
                updated_filters,
                turn_history_updated,
                intent,
                chat_history_tuples,
                bearer_token=bearer_token,
                unfyd_chat_id=unfyd_chat_id,
            )

            return QueryResponse(
                intent="INVENTORY",
                answer=answer,
                data=data_json,
                sources=None,
                citations=None,
                session_id=request.session_id,
                topic_shift=False,
            )

        # ── RAG PATH — Unfyd Enlighten API ────────────────────────────────
        else:
            # Step A: Get token if we don't have one yet
            if not bearer_token:
                bearer_token = get_bearer_token()

            # Step B: Is this a follow-up RAG question?
            is_followup = (last_intent == "RAG" and unfyd_chat_id is not None)

            # Step C: Call Unfyd
            answer, citations, new_chat_id, refreshed_token = call_unfyd_rag(
                question=request.question,
                token=bearer_token,
                chat_id=unfyd_chat_id if is_followup else None,
                is_followup=is_followup,
            )

            # Step D: Save refreshed token + chat_id for next turn
            sources = list({c["title"] for c in citations if c.get("title")})

            update_session(
                request.session_id,
                accumulated_filters,
                turn_history,
                intent,
                chat_history_tuples,
                bearer_token=refreshed_token,   # always refresh
                unfyd_chat_id=new_chat_id,       # link next turn
            )

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
        "version": "UNFYD RAG",
        "rag_engine": "Unfyd Enlighten API (vendor)",
        "inventory_engine": "DuckDB (local)",
        "model": settings.llm_model,
        "inventory": settings.inventory_path,
        "unfyd_base_url": UNFYD_BASE_URL,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main_unfyd_rag:app", host="0.0.0.0", port=8000, reload=False)
