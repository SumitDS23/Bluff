"""
api/main.py
COE Analytics Chatbot API — Final Version

Key improvements:
- Universal Query Rewriter (LLM-based) used by both INVENTORY and RAG
- Cross-intent context bridge (resolves pronouns across intents)
- Cross-encoder reranking for RAG (in chain.py)
- Strict no-blending RAG prompt (in chain.py)
- Data sanitization on inventory load (in inventory.py)
- Topic switch via Unfyd's isFollowUp flag (no brittle word-overlap)

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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="COE Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global exception handler
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled Error: {exc}")
    return JSONResponse(
        status_code=500,
        content=format_error_response("Internal Server Error", 500)
    )

# -----------------------------------------------------------------------------
# Load RAG at startup
# -----------------------------------------------------------------------------
retriever   = None
chain       = None
vectorstore = None

@app.on_event("startup")
def load_rag():
    global retriever, chain, vectorstore
    try:
        from rag.retriever import load_retriever
        from rag.chain import build_chain
        retriever, vectorstore = load_retriever()
        chain = build_chain(retriever)
        logger.info("RAG loaded successfully")
    except Exception as e:
        logger.warning(f"RAG not loaded: {e}")

# -----------------------------------------------------------------------------
# Request Model — Unfyd payload structure
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query:              str
    userID:             str
    chatID:             str
    isFollowUp:         bool          = False
    isChatIntentSwitch: bool          = False
    processID:          Optional[int] = None
    LOB:                Optional[int] = None
    departmentID:       Optional[int] = None
    browserTime:        Optional[str] = None
    isfaq:              Optional[bool] = None
    userDepartmentType: Optional[str] = None
    isThemeFirstChat:   Optional[bool] = None
    themeID:            Optional[int]  = None
    theme:              Optional[str]  = None

# -----------------------------------------------------------------------------
# Session Store
# -----------------------------------------------------------------------------
session_store = {}

def get_session(chat_id: str) -> dict:
    if chat_id not in session_store:
        session_store[chat_id] = {
            "filters":         {},   # inventory filters
            "history":         [],   # inventory turn history
            "last_intent":     "",
            "chat":            [],   # RAG history (tuples)
            "last_subjects":   [],   # entities for cross-intent resolution
            "last_offer":      None, # tracks what bot offered (for yes/no handling)
        }
    return session_store[chat_id]

def update_session(chat_id: str, filters: dict, history: list,
                   intent: str, chat: list, last_subjects: list = None,
                   last_offer: dict = None):
    existing = session_store.get(chat_id, {})
    session_store[chat_id] = {
        "filters":         filters,
        "history":         history[-10:],
        "last_intent":     intent,
        "chat":            chat[-10:],
        "last_subjects":   (last_subjects or existing.get("last_subjects", []))[-5:],
        "last_offer":      last_offer,  # None or {"type": "list_models", "data": {...}}
    }

def clear_session(chat_id: str):
    if chat_id in session_store:
        del session_store[chat_id]
        logger.info(f"[{chat_id}] Session fully cleared")

# -----------------------------------------------------------------------------
# Citation helpers
# -----------------------------------------------------------------------------
def clean_citation_source(source: str):
    if not source:
        return "", ""
    filename = source.split("\\")[-1].split("/")[-1]
    if source.startswith(("D:\\", "C:\\", "/")):
        return filename, ""
    return filename, source

def deduplicate_citations(citations: list) -> list:
    unique = []
    seen   = set()
    for c in citations:
        key = (c["title"], c["pages"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# -----------------------------------------------------------------------------
# HTML Table Formatter
# -----------------------------------------------------------------------------
def df_to_html_table(df) -> str:
    if df is None or df.empty:
        return ""

    cols = list(df.columns)
    if "model_count" in cols:
        display_cols = cols
    else:
        display_cols = [
            c for c in [
                "Name_of_the_Model", "LOB", "Function", "Status",
                "Owner", "Time_Line", "ML_Non_ML", "Document_Availability",
            ]
            if c in cols
        ]
        if not display_cols:
            display_cols = cols

    df_display = df[display_cols].copy()

    for col in display_cols:
        df_display[col] = (
            df_display[col]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
            .str.strip()
        )

    header = "<tr>" + "".join(
        f"<th>{col.replace('_', ' ')}</th>"
        for col in display_cols
    ) + "</tr>"

    rows = []
    for _, row in df_display.iterrows():
        data_row = "<tr>" + "".join(
            f"<td>{row[col]}</td>"
            for col in display_cols
        ) + "</tr>"
        rows.append(data_row)

    return (
        "<table border='1' cellpadding='5' cellspacing='0'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

# -----------------------------------------------------------------------------
# Cross-intent subject extraction
# -----------------------------------------------------------------------------
def extract_subjects_from_df(df) -> list:
    """Extract model names from inventory results for cross-intent context."""
    if df is None or df.empty:
        return []
    if "Name_of_the_Model" in df.columns:
        return df["Name_of_the_Model"].head(5).tolist()
    return []

# -----------------------------------------------------------------------------
# Response Formatters
# -----------------------------------------------------------------------------
def format_response(chat_id, msg_id, question, intent, answer,
                     citations=None, is_intent_switch=False, user_id=0,
                     usage=None):
    if is_intent_switch:
        return {
            "success":           True,
            "error":             False,
            "message":           "Intent switch detected",
            "code":              200,
            "data":              [],
            "TokenIndex":        {"token": ""},
            "isIntentSwitch":    True,
            "isOpeanAIResponse": False,
            "usage": {
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "total_tokens":      0,
            },
        }

    # Default usage if not provided
    if usage is None:
        usage = {
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        }

    return {
        "success": True,
        "error":   False,
        "message": "success",
        "code":    200,
        "data": [{
            "userid":            user_id,
            "chatid":            chat_id,
            "msgid":             msg_id,
            "intent":            intent,
            "usermsg":           question,
            "systemmsg":         "",
            "assistantmsg":      answer,
            "browsertime":       datetime.utcnow().isoformat(),
            "likedislikestatus": None,
            "citation":          json.dumps(citations or []),
            "reference":         None,
        }],
        "TokenIndex":        {"token": ""},
        "isIntentSwitch":    False,
        "isOpeanAIResponse": True,
        "usage": usage,
    }

def format_error_response(message: str, code: int):
    return {
        "success":           False,
        "error":             True,
        "message":           message,
        "code":              code,
        "data":              [],
        "TokenIndex":        {"token": ""},
        "isIntentSwitch":    False,
        "isOpeanAIResponse": False,
    }

# -----------------------------------------------------------------------------
# Main Endpoint
# -----------------------------------------------------------------------------
@app.post("/query")
def query(request: QueryRequest):
    try:
        question       = request.query
        chat_id        = request.chatID
        msg_id         = str(abs(hash(question)))[:8]
        confirm_switch = request.isChatIntentSwitch
        is_followup    = request.isFollowUp
        user_id        = int(request.userID) if request.userID else 0

        logger.info(
            f"[{chat_id}] Received | confirm_switch={confirm_switch} | "
            f"is_followup={is_followup} | Q={question}"
        )

        # ── Step 1: Handle topic switch confirmation ──────────────────────
        if confirm_switch:
            logger.info(f"[{chat_id}] Topic switch confirmed — clearing session")
            clear_session(chat_id)

        session = get_session(chat_id)

        logger.info(
            f"[{chat_id}] Session | filters={session['filters']} | "
            f"last_intent={session['last_intent']} | "
            f"history_len={len(session['history'])} | chat_len={len(session['chat'])}"
        )

        # ── Step 2: UNIVERSAL QUERY REWRITER ──────────────────────────────
        # Rewrites question to be self-contained using BOTH RAG history
        # and inventory filters. Solves all cross-intent follow-ups.
        from rag.chain import rewrite_query_with_context

        # ── Step 2a: Check for affirmative/negative responses ─────────────
        # If bot offered something and user responds with yes/no/sure/etc.
        question_lower = question.strip().lower()
        affirmative_patterns = [
            "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", 
            "please", "show me", "list them", "tell me", "give me",
            "i would", "i'd like", "sounds good"
        ]
        negative_patterns = [
            "no", "nope", "nah", "not now", "skip", "later", 
            "don't", "do not", "never mind"
        ]
        
        is_affirmative = any(p in question_lower for p in affirmative_patterns)
        is_negative = any(p in question_lower for p in negative_patterns)
        is_short_response = len(question.strip().split()) <= 3
        
        # If user responds affirmatively/negatively to a bot offer
        if session.get("last_offer") and is_short_response and (is_affirmative or is_negative):
            
            if is_negative:
                # User declined the offer
                session["last_offer"] = None
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="GREETING",
                    answer="No problem! Let me know if you need anything else.",
                    citations=[], is_intent_switch=False, user_id=user_id,
                )
            
            # User accepted the offer — reconstruct the query
            offer = session["last_offer"]
            if offer["type"] == "list_models":
                # Reconstruct query: "List [models from previous context]"
                filters_str = ", ".join(f"{k}={v}" for k, v in offer["filters"].items() if v)
                reconstructed_query = f"List models with filters: {filters_str}"
                logger.info(f"[{chat_id}] Affirmative response → reconstructing: '{reconstructed_query}'")
                question = reconstructed_query
                session["last_offer"] = None  # Clear offer after accepting
            
            elif offer["type"] == "more_details":
                # User wants more details about specific model/topic
                subject = offer.get("subject", "")
                reconstructed_query = f"Give me more details about {subject}"
                logger.info(f"[{chat_id}] Affirmative response → reconstructing: '{reconstructed_query}'")
                question = reconstructed_query
                session["last_offer"] = None

        # Combine history for rewriter context
        combined_history = []
        # Add RAG history as tuples
        for q, a in session["chat"][-2:]:
            combined_history.append((q, a))
        # Add inventory history as Q-only entries
        for h in session["history"][-2:]:
            if isinstance(h, dict):
                combined_history.append((h.get("question", ""), ""))

        rewrite_result = rewrite_query_with_context(
            question=question,
            history=combined_history,
            inventory_filters=session["filters"],
        )
        rewritten_q = rewrite_result.get("rewritten", question)
        logger.info(f"[{chat_id}] Rewritten: '{rewritten_q}'")

        # ── Step 3: Intent classification ─────────────────────────────────
        from rag.router import classify_intent, get_greeting_response
        decision = classify_intent(rewritten_q, last_intent=session["last_intent"])
        intent = decision.intent.value
        logger.info(f"[{chat_id}] Intent: {intent}")

        # ── GREETING FLOW ──────────────────────────────────────────────────
        if intent == "GREETING":
            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="GREETING",
                answer=get_greeting_response(question),
                citations=[], is_intent_switch=False, user_id=user_id,
            )

        # ── INVENTORY FLOW ─────────────────────────────────────────────────
        if intent == "INVENTORY":

            # RAG → INVENTORY switch (only on follow-up messages with RAG history)
            if (session["last_intent"] == "RAG"
                    and session["chat"]
                    and not confirm_switch
                    and is_followup):

                logger.info(f"[{chat_id}] RAG→INVENTORY switch detected")
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="INVENTORY",
                    answer=(
                        "You were in a document conversation. "
                        "Switching to inventory search will clear your "
                        "document context. Do you want to continue?"
                    ),
                    citations=[], is_intent_switch=True, user_id=user_id,
                )

            from rag.inventory import run_inventory_query
            answer, df, filters, topic_shift, parsed, usage, offer = run_inventory_query(
                question=question,
                accumulated_filters=session["filters"],
                turn_history=session["history"],
                rewritten_question=rewritten_q,
            )

            # LOB shift within INVENTORY (only on follow-up)
            if topic_shift and not confirm_switch and is_followup:
                old_lob = session["filters"].get("LOB", "previous context")
                new_lob = (filters.get("LOB", "new topic")
                           if filters else "new topic")
                logger.info(f"[{chat_id}] LOB shift | {old_lob} → {new_lob}")

                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="INVENTORY",
                    answer=(
                        f"It looks like you are switching from {old_lob} "
                        f"to {new_lob}. Your previous context will be cleared. "
                        f"Do you want to continue?"
                    ),
                    citations=[], is_intent_switch=True, user_id=user_id,
                )

            # Build response with HTML table
            if df is not None and not df.empty:
                html_table  = df_to_html_table(df)
                full_answer = f"{answer}<br><br>{html_table}"
            else:
                full_answer = answer or "No models found matching your criteria."

            # Track subjects for cross-intent context
            subjects = extract_subjects_from_df(df)
            merged_subjects = list(dict.fromkeys(
                subjects + session.get("last_subjects", [])
            ))[:5]

            update_session(
                chat_id=chat_id,
                filters=filters,
                history=session["history"] + [{
                    "question":     question,
                    "filters":      filters,
                    "result_count": len(df) if df is not None else 0,
                    "query_type":   parsed.get("query_type", "list"),
                    "is_followup":  parsed.get("is_followup", False),
                }],
                intent=intent,
                chat=session["chat"],
                last_subjects=merged_subjects,
                last_offer=offer,  # Track offer if bot asked "want more details?"
            )

            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="INVENTORY",
                answer=full_answer,
                citations=[], is_intent_switch=False, user_id=user_id,
                usage=usage,
            )

        # ── RAG FLOW ───────────────────────────────────────────────────────
        else:
            if chain is None or retriever is None:
                raise HTTPException(
                    status_code=503,
                    detail="RAG not available. Run ingest first."
                )

            from rag.chain import run_rag

            # The rewritten query already includes resolution of
            # pronouns and cross-intent context. Pass it to run_rag.
            answer, docs, usage, offer = run_rag(
                chain=chain,
                retriever=retriever,
                question=question,
                history=session["chat"],
                rewritten_query=rewritten_q,
            )

            # Build citations
            citations = []
            for doc in docs:
                source = doc.metadata.get("source", "")
                page   = doc.metadata.get("page", "")
                title, url = clean_citation_source(source)
                citations.append({
                    "title": title,
                    "pages": str(page) if page else "",
                    "url":   url,
                })
            citations = deduplicate_citations(citations)

            update_session(
                chat_id=chat_id,
                filters=session["filters"],
                history=session["history"],
                intent=intent,
                chat=session["chat"] + [(question, answer)],
                last_subjects=session.get("last_subjects", []),
                last_offer=offer,  # Track RAG offers too
            )

            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="RAG",
                answer=answer,
                citations=citations,
                is_intent_switch=False, user_id=user_id,
                usage=usage,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request.chatID}] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=format_error_response(str(e), 500)
        )

# -----------------------------------------------------------------------------
# Health & Data Quality
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "rag_ready":        chain is not None,
        "rag_engine":       "FAISS + LangChain (local) + Cross-encoder reranking",
        "inventory_engine": "DuckDB (local) with sanitization",
        "model":            settings.llm_model,
        "active_sessions":  len(session_store),
    }

@app.get("/data-quality")
def data_quality():
    """Inventory data quality report — for spotting issues before they affect users."""
    try:
        from rag.inventory import get_data_quality_report
        return {"success": True, "report": get_data_quality_report()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
