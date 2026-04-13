"""
api/main.py
COE Analytics Chatbot API — Final Version- INVENTORY intent → Text-to-SQL → DuckDB (local) → HTML table in assistantmsg- RAG intent       
→ FAISS + LangChain + Azure OpenAI (local) → citations- Intent switch    - Cross-intent     
→ isChatIntentSwitch flag from Unfyd
→ inventory filters preserved through RAG turns
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
retriever = None
chain     = None
@app.on_event("startup")
def load_rag():
    global retriever, chain
    try:
        from rag.retriever import load_retriever
        from rag.chain import build_chain
        retriever = load_retriever()
        chain     = build_chain(retriever)
        logger.info(" RAG loaded successfully")
    except Exception as e:
        logger.warning(f" RAG not loaded: {e}")
# -----------------------------------------------------------------------------
# Request Model — Unfyd payload structure
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query:              str
    userID:             str
    chatID:             str
    isFollowUp:         bool         = False
    isChatIntentSwitch: bool         = False
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
# Session Store — keyed by chatID from Unfyd
# -----------------------------------------------------------------------------
session_store = {}
def get_session(chat_id: str) -> dict:
    if chat_id not in session_store:
        session_store[chat_id] = {
            "filters":     {},   # inventory filters (LOB, Status etc)
            "history":     [],   # inventory turn history (list of dicts)
            "last_intent": "",   # last intent for router context
            "chat":        [],   # RAG conversation history (list of tuples)
        }
    return session_store[chat_id]
def update_session(chat_id: str, filters: dict, history: list,
                   intent: str, chat: list):
    session_store[chat_id] = {
        "filters":     filters,
        "history":     history[-10:],
        "last_intent": intent,
        "chat":        chat[-10:],
    }
def clear_session(chat_id: str):
    """Completely delete session — used on topic switch confirmation."""
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
    """Convert DataFrame to HTML table string for frontend rendering."""
    if df is None or df.empty:
        return ""
    # display_cols = [
        # c for c in [
            # "Name_of_the_Model",
            # "LOB",
            # "Function",
            # "Status",
            # "Owner",
            # "Time_Line",
            # "ML_Non_ML",
            # "Document_Availability",
        # ]
        # if c in df.columns
    # ]
    # if not display_cols:
        # display_cols = list(df.columns)
    # df_display = df[display_cols]
    # header = "<tr>" + "".join(
        # f"<th>{col.replace('_', ' ')}</th>"
        # for col in display_cols
    # ) + "</tr>"
    # rows = []
    # for _, row in df_display.iterrows():
        # data_row = "<tr>" + "".join(
            # f"<td>{row[col]}</td>"
            # for col in display_cols
        # ) + "</tr>"
        # rows.append(data_row)
    # return (
        # "<table border='1' cellpadding='5' cellspacing='0'>"
        # f"<thead>{header}</thead>"
        # f"<tbody>{''.join(rows)}</tbody>"
        # "</table>"
    # )
    
    # ── FIX 1: Clean column selection ────────────────
    # For breakdown queries show all columns
    # For list queries show selected columns
    cols = list(df.columns)

    # If breakdown (has model_count) → show all columns
    if "model_count" in cols:
        display_cols = cols    # ← show Function + model_count
    else:
        # For list queries — preferred columns
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
            if c in cols
        ]
        if not display_cols:
            display_cols = cols
    # ──────────────────────────────────────────────────

    df_display = df[display_cols].copy()

    # ── FIX 2: Clean \n from cell values ─────────────
    for col in display_cols:
        df_display[col] = (
            df_display[col]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
            .str.strip()
        )
    # ──────────────────────────────────────────────────

    # Build header
    header = "<tr>" + "".join(
        f"<th>{col.replace('_', ' ')}</th>"
        for col in display_cols
    ) + "</tr>"

    # Build rows
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

# -----------------------------------------------------------------------------
# Response Formatters
# -----------------------------------------------------------------------------
def format_response(
    chat_id:          str,
    msg_id:           str,
    question:         str,
    intent:           str,
    answer:           str,
    citations:        list = None,
    is_intent_switch: bool = False,
    user_id:          int  = 0,
):
   # If intent switch → return empty data
    if is_intent_switch:
       return {
           "success": True,
           "error": False,
           "message": "Intent switch detected",
           "code": 200,
           "data": [],   #  EMPTY ARRAY
           "TokenIndex": {"token": ""},
           "isIntentSwitch": True,
           "isOpeanAIResponse": False
       }
    return {
        "success":True,
        "error":False,
        "message":"success",
        "code":200,
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
        }],
            "reference":         None,
        "TokenIndex":        {"token": ""},
        "isIntentSwitch":    is_intent_switch,
        "isOpeanAIResponse": True,
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
        # ── MAPPING LAYER (Unfyd → Internal) ──────────────────────────────
        question       = request.query
        chat_id        = request.chatID
        msg_id         = str(abs(hash(question)))[:8]
        confirm_switch = request.isChatIntentSwitch
        is_followup    = request.isFollowUp
        user_id        = int(request.userID) if request.userID else 0
        logger.info(
            f"[{chat_id}] Received | "
            f"confirm_switch={confirm_switch} | "
            f"is_followup={is_followup} | "
            f"Q={question}"
        )
        # ── Step 1: Handle topic switch confirmation ───────────────────────
        # Unfyd sends isChatIntentSwitch=true when user confirms popup
        if confirm_switch:
            logger.info(f"[{chat_id}] Topic switch confirmed — clearing session")
            clear_session(chat_id)          # completely delete old context
        # Always get fresh session after potential clear
        session = get_session(chat_id)
        logger.info(
            f"[{chat_id}] Session | "
            f"filters={session['filters']} | "
            f"last_intent={session['last_intent']} | "
            f"history_len={len(session['history'])} | "
            f"chat_len={len(session['chat'])}"
        )
        # ── Step 2: Intent classification ─────────────────────────────────
        from rag.router import classify_intent, get_greeting_response
        decision = classify_intent(
            question,
            last_intent=session["last_intent"]
        )
        intent = decision.intent.value
        logger.info(f"[{chat_id}] Intent: {intent}")
        
        # ── GREETING FLOW ─────────────────────────────────────────────────       
        if intent == "GREETING":
            return format_response(
                chat_id=chat_id,
                msg_id=msg_id,
                question=question,
                intent="GREETING",
                answer=get_greeting_response(question),
                citations=[],
                is_intent_switch=False,
                user_id=user_id,
            )
        # ── INVENTORY FLOW ─────────────────────────────────────────────────
        if intent == "INVENTORY":
                    
            # ── FIX 3: RAG→INVENTORY switch detection ────────
            if (session["last_intent"] == "RAG"
                    and session["chat"]
                    and not confirm_switch):

                logger.info(
                    f"[{chat_id}] RAG→INVENTORY switch detected"
                )
                return format_response(
                    chat_id=chat_id,
                    msg_id=msg_id,
                    question=question,
                    intent="INVENTORY",
                    answer=(
                        "You were in a document conversation. "
                        "Switching to inventory search will "
                        "clear your document context. "
                        "Do you want to continue?"
                    ),
                    citations=[],
                    is_intent_switch=True,
                    user_id=user_id,
                )

            from rag.inventory import run_inventory_query
            answer, df, filters, topic_shift, parsed = run_inventory_query(
                question=question,
                accumulated_filters=session["filters"],
                turn_history=session["history"],
            )
            # Topic shift detected AND not yet confirmed by user
            if topic_shift and not confirm_switch:
                old_lob = session["filters"].get("LOB", "previous context")
                new_lob = (
                    filters.get("LOB", "new topic")
                    if filters else "new topic"
                )
                logger.info(
                    f"[{chat_id}] Topic shift | "
                    f"{old_lob} → {new_lob}"
                )
                # Do NOT update session — preserve old context
                return format_response(
                    chat_id=chat_id,
                    msg_id=msg_id,
                    question=question,
                    intent="INVENTORY",
                    answer=(
                        f"It looks like you are switching from {old_lob} "
                        f"to {new_lob}. "
                        f"Your previous context will be cleared. "
                        f"Do you want to continue?"
                    ),
                    citations=[],
                    is_intent_switch=True,      # Unfyd shows popup
                    user_id=user_id,
                )
            # Build answer with HTML table if results exist
            if df is not None and not df.empty:
                html_table  = df_to_html_table(df)
                full_answer = f"{answer}<br><br>{html_table}"
            else:
                full_answer = answer or "No models found matching your criteria."
            # Update inventory context
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
                chat=session["chat"],       # RAG history preserved
            )
            return format_response(
                chat_id=chat_id,
                msg_id=msg_id,
                question=question,
                intent="INVENTORY",
                answer=full_answer,
                citations=[],
                is_intent_switch=False,
                user_id=user_id,
            )
        # ── RAG FLOW ───────────────────────────────────────────────────────
        else:
            if chain is None or retriever is None:
                raise HTTPException(
                    status_code=503,
                    detail="RAG not available. Run ingest first."
                )
            from rag.chain import run_rag
            # Enrich question with inventory context for cross-intent follow-ups
            # e.g. "What is their SOP?" after "Show me ABCD models"
            inventory_context = ""
            if session["filters"]:
                parts = []
                if session["filters"].get("LOB"):
                    parts.append(f"LOB: {session['filters']['LOB']}")
                if session["filters"].get("Function"):
                    parts.append(f"Function: {session['filters']['Function']}")
                if session["filters"].get("Status"):
                    parts.append(f"Status: {session['filters']['Status']}")
                if parts:
                    inventory_context = ", ".join(parts)
                    logger.info(
                        f"[{chat_id}] RAG enriched with inventory context: "
                        f"{inventory_context}"
                    )
             # ── RAG ---> RAG topic switch detection ───────────────────────────────────────────────────────        
            if (session["last_intent"] == "RAG" 
                    and session["chat"] 
                    and not confirm_switch
                    and not is_followup):
                 
                last_rag_q = session["chat"][-1][0]

                stop_words = {
                    "what", "is", "the", "a", "an", "of", "in",
                    "how", "are", "tell", "me", "about", "can",
                    "you", "do", "does", "which", "who", "where",
                    "give", "explain", "describe", "list", "show"
                }

                last_words = set(last_rag_q.lower().split()) - stop_words
                curr_words  = set(question.lower().split()) - stop_words
                overlap     = last_words & curr_words

                if not overlap and last_words and curr_words:
                    logger.info(
                        f"[{chat_id}] RAG→RAG topic switch detected | "
                        f"prev: '{last_rag_q}' | new: '{question}'"
                    )
                    return format_response(
                        chat_id=chat_id,
                        msg_id=msg_id,
                        question=question,
                        intent="RAG",
                        answer=(
                            "It looks like you are asking about a "
                            "different topic. Should I start a "
                            "fresh conversation?"
                        ),
                        citations=[],
                        is_intent_switch=True,
                        user_id=user_id,
                    )

            if session["last_intent"] == "RAG" and inventory_context:
                enriched_question = (
                    f"{question} (Context: {inventory_context})"
                )
            else:
                enriched_question= question
                
            answer, docs = run_rag(
                chain=chain,
                retriever=retriever,
                question=enriched_question,
                history=session["chat"],
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
            # Update session — inventory filters and history PRESERVED
            update_session(
                chat_id=chat_id,
                filters=session["filters"],         # preserve inventory filters
                history=session["history"],         # preserve inventory history
                intent=intent,
                chat=session["chat"] + [(question, answer)],
            )
            return format_response(
                chat_id=chat_id,
                msg_id=msg_id,
                question=question,
                intent="RAG",
                answer=answer,
                citations=citations,
                is_intent_switch=False,
                user_id=user_id,
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
# Health Check
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "rag_ready":        chain is not None,
        "rag_engine":       "FAISS + LangChain (local)",
        "inventory_engine": "DuckDB (local)",
        "model":            settings.llm_model,
        "active_sessions":  len(session_store),
    }
# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)