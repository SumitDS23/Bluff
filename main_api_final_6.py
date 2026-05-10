"""
api/main.py

COE Analytics Chatbot API — Final Version
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

# Load environment
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
        content=format_error_response("Internal Server Error", 500),
    )

# -----------------------------------------------------------------------------
# Load RAG at startup
# -----------------------------------------------------------------------------
retriever = None
chain = None
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
# Request Model
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    userID: str
    chatID: str
    isFollowUp: bool = False
    isChatIntentSwitch: bool = False
    processID: Optional[int] = None
    LOB: Optional[int] = None
    departmentID: Optional[int] = None
    browserTime: Optional[str] = None
    isfaq: Optional[bool] = None
    userDepartmentType: Optional[str] = None
    isThemeFirstChat: Optional[bool] = None
    themeID: Optional[int] = None
    theme: Optional[str] = None


# -----------------------------------------------------------------------------
# Session Store
# -----------------------------------------------------------------------------
session_store = {}


def get_session(chat_id: str) -> dict:
    if chat_id not in session_store:
        session_store[chat_id] = {
            "filters": {},
            "history": [],
            "last_intent": "",
            "chat": [],
        }
    return session_store[chat_id]


def update_session(chat_id: str, filters: dict, history: list, intent: str, chat: list):
    session_store[chat_id] = {
        "filters": filters,
        "history": history[-10:],
        "last_intent": intent,
        "chat": chat[-10:],
    }


def clear_session(chat_id: str):
    if chat_id in session_store:
        del session_store[chat_id]
    logger.info(f"[{chat_id}] Session cleared")


# -----------------------------------------------------------------------------
# Helpers
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
    seen = set()

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
            c
            for c in [
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
        f"<th>{col.replace('_', ' ')}</th>" for col in display_cols
    ) + "</tr>"

    rows = []
    for _, row in df_display.iterrows():
        rows.append(
            "<tr>" + "".join(f"<td>{row[col]}</td>" for col in display_cols) + "</tr>"
        )

    return (
        "<table border='1' cellpadding='5' cellspacing='0'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# -----------------------------------------------------------------------------
# Response Formatters
# -----------------------------------------------------------------------------
def format_response(
    chat_id,
    msg_id,
    question,
    intent,
    answer,
    citations=None,
    is_intent_switch=False,
    user_id=0,
):
    if is_intent_switch:
        return {
            "success": True,
            "error": False,
            "message": "Intent switch detected",
            "code": 200,
            "data": [],
            "TokenIndex": {"token": ""},
            "isIntentSwitch": True,
            "isOpeanAIResponse": False,
        }

    return {
        "success": True,
        "error": False,
        "message": "success",
        "code": 200,
        "data": [
            {
                "userid": user_id,
                "chatid": chat_id,
                "msgid": msg_id,
                "intent": intent,
                "usermsg": question,
                "assistantmsg": answer,
                "browsertime": datetime.utcnow().isoformat(),
                "citation": json.dumps(citations or []),
            }
        ],
        "TokenIndex": {"token": ""},
        "isIntentSwitch": False,
        "isOpeanAIResponse": True,
    }


def format_error_response(message, code):
    return {
        "success": False,
        "error": True,
        "message": message,
        "code": code,
        "data": [],
    }
# -----------------------------------------------------------------------------
# Main Endpoint
# -----------------------------------------------------------------------------
@app.post("/query")
def query(request: QueryRequest):
    try:
        # ── MAPPING LAYER (Unfyd → Internal) ──────────────────────────────
        question = request.query
        chat_id = request.chatID
        msg_id = str(abs(hash(question)))[:8]
        confirm_switch = request.isChatIntentSwitch
        is_followup = request.isFollowUp
        user_id = int(request.userID) if request.userID else 0

        logger.info(
            f"[{chat_id}] Received | "
            f"confirm_switch={confirm_switch} | "
            f"is_followup={is_followup} | "
            f"Q={question}"
        )

        # ── Step 1: Handle topic switch confirmation ───────────────────────
        if confirm_switch:
            logger.info(f"[{chat_id}] Topic switch confirmed — clearing session")
            clear_session(chat_id)

        session = get_session(chat_id)
        
        if not is_followup and (session["filters"] or session["chat"] or session["last_intent"]):
            logger.info(f"[{chat_id}] Fresh chat detected - clearing old session")
            clear_session(chat_id)
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

        # ── GREETING FLOW ──────────────────────────────────────────────────
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

            if (
                session["last_intent"] == "RAG"
                and session["chat"]
                and not confirm_switch
                and not is_followup
            ):
                logger.info(f"[{chat_id}] RAG→INVENTORY switch detected")

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

            if topic_shift and not confirm_switch:     #and not is_followup
                old_lob = session["filters"].get("LOB", "previous context")
                new_lob = filters.get("LOB", "new topic") if filters else "new topic"

                logger.info(f"[{chat_id}] LOB shift | {old_lob} → {new_lob}")

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
                    is_intent_switch=True,
                    user_id=user_id,
                )

            if df is not None and not df.empty:
                html_table = df_to_html_table(df)
                full_answer = f"{answer}<br><br>{html_table}"
            else:
                full_answer = answer or "No models found matching your criteria."

            update_session(
                chat_id=chat_id,
                filters=filters,
                history=session["history"] + [{
                    "question": question,
                    "filters": filters,
                    "result_count": len(df) if df is not None else 0,
                    "query_type": parsed.get("query_type", "list"),
                    "is_followup": parsed.get("is_followup", False),
                }],
                intent=intent,
                chat=session["chat"],
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

            # if (
                # session["last_intent"] == "RAG"
                # and session["chat"]
                # and not confirm_switch
                # and is_followup
            # ):
                # logger.info(
                    # f"[{chat_id}] RAG→RAG topic switch detected | "
                    # f"prev: '{session['chat'][-1][0]}' | "
                    # f"new: '{question}'"
                # )

                # return format_response(
                    # chat_id=chat_id,
                    # msg_id=msg_id,
                    # question=question,
                    # intent="RAG",
                    # answer=(
                        # "It looks like you are asking about a "
                        # "different topic. Should I start a "
                        # "fresh conversation?"
                    # ),
                    # citations=[],
                    # is_intent_switch=True,
                    # user_id=user_id,
                # )
            lob_map = {
                "ABFL-RA": "ABFL-RA",
                "ABFL-CA": "ABFL-CA",
                "ABFL" : "ABFL-RA",
                "ABSLI": "ABSLI",
                "ABHI": "ABHI",
                "ABHFL": "ABHFL",
                "ABSLAMC": "ABSLAMC",
                "CAU": "CAU",
                "ABCD": "ABCD",
                }
            # detected_lob = None
            # question_upper = question.upper()
            # for key, val in lob_map.items():
                # if key in question_upper:
                    # detected_lob = val
                    # break
            # lob_filter = detected_lob  # only LOB mentioned in the question
            
            # Detect ALL LOBs mentioned in question
            question_upper = question.upper()
            detected_lobs = [val for key, val in lob_map.items() 
                             if key in question_upper]
            detected_lobs = list(dict.fromkeys(detected_lobs))  # deduplicate

            if len(detected_lobs) > 1:
                # Cross-LOB comparison → no filter, search globally
                lob_filter = None
                logger.info(f"[{chat_id}] Cross-LOB query detected: {detected_lobs} → global search")
            elif len(detected_lobs) == 1:
                # Single LOB → filter to that LOB
                lob_filter = detected_lobs[0]
                logger.info(f"[{chat_id}] Single LOB detected: {lob_filter}")
            else:
                # No LOB mentioned → global search
                lob_filter = None
                logger.info(f"[{chat_id}] No LOB detected → global search")
                
            enriched_question = question
            if session["filters"]:
                pronouns = {"their", "these", "those", "them", "it", "its", "this", "such"}
                question_words = set(question.lower().split())

                if question_words & pronouns:
                    parts = []

                    if session["filters"].get("LOB"):
                        parts.append(f"LOB: {session['filters']['LOB']}")

                    if session["filters"].get("Function"):
                        parts.append(f"Function: {session['filters']['Function']}")

                    if parts:
                        inventory_context = ", ".join(parts)
                        enriched_question = f"{question} (Context: {inventory_context})"

                        logger.info(f"[{chat_id}] RAG enriched: {inventory_context}")
            # Use filtered retriever if LOB context exists
            from rag.retriever import get_filtered_retriever

            function_filter = session["filters"].get("Function")

            if lob_filter or function_filter:
                filtered_retriever = get_filtered_retriever(
                    vectorstore,
                    lob=lob_filter,
                    function=function_filter
                )
                logger.info(f"[{chat_id}] Filtered retrieval | LOB={lob_filter} | Function={function_filter}")
            else:
                filtered_retriever = retriever  # use global unfiltered

            answer, docs = run_rag(
                chain=chain,
                retriever=filtered_retriever,
                question=enriched_question,
                history=session["chat"],
            )
            # answer, docs = run_rag(
                # chain=chain,
                # retriever=retriever,
                # question=enriched_question,
                # history=session["chat"],
            # )

            citations = []
            for doc in docs:
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page", "")
                title, url = clean_citation_source(source)

                citations.append({
                    "title": title,
                    "pages": str(page) if page else "",
                    "url": url,
                })

            citations = deduplicate_citations(citations)

            update_session(
                chat_id=chat_id,
                filters=session["filters"],
                history=session["history"],
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
        "status": "ok",
        "rag_ready": chain is not None,
        "model": settings.llm_model,
        "active_sessions": len(session_store),
    }


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8030)
