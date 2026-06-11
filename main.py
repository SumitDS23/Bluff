"""
api/main.py
COE Analytics Chatbot API — Unified Version

Features:
- Universal Query Rewriter (LLM-based) for INVENTORY and RAG
- Cross-intent context bridge (resolves pronouns across intents)
- Cross-encoder reranking for RAG (in chain.py)
- Data sanitization on inventory load (in inventory.py)
- Topic switch via Unfyd's isFollowUp flag
- Dashboard Inventory support (Text-to-SQL)
- Model Inventory support (DuckDB Excel queries)
- Utility Inventory support (FAISS General store + SharePoint link)  ← NEW
- Multi-LOB detection and filtering for RAG
- Session management with offer tracking
- Dual FAISS stores: ML store (model_doc/cde/framework/general)      ← NEW
                     General store (newsletter/utility)               ← NEW
- Category-driven SharePoint link injection                           ← NEW

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


# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="COE Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled Error: {exc}")
    return JSONResponse(
        status_code=500,
        content=format_error_response("Internal Server Error", 500),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Category → FAISS store mapping (Option A: deterministic, zero extra LLM calls)
# ──────────────────────────────────────────────────────────────────────────────

ML_CATEGORIES = {
    "model_inventory",      # INVENTORY intent, but RAG fallback needs the ML store
    "model_documentation",
    "standards",
    "cde",
    "framework",
}
# Everything else ("newsletter", "utility") → General store

# Trigger words that cause a SharePoint link to be appended to the response.
# For UTILITY intent the link is always appended (no trigger word needed).
LINK_TRIGGERS = (
    "link", "sharepoint", "folder", "document", "where can i find",
    "open", "access", "file", "download",
)


def _should_append_link(question: str) -> bool:
    """Return True when the user is explicitly asking for a file / location."""
    return any(t in question.lower() for t in LINK_TRIGGERS)


# ──────────────────────────────────────────────────────────────────────────────
# Global state — two retriever/chain pairs
# ──────────────────────────────────────────────────────────────────────────────

ml_retriever        = None
ml_vectorstore      = None
ml_chain            = None

general_retriever   = None
general_vectorstore = None
general_chain       = None

dashboard_engine    = None


@app.on_event("startup")
def load_rag():
    global ml_retriever, ml_vectorstore, ml_chain
    global general_retriever, general_vectorstore, general_chain
    global dashboard_engine

    # ── ML store (model_doc, cde, framework, general) ─────────────────────
    try:
        from rag.retriever import load_retriever
        from rag.chain import build_chain
        ml_retriever, ml_vectorstore = load_retriever("ml")
        ml_chain = build_chain(ml_retriever)
        logger.info("✅ ML FAISS store loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ ML FAISS store not loaded: {e}")

    # ── General store (newsletter, utility) ───────────────────────────────
    try:
        from rag.retriever import load_retriever
        from rag.chain import build_chain
        general_retriever, general_vectorstore = load_retriever("general")
        general_chain = build_chain(general_retriever)
        logger.info("✅ General FAISS store loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ General FAISS store not loaded: {e}")

    # ── Model Inventory (lazy-load on first query) ─────────────────────────
    try:
        inventory_path = Path(settings.inventory_path)
        if inventory_path.exists():
            logger.info(f"✅ Model Inventory file found: {inventory_path.name} (lazy load)")
        else:
            logger.error(f"❌ Model Inventory file NOT found: {inventory_path}")
    except Exception as e:
        logger.warning(f"⚠️ Model Inventory validation failed: {e}")

    # ── Dashboard Inventory ────────────────────────────────────────────────
    try:
        from rag.dashboard_inventory import create_dashboard_engine
        dashboard_engine = create_dashboard_engine()
        logger.info("✅ Dashboard Inventory loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Dashboard Inventory not loaded: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Request Model — Unfyd payload structure
# ──────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:              str
    userID:             str
    chatID:             str
    isFollowUp:         bool           = False
    isChatIntentSwitch: bool           = False
    processID:          Optional[int]  = None
    LOB:                Optional[int]  = None
    departmentID:       Optional[int]  = None
    browserTime:        Optional[str]  = None
    isfaq:              Optional[bool] = None
    userDepartmentType: Optional[str]  = None
    isThemeFirstChat:   Optional[bool] = None
    themeID:            Optional[int]  = None
    theme:              Optional[str]  = None


# ──────────────────────────────────────────────────────────────────────────────
# Session store
# ──────────────────────────────────────────────────────────────────────────────

session_store = {}


def get_session(chat_id: str) -> dict:
    if chat_id not in session_store:
        session_store[chat_id] = {
            "filters":       {},
            "history":       [],
            "last_intent":   "",
            "last_category": "utility",
            "chat":          [],
            "last_subjects": [],
            "last_offer":    None,
        }
    return session_store[chat_id]


def update_session(chat_id: str, filters: dict, history: list,
                   intent: str, chat: list,
                   last_subjects: list = None,
                   last_offer: dict = None,
                   last_category: str = "utility"):
    existing = session_store.get(chat_id, {})
    session_store[chat_id] = {
        "filters":       filters,
        "history":       history[-10:],
        "last_intent":   intent,
        "last_category": last_category,
        "chat":          chat[-10:],
        "last_subjects": (last_subjects or existing.get("last_subjects", []))[-5:],
        "last_offer":    last_offer,
    }


def clear_session(chat_id: str):
    if chat_id in session_store:
        del session_store[chat_id]
        logger.info(f"[{chat_id}] Session fully cleared")


# ──────────────────────────────────────────────────────────────────────────────
# Citation helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# HTML Table Formatter
# ──────────────────────────────────────────────────────────────────────────────

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
        f"<th>{col.replace('_', ' ')}</th>" for col in display_cols
    ) + "</tr>"

    rows = []
    for _, row in df_display.iterrows():
        data_row = "<tr>" + "".join(
            f"<td>{row[col]}</td>" for col in display_cols
        ) + "</tr>"
        rows.append(data_row)

    return (
        "<table border='1' cellpadding='5' cellspacing='0'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-intent subject extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_subjects_from_df(df) -> list:
    if df is None or df.empty:
        return []
    if "Name_of_the_Model" in df.columns:
        return df["Name_of_the_Model"].head(5).tolist()
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Response Formatters
# ──────────────────────────────────────────────────────────────────────────────

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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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
        "usage":             usage,
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


# ──────────────────────────────────────────────────────────────────────────────
# Main endpoint
# ──────────────────────────────────────────────────────────────────────────────

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

        # ── Topic switch confirmation ─────────────────────────────────────
        if confirm_switch:
            logger.info(f"[{chat_id}] Topic switch confirmed — clearing session")
            clear_session(chat_id)

        session = get_session(chat_id)

        # Fresh chat guard
        if not is_followup and (session["filters"] or session["chat"] or session["last_intent"]):
            logger.info(f"[{chat_id}] Fresh chat — clearing old session")
            clear_session(chat_id)
            session = get_session(chat_id)

        logger.info(
            f"[{chat_id}] Session | filters={session['filters']} | "
            f"last_intent={session['last_intent']} | "
            f"history_len={len(session['history'])} | chat_len={len(session['chat'])}"
        )

        # ── Universal Query Rewriter ──────────────────────────────────────
        from rag.chain import rewrite_query_with_context

        question_lower = question.strip().lower()
        affirmative_patterns = [
            "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead",
            "please", "show me", "list them", "tell me", "give me",
            "i would", "i'd like", "sounds good",
        ]
        negative_patterns = [
            "no", "nope", "nah", "not now", "skip", "later",
            "don't", "do not", "never mind",
        ]

        is_affirmative  = any(p in question_lower for p in affirmative_patterns)
        is_negative     = any(p in question_lower for p in negative_patterns)
        is_short_response = len(question.strip().split()) <= 3

        # Handle yes/no responses to bot offers
        if session.get("last_offer") and is_short_response and (is_affirmative or is_negative):
            if is_negative:
                session["last_offer"] = None
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="GREETING",
                    answer="No problem! Let me know if you need anything else.",
                    citations=[], is_intent_switch=False, user_id=user_id,
                )

            offer = session["last_offer"]
            if offer["type"] == "list_models":
                filters_str  = ", ".join(f"{k}={v}" for k, v in offer["filters"].items() if v)
                question     = f"List models with filters: {filters_str}"
                session["last_offer"] = None
                logger.info(f"[{chat_id}] Affirmative → reconstructed: '{question}'")

            elif offer["type"] == "more_details":
                subject  = offer.get("subject", "")
                question = f"Give me more details about {subject}"
                session["last_offer"] = None
                logger.info(f"[{chat_id}] Affirmative → reconstructed: '{question}'")

        # Combine histories for rewriter
        combined_history = []
        for q, a in session["chat"][-2:]:
            combined_history.append((q, a))
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

        # ── Intent classification ─────────────────────────────────────────
        from rag.router import classify_intent, get_greeting_response, format_link_html

        decision = classify_intent(rewritten_q, last_intent=session["last_intent"])
        intent   = decision.intent.value
        category = decision.category          # drives store selection + SharePoint link

        logger.info(f"[{chat_id}] Intent={intent} | Category={category}")

        # ══════════════════════════════════════════════════════════════════
        # GREETING FLOW
        # ══════════════════════════════════════════════════════════════════
        if intent == "GREETING":
            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="GREETING",
                answer=get_greeting_response(question),
                citations=[], is_intent_switch=False, user_id=user_id,
            )

        # ══════════════════════════════════════════════════════════════════
        # DASHBOARD FLOW
        # ══════════════════════════════════════════════════════════════════
        if intent == "DASHBOARD":
            if dashboard_engine is None:
                raise HTTPException(
                    status_code=503,
                    detail="Dashboard Inventory not available. Check CSV file path.",
                )

            # Intent switch guard
            if (session["last_intent"] in ["RAG", "INVENTORY", "UTILITY"]
                    and (session["chat"] or session["history"])
                    and not confirm_switch
                    and is_followup):

                source = {
                    "RAG":     "document conversation",
                    "INVENTORY": "model inventory",
                    "UTILITY": "utility search",
                }.get(session["last_intent"], "previous conversation")

                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="DASHBOARD",
                    answer=(
                        f"You were in a {source}. "
                        "Switching to dashboard inventory will clear your context. "
                        "Do you want to continue?"
                    ),
                    citations=[], is_intent_switch=True, user_id=user_id,
                )

            result = dashboard_engine.answer_question(rewritten_q)

            if not result["success"]:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"[{chat_id}] Dashboard query failed: {error_msg}")
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="DASHBOARD",
                    answer=f"I encountered an error querying the dashboard inventory: {error_msg}",
                    citations=[], is_intent_switch=False, user_id=user_id,
                )

            answer_text = result["answer"]

            if result.get("result_df") is not None and not result["result_df"].empty:
                df = result["result_df"]
                html_table = df.to_html(
                    index=False, border=1,
                    classes="dashboard-table", escape=False,
                ).replace('<table', '<table border="1" cellpadding="5" cellspacing="0"')
                answer_text += f"<br><br>{html_table}"

            # Append SharePoint link if user asked for it
            if _should_append_link(question):
                answer_text += "<br><br>" + format_link_html(category)

            update_session(
                chat_id=chat_id, filters=session["filters"],
                history=session["history"], intent=intent,
                chat=session["chat"],
                last_subjects=session.get("last_subjects", []),
                last_offer=None, last_category=category,
            )

            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="DASHBOARD",
                answer=answer_text,
                citations=[], is_intent_switch=False, user_id=user_id,
                usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
            )

        # ══════════════════════════════════════════════════════════════════
        # INVENTORY FLOW
        # ══════════════════════════════════════════════════════════════════
        if intent == "INVENTORY":

            # Intent switch guard
            if (session["last_intent"] in ["RAG", "UTILITY"]
                    and session["chat"]
                    and not confirm_switch
                    and is_followup):

                source = "utility search" if session["last_intent"] == "UTILITY" else "document conversation"
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="INVENTORY",
                    answer=(
                        f"You were in a {source}. "
                        "Switching to inventory search will clear your context. "
                        "Do you want to continue?"
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

            # LOB shift within INVENTORY
            if topic_shift and not confirm_switch and is_followup:
                old_lob = session["filters"].get("LOB", "previous context")
                new_lob = filters.get("LOB", "new topic") if filters else "new topic"
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="INVENTORY",
                    answer=(
                        f"It looks like you are switching from {old_lob} to {new_lob}. "
                        "Your previous context will be cleared. Do you want to continue?"
                    ),
                    citations=[], is_intent_switch=True, user_id=user_id,
                )

            if df is not None and not df.empty:
                full_answer = f"{answer}<br><br>{df_to_html_table(df)}"
            else:
                full_answer = answer or "No models found matching your criteria."

            # Append SharePoint link if user asked for files/location
            if _should_append_link(question):
                full_answer += "<br><br>" + format_link_html(category)

            subjects = extract_subjects_from_df(df)
            merged_subjects = list(dict.fromkeys(
                subjects + session.get("last_subjects", [])
            ))[:5]

            update_session(
                chat_id=chat_id, filters=filters,
                history=session["history"] + [{
                    "question":     question,
                    "filters":      filters,
                    "result_count": len(df) if df is not None else 0,
                    "query_type":   parsed.get("query_type", "list"),
                    "is_followup":  parsed.get("is_followup", False),
                }],
                intent=intent, chat=session["chat"],
                last_subjects=merged_subjects,
                last_offer=offer, last_category=category,
            )

            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="INVENTORY",
                answer=full_answer,
                citations=[], is_intent_switch=False,
                user_id=user_id, usage=usage,
            )

        # ══════════════════════════════════════════════════════════════════
        # UTILITY FLOW  ← NEW
        # ══════════════════════════════════════════════════════════════════
        if intent == "UTILITY":
            if general_chain is None:
                return format_response(
                    chat_id=chat_id, msg_id=msg_id,
                    question=question, intent="UTILITY",
                    answer=(
                        "The Utility knowledge base is not loaded. "
                        "Please run ingest/ingest.py first."
                    ),
                    citations=[], is_intent_switch=False, user_id=user_id,
                )

            from rag.chain import run_rag

            answer, docs, usage, offer = run_rag(
                chain=general_chain,
                retriever=general_retriever,
                question=question,
                history=session["chat"],
                rewritten_query=rewritten_q,
            )

            answer_html = answer.replace("\n", "<br>")
            # Always append SharePoint link for utility questions
            answer_html += "<br><br>" + format_link_html("utility", "📁 Open Utility Folder (SharePoint)")

            # Build citations from retrieved docs
            citations = deduplicate_citations([
                {
                    "title": clean_citation_source(d.metadata.get("source", ""))[0],
                    "pages": str(d.metadata.get("page", "")),
                    "url":   clean_citation_source(d.metadata.get("source", ""))[1],
                }
                for d in docs
            ])

            update_session(
                chat_id=chat_id, filters=session["filters"],
                history=session["history"], intent=intent,
                chat=session["chat"] + [(question, answer)],
                last_subjects=session.get("last_subjects", []),
                last_offer=offer, last_category=category,
            )

            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="UTILITY",
                answer=answer_html,
                citations=citations,
                is_intent_switch=False, user_id=user_id, usage=usage,
            )

        # ══════════════════════════════════════════════════════════════════
        # RAG FLOW — category drives store selection (Option A)
        # ══════════════════════════════════════════════════════════════════
        else:
            # Select store based on category
            if category in ML_CATEGORIES:
                active_chain      = ml_chain
                active_retriever  = ml_retriever
                active_vectorstore = ml_vectorstore
                store_label       = "ML"
            else:
                active_chain      = general_chain
                active_retriever  = general_retriever
                active_vectorstore = general_vectorstore
                store_label       = "General"

            logger.info(f"[{chat_id}] RAG → using {store_label} store (category={category})")

            if active_chain is None or active_retriever is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"RAG [{store_label}] store not available. Run ingest first.",
                )

            from rag.chain import run_rag
            from rag.retriever import get_filtered_retriever

            # Multi-LOB detection
            lob_map = {
                "ABFL-RA": "ABFL-RA", "ABFL-CA": "ABFL-CA", "ABFL": "ABFL-RA",
                "ABSLI": "ABSLI", "ABHI": "ABHI", "ABHFL": "ABHFL",
                "ABSLAMC": "ABSLAMC", "CAU": "CAU", "ABCD": "ABCD",
            }

            question_upper = question.upper()
            detected_lobs  = list(dict.fromkeys(
                val for key, val in lob_map.items() if key in question_upper
            ))

            if len(detected_lobs) > 1:
                lob_filter = None
                logger.info(f"[{chat_id}] Cross-LOB query: {detected_lobs} → global search")
            elif len(detected_lobs) == 1:
                lob_filter = detected_lobs[0]
                logger.info(f"[{chat_id}] Single LOB: {lob_filter}")
            else:
                lob_filter = None
                logger.info(f"[{chat_id}] No LOB detected → global search")

            # Pronoun enrichment from inventory context
            enriched_question = question
            if session["filters"]:
                pronouns      = {"their", "these", "those", "them", "it", "its", "this", "such"}
                question_words = set(question.lower().split())
                if question_words & pronouns:
                    parts = []
                    if session["filters"].get("LOB"):
                        parts.append(f"LOB: {session['filters']['LOB']}")
                    if session["filters"].get("Function"):
                        parts.append(f"Function: {session['filters']['Function']}")
                    if parts:
                        enriched_question = f"{question} (Context: {', '.join(parts)})"
                        logger.info(f"[{chat_id}] RAG enriched with: {parts}")

            # Apply LOB/Function filter only on ML store (general store has no LOB metadata)
            function_filter = session["filters"].get("Function")
            if store_label == "ML" and (lob_filter or function_filter):
                filtered_retriever = get_filtered_retriever(
                    active_vectorstore,
                    lob=lob_filter,
                    function=function_filter,
                )
                logger.info(f"[{chat_id}] Filtered retrieval | LOB={lob_filter} | Function={function_filter}")
            else:
                filtered_retriever = active_retriever

            answer, docs, usage, offer = run_rag(
                chain=active_chain,
                retriever=filtered_retriever,
                question=enriched_question,
                history=session["chat"],
                rewritten_query=rewritten_q,
            )

            # Build citations
            citations = deduplicate_citations([
                {
                    "title": clean_citation_source(d.metadata.get("source", ""))[0],
                    "pages": str(d.metadata.get("page", "")),
                    "url":   clean_citation_source(d.metadata.get("source", ""))[1],
                }
                for d in docs
            ])

            answer_html = answer

            # Append SharePoint link when user asks for file/location
            if _should_append_link(question):
                answer_html += "<br><br>" + format_link_html(category)

            update_session(
                chat_id=chat_id, filters=session["filters"],
                history=session["history"], intent=intent,
                chat=session["chat"] + [(question, answer)],
                last_subjects=session.get("last_subjects", []),
                last_offer=offer, last_category=category,
            )

            return format_response(
                chat_id=chat_id, msg_id=msg_id,
                question=question, intent="RAG",
                answer=answer_html,
                citations=citations,
                is_intent_switch=False, user_id=user_id, usage=usage,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request.chatID}] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=format_error_response(str(e), 500),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Health & Data Quality
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":              "ok",
        "ml_rag_ready":        ml_chain is not None,
        "general_rag_ready":   general_chain is not None,
        "dashboard_ready":     dashboard_engine is not None,
        "rag_engine":          "FAISS + LangChain (local) + Cross-encoder reranking",
        "inventory_engine":    "DuckDB (local) with sanitization",
        "dashboard_engine":    "DuckDB (local) + Text-to-SQL",
        "ml_store_path":       settings.local_index_path_ml,
        "general_store_path":  settings.local_index_path_general,
        "model":               settings.llm_model,
        "active_sessions":     len(session_store),
    }


@app.get("/data-quality")
def data_quality():
    try:
        from rag.inventory import get_data_quality_report
        return {"success": True, "report": get_data_quality_report()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
