"""
main.py integration example
Shows how to integrate Dashboard, Governance, and Model Doc routing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.query_router import QueryRouter
from rag.dashboard_inventory import create_dashboard_engine
# from rag.governance_rag import answer_governance_question  # When ready
# from rag.chain import run_rag  # Your existing RAG

app = FastAPI()

# Initialize engines
router = QueryRouter()
dashboard_engine = create_dashboard_engine()  # Loads CSV on startup

class QueryRequest(BaseModel):
    query: str
    chatID: str
    userID: str = None


@app.post("/query")
def query(request: QueryRequest):
    """
    Unified query endpoint with intelligent routing.
    """
    question = request.query
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Route the query
    # ═══════════════════════════════════════════════════════════════
    routing = router.route_query(question)
    intent = routing["intent"]
    
    logger.info(f"Query: {question}")
    logger.info(f"Routed to: {intent}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Handle based on intent
    # ═══════════════════════════════════════════════════════════════
    
    # ─────────────────────────────────────────────────────────────
    # DASHBOARD INVENTORY
    # ─────────────────────────────────────────────────────────────
    if intent == "DASHBOARD":
        try:
            result = dashboard_engine.answer_question(question)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Format response
            answer_text = result["answer"]
            
            # Add SQL transparency (optional)
            if result["sql"]:
                answer_text += f"\n\n<details>\n<summary>SQL Query</summary>\n\n```sql\n{result['sql']}\n```\n</details>"
            
            return format_response(
                chat_id=request.chatID,
                msg_id=generate_msg_id(),
                question=question,
                intent="DASHBOARD_INVENTORY",
                answer=answer_text,
                usage={
                    "prompt_tokens": 500,  # Approximate for Text-to-SQL
                    "completion_tokens": 100,
                    "total_tokens": 600
                }
            )
        
        except Exception as e:
            logger.error(f"Dashboard query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ─────────────────────────────────────────────────────────────
    # GOVERNANCE TABLES
    # ─────────────────────────────────────────────────────────────
    elif intent == "GOVERNANCE":
        try:
            # Extract LOB from question
            lob = extract_lob(question)
            
            # Query governance RAG
            result = answer_governance_question(
                question=question,
                vectorstore=governance_vectorstore,
                lob=lob
            )
            
            return format_response(
                chat_id=request.chatID,
                msg_id=generate_msg_id(),
                question=question,
                intent="GOVERNANCE",
                answer=result["answer"],
                usage=result.get("usage", {})
            )
        
        except Exception as e:
            logger.error(f"Governance query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ─────────────────────────────────────────────────────────────
    # MODEL DOCUMENTATION
    # ─────────────────────────────────────────────────────────────
    elif intent == "MODEL_DOC":
        try:
            # Your existing RAG flow
            lob = extract_lob(question)
            
            retriever = get_retriever(
                vectorstore=model_doc_vectorstore,
                question=question,
                lob=lob
            )
            
            answer, docs, usage, offer = run_rag(
                chain=chain,
                retriever=retriever,
                question=question,
                history=[],
                rewritten_query=question
            )
            
            return format_response(
                chat_id=request.chatID,
                msg_id=generate_msg_id(),
                question=question,
                intent="MODEL_DOC",
                answer=answer,
                usage=usage
            )
        
        except Exception as e:
            logger.error(f"Model doc query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ─────────────────────────────────────────────────────────────
    # GREETING
    # ─────────────────────────────────────────────────────────────
    elif intent == "GREETING":
        answer = handle_greeting(question)
        
        return format_response(
            chat_id=request.chatID,
            msg_id=generate_msg_id(),
            question=question,
            intent="GREETING",
            answer=answer,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
    
    # ─────────────────────────────────────────────────────────────
    # UNKNOWN / FALLBACK
    # ─────────────────────────────────────────────────────────────
    else:
        answer = """I'm not sure what you're asking about. I can help you with:

1. **Dashboard Inventory**: Questions about dashboards, MIS reports, automation levels, etc.
   Example: "How many dashboards are there in ABHI?"

2. **Governance Data**: Questions about source systems, CDEs, DQ rules
   Example: "Which source systems are critical in ABHFL?"

3. **Model Documentation**: Questions about analytical models
   Example: "What is the EWI model in ABSLI?"

Could you please rephrase your question?"""
        
        return format_response(
            chat_id=request.chatID,
            msg_id=generate_msg_id(),
            question=question,
            intent="UNKNOWN",
            answer=answer,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )


def format_response(chat_id, msg_id, question, intent, answer, usage):
    """Format the API response."""
    return {
        "chatID": chat_id,
        "msgID": msg_id,
        "query": question,
        "intent": intent,
        "answer": answer,
        "usage": usage,
        "timestamp": datetime.now().isoformat()
    }


def handle_greeting(question: str) -> str:
    """Handle greeting queries."""
    q_lower = question.lower()
    
    if any(word in q_lower for word in ["hi", "hello", "hey"]):
        return "Hello! I'm the COE Analytics assistant. I can help you with questions about dashboards, governance data, and analytical models. What would you like to know?"
    
    elif any(word in q_lower for word in ["thanks", "thank you"]):
        return "You're welcome! Let me know if you have any other questions."
    
    elif any(word in q_lower for word in ["bye", "goodbye"]):
        return "Goodbye! Feel free to come back anytime you have questions."
    
    else:
        return "Hello! How can I assist you today?"


def extract_lob(question: str) -> str:
    """Extract LOB from question."""
    import re
    lobs = {
        "ABCD": ["abcd", "finserv", "app", "digital"],
        "AMC": ["amc", "mutual fund"],
        "ABHFL": ["abhfl", "housing", "hfl", "home loan"],
        "ABHI": ["abhi", "health insurance"],
        "ABSLI": ["absli", "life insurance", "sli"],
        "ABCL BIU": ["abcl biu", "biu", "business intelligence"],
        "CAU": ["cau", "central analytics"],
    }
    
    q_lower = question.lower()
    
    for lob, aliases in lobs.items():
        if any(alias in q_lower for alias in aliases):
            return lob
    
    return "Unknown"


def generate_msg_id() -> str:
    """Generate unique message ID."""
    import uuid
    return str(uuid.uuid4())
