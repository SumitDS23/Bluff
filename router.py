"""
rag/router.py
Intents: INVENTORY, RAG, or GREETING.
Modified to use Azure OpenAI instead of Gemini/Gemma.
"""

import logging
import json
import re
from enum import Enum
from pydantic import BaseModel
from openai import AzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Intent Enum — added GREETING
# ------------------------------------------------------------------------------

class Intent(str, Enum):
    INVENTORY = "INVENTORY"
    RAG       = "RAG"
    GREETING  = "GREETING"


class RouterDecision(BaseModel):
    intent:    Intent
    reasoning: str

# ------------------------------------------------------------------------------
# Button click patterns — exact matches from Unfyd frontend
# ------------------------------------------------------------------------------

BUTTON_CLICKS = {
    "model inventory": {
        "intent": Intent.GREETING,
        "message": (
            "You are now in **Model Inventory** mode. "
            "I can help you with:\n"
            "- Count of models by LOB, Function, Status\n"
            "- Live vs WIP model breakdown\n"
            "- Model ownership and timelines\n"
            "- Documentation availability\n\n"
            "What would you like to know?"
        )
    },
    "model documentation": {
        "intent": Intent.GREETING,
        "message": (
            "You are now in **Model Documentation** mode. "
            "I can help you with:\n"
            "- Model methodology and approach\n"
            "- How specific models work\n"
            "- Process explanations\n"
            "- Technical documentation\n\n"
            "What would you like to know?"
        )
    },
    "standards & best practices": {
        "intent": Intent.GREETING,
        "message": (
            "You are now in **Standards & Best Practices** mode. "
            "I can help you with:\n"
            "- Analytics standards and guidelines\n"
            "- Best practices for model development\n"
            "- Policy and compliance content\n\n"
            "What would you like to know?"
        )
    },
    "enterprise reports & dashboards inventory": {
        "intent": Intent.GREETING,
        "message": (
            "You are now in **Enterprise Reports & Dashboards** mode. "
            "I can help you with:\n"
            "- Enterprise reporting standards\n"
            "- Dashboard inventory and guidelines\n"
            "- Report documentation\n\n"
            "What would you like to know?"
        )
    },
}

# ------------------------------------------------------------------------------
# Greeting patterns — detected before calling LLM
# ------------------------------------------------------------------------------

GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "good night", "how are you", "what can you do",
    "who are you", "help", "what do you do", "thanks", "thank you",
    "bye", "goodbye", "ok", "okay", "sure", "great", "awesome",
    "nice", "cool", "got it", "understood", "welcome",
]

# ------------------------------------------------------------------------------
# Router Prompt — updated to include GREETING
# ------------------------------------------------------------------------------

ROUTER_PROMPT = """
You are an intent classifier for a COE Analytics chatbot with THREE intents:

1. GREETING  - Casual conversation, greetings, thanks, help requests,
               button clicks like "Model Inventory", "Model Documentation",
               "Standards & Best Practices", "Enterprise Reports & Dashboards",
               or any message that is not a specific data question.

2. INVENTORY (Excel) - Enterprise Model Inventory containing model names,
   LOBs, Functions, Status, Owners, Timelines, Documentation availability.

3. RAG (Vector Store) - Internal documents about analytics methodologies,
   model processes, documentation content, company policies.

ALWAYS ROUTE TO GREETING for:
- Hi, Hello, Hey, Good morning/afternoon/evening
- How are you, What can you do, Who are you
- Thank you, Thanks, Bye, Goodbye
- Button clicks: "Model Inventory", "Model Documentation",
  "Standards & Best Practices", "Enterprise Reports & Dashboards Inventory"
- Any vague or non-specific message

ALWAYS ROUTE TO INVENTORY for ANY question that involves:
- Counting models: "how many", "count", "total", "number of"
- Listing models: "list", "show", "which models", "what models"
- Model status: "live", "WIP", "in progress"
- Model ownership: "who owns", "owned by"
- Filtering by LOB: SLI, AMC, HI, HFC, Life Insurance, Mutual Funds, etc.
- Filtering by Function: Cross-Sell, HR, Fraud, etc.
- Model details from the inventory: timeline, documentation availability

ONLY ROUTE TO RAG for questions about:
- How a model works technically
- Methodology or approach used
- Process explanations
- Policy or guideline content
- Concepts that require reading a document

CRITICAL EXAMPLES:
"Hi"                                          -> GREETING
"Hello"                                       -> GREETING
"Good afternoon"                              -> GREETING
"What can you help me with?"                  -> GREETING
"Model Inventory"                             -> GREETING
"Model Documentation"                         -> GREETING
"Standards & Best Practices"                  -> GREETING
"Enterprise Reports & Dashboards Inventory"   -> GREETING
"How many live models are there in SLI?"      -> INVENTORY
"How many models does Life Insurance have?"   -> INVENTORY
"List all live SLI models"                    -> INVENTORY
"How many WIP models are in Cross-Sell?"      -> INVENTORY
"Which AMC models have documentation?"        -> INVENTORY
"Who owns the fraud detection model?"         -> INVENTORY
"How many models are live?"                   -> INVENTORY
"How does the persistency model work?"        -> RAG
"What methodology is used for fraud?"         -> RAG
"Explain the churn prediction approach"       -> RAG

Return valid JSON only in this format:
{"intent": "GREETING" or "INVENTORY" or "RAG", "reasoning": "your reasoning here"}
"""

# ------------------------------------------------------------------------------
# Greeting responses for general greetings
# ------------------------------------------------------------------------------

GENERAL_GREETING_RESPONSE = (
    "Hello! I am Converge Knowledge, your COE Analytics AI assistant. "
    "I can help you with:\n\n"
    "📊 **Model Inventory** — Count, list, and filter AI/ML models "
    "by LOB, function, status, owner, and timeline.\n\n"
    "📄 **Model Documentation** — Learn how models work, their "
    "methodology, processes, and business objectives.\n\n"
    "What would you like to explore today?"
)

# ------------------------------------------------------------------------------
# classify_intent — main function
# ------------------------------------------------------------------------------

def classify_intent(question: str, last_intent: str = "") -> RouterDecision:

    question_lower = question.strip().lower()

    # ── Step 1: Check button clicks first (exact match) ───────────────────
    for button_text, config in BUTTON_CLICKS.items():
        if question_lower == button_text or question_lower.startswith(button_text):
            logger.info(f"Button click detected: '{question}' → GREETING")
            return RouterDecision(
                intent=config["intent"],
                reasoning=f"Button click: {question}"
            )

    # ── Step 2: Check greeting patterns (before calling LLM) ──────────────
    for pattern in GREETING_PATTERNS:
        if question_lower == pattern or question_lower.startswith(pattern + " ") \
                or question_lower.endswith(" " + pattern):
            logger.info(f"Greeting pattern detected: '{question}' → GREETING")
            return RouterDecision(
                intent=Intent.GREETING,
                reasoning=f"Greeting pattern matched: {pattern}"
            )

    # ── Step 3: Very short messages → GREETING ────────────────────────────
    if len(question.strip()) <= 10 and not any(
        kw in question_lower for kw in [
            "live", "wip", "model", "count", "list",
            "how many", "show", "absli", "abhi", "abcd"
        ]
    ):
        logger.info(f"Short message detected: '{question}' → GREETING")
        return RouterDecision(
            intent=Intent.GREETING,
            reasoning="Short non-specific message"
        )

    # ── Step 4: Build context hint based on last intent ───────────────────
    context_hint = ""
    if last_intent == "INVENTORY":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the INVENTORY. "
            "If this question is a follow-up or continuation "
            "(e.g. 'list all', 'show me', 'which ones', 'can you list'), "
            "route to INVENTORY."
        )
    elif last_intent == "RAG":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the RAG documents. "
            "If this question is a follow-up about the same topic, route to RAG."
        )

    # ── Step 5: Call Azure OpenAI for classification ───────────────────────
    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    messages = [
        {
            "role": "system",
            "content": ROUTER_PROMPT + context_hint
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    # ── Step 6: Parse response ─────────────────────────────────────────────
    text = response.choices[0].message.content.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    try:
        data     = json.loads(text)
        decision = RouterDecision(
            intent=Intent(data["intent"].upper()),
            reasoning=data.get("reasoning", ""),
        )
        logger.info(
            f"Routed to: {decision.intent} | "
            f"Reason: {decision.reasoning}"
        )
        return decision

    except Exception as e:
        logger.error(f"Router parse failed: {e} | raw: {text}")
        return RouterDecision(
            intent=Intent.RAG,
            reasoning="parse fallback"
        )


# ------------------------------------------------------------------------------
# Helper — get greeting response for main.py
# ------------------------------------------------------------------------------

def get_greeting_response(question: str) -> str:
    """
    Returns appropriate greeting message based on question.
    Called from main.py when intent == GREETING.
    """
    question_lower = question.strip().lower()

    # Check button clicks first
    for button_text, config in BUTTON_CLICKS.items():
        if question_lower == button_text or \
                question_lower.startswith(button_text):
            return config["message"]

    # General greeting
    return GENERAL_GREETING_RESPONSE
