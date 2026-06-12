"""
rag/router.py
Intents: INVENTORY, RAG, DASHBOARD, UTILITY, or GREETING.
Modified to use Azure OpenAI.

Changes vs previous version:
- Added Intent.UTILITY
- Added RouterDecision.category field (drives both SharePoint link + store selection)
- Replaced hardcoded utility_link with SHAREPOINT_LINKS registry
- Added format_link_html() helper consumed by main.py
- Updated ROUTER_PROMPT: UTILITY intent + category instruction + disambiguation rules
- Removed URL from prompt (LLM never generates URLs)
- Updated last_intent context hint for UTILITY
- BUTTON_CLICKS now use format_link_html() per category
- Fix 1: UTILITY_KEYWORDS pre-LLM keyword list covers business-function language
         and all actual Category_Tags from the utility inventory CSV
- Fix 2: ROUTER_PROMPT examples expanded with real utility names and natural phrasing
- Fix 3: UTILITY vs RAG disambiguation rule added (existence query vs concept query)
"""

import logging
import json
import re
from enum import Enum
from pydantic import BaseModel
from openai import AzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SharePoint link registry — single source of truth
# Never put URLs inside the LLM prompt; always do a deterministic lookup here.
# ──────────────────────────────────────────────────────────────────────────────

SHAREPOINT_LINKS: dict = {
    "model_inventory":     "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FModel%20Inventory",
    "model_documentation": "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FModel%20Documentation",
    "standards":           "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FStandards",
    "dashboards":          "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FDashboards",
    "cde":                 "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FCDE",
    "framework":           "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FFramework",
    "newsletter":          "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FNewsletter",
    "utility":             "https://adityabirlacapital.sharepoint.com/sites/Analytics_CoE/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=He1O87&TeamsCID=f101356f%2D9637%2D4954%2Da8ab%2Dae9954837ccc&CID=e7e7c96d%2D66af%2D4e6c%2D9d65%2De583cda62c73&FolderCTID=0x01200055834EDD23BAC84C8E8885F55F041730&id=%2Fsites%2FAnalytics%5FCoE%2FShared%20Documents%2FCOE%20Initiatives%2FKnowledge%5FBase%5FJune%2726%2FUtility",
}

_FALLBACK_LINK = SHAREPOINT_LINKS["utility"]   # generic fallback


def get_sharepoint_link(category: str) -> str:
    """Deterministic URL lookup. Never call the LLM for a URL."""
    return SHAREPOINT_LINKS.get(category.lower(), _FALLBACK_LINK)


def format_link_html(category: str, label: str = "📁 Open in SharePoint") -> str:
    """Return a ready-to-embed HTML anchor for the given category."""
    url = get_sharepoint_link(category)
    return f"<a href='{url}' target='_blank'>{label}</a>"


# ──────────────────────────────────────────────────────────────────────────────
# Intent Enum
# ──────────────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    INVENTORY = "INVENTORY"
    RAG       = "RAG"
    DASHBOARD = "DASHBOARD"
    UTILITY   = "UTILITY"       # ← NEW
    GREETING  = "GREETING"


class RouterDecision(BaseModel):
    intent:    Intent
    reasoning: str
    category:  str = "utility"  # safe default — never breaks old callers


# ──────────────────────────────────────────────────────────────────────────────
# Button click patterns — exact matches from Unfyd frontend
# ──────────────────────────────────────────────────────────────────────────────

BUTTON_CLICKS = {
    "model inventory": {
        "intent":  Intent.GREETING,
        "message": (
            "You are now in **Model Inventory** mode. "
            "I can help you with the questions similar to:\n"
            "- Can you share me the count of Models across all LOBs?\n"
            "- Can you share me break up of the Models across LOB?\n"
            "- Hi, I am a new joinee at ABSLAMC, can you tell me which all models are created by my team?\n"
            "- Hi, I am an existing ABCD member, can you tell me how many models are created by my team and across which functions?\n\n"
            "What would you like to know?\n"
            + format_link_html("model_inventory", "📁 Open Model Inventory (SharePoint)")
        ),
    },
    "model documentation": {
        "intent":  Intent.GREETING,
        "message": (
            "You are now in **Model Documentation** mode. "
            "I can help you with:\n"
            "- Can you give me overview of the ABHI_FLS Attrition Prediction?\n"
            "- Can you share explainability & insights about the model?\n"
            "- Give me the objective of ABSLAMC_SIP Cancellation_V1\n"
            "- Can you share details of Methodology used for this model?\n\n"
            "What would you like to know?\n"
            + format_link_html("model_documentation", "📁 Open Model Documentation (SharePoint)")
        ),
    },
    "standards & best practices": {
        "intent":  Intent.GREETING,
        "message": (
            "You are now in **Standards & Best Practices** mode. "
            "I can help you with:\n"
            "- What is MLOps?\n"
            "- What are the Key Pillars of MLOps?\n"
            "- How do I implement Model Monitoring?\n\n"
            "What would you like to know?\n"
            + format_link_html("standards", "📁 Open Standards (SharePoint)")
        ),
    },
    "enterprise reports & dashboards inventory": {
        "intent":  Intent.GREETING,
        "message": (
            "You are now in **Enterprise Reports & Dashboards** mode. "
            "I can help you with:\n"
            "- How many dashboards are there in ABHI?\n"
            "- Which LOB has the most dashboards?\n"
            "- List all live sales dashboards in ABSLI\n"
            "- What are the data sources for Operations dashboards?\n\n"
            "What would you like to know?\n"
            + format_link_html("dashboards", "📁 Open Dashboard Inventory (SharePoint)")
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Greeting patterns — detected before calling LLM
# ──────────────────────────────────────────────────────────────────────────────

GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "good night", "how are you",
    "who are you", "thanks", "thank you",
    "bye", "goodbye",
]


# ──────────────────────────────────────────────────────────────────────────────
# Router Prompt
# ──────────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """
You are an intent classifier for a COE Analytics chatbot with FIVE intents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GREETING
   - Casual conversation, greetings, thanks, help requests
   - Button clicks: "Model Inventory", "Model Documentation",
     "Standards & Best Practices", "Enterprise Reports & Dashboards Inventory"
   - Any vague or non-specific message

2. INVENTORY (Excel / DuckDB)
   - Enterprise Model Inventory: model names, LOBs, Functions,
     Status, Owners, Timelines, Documentation availability
   - Counting, listing, filtering MODELS by any dimension

3. DASHBOARD (CSV / DuckDB)
   - Dashboard/MIS/Report repository: dashboard names, LOBs,
     Business Functions, Status, Data Sources, Frequency,
     Automation Level, Distribution Method, Teams, Stakeholders
   - Counting, listing, filtering DASHBOARDS / MIS / Reports

4. UTILITY (DuckDB — Utility Inventory)
   The utility inventory contains internal COE-built solutions including:
     - AI/ML platforms, automation scripts, NLP/voice engines
     - Credit assessment tools, risk monitoring systems
     - Data extraction & web scraping utilities
     - Model documentation generators
     - Customer analytics & linkage tools
     - OCR and document intelligence platforms

   Route to UTILITY when the user asks about:
   a) EXISTENCE — "is there a tool/solution/utility for X?"
      "do we have anything for X?", "any automation for X?"
   b) DISCOVERY — "what tools/utilities/solutions does [LOB] have?"
      "list all COE tools", "show me what's available"
   c) SPECIFIC NAMED TOOLS — any question about a tool by name:
      "Voice Analytics Engine", "Family ID", "Web Scraping Utility",
      "CAM Automation", "FCHR", "Multi Enquiry Trigger",
      "Model Documentation Utility", "Property Document Intelligence"
   d) SPOC / OWNERSHIP — "who built X?", "who is the SPOC for X?",
      "which team owns the web scraping tool?"
   e) STATUS / METRICS — "which utilities are live?",
      "what are the success metrics for COE tools?",
      "how many utilities are in testing phase?"
   f) BUSINESS FUNCTION SEARCH — "any tool for credit underwriting?",
      "is there something for risk monitoring?",
      "what does COE have for document automation?"

5. RAG (Vector Store — ML)
   - How a specific model works: methodology, algorithms, features
   - Model documentation content: objectives, target variables
   - Standards, best practices, MLOps guidelines
   - Newsletter content, CDE definitions, governance policies
   - Conceptual explanations: "what is X?", "how does X work?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALWAYS ROUTE TO GREETING for:
- Hi, Hello, Hey, Good morning/afternoon/evening
- How are you, What can you do, Who are you
- Thank you, Thanks, Bye, Goodbye
- Button clicks (listed above)
- Any vague or non-specific message
- Short continuations like "yes", "no", "sure", "please",
  "tell me more", "show me", "list them", "go ahead"
  MUST use the previous intent context — NEVER classify as GREETING.

ALWAYS ROUTE TO INVENTORY for ANY question involving AI/ML MODELS:
- Counting: "how many models", "count models", "total models"
- Listing:  "list models", "show models", "which models"
- Status:   "live models", "WIP models", "in progress models"
- Ownership: "who owns the model", "models owned by"
- Filtering by LOB, Function, Status, Timeline, Documentation

ALWAYS ROUTE TO DASHBOARD for ANY question involving DASHBOARDS/MIS:
- "how many dashboards", "list dashboards", "which dashboards"
- Dashboard status, automation level, frequency, distribution
- Data sources for dashboards (SAP BO, Databricks, BigQuery, FinnCorp)
- Business function: "sales dashboards", "operations dashboards"

ALWAYS ROUTE TO UTILITY for:
- ANY question about whether a tool/solution/automation EXISTS
- ANY question about a named COE utility or platform
- ANY "do we have / is there / any tool for" phrasing
- Listing, counting, or filtering COE-built tools/solutions
- SPOC, ownership, status, or success metrics of utilities

ONLY ROUTE TO RAG for:
- How a specific AI/ML model works technically (methodology)
- Conceptual questions: "what is MLOps?", "explain X"
- Policy, governance, or standards document content
- Newsletter content, CDE field definitions
- Exhaustive model details requiring document reading

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DISAMBIGUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEYWORD PRIORITY:
  "model" / "models"             → INVENTORY  (AI/ML model registry)
  "dashboard" / "MIS" / "report" → DASHBOARD
  "utility" / "tool" / "script" / "solution" / "platform" / "engine"
                                 → UTILITY
  "how does" / "methodology" / "explain" / "what is"
                                 → RAG

UTILITY vs RAG — THE KEY DISTINCTION:
  The question is: does the user want to FIND something, or UNDERSTAND something?

  FIND (existence / discovery / capability) → UTILITY
    "Is there a voice analytics solution?"           → UTILITY
    "Do we have an OCR tool for property documents?" → UTILITY
    "What does the Voice Analytics Engine do?"       → UTILITY  ← named tool = UTILITY
    "Tell me about the Family ID utility"            → UTILITY
    "Who built the CAM automation platform?"         → UTILITY
    "Which COE tools are in testing phase?"          → UTILITY
    "Any automation for credit assessment?"          → UTILITY

  UNDERSTAND (concept / methodology) → RAG
    "How does voice analytics work?"                 → RAG  ← generic concept
    "What is sentiment analysis?"                    → RAG
    "Explain OCR technology"                         → RAG

  RULE: If a specific named COE tool/platform is mentioned → always UTILITY.
        If the question is generic / conceptual with no named tool → RAG.

DATA SOURCE CONTEXT:
  "Data sources used in [LOB]" (no other context) → DASHBOARD
  "Data sources for the [MODEL] model"             → RAG
  "Which models use [DATA_SOURCE]"                 → INVENTORY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GREETING:
"Hi"                                                -> GREETING
"Hello"                                             -> GREETING
"Good afternoon"                                    -> GREETING
"What can you help me with?"                        -> GREETING
"Model Inventory"                                   -> GREETING
"Model Documentation"                               -> GREETING
"Standards & Best Practices"                        -> GREETING
"Enterprise Reports & Dashboards Inventory"         -> GREETING

INVENTORY:
"How many live models are there in SLI?"            -> INVENTORY
"How many models does Life Insurance have?"         -> INVENTORY
"List all live SLI models"                          -> INVENTORY
"How many WIP models are in Cross-Sell?"            -> INVENTORY
"Which AMC models have documentation?"              -> INVENTORY
"Who owns the fraud detection model?"               -> INVENTORY

DASHBOARD:
"How many dashboards are there in ABHI?"            -> DASHBOARD
"Which LOB has the most dashboards?"                -> DASHBOARD
"List all live sales dashboards in ABSLI"           -> DASHBOARD
"How many dashboards are fully automated?"          -> DASHBOARD
"What are the data sources for sales dashboards?"   -> DASHBOARD
"Which dashboards are updated daily?"               -> DASHBOARD

UTILITY — natural business language (no "utility" keyword needed):
"What utilities are available for ABCD?"            -> UTILITY
"Is there a tool for automated CAM generation?"     -> UTILITY
"Do we have anything for web scraping?"             -> UTILITY
"Is there an OCR solution for property documents?"  -> UTILITY
"What does the Voice Analytics Engine do?"          -> UTILITY
"Tell me about the Family ID utility"               -> UTILITY
"Is there something for credit underwriting automation?" -> UTILITY
"Which COE tools are in testing phase?"             -> UTILITY
"Who is the SPOC for the web scraping tool?"        -> UTILITY
"What are the success metrics for COE utilities?"   -> UTILITY
"Any tool that helps prevent risky disbursals?"     -> UTILITY
"Show me document intelligence solutions"           -> UTILITY
"What automation tools does COE have?"              -> UTILITY
"List all utilities built by COE"                   -> UTILITY
"Which utilities does ABFL-CA have?"                -> UTILITY
"Is there a model documentation generator?"         -> UTILITY

RAG — concepts, methodology, standards:
"How does the persistency model work?"              -> RAG
"What methodology is used for fraud?"               -> RAG
"Explain the churn prediction approach"             -> RAG
"Give me details of the App-Digi Intent model"      -> RAG
"Share exhaustive details for the MF Intent Model"  -> RAG
"What is MLOps?"                                    -> RAG
"What was in the Dec25 newsletter?"                 -> RAG
"How does voice analytics work?"                    -> RAG
"What is sentiment analysis?"                       -> RAG
"Explain OCR technology"                            -> RAG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT FLOWCHART
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: GREETING keywords present?                  → GREETING
Step 2: "model" / "models" keyword?                 → INVENTORY
Step 3: "dashboard" / "MIS" keyword?                → DASHBOARD
Step 4: Named COE tool/platform mentioned?          → UTILITY
Step 5: "is there" / "do we have" / "any tool for"? → UTILITY
Step 6: "utility" / "tool" / "solution" / "engine"? → UTILITY
Step 7: "how does" / "methodology" / "explain" / "what is"? → RAG
Step 8: Data source context (see above)
Step 9: Default (no keywords matched)               → INVENTORY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY FIELD — REQUIRED IN EVERY RESPONSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Always include a "category" field identifying the knowledge area.
This drives SharePoint link selection — do NOT put any URL in your response.

Allowed values:
  "model_inventory"     → model counts, lists, status, owners
  "model_documentation" → how a model works, methodology
  "standards"           → MLOps, governance, best practices
  "dashboards"          → dashboard/MIS/report inventory
  "cde"                 → critical data element definitions
  "framework"           → data & analytics framework docs
  "newsletter"          → COE newsletter content
  "utility"             → internal tools, calculators, utilities

Category examples:
"How many models in ABSLI?"                    → "model_inventory"
"Methodology of the PL Intent model"           → "model_documentation"
"What is MLOps?"                               → "standards"
"List dashboards using Databricks"             → "dashboards"
"What utilities exist for campaign sizing?"    → "utility"
"What was in the Dec25 newsletter?"            → "newsletter"
"What are the CDE definitions for ABHI?"       → "cde"
"Tell me about the analytics framework"        → "framework"

Return valid JSON only — no markdown, no preamble:
{"intent": "GREETING|INVENTORY|DASHBOARD|UTILITY|RAG", "reasoning": "...", "category": "..."}
"""


# ──────────────────────────────────────────────────────────────────────────────
# General greeting response
# ──────────────────────────────────────────────────────────────────────────────

GENERAL_GREETING_RESPONSE = (
    "Hello! I am Converge Knowledge, your COE Analytics AI assistant. "
    "I can help you with:\n\n"
    "**Model Inventory** — Count, list, and filter AI/ML models "
    "by LOB, function, status, owner, and timeline.\n\n"
    "**Dashboard Inventory** — Count, list, and filter dashboards, MIS reports "
    "by LOB, business function, status, automation level, and data sources.\n\n"
    "**Model Documentation** — Learn how models work, their "
    "methodology, processes, and business objectives.\n\n"
    "**MLOps & Standards** — Learn how to operationalize ML models in production "
    "through orchestration, monitoring & governance.\n\n"
    "**Utilities** — Discover internal COE tools, calculators, "
    "scripts, and templates available across LOBs.\n\n"
    "What would you like to explore today?"
)


# ──────────────────────────────────────────────────────────────────────────────
# classify_intent — main function
# ──────────────────────────────────────────────────────────────────────────────

def classify_intent(question: str, last_intent: str = "") -> RouterDecision:

    question_lower = question.strip().lower()

    # ── Step 1: Button clicks (exact match, before LLM) ──────────────────
    for button_text, config in BUTTON_CLICKS.items():
        if question_lower == button_text or question_lower.startswith(button_text):
            logger.info(f"Button click detected: '{question}' → GREETING")
            return RouterDecision(
                intent=config["intent"],
                reasoning=f"Button click: {question}",
                category="utility",        # default; button responses carry their own link
            )

    # ── Step 2: Greeting patterns (before LLM) ────────────────────────────
    for pattern in GREETING_PATTERNS:
        if (question_lower == pattern
                or question_lower.startswith(pattern + " ")
                or question_lower.endswith(" " + pattern)):
            logger.info(f"Greeting pattern detected: '{question}' → GREETING")
            return RouterDecision(
                intent=Intent.GREETING,
                reasoning=f"Greeting pattern matched: {pattern}",
                category="utility",
            )

    # ── Step 3: Very short messages → GREETING ────────────────────────────
    data_keywords = {
        "live", "wip", "model", "count", "list", "how many", "show",
        "absli", "abhi", "abcd", "dashboard", "mis", "report",
        "utility", "tool", "calculator", "solution", "platform", "engine",
        "script", "automat",
    }
    if len(question.strip()) <= 10 and not any(kw in question_lower for kw in data_keywords):
        logger.info(f"Short message detected: '{question}' → GREETING")
        return RouterDecision(
            intent=Intent.GREETING,
            reasoning="Short non-specific message",
            category="utility",
        )

    # ── Step 3b: UTILITY keyword pre-check (before LLM) ──────────────────
    # Covers business-function language users naturally use — they won't say
    # "utility", they'll say "is there a tool for X" or name the platform.
    # Ordered from most specific (named tools) to most general.
    UTILITY_KEYWORDS = [
        # Named COE tools from the utility inventory
        "voice analytics engine", "family id", "web scraping utility",
        "cam automation", "fchr", "multi enquiry trigger",
        "model documentation utility", "property document intelligence",
        "bl fchr", "bل fchr",
        # Category tags
        "document intelligence", "ocr solution", "ocr tool",
        "voice analytics", "sentiment analytics", "credit assessment tool",
        "risk monitoring tool", "data extraction tool",
        # Existence / discovery phrasing — strongest UTILITY signals
        "is there a tool", "is there an", "do we have a", "do we have an",
        "any tool for", "any utility for", "any solution for",
        "any automation for", "something for automating",
        "is there something", "do we have something",
        # General capability search
        "what tools", "what utilities", "what solutions", "what automation",
        "list all utilities", "list all tools", "list coe utilities",
        "show me utilities", "show me tools", "available utilities",
        "available tools", "coe tools", "coe utilities",
        # SPOC / ownership of tools
        "spoc for the", "who built the", "who owns the tool",
        "technical spoc", "who developed the",
        # Status / metrics of tools
        "utilities in testing", "tools in testing", "tools in development",
        "success metrics for", "utilities deployed", "tools deployed",
    ]
    if any(kw in question_lower for kw in UTILITY_KEYWORDS):
        logger.info(f"Utility keyword pre-check matched: '{question}' → UTILITY")
        return RouterDecision(
            intent=Intent.UTILITY,
            reasoning="Utility keyword pre-check matched",
            category="utility",
        )

    # ── Step 4: Build last-intent context hint ────────────────────────────
    context_hint = ""
    if last_intent == "INVENTORY":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the MODEL INVENTORY. "
            "If this is a follow-up (e.g. 'list all', 'show me', 'which ones'), "
            "route to INVENTORY."
        )
    elif last_intent == "DASHBOARD":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the DASHBOARD INVENTORY. "
            "If this is a follow-up, route to DASHBOARD."
        )
    elif last_intent == "RAG":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the RAG documents. "
            "If this is a follow-up about the same topic, route to RAG."
        )
    elif last_intent == "UTILITY":
        context_hint = (
            "\nCONTEXT: The previous question was about COE Utilities. "
            "If this is a follow-up about utilities or tools, route to UTILITY."
        )

    # ── Step 5: Call Azure OpenAI for classification ──────────────────────
    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    messages = [
        {"role": "system", "content": ROUTER_PROMPT + context_hint},
        {"role": "user",   "content": question},
    ]

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    # ── Step 6: Parse response ────────────────────────────────────────────
    text = response.choices[0].message.content.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    try:
        data     = json.loads(text)
        decision = RouterDecision(
            intent=Intent(data["intent"].upper()),
            reasoning=data.get("reasoning", ""),
            category=data.get("category", "utility").lower(),   # ← NEW
        )
        logger.info(
            f"Routed → intent={decision.intent} | "
            f"category={decision.category} | "
            f"reason={decision.reasoning}"
        )
        return decision

    except Exception as e:
        logger.error(f"Router parse failed: {e} | raw: {text}")
        return RouterDecision(
            intent=Intent.RAG,
            reasoning="parse fallback",
            category="utility",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helper — get greeting response for main.py
# ──────────────────────────────────────────────────────────────────────────────

def get_greeting_response(question: str) -> str:
    """
    Returns appropriate greeting message based on question.
    Called from main.py when intent == GREETING.
    """
    question_lower = question.strip().lower()

    for button_text, config in BUTTON_CLICKS.items():
        if question_lower == button_text or question_lower.startswith(button_text):
            return config["message"]

    return GENERAL_GREETING_RESPONSE
