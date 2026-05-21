"""
rag/query_router.py
Intelligent router that detects query intent and routes to the appropriate engine:
- Dashboard Inventory (Text-to-SQL)
- Governance Tables (Table-RAG)
- Model Documentation (Standard RAG)
"""

import sys
import logging
from pathlib import Path
from typing import Literal, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Query intent types
QueryIntent = Literal["DASHBOARD", "GOVERNANCE", "MODEL_DOC", "GREETING", "UNKNOWN"]


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD PATTERNS FOR INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_KEYWORDS = {
    # Direct mentions
    "dashboard", "dashboards", "mis", "report", "reports",
    
    # Dashboard-specific columns
    "automation level", "automated", "distribution method",
    "frequency", "daily", "weekly", "monthly", "real-time",
    "business function", "data source", "data sources",
    "kpi", "kpis", "metrics tracked",
    "spoc", "stakeholders", "key users",
    
    # Dashboard-specific actions
    "how many dashboards", "count dashboards", "list dashboards",
    "which team", "which lob", "live dashboards", "wip dashboards",
    
    # Output types
    "dashboard", "mis", "dump", "data dump",
    
    # LOB names (in dashboard context)
    "abcd", "amc", "abhfl", "abhi", "absli", "abcl", "cau",
    
    # Status
    "live", "wip", "on hold", "discontinued",
    
    # Data sources
    "sap bo", "databricks", "bigquery", "finncorp", "finnrtl",
    "manual excel", "power bi",
}

GOVERNANCE_KEYWORDS = {
    # Direct mentions
    "source system", "source systems", "cde", "cdes",
    "critical data element", "critical data elements",
    "data quality", "dq rule", "dq rules",
    
    # Governance-specific terms
    "identified", "prioritized", "critical system",
    "field count", "fields identified",
    "governance", "data governance",
    
    # Specific source systems (governance context)
    "a3s", "finverse", "sfdc", "crm", "servosys",
    "jocata", "kepler", "sap",
    
    # Actions
    "which systems are critical", "how many cdes",
    "percentage of prioritized", "identify systems",
}

MODEL_DOC_KEYWORDS = {
    # Direct mentions
    "model", "models", "model documentation",
    "ewi", "retention", "churn", "propensity",
    "activation", "productivity", "engagement",
    
    # Model-specific terms
    "algorithm", "features", "training", "prediction",
    "target variable", "performance", "accuracy",
    "data pipeline", "feature engineering",
    
    # Actions
    "explain the model", "how does the model work",
    "what features", "what is the target",
}

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "thanks", "thank you", "bye", "goodbye",
}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class QueryRouter:
    """
    Intelligent router that detects query intent and routes to appropriate engine.
    """
    
    def __init__(self):
        """Initialize the router."""
        self.dashboard_keywords = DASHBOARD_KEYWORDS
        self.governance_keywords = GOVERNANCE_KEYWORDS
        self.model_doc_keywords = MODEL_DOC_KEYWORDS
        self.greeting_keywords = GREETING_KEYWORDS
    
    def detect_intent(self, question: str) -> QueryIntent:
        """
        Detect the intent of a user query.
        
        Args:
            question: User's natural language question
            
        Returns:
            QueryIntent enum value
        """
        q_lower = question.lower()
        
        # Count keyword matches for each intent
        dashboard_score = self._count_matches(q_lower, self.dashboard_keywords)
        governance_score = self._count_matches(q_lower, self.governance_keywords)
        model_doc_score = self._count_matches(q_lower, self.model_doc_keywords)
        greeting_score = self._count_matches(q_lower, self.greeting_keywords)
        
        logger.debug(f"Intent scores - Dashboard: {dashboard_score}, "
                    f"Governance: {governance_score}, "
                    f"Model Doc: {model_doc_score}, "
                    f"Greeting: {greeting_score}")
        
        # Greeting detection (highest priority)
        if greeting_score > 0 and len(q_lower.split()) <= 5:
            return "GREETING"
        
        # Dashboard detection
        if dashboard_score > 0:
            # Strong dashboard signals
            if any(phrase in q_lower for phrase in [
                "dashboard", "mis", "how many dashboards",
                "list dashboards", "count dashboards",
                "automation level", "distribution method"
            ]):
                return "DASHBOARD"
            
            # If only LOB name, check context
            if dashboard_score == 1 and governance_score > 0:
                # Could be either - check for governance-specific terms
                if any(term in q_lower for term in ["cde", "source system", "critical"]):
                    return "GOVERNANCE"
                else:
                    return "DASHBOARD"
        
        # Governance detection
        if governance_score > 0:
            if any(phrase in q_lower for phrase in [
                "source system", "cde", "dq rule",
                "critical system", "identified", "prioritized"
            ]):
                return "GOVERNANCE"
        
        # Model documentation detection
        if model_doc_score > 0:
            if any(phrase in q_lower for phrase in [
                "model", "algorithm", "features", "prediction"
            ]):
                return "MODEL_DOC"
        
        # Default: If mentions LOB without specific context, assume dashboard
        lobs = ["abcd", "amc", "abhfl", "abhi", "absli", "abcl", "cau"]
        if any(lob in q_lower for lob in lobs):
            return "DASHBOARD"
        
        # Unknown intent
        return "UNKNOWN"
    
    def _count_matches(self, text: str, keywords: set) -> int:
        """
        Count how many keywords match in the text.
        
        Args:
            text: Text to search (lowercase)
            keywords: Set of keywords to match
            
        Returns:
            Count of matching keywords
        """
        count = 0
        for keyword in keywords:
            if keyword in text:
                count += 1
        return count
    
    def route_query(self, question: str) -> Dict[str, Any]:
        """
        Detect intent and return routing information.
        
        Args:
            question: User's natural language question
            
        Returns:
            Dict with intent and routing metadata
        """
        intent = self.detect_intent(question)
        
        routing_info = {
            "intent": intent,
            "question": question,
        }
        
        # Add intent-specific metadata
        if intent == "DASHBOARD":
            routing_info["engine"] = "dashboard_inventory"
            routing_info["method"] = "text_to_sql"
            routing_info["description"] = "Query will be converted to SQL and executed against Dashboard Inventory"
        
        elif intent == "GOVERNANCE":
            routing_info["engine"] = "governance_rag"
            routing_info["method"] = "table_rag"
            routing_info["description"] = "Query will retrieve governance tables via RAG"
        
        elif intent == "MODEL_DOC":
            routing_info["engine"] = "model_doc_rag"
            routing_info["method"] = "standard_rag"
            routing_info["description"] = "Query will retrieve model documentation via RAG"
        
        elif intent == "GREETING":
            routing_info["engine"] = "greeting_handler"
            routing_info["method"] = "template"
            routing_info["description"] = "Simple greeting response"
        
        else:
            routing_info["engine"] = "fallback"
            routing_info["method"] = "clarification"
            routing_info["description"] = "Intent unclear, will ask for clarification"
        
        logger.info(f"Routed query to: {intent} ({routing_info['engine']})")
        
        return routing_info


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_query_intent(question: str) -> QueryIntent:
    """
    Quick function to detect query intent.
    
    Args:
        question: User's question
        
    Returns:
        QueryIntent enum value
    """
    router = QueryRouter()
    return router.detect_intent(question)


def route_query(question: str) -> Dict[str, Any]:
    """
    Quick function to get routing information.
    
    Args:
        question: User's question
        
    Returns:
        Routing information dict
    """
    router = QueryRouter()
    return router.route_query(question)


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test queries
    test_queries = [
        # Dashboard queries
        "How many dashboards are there in ABHI?",
        "List all live dashboards in ABSLI",
        "Which LOB has the most dashboards?",
        "Show me all fully automated dashboards",
        "What are the data sources for sales dashboards?",
        
        # Governance queries
        "Which source systems are critical in ABHFL?",
        "How many CDEs were identified for A3S?",
        "What percentage of CDEs are prioritized?",
        "List all critical systems with their CDE counts",
        
        # Model documentation queries
        "What is the EWI model in ABSLI?",
        "Explain the churn prediction model",
        "What features are used in the retention model?",
        
        # Greetings
        "Hi",
        "Hello, how are you?",
        "Thanks!",
        
        # Ambiguous
        "Tell me about ABHI",
        "What do you know about sales?",
    ]
    
    router = QueryRouter()
    
    print("="*80)
    print("QUERY ROUTING TESTS")
    print("="*80)
    
    for query in test_queries:
        routing = router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {routing['intent']}")
        print(f"Engine: {routing['engine']}")
        print(f"Method: {routing['method']}")
        print("-"*80)
