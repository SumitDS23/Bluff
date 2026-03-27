"""
rag/inventory.py
Text-to-SQL inventory engine.
Primary: LLM generates SQL from natural language question.
Fallback: filter-based query.
Uses Azure OpenAI.
"""

import json
import logging
import re
import pandas as pd
import duckdb
from pathlib import Path
from openai import AzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

_df: pd.DataFrame = None

# ------------------------------------------------------------------------------
# Azure OpenAI client
# ------------------------------------------------------------------------------

def _get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

# ------------------------------------------------------------------------------
# SCHEMA
# ------------------------------------------------------------------------------

SCHEMA_INFO = """
You are querying a DuckDB table called `df` with these exact columns:

| Column                | Type    | Known Values                                                                 |
|-----------------------|---------|------------------------------------------------------------------------------|
| S_No                  | INTEGER | Row identifier                                                               |
| LOB                   | TEXT    | ABCD, ABFL-CA, ABFL-RA, ABHFL, ABHI, ABSLAMC, ABSLI, CAU                    |
| Function              | TEXT    | Collection, Cross-Sell, Distribution, Engagement, Finance & Planning,        |
|                       |         | Foreclosure, Health Management, HR, Login, Onboarding, Pre-Sales,            |
|                       |         | Prospecting, Revenue Assurance, Revenue Retention, Risk Management,          |
|                       |         | Service, Underwriting, Up-Sell                                               |
| Name_of_the_Model     | TEXT    | (free text)                                                                  |
| Model_Description     | TEXT    | (free text)                                                                  |
| ML_Non_ML             | TEXT    | ML, Non-ML                                                                   |
| Status                | TEXT    | LIVE, WIP                                                                    |
| Time_Line             | TEXT    | FY25, FY26 End, etc.                                                         |
| Owner                 | TEXT    | (free text)                                                                  |
| Document_Availability | TEXT    | Yes, No                                                                      |

IMPORTANT:
- Column names must match EXACTLY as above.
- "Name of the Model" → Name_of_the_Model
- "ML/Non-ML"        → ML_Non_ML
- "Time Line"        → Time_Line
- "Document Availability" → Document_Availability

LOB Mapping (use EXACT values in SQL):
- ABSLI / Life Insurance / SLI / Sun Life     -> 'ABSLI'
- ABHI  / Health Insurance / HI               -> 'ABHI'
- ABHFL / Housing Finance / HFL / HFC         -> 'ABHFL'
- ABSLAMC / Mutual Funds / AMC                -> 'ABSLAMC'
- ABCD / Bajaj Digital                        -> 'ABCD'
- CAU  / Central Analytics Unit               -> 'CAU'
- ABFL-CA / Finance Consumer / FL CA          -> 'ABFL-CA'
- ABFL-RA / Finance Risk / FL RA              -> 'ABFL-RA'

Status aliases:
- live / running      -> 'LIVE'
- wip / in progress   -> 'WIP'

ML aliases:
- ml / machine learning  -> 'ML'
- non ml / non-ml        -> 'Non-ML'

Function normalization:
- "Finance & Planning" should be matched with LIKE '%finance%planning%' or '%finance & planning%'
- Multi-word functions like "Revenue Retention", "Risk Management", "Health Management"
  should use LIKE with % on both sides
"""

TEXT_TO_SQL_PROMPT = f"""
{SCHEMA_INFO}

You are a Text-to-SQL expert. Convert the user's natural language question into a
valid DuckDB SQL query against the `df` table.

RULES:
1. Always use LOWER() with LIKE for LOB, Function, Owner, Name_of_the_Model comparisons
   e.g. LOWER(LOB) LIKE '%absli%'
2. For Status and ML_Non_ML use exact case-insensitive match:
   e.g. LOWER(Status) = 'live'
3. For COUNT queries   : SELECT COUNT(*) as count FROM df WHERE ...
4. For LIST queries    : SELECT Name_of_the_Model, LOB, Function, Status, Owner, Time_Line FROM df WHERE ...
5. For DESCRIBE queries: SELECT * FROM df WHERE ...
6. For BREAKDOWN       : SELECT <column>, COUNT(*) as model_count FROM df WHERE ... GROUP BY <column> ORDER BY model_count DESC
7. Only generate SELECT statements — never DROP, INSERT, UPDATE, DELETE
8. Return ONLY the SQL query — no explanation, no markdown, no backticks

FOLLOW-UP CONTEXT:
If the conversation history shows prior filters (e.g. LOB was ABSLI),
carry those filters forward into the WHERE clause unless:
1. The new question explicitly mentions a DIFFERENT LOB   -> topic shift
2. The new question contains global signals like "across all LOBs",
   "all lines of business", "overall", "entire inventory", "all LOBs" -> DROP LOB filter
3. The new question explicitly mentions "all" before any entity -> don't restrict by LOB

EXAMPLES:
"How many live models in ABSLI?"
SELECT COUNT(*) as count FROM df WHERE LOWER(LOB) LIKE '%absli%' AND LOWER(Status) = 'live'

"List all WIP models in Cross-Sell"
SELECT Name_of_the_Model, LOB, Function, Status, Owner, Time_Line FROM df WHERE LOWER(Function) LIKE '%cross-sell%' AND LOWER(Status) = 'wip'

"Which owner has the most models?"
SELECT Owner, COUNT(*) as model_count FROM df GROUP BY Owner ORDER BY model_count DESC LIMIT 1

"Show ML models going live in FY26"
SELECT Name_of_the_Model, LOB, Function, Status, Owner, Time_Line FROM df WHERE LOWER(ML_Non_ML) = 'ml' AND LOWER(Time_Line) LIKE '%fy26%'

"Which functions have more WIP than Live models?"
SELECT Function FROM df GROUP BY Function HAVING SUM(CASE WHEN Status='WIP' THEN 1 ELSE 0 END) > SUM(CASE WHEN Status='LIVE' THEN 1 ELSE 0 END)

"Which LOB has the highest ratio of WIP to Live models?"
SELECT LOB, CAST(SUM(CASE WHEN Status='WIP' THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(SUM(CASE WHEN Status='LIVE' THEN 1 ELSE 0 END), 0) as ratio FROM df GROUP BY LOB ORDER BY ratio DESC LIMIT 1
"""

FOLLOWUP_SIGNALS = [
    "of these", "of them", "out of these", "out of them",
    "how many of", "among these", "among them", "within these",
    "from these", "from them", "these models", "those models",
    "in them", "further", "also", "and how many", "what about",
    "which of", "do any of", "are any of"
]

GLOBAL_SIGNALS = [
    "across all lobs", "all lines of business", "all lobs",
    "across all", "overall", "entire inventory", "all models",
    "every lob", "each lob", "by lob", "per lob"
]

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _is_global_query(question: str) -> bool:
    return any(signal in question.lower() for signal in GLOBAL_SIGNALS)


def _infer_filters_from_sql(sql: str, accumulated_filters: dict) -> dict:
    filters = accumulated_filters.copy()
    sql_lower = sql.lower()

    # LOB map — matches actual table values
    lob_map = {
        "absli":    "ABSLI",
        "abhi":     "ABHI",
        "abhfl":    "ABHFL",
        "abslamc":  "ABSLAMC",
        "abcd":     "ABCD",
        "cau":      "CAU",
        "abfl-ca":  "ABFL-CA",
        "abfl-ra":  "ABFL-RA",
    }
    for key, val in lob_map.items():
        if key in sql_lower:
            filters["LOB"] = val
            break

    # Status — table uses uppercase LIVE/WIP
    if "= 'live'" in sql_lower or "='live'" in sql_lower:
        filters["Status"] = "LIVE"
    elif "= 'wip'" in sql_lower or "='wip'" in sql_lower:
        filters["Status"] = "WIP"

    # Function
    function_list = [
        "cross-sell", "hr", "pre-sales", "revenue retention", "risk management",
        "up-sell", "fraud", "engagement", "claims", "underwriting",
        "health management", "collection", "distribution", "onboarding",
        "prospecting", "service", "login", "foreclosure", "finance & planning",
        "revenue assurance"
    ]
    for func in function_list:
        if func in sql_lower:
            filters["Function"] = func.title()
            break

    # ML_Non_ML
    if "= 'ml'" in sql_lower:
        filters["ML_Non_ML"] = "ML"
    elif "= 'non-ml'" in sql_lower:
        filters["ML_Non_ML"] = "Non-ML"

    # Document_Availability
    if "= 'yes'" in sql_lower and "document" in sql_lower:
        filters["Document_Availability"] = "Yes"

    return filters

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_inventory() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df
    path = settings.inventory_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Inventory file not found at '{path}'. Set INVENTORY_PATH in your .env file."
        )
    df = pd.read_excel(path)

    # Normalize column names
    df.columns = [
        c.strip()
         .replace(" ", "_")
         .replace("/", "_")
         .replace("-", "_")
         .replace(".", "_")
        for c in df.columns
    ]

    _df = df
    logger.info(f"Inventory loaded: {len(df)} rows from {path}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def get_inventory_stats() -> dict:
    try:
        df = load_inventory()
        return {
            "total": len(df),
            "live":  len(df[df["Status"] == "LIVE"]),
            "wip":   len(df[df["Status"] == "WIP"]),
            "lobs":  df["LOB"].nunique(),
        }
    except Exception as e:
        logger.warning(f"Could not load inventory stats: {e}")
        return {"total": "N/A", "live": "N/A", "wip": "N/A", "lobs": "N/A"}

# ------------------------------------------------------------------------------
# Step 1 — Text to SQL
# ------------------------------------------------------------------------------

def generate_sql(question: str, turn_history: list, accumulated_filters: dict) -> str:
    client = _get_azure_client()

    history_text = "\n".join(
        f"Turn {i+1}: Q='{h['question']}' | filters={h['filters']} | rows={h['result_count']}"
        for i, h in enumerate(turn_history[-4:])
    ) if turn_history else "None - first question."

    filters_text = json.dumps(accumulated_filters) if accumulated_filters else "None"

    signals = [s for s in FOLLOWUP_SIGNALS if s in question.lower()]
    followup_hint = (
        f"NOTE: This appears to be a follow-up question (detected: {signals}). "
        f"Carry forward these active filters: {filters_text}"
        if signals else
        f"Active filters from conversation: {filters_text}"
    )

    full_prompt = f"""
{TEXT_TO_SQL_PROMPT}

CONVERSATION HISTORY:
{history_text}

{followup_hint}

USER QUESTION: {question}

Return ONLY the SQL query:
"""

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0,
    )

    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```(?:sql)?", "", sql).strip().rstrip("```").strip()
    return sql


def is_safe_sql(sql: str) -> bool:
    forbidden = ["drop", "insert", "update", "delete", "alter", "create", "truncate"]
    sql_lower = sql.lower().strip()
    return sql_lower.startswith("select") and not any(kw in sql_lower for kw in forbidden)

# ------------------------------------------------------------------------------
# Step 2 — Fallback: filter-based query
# ------------------------------------------------------------------------------

def _fallback_parse_question(question: str, accumulated_filters: dict,
                              turn_history: list) -> dict:
    client = _get_azure_client()

    history_text = "\n".join(
        f"Turn {i+1}: Q='{h['question']}' filters={h['filters']} results={h['result_count']}"
        for i, h in enumerate(turn_history[-4:])
    ) if turn_history else "None."

    prompt = f"""
{SCHEMA_INFO}

CONVERSATION HISTORY: {history_text}
ACCUMULATED FILTERS: {json.dumps(accumulated_filters) if accumulated_filters else "{{}}"}
QUESTION: "{question}"

Return ONLY JSON:
{{
  "is_followup": <true|false>,
  "topic_shift": <true|false>,
  "new_filters": {{}},
  "final_filters": {{}},
  "query_type": "<count|list|describe|breakdown>",
  "group_by": "<column or null>",
  "explanation": "<one line>"
}}
"""
    try:
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Fallback parse failed: {e}")
        return {
            "is_followup": False, "topic_shift": False,
            "new_filters": {}, "final_filters": {},
            "query_type": "list", "group_by": None, "explanation": str(e),
        }


def _build_fallback_sql(parsed: dict) -> str:
    final_filters = parsed.get("final_filters", {})
    query_type    = parsed.get("query_type", "list")
    group_by      = parsed.get("group_by")

    EXACT_MATCH_COLS = {"ML_Non_ML", "Status", "Document_Availability"}

    conditions = []
    for col, val in final_filters.items():
        if not val:
            continue
        safe_val = val.replace("'", "''")
        if col in EXACT_MATCH_COLS:
            conditions.append(f"LOWER({col}) = '{safe_val.lower()}'")
        else:
            conditions.append(f"LOWER({col}) LIKE '%{safe_val.lower()}%'")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    if query_type == "count":
        return f"SELECT COUNT(*) as count FROM df {where}"
    elif query_type == "breakdown":
        cols = ", ".join(group_by) if isinstance(group_by, list) else (group_by or "Function")
        return f"SELECT {cols}, COUNT(*) as model_count FROM df {where} GROUP BY {cols} ORDER BY model_count DESC"
    elif query_type == "describe":
        return f"SELECT * FROM df {where} LIMIT 1"
    else:
        return f"SELECT Name_of_the_Model, LOB, Function, Status, Owner, Time_Line FROM df {where}"

# ------------------------------------------------------------------------------
# Step 3 — Natural language response
# ------------------------------------------------------------------------------

def generate_nl_response(question: str, result_df: pd.DataFrame,
                          turn_history: list, final_filters: dict,
                          sql_used: str) -> str:
    client = _get_azure_client()

    history_text = "\n".join(
        f"Q: {h['question']}" for h in turn_history[-4:]
    ) if turn_history else "First question."

    if result_df.empty:
        result_summary = "No results found."
    elif list(result_df.columns) == ["count"] and len(result_df) == 1:
        result_summary = f"Count result: {result_df.iloc[0, 0]}"
    elif len(result_df.columns) <= 2:
        result_summary = result_df.to_string(index=False)
    else:
        preview_cols = [
            c for c in [
                "Name_of_the_Model", "LOB", "Function", "Status",
                "Owner", "Time_Line", "Model_Description"
            ]
            if c in result_df.columns
        ]
        result_summary = result_df[preview_cols].to_string(index=False)

    prompt = f"""
You are Converge Knowledge, a friendly and professional COE Analytics assistant.

CONVERSATION SO FAR:
{history_text}

USER QUESTION: "{question}"
ACTIVE FILTERS: {json.dumps(final_filters) if final_filters else "none"}

DATABASE RESULTS:
{result_summary}

Respond in natural, conversational language:
- For counts: state the number clearly with context.
- For lists <= 5 results: mention all model names naturally in your response.
- For lists > 5 results: give total count and key patterns. End with "Here's the full list below."
- For breakdowns: highlight the most interesting insight.
- For describes: give a clear readable summary.
- For empty results: explain what was searched and suggest broader criteria.
- Never mention SQL, DuckDB, DataFrames, or technical internals.
- Keep it concise — 2 to 4 sentences max unless describing a specific model.
- Do NOT include the table in your response — it will be appended separately.
"""

    try:
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"NL response failed: {e}")
        if result_df.empty:
            return "No models found matching your criteria. Try broadening your search."
        if list(result_df.columns) == ["count"]:
            return f"Found **{result_df.iloc[0, 0]}** model(s) matching your criteria."
        return f"Found **{len(result_df)}** model(s) matching your criteria."

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

def run_inventory_query(question: str, accumulated_filters: dict,
                        turn_history: list) -> tuple:
    """
    Full Text-to-SQL pipeline with filter-based fallback.
    Returns: (nl_answer, result_df_or_None, updated_filters, topic_shift, parsed)
    """
    try:
        df = load_inventory()
    except FileNotFoundError as e:
        return str(e), None, accumulated_filters, False, {}

    sql         = None
    result_df   = None
    used_fallback = False
    topic_shift = False
    parsed      = {}

    # Global query check
    if _is_global_query(question) and accumulated_filters.get("LOB"):
        return (
            "",
            None,
            accumulated_filters,
            True,
            {"new_filters": {"LOB": "All LOBs"}}
        )

    effective_filters = accumulated_filters.copy()

    # ── Primary: Text-to-SQL ──────────────────────────────────────────────
    try:
        sql = generate_sql(question, turn_history, effective_filters)
        logger.info(f"[Text-to-SQL] Generated: {sql}")

        if not is_safe_sql(sql):
            raise ValueError(f"Unsafe SQL blocked: {sql}")

        prospective_filters = _infer_filters_from_sql(sql, {})
        new_lob = prospective_filters.get("LOB")
        old_lob = accumulated_filters.get("LOB")

        if new_lob and old_lob and new_lob != old_lob:
            return (
                "",
                None,
                accumulated_filters,
                True,
                {"new_filters": prospective_filters}
            )

        result_df = duckdb.query(sql).to_df()
        logger.info(f"[Text-to-SQL] SUCCESS — {len(result_df)} rows returned")

    except Exception as primary_error:
        logger.warning(f"[Text-to-SQL] FAILED: {primary_error} | Trying fallback...")
        used_fallback = True

        try:
            parsed = _fallback_parse_question(question, accumulated_filters, turn_history)
            topic_shift = parsed.get("topic_shift", False)
            fallback_sql = _build_fallback_sql(parsed)
            logger.info(f"[Fallback] SQL: {fallback_sql}")
            result_df = duckdb.query(fallback_sql).to_df()
            sql = fallback_sql
            logger.info(f"[Fallback] SUCCESS — {len(result_df)} rows returned")
        except Exception as fallback_error:
            logger.error(f"[Fallback] ALSO FAILED: {fallback_error}")
            return (
                "I had trouble retrieving that information. Could you rephrase your question?",
                None, accumulated_filters, False, parsed
            )

    # ── Update filters ────────────────────────────────────────────────────
    if not used_fallback:
        updated_filters = _infer_filters_from_sql(sql, accumulated_filters)
        new_lob = updated_filters.get("LOB")
        old_lob = accumulated_filters.get("LOB")
        if new_lob and old_lob and new_lob != old_lob:
            topic_shift = True
    else:
        updated_filters = parsed.get("final_filters", accumulated_filters)

    # ── Natural language answer ───────────────────────────────────────────
    nl_answer = generate_nl_response(
        question=question,
        result_df=result_df,
        turn_history=turn_history,
        final_filters=updated_filters,
        sql_used=sql,
    )

    # ── Decide whether to return df for HTML table ────────────────────────
    show_df = None
    if result_df is not None:
        cols     = list(result_df.columns)
        is_count = cols == ["count"] and len(result_df) == 1
        is_breakdown = len(cols) == 2 and "model_count" in cols
        is_list  = not is_count and len(result_df) > 0

        if is_list or is_breakdown:
            show_df = result_df

    return nl_answer, show_df, updated_filters, topic_shift, parsed
