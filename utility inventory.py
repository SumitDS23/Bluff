"""
rag/utility_inventory.py
DuckDB-backed Text-to-SQL engine for the Utility Inventory CSV.

Mirrors the dashboard_inventory.py pattern exactly.

Actual CSV columns (as per COE utility schema):
  Sr_No, COE_LOB, Category_Tag, Utility_Name_&_Description,
  Business_Problem_Statement_&_Use_Cases, Utility_Category,
  Business_LOB_&_Sub_Function, Stakeholder_Name, Technical_SPOC,
  Current_Status, Success_Metrics

Usage:
    from rag.utility_inventory import create_utility_engine
    engine = create_utility_engine()
    result = engine.answer_question("What utilities does ABSLI have?")
"""

import logging
import re
import json
import duckdb
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Column descriptions — update to match your actual CSV headers
# ──────────────────────────────────────────────────────────────────────────────

COLUMN_DESCRIPTIONS = {
    "Sr_No":                                  "Serial number / row identifier",
    "COE_LOB":                                "Owner of the utility — either 'COE' (Center of Excellence) or a specific LOB (e.g. ABFL-CA, ABSLAMC, CoE). Use this for 'which team built it' questions.",
    "Category_Tag":                           "High-level technology/capability tag (e.g. Document Intelligence & OCR, Voice & Sentiment Analytics, Customer Analytics & Linkage, Risk Monitoring & Early Warning, Data Extraction & Automation, Model Documentation & Governance, Credit Assessment & Underwriting)",
    "Utility_Name_&_Description":             "Name of the utility followed by a short description of what it does. Use for name-based searches and 'what is X' questions.",
    "Business_Problem_Statement_&_Use_Cases": "Detailed description of the business problem the utility solves and its key use cases. Use for 'what problem does X solve', 'use cases', or topic-based searches.",
    "Utility_Category":                       "Technical classification of the utility (e.g. AI/ML/Non-ML, Non-ML / Automation, AI / NLP / Voice Analytics, Risk Analytics / Bureau Intelligence / Early Warning System, Automation)",
    "Business_LOB_&_Sub_Function":            "The LOB and business function that consumes or benefits from the utility (e.g. Secured Lending (Credit Risk / Operations / Legal), AMC Analytics, NBFC-RA & ABCD (Credit Risk)). Use for 'which LOB uses X' questions.",
    "Stakeholder_Name":                       "Business stakeholder / sponsor name (e.g. Head of Secured Lending, Saket Rakhe, Aman Gupta)",
    "Technical_SPOC":                         "Technical single point of contact / developer name (e.g. AI Engineering Lead, Makineni Vamsi, Angshuman Pandey, Harsh Jaykumar)",
    "Current_Status":                         "Current deployment status of the utility. Values include: Testing Phase, Development, POC In-Progress, Implemented, Developed and Deployed, In Development, Pilot Use Case",
    "Success_Metrics":                        "Quantified or qualitative outcomes achieved (e.g. 17 man hours saved, 80% cost reduction, ₹24.6 Cr risky disbursals prevented). May be empty for early-stage utilities.",
}


# ──────────────────────────────────────────────────────────────────────────────
# SQL generation prompt
# ──────────────────────────────────────────────────────────────────────────────

def _build_sql_prompt(question: str, columns: list, sample_rows: str) -> str:
    col_info = "\n".join(
        f"  - {col}: {COLUMN_DESCRIPTIONS.get(col, 'no description')}"
        for col in columns
    )
    return f"""
You are a SQL expert. Generate a DuckDB SQL query to answer the user's question
about an internal COE Utility Inventory table called `utility_inventory`.

Table columns:
{col_info}

Sample data (first 3 rows):
{sample_rows}

Rules:
1. Use ONLY column names listed above — no invented columns.
2. All string comparisons MUST be case-insensitive: use LOWER(col) LIKE LOWER('%value%')
3. LOB / owner filtering:
   - "COE utilities"     → LOWER("COE_LOB") LIKE '%coe%'
   - "ABFL-CA utilities" → LOWER("COE_LOB") LIKE '%abfl%' OR LOWER("Business_LOB_&_Sub_Function") LIKE '%abfl%'
   - Always check BOTH COE_LOB and Business_LOB_&_Sub_Function for LOB questions
4. Status filtering:
   - "live" / "deployed"    → LOWER("Current_Status") LIKE '%implemented%' OR LOWER("Current_Status") LIKE '%deployed%'
   - "in progress" / "WIP"  → LOWER("Current_Status") LIKE '%development%' OR LOWER("Current_Status") LIKE '%progress%' OR LOWER("Current_Status") LIKE '%testing%'
5. COUNT queries:
   SELECT COUNT(*) AS utility_count FROM utility_inventory WHERE ...
6. LIST queries (default columns for listing):
   SELECT "Sr_No", "COE_LOB", "Category_Tag", "Utility_Name_&_Description", "Current_Status"
   FROM utility_inventory WHERE ... ORDER BY "COE_LOB", "Sr_No"
7. DETAIL queries ("what does X do", "describe X", "use cases for X"):
   SELECT "Utility_Name_&_Description", "Business_Problem_Statement_&_Use_Cases",
          "Utility_Category", "Business_LOB_&_Sub_Function",
          "Stakeholder_Name", "Technical_SPOC", "Current_Status", "Success_Metrics"
   FROM utility_inventory WHERE LOWER("Utility_Name_&_Description") LIKE '%keyword%'
8. SPOC / stakeholder queries:
   SELECT "Utility_Name_&_Description", "Stakeholder_Name", "Technical_SPOC", "Current_Status"
   FROM utility_inventory WHERE ...
9. SUCCESS METRICS queries:
   SELECT "Utility_Name_&_Description", "Success_Metrics", "Current_Status"
   FROM utility_inventory WHERE "Success_Metrics" IS NOT NULL AND "Success_Metrics" != 'nan'
10. Return ALL matching rows — do NOT add LIMIT unless user asks for top-N.
11. Column names that contain special characters (&, /) MUST be double-quoted in SQL.
    e.g. "Utility_Name_&_Description", "Business_LOB_&_Sub_Function"
12. Return ONLY the raw SQL — no markdown fences, no explanation, no backticks.

User question: {question}
SQL:"""


# ──────────────────────────────────────────────────────────────────────────────
# Natural language answer prompt
# ──────────────────────────────────────────────────────────────────────────────

def _build_answer_prompt(question: str, sql: str, result_df: pd.DataFrame) -> str:
    if result_df.empty:
        result_text = "No results found."
    else:
        result_text = result_df.to_string(index=False)

    return f"""
You are a helpful COE Analytics assistant answering questions about internal
COE utilities — tools, scripts, AI/ML solutions, and automation built by the
Center of Excellence and LOB analytics teams.

User question: {question}

SQL executed: {sql}

Query result:
{result_text}

Instructions:
- If no results: say no matching utilities were found and suggest the user
  check the SharePoint Utility folder for the latest inventory.
- If results exist: give a clear, concise natural language summary first,
  then reference the table shown below for full details.
- For COUNT questions: state the number directly
  e.g. "There are 3 utilities owned by COE."
- For LIST questions: give a 1-line intro then mention the table has the details.
  Do NOT repeat every row in prose.
- For DETAIL / "what does X do" questions: summarise the utility purpose
  (2-3 sentences), mention the business problem it solves, key use cases,
  current status, and SPOC if available.
- For SUCCESS METRICS questions: highlight the quantified outcomes
  (man-hours saved, cost reduction, disbursals prevented etc.).
- For SPOC / stakeholder questions: list name and their associated utility clearly.
- Do NOT mention SQL, DuckDB, or any technical query internals.
- Do NOT fabricate information not present in the query result.
- Keep tone professional but conversational.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Utility Inventory Engine
# ──────────────────────────────────────────────────────────────────────────────

class UtilityInventoryEngine:

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.conn     = duckdb.connect(":memory:")
        self.columns  = []
        self._load()

    # ── Load CSV into DuckDB ──────────────────────────────────────────────

    def _load(self):
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Utility inventory CSV not found at '{self.csv_path}'. "
                "Set UTILITY_INVENTORY_PATH in your .env file."
            )

        df = pd.read_csv(path)

        # Normalise column names:
        # strip whitespace, replace spaces → underscores
        # keep & and / as-is (DuckDB handles them fine when double-quoted)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        # Clean string columns:
        # - strip leading/trailing whitespace
        # - collapse internal newlines (multiline cells from Excel exports)
        for col in df.select_dtypes(include="object").columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\n+", " ", regex=True)
                .str.replace(r"\r+", " ", regex=True)
            )

        # Replace literal "nan" strings (from pd.read_csv on empty cells) with None
        df = df.replace("nan", None)

        self.conn.execute(
            "CREATE TABLE utility_inventory AS SELECT * FROM df"
        )
        self.columns = list(df.columns)

        logger.info(
            f"Utility Inventory loaded: {len(df)} rows | "
            f"Columns: {self.columns}"
        )

    # ── Sample rows for prompt context ───────────────────────────────────

    def _sample_rows(self, n: int = 3) -> str:
        try:
            df = self.conn.execute(
                f"SELECT * FROM utility_inventory LIMIT {n}"
            ).df()
            return df.to_string(index=False)
        except Exception:
            return "(sample unavailable)"

    # ── Step 1: Generate SQL ──────────────────────────────────────────────

    def _generate_sql(self, question: str) -> str:
        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

        prompt = _build_sql_prompt(question, self.columns, self._sample_rows())

        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        sql = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        sql = re.sub(r"```(?:sql)?", "", sql).strip().rstrip("```").strip()
        logger.info(f"Generated SQL: {sql}")
        return sql

    # ── Step 2: Execute SQL ───────────────────────────────────────────────

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    # ── Step 3: Generate natural language answer ──────────────────────────

    def _generate_answer(self, question: str, sql: str,
                         result_df: pd.DataFrame) -> str:
        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

        prompt = _build_answer_prompt(question, sql, result_df)

        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    # ── Public entry point ────────────────────────────────────────────────

    def answer_question(self, question: str) -> dict:
        """
        Full Text-to-SQL pipeline for a utility question.

        Returns
        -------
        {
            "success":   bool,
            "answer":    str,          # natural language summary
            "sql":       str,          # generated SQL (for logging/debug)
            "result_df": pd.DataFrame, # raw query result (for HTML table)
            "error":     str | None,
        }
        """
        try:
            sql       = self._generate_sql(question)
            result_df = self._execute_sql(sql)
            answer    = self._generate_answer(question, sql, result_df)

            return {
                "success":   True,
                "answer":    answer,
                "sql":       sql,
                "result_df": result_df,
                "error":     None,
            }

        except Exception as e:
            logger.error(f"Utility inventory query failed: {e}", exc_info=True)
            return {
                "success":   False,
                "answer":    "",
                "sql":       "",
                "result_df": pd.DataFrame(),
                "error":     str(e),
            }


# ──────────────────────────────────────────────────────────────────────────────
# Factory — called from main.py startup
# ──────────────────────────────────────────────────────────────────────────────

def create_utility_engine() -> UtilityInventoryEngine:
    """Instantiate and return a UtilityInventoryEngine."""
    return UtilityInventoryEngine(csv_path=settings.utility_inventory_path)
