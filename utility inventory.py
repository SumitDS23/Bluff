"""
rag/utility_inventory.py
DuckDB-backed Text-to-SQL engine for the Utility Inventory CSV.

Mirrors the dashboard_inventory.py pattern exactly.

Expected CSV columns (adjust COLUMN_DESCRIPTIONS if yours differ):
  Utility_Name, LOB, Category, Description, Owner, Status,
  Technology, Access_Link, Last_Updated

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
    "Utility_Name":   "Name of the internal COE tool, script, notebook, or calculator",
    "LOB":            "Line of Business the utility belongs to (ABSLI, ABHI, ABCD, ABSLAMC, ABHFL, CAU, ABFL-CA, ABFL-RA, General)",
    "Category":       "Type of utility (e.g. Campaign, Monitoring, Reporting, Data Prep, Modelling)",
    "Description":    "What the utility does and its business purpose",
    "Owner":          "Team or person responsible for maintaining the utility",
    "Status":         "Current state: Live, WIP, Deprecated",
    "Technology":     "Tech stack used (Python, Excel, SQL, Power BI, etc.)",
    "Access_Link":    "SharePoint or network path to access the utility",
    "Last_Updated":   "Date the utility was last updated",
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
2. All string comparisons must be case-insensitive: use LOWER(col) LIKE LOWER('%value%')
3. For LOB filtering: match partial names too
   e.g. LOWER(LOB) LIKE '%absli%' catches "ABSLI", "ABSLI Team" etc.
4. COUNT queries: SELECT COUNT(*) AS utility_count FROM utility_inventory WHERE ...
5. LIST queries:  SELECT Utility_Name, LOB, Category, Description, Status
                  FROM utility_inventory WHERE ... ORDER BY LOB, Utility_Name
6. For "what does X do" or "describe X": fetch Description, Owner, Technology, Access_Link
7. Return ALL matching rows — do NOT add LIMIT unless the user asks for top-N.
8. Return ONLY the raw SQL query — no markdown, no explanation, no backticks.

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
You are a helpful COE Analytics assistant answering questions about internal utilities and tools.

User question: {question}

SQL executed: {sql}

Query result:
{result_text}

Instructions:
- If no results: say no matching utilities were found and suggest the user check the SharePoint folder.
- If results exist: give a clear, concise natural language summary.
- For counts: state the number directly ("There are X utilities in ABSLI").
- For lists: briefly introduce the list, then reference the table below (do not repeat rows).
- For descriptions: explain what the utility does in 2-3 sentences.
- Do NOT mention SQL, DuckDB, or technical internals.
- Keep the tone professional but conversational.
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

        # Normalise column names — strip spaces, replace spaces with underscores
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        # Strip whitespace from all string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

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
