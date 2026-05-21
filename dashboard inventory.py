"""
rag/dashboard_inventory.py
Text-to-SQL engine for Dashboard Repository queries.

Uses DuckDB in-memory database with comprehensive schema knowledge.
Handles natural language queries about dashboards, MIS reports, and data dumps.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from openai import AzureOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE SCHEMA INFORMATION
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_INFO = """
You are querying a DuckDB table called `df` with these exact columns (15 total):

| Column                     | Type   | Description / Known Values                                                |
|----------------------------|--------|---------------------------------------------------------------------------|
| Sr_No                      | INT    | Serial number (1-2167), Note: ~673 rows have NULL values                  |
| Dashboard_MIS_Name         | TEXT   | Name of the Dashboard/MIS report (free text, ~2166 entries)               |
| Business_Function          | TEXT   | Business function/team owning the report (266 unique values)              |
| Purpose_Business_Impact    | TEXT   | Goal/description of the dashboard (free text)                             |
| Data_Analytics_SPOC        | TEXT   | Single Point of Contact names (free text)                                 |
| Key_Users_Stakeholders     | TEXT   | Teams/Roles relying on the report (free text)                             |
| Data_Sources               | TEXT   | Source systems for data (85 unique values)                                |
| KPIs_Metrics_Tracked       | TEXT   | List of tracked metrics/indicators (free text)                            |
| Frequency                  | TEXT   | Update frequency (46 unique values)                                       |
| Automation_Level           | TEXT   | Level of automation (17 variations)                                       |
| Distribution_Method        | TEXT   | How report is delivered (39 unique values)                                |
| LOB                        | TEXT   | Line of Business (10 unique values)                                       |
| Team                       | TEXT   | Team who developed/manages the dashboard (78 unique values)               |
| Status                     | TEXT   | Current status of dashboard (6 values)                                    |
| Dashboard_MIS_Dump         | TEXT   | Type of output (5 values)                                                 |

================================================================================
LOB (Line of Business) - 10 UNIQUE VALUES
================================================================================
| Value          | Full Name / Description                                      |
|----------------|--------------------------------------------------------------|
| ABCD           | Aditya Birla Capital Digital / Finserv App                   |
| AMC            | Asset Management Company / Mutual Funds [Note: 'AMC ' has trailing space] |
| ABHFL          | Aditya Birla Housing Finance Limited                         |
| ABHI           | Aditya Birla Health Insurance                                |
| ABSLI          | Aditya Birla Sun Life Insurance                              |
| ABCL BIU       | ABCL Business Intelligence Unit                              |
| ABCL Consumer  | ABCL Consumer Analytics                                      |
| ABCL Data      | ABCL Data Team                                               |
| ABCL Risk      | ABCL Risk Team                                               |
| CAU            | Central Analytics Unit                                       |

LOB Aliases:
- finserv/app/digital         -> 'ABCD'
- mutual funds/amc            -> 'AMC ' (note trailing space)
- housing/hfl/home loan       -> 'ABHFL'
- health insurance            -> 'ABHI'
- life insurance/sli          -> 'ABSLI'
- biu/business intelligence   -> 'ABCL BIU'

================================================================================
STATUS - 6 VALUES
================================================================================
Live, WIP, On Hold, Discontinued

Use TRIM() to handle trailing spaces: WHERE TRIM(Status) = 'Live'

================================================================================
DASHBOARD_MIS_DUMP (Output Type) - 5 VALUES
================================================================================
Dashboard, MIS, Dump, Data Dump

================================================================================
KEY QUERY PATTERNS
================================================================================

1. CASE-INSENSITIVE MATCHING:
   Use ILIKE for all text comparisons
   WHERE LOB ILIKE '%abhi%'
   WHERE Business_Function ILIKE '%sales%'

2. HANDLE TRAILING/LEADING SPACES:
   Use TRIM() for exact matches
   WHERE TRIM(LOB) = 'AMC'
   WHERE TRIM(Status) = 'Live'

3. PARTIAL MATCHING:
   Use LIKE or ILIKE with wildcards
   WHERE Dashboard_MIS_Name ILIKE '%sales%'
   WHERE Data_Sources ILIKE '%databricks%'

4. COUNT QUERIES:
   SELECT LOB, COUNT(*) as count 
   FROM df 
   GROUP BY LOB 
   ORDER BY count DESC

5. FILTER BY MULTIPLE CONDITIONS:
   SELECT * FROM df 
   WHERE LOB ILIKE '%abhi%' 
   AND TRIM(Status) = 'Live'
   AND Automation_Level ILIKE '%fully%automated%'

6. BANK-SPECIFIC QUERIES:
   WHERE Business_Function LIKE '%AXIS Bank%'
   WHERE Business_Function LIKE '%HDFC%'

7. NULL HANDLING:
   WHERE Status IS NOT NULL AND TRIM(Status) != ''
"""

# ══════════════════════════════════════════════════════════════════════════════
# TEXT-TO-SQL PROMPT
# ══════════════════════════════════════════════════════════════════════════════

TEXT_TO_SQL_PROMPT_TEMPLATE = """You are an expert SQL query generator for a Dashboard Inventory system.

{schema_info}

USER QUESTION: "{question}"

INSTRUCTIONS:
1. Generate a valid DuckDB SQL query to answer the question
2. Use ILIKE for case-insensitive text matching
3. Use TRIM() to handle trailing/leading spaces
4. Use COUNT(*), GROUP BY, ORDER BY as needed
5. Return ONLY the SQL query, no explanation or markdown
6. For counting questions, always include ORDER BY to sort results
7. For "list" questions, select relevant columns (not SELECT *)
8. For "which" questions, return the specific column requested

EXAMPLES:

Q: "How many dashboards are there in ABHI?"
SQL: SELECT COUNT(*) as count FROM df WHERE LOB ILIKE '%abhi%'

Q: "List all live dashboards in ABHFL"
SQL: SELECT Dashboard_MIS_Name, Business_Function, Team FROM df WHERE LOB ILIKE '%abhfl%' AND TRIM(Status) = 'Live'

Q: "Which LOB has the most dashboards?"
SQL: SELECT LOB, COUNT(*) as count FROM df GROUP BY LOB ORDER BY count DESC LIMIT 1

Q: "Show me all sales dashboards that are fully automated"
SQL: SELECT Dashboard_MIS_Name, LOB, Team FROM df WHERE Business_Function ILIKE '%sales%' AND Automation_Level ILIKE '%fully%automated%'

Q: "Count dashboards by status in ABHI"
SQL: SELECT TRIM(Status) as Status, COUNT(*) as count FROM df WHERE LOB ILIKE '%abhi%' GROUP BY TRIM(Status) ORDER BY count DESC

Q: "Which teams manage the most dashboards?"
SQL: SELECT Team, COUNT(*) as count FROM df WHERE Team IS NOT NULL AND TRIM(Team) != '' GROUP BY Team ORDER BY count DESC LIMIT 10

Q: "What are the data sources for sales dashboards in ABSLI?"
SQL: SELECT DISTINCT Data_Sources FROM df WHERE LOB ILIKE '%absli%' AND Business_Function ILIKE '%sales%'

Q: "How many dashboards are updated daily?"
SQL: SELECT COUNT(*) as count FROM df WHERE Frequency ILIKE '%daily%'

Generate the SQL query:
"""


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD INVENTORY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class DashboardInventoryEngine:
    """
    Text-to-SQL engine for Dashboard Repository queries.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize with path to dashboard inventory CSV.
        
        Args:
            csv_path: Path to the dashboard inventory CSV file
        """
        self.csv_path = Path(csv_path)
        self.conn = None
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        
        # Load data into DuckDB
        self._load_data()
    
    def _load_data(self):
        """Load CSV into DuckDB in-memory database."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dashboard CSV not found at {self.csv_path}")
        
        logger.info(f"Loading dashboard inventory from {self.csv_path}")
        
        # Create in-memory DuckDB connection
        self.conn = duckdb.connect(":memory:")
        
        # Read CSV with pandas (handles encoding issues better)
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        # Register as DuckDB table
        self.conn.register("df", df)
        
        # Verify
        count = self.conn.execute("SELECT COUNT(*) FROM df").fetchone()[0]
        logger.info(f"✅ Loaded {count} dashboard records into DuckDB")
    
    def generate_sql(self, question: str) -> str:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            SQL query string
        """
        prompt = TEXT_TO_SQL_PROMPT_TEMPLATE.format(
            schema_info=SCHEMA_INFO,
            question=question
        )
        
        try:
            response = self.client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic for SQL generation
                max_tokens=500,
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            logger.info(f"Generated SQL: {sql}")
            
            return sql
        
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            raise
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            
        Returns:
            Results as pandas DataFrame
        """
        try:
            result = self.conn.execute(sql).fetchdf()
            logger.info(f"Query returned {len(result)} row(s)")
            return result
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer natural language question about dashboards.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with answer, SQL, results, and metadata
        """
        try:
            # Generate SQL
            sql = self.generate_sql(question)
            
            # Execute query
            result_df = self.execute_query(sql)
            
            # Format answer based on result type
            answer = self._format_answer(question, result_df)
            
            return {
                "answer": answer,
                "sql": sql,
                "result_df": result_df,
                "row_count": len(result_df),
                "success": True,
            }
        
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "answer": f"I encountered an error while querying the dashboard inventory: {str(e)}",
                "sql": None,
                "result_df": None,
                "row_count": 0,
                "success": False,
                "error": str(e),
            }
    
    def _format_answer(self, question: str, result_df: pd.DataFrame) -> str:
        """
        Format query results into natural language answer.
        
        Args:
            question: Original question
            result_df: Query results
            
        Returns:
            Formatted answer string
        """
        if result_df.empty:
            return "No results found for your query."
        
        # Single value result (count, max, etc.)
        if len(result_df) == 1 and len(result_df.columns) == 1:
            value = result_df.iloc[0, 0]
            return f"The answer is: **{value}**"
        
        # Single row with multiple columns
        if len(result_df) == 1:
            answer = "Here's what I found:\n\n"
            for col in result_df.columns:
                answer += f"- **{col}**: {result_df.iloc[0][col]}\n"
            return answer
        
        # Multiple rows - return as table
        if len(result_df) <= 20:
            # Small result - show full table
            answer = f"Found {len(result_df)} result(s):\n\n"
            answer += result_df.to_markdown(index=False)
            return answer
        else:
            # Large result - show summary + first 20
            answer = f"Found {len(result_df)} result(s). Showing first 20:\n\n"
            answer += result_df.head(20).to_markdown(index=False)
            answer += f"\n\n... and {len(result_df) - 20} more rows."
            return answer


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_dashboard_engine(csv_path: str = None) -> DashboardInventoryEngine:
    """
    Factory function to create dashboard engine instance.
    
    Args:
        csv_path: Path to dashboard CSV. If None, uses default from settings.
        
    Returns:
        DashboardInventoryEngine instance
    """
    if csv_path is None:
        # Use default path from settings
        csv_path = settings.dashboard_inventory_path
    
    return DashboardInventoryEngine(csv_path)


def query_dashboards(question: str, csv_path: str = None) -> Dict[str, Any]:
    """
    One-shot function to query dashboards.
    
    Args:
        question: Natural language question
        csv_path: Path to dashboard CSV
        
    Returns:
        Query results dict
    """
    engine = create_dashboard_engine(csv_path)
    return engine.answer_question(question)


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test queries
    test_questions = [
        "How many dashboards are there in ABHI?",
        "Which LOB has the most dashboards?",
        "List all live sales dashboards in ABSLI",
        "How many dashboards are fully automated?",
        "What are the top 5 teams managing the most dashboards?",
        "Count dashboards by status",
    ]
    
    # Initialize engine
    csv_path = "path/to/dashboard_inventory.csv"  # Update this
    engine = create_dashboard_engine(csv_path)
    
    # Run tests
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)
        
        result = engine.answer_question(question)
        
        print(f"SQL: {result['sql']}")
        print(f"\nAnswer:\n{result['answer']}")
