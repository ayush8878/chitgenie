# BigQuery + Gemini API using GOOGLE_API_KEY from .env
# Much simpler setup - no complex Vertex AI configuration needed!

# pip install google-generativeai google-cloud-bigquery streamlit pandas python-dotenv

import streamlit as st
import pandas as pd
from google.cloud import bigquery
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import re
import time

# Load environment variables
load_dotenv()


class GeminiBigQueryBot:
    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

        # Initialize Gemini API with API key from .env
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception(
                "GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        # Initialize Gemini model
        self.llm = genai.GenerativeModel('gemini-2.5-flash-lite')

        # Initialize BigQuery (still needs service account for BigQuery access)
        self.bq_client = bigquery.Client(project=project_id)

        # Get table schema for context
        self.table_schema = self.get_table_schema()

    def get_table_schema(self):
        """Get the schema of the BigQuery table"""
        try:
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            table = self.bq_client.get_table(table_ref)

            schema_info = []
            for field in table.schema:
                schema_info.append(
                    f"- {field.name} ({field.field_type}): {field.description or 'No description'}")

            return "\n".join(schema_info)
        except Exception as e:
            return f"Could not fetch schema: {str(e)}"

    def generate_sql(self, natural_language_query: str) -> str:
        """Convert natural language to SQL using Gemini API"""

        prompt = f"""
You are an expert read only SQL generator for BigQuery. Convert the following natural language query to SQL.

Table Information:
- Table: `{self.project_id}.{self.dataset_id}.{self.table_id}`
- Schema:
{self.table_schema}

Business Context:
- This is an invoices table with financial data
- "overdue" means status = 'overdue' or due_date < CURRENT_DATE() and status != 'paid'
- "pending" means status = 'pending'
- "paid" means status = 'paid'
- Use proper BigQuery functions like DATE_SUB, DATE_TRUNC for date operations
- Always limit results to prevent large queries (use LIMIT 100 unless specifically asked for more)

Natural Language Query: "{natural_language_query}"

Generate ONLY the SQL query, no explanation. Use proper BigQuery syntax.

SQL:
"""

        try:
            # Generate content using Gemini API
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                    top_p=0.8,
                    top_k=40
                )
            )

            # Clean up the response
            sql = response.text.strip()

            print(
                f'The generate query for question "{natural_language_query}" is: {sql}')
            # Remove any markdown formatting
            sql = re.sub(r'```sql\n?', '', sql)
            sql = re.sub(r'```\n?', '', sql)

            return sql.strip()

        except Exception as e:
            raise Exception(f"Failed to generate SQL: {str(e)}")

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL query on BigQuery"""
        try:
            # Add safety checks
            if not self.is_safe_query(sql):
                raise Exception("Query contains potentially unsafe operations")

            # Execute query
            job_config = bigquery.QueryJobConfig(
                maximum_bytes_billed=1000000000,  # 1GB limit
                use_query_cache=True,
                job_timeout_ms=30000  # 30 seconds
            )

            query_job = self.bq_client.query(sql, job_config=job_config)
            results = query_job.result()

            # Convert to DataFrame
            df = results.to_dataframe()
            return df

        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")

    def is_safe_query(self, sql: str) -> bool:
        """
        Enhanced safety checks for SQL queries.

        This method checks for potentially dangerous SQL operations while allowing
        legitimate queries with subqueries and common patterns.

        Args:
            sql: The SQL query to validate

        Returns:
            bool: True if the query appears safe, False if it contains potentially dangerous operations
        """
        import re

        # Normalize SQL: remove string literals and comments first
        # Remove string literals
        sql_normalized = re.sub(r"'.*?'", "''", sql, flags=re.DOTALL)
        sql_normalized = re.sub(
            r'--.*?$', '', sql_normalized, flags=re.MULTILINE)  # Remove -- comments
        # Remove /* */ comments
        sql_normalized = re.sub(
            r'/\*.*?\*/', '', sql_normalized, flags=re.DOTALL)

        # Remove subqueries in parentheses to avoid false positives
        # This helps when checking for patterns that might appear in subqueries
        sql_for_validation = re.sub(r'\(\s*SELECT\s+.*?\s+FROM\s+.*?\s*(?:WHERE\s+.*?)?\)',
                                    '(SELECT ...)',
                                    sql_normalized,
                                    flags=re.IGNORECASE | re.DOTALL)

        # Check for DDL and other dangerous operations
        # These patterns look for SQL keywords at the start of the query or after a semicolon
        dangerous_patterns = [
            # DDL operations
            r'(?i)(?:^|;)\s*(?:alter|create|drop|truncate|rename|replace)\s+',

            # DML operations that modify data
            r'(?i)(?:^|;)\s*(delete|insert|update|merge|set|grant|revoke|deny)(?!\s*\(|\s+@|\s+\w+\s*,\s*\w+\s*\()\s+',

            # Potentially dangerous procedures/execution
            r'(?i)(?:^|;)\s*(exec|execute|execute\s+immediate|prepare|shutdown|use|lock|backup|restore|load\s+data)\s+',

            # System table access
            r'(?i)(?:^|;)\s*select\s+\*\s+from\s+(information_schema|pg_|sys\.)',

            # Potentially dangerous SQL Server procedures
            r'(?i)\b(xp_cmdshell|sp_configure|xp_reg(?:read|write|delete(?:value|key))|sp_oacreate|sp_oamethod|sp_adduser|sp_addrole)\b',

            # Multiple statements (except when used in a CASE statement)
            r'(?i)(?:^|;)\s*(?!.*?case\s+when.*?then)(select|update|delete|insert|create|alter|drop|truncate|grant|revoke|deny|exec|execute|shutdown|use|backup|restore|load\s+data)\s+.*?;\s*(select|update|delete|insert|create|alter|drop|truncate|grant|revoke|deny|exec|execute|shutdown|use|backup|restore|load\s+data)',

            # Potentially dangerous functions
            r'(?i)\b(?:waitfor\s+delay|sleep\s*\(|pg_sleep\s*\(|benchmark\s*\()',

            # MySQL specific dangerous patterns
            r'(?i)\b(into\s+(outfile|dumpfile))\b',

            # SQL injection patterns
            r'(?i)(?:\b(?:select|union|where|and|or|not)\b\s*[()\d+\-*/=]?\s*){5,}',

            # Suspicious comments that might bypass filters
            r'(?i)/\*![\d]',

            # Potentially dangerous string concatenation
            r'(?i)char\(\s*\d+\s*(,\s*\d+\s*)+\)',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sql_for_validation):
                print(f"Blocked by pattern: {pattern}")
                return False

        # Additional checks for specific patterns

        # Check for potential DML without proper WHERE clauses on large tables
        # But allow if there's a LIMIT clause or it's a simple SELECT
        if re.search(r'(?i)(?:^|;)\s*(update|delete|insert|merge)\s+\w+\s+(?!where)', sql_for_validation):
            if not re.search(r'(?i)(?:\slimit\s+\d+|\s*\bwhere\b|\s*\bvalues\s*\()', sql_for_validation):
                print("Blocked: DML without WHERE or LIMIT")
                return False

        # Check for potentially expensive queries (SELECT * without LIMIT or WHERE)
        # But only for top-level queries, not subqueries
        if re.search(r'(?i)^\s*select\s+\*\s+from\s+\w+\s*(?:;|$)', sql_for_validation):
            if not (re.search(r'(?i)\bwhere\b|\blimit\s+\d+', sql_for_validation)):
                print("Blocked: Unbounded SELECT *")
                return False

        # Check for suspicious string concatenation that might be used for SQL injection
        if re.search(r'(?i)(?:concat_ws?|\|\||\+)\s*\([^)]*\b(?:select|union|from|where|0x[0-9a-f]|char\()', sql_for_validation):
            print("Blocked: Suspicious string concatenation")
            return False

        return True

    def generate_explanation(self, query: str, sql: str, results: pd.DataFrame) -> str:
        """Generate natural language explanation of results"""

        # Create summary of results
        result_summary = f"""
Results Summary:
- Number of rows returned: {len(results)}
- Columns: {', '.join(results.columns.tolist())}
"""

        if len(results) > 0:
            # Add some basic stats
            numeric_cols = results.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                result_summary += f"\nNumeric Summary:\n"
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    result_summary += f"- {col}: Total = {results[col].sum():.2f}, Average = {results[col].mean():.2f}\n"

        prompt = f"""
User asked: "{query}"
SQL generated: {sql}
{result_summary}

Generate a clear, business-friendly explanation that:
1. Directly answers the user's question
2. Highlights key insights from the data
3. Uses professional language
4. Is concise but informative

Explanation:
"""

        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=300
                )
            )
            return response.text.strip()
        except:
            return "Query completed successfully. Please review the results above."

    def query(self, natural_language_query: str, max_retries: int = 3):
        """Main method to process natural language query with retry logic

        Args:
            natural_language_query: The query in natural language
            max_retries: Maximum number of retry attempts for query execution

        Returns:
            dict: Dictionary containing query results and metadata
        """
        attempt = 0
        last_error = None

        while attempt <= max_retries:
            try:
                # Step 1: Generate SQL
                sql = self.generate_sql(natural_language_query)

                # Validate generated SQL
                if not sql or not sql.strip():
                    raise ValueError("Generated SQL query is empty")

                # Step 2: Execute SQL with retry
                results_df = self.execute_sql(sql)

                # Step 3: Generate explanation if we have results
                explanation = self.generate_explanation(
                    natural_language_query, sql, results_df)

                return {
                    "success": True,
                    "sql": sql,
                    "data": results_df,
                    "explanation": explanation,
                    "error": None,
                    "attempts": attempt + 1
                }

            except Exception as e:
                last_error = str(e)
                attempt += 1

                # Log the error for debugging
                print(f"Attempt {attempt} failed: {last_error}")

                # If we've reached max retries, return the error
                if attempt > max_retries:
                    error_msg = f"Failed after {max_retries} attempts. Last error: {last_error}"
                    print(error_msg)

                    # Try to get a simplified explanation of the error
                    error_explanation = self._get_error_explanation(
                        last_error, natural_language_query)

                    return {
                        "success": False,
                        "sql": sql if 'sql' in locals() else None,
                        "data": None,
                        "explanation": error_explanation,
                        "error": error_msg,
                        "attempts": attempt
                    }

                # Add a small delay before retry (exponential backoff)
                time.sleep(min(2 ** attempt, 5))  # Cap at 5 seconds

    def _get_error_explanation(self, error: str, query: str) -> str:
        """Generate a user-friendly explanation of the error"""
        try:
            prompt = f"""
            A database query failed with the following error:
            Error: {error}
            
            The user's original question was: "{query}"
            
            Please provide a clear, non-technical explanation of what went wrong 
            and suggest how the user might rephrase their question.
            """

            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=200
                )
            )
            return response.text.strip()
        except Exception as e:
            # Fallback error message if we can't generate an explanation
            return "An error occurred while processing your query. Please try rephrasing your question or check your input data."

# Streamlit Interface


def main():
    st.set_page_config(
        page_title="Gemini API Invoice Query Bot",
        page_icon="🔥",
        layout="wide"
    )

    st.title("🔥 Gemini API Invoice Query Bot")
    st.markdown(
        "Using GOOGLE_API_KEY from .env file - No complex Vertex AI setup needed!")

    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found in environment variables!")
        st.markdown("""
        **Setup Required:**
        1. Create a `.env` file in your project directory
        2. Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`
        3. Get your API key from: https://makersuite.google.com/app/apikey
        """)
        return

    # Show API key status (masked)
    api_key = os.getenv("GOOGLE_API_KEY")
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if api_key else "Not set"
    st.sidebar.success(f"✅ API Key: {masked_key}")

    # Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        project_id = st.text_input(
            "GCP Project ID", value=os.getenv("GCP_PROJECT_ID"))
        dataset_id = st.text_input("Dataset ID", value=os.getenv("DATASET_ID"))
        table_id = st.text_input("Table ID", value=os.getenv("TABLE_ID"))

        if st.button("🚀 Initialize Bot"):
            if project_id and dataset_id and table_id:
                try:
                    st.session_state.bot = GeminiBigQueryBot(
                        project_id, dataset_id, table_id)
                    st.session_state.bot_ready = True
                    st.success("✅ Bot initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
            else:
                st.error("Please fill in all configuration fields")

    # Main interface
    if not hasattr(st.session_state, 'bot_ready'):
        st.info("👈 Please configure and initialize the bot in the sidebar")
        return

    bot = st.session_state.bot

    # Query interface
    st.header("💬 Ask Your Question")

    # Show table schema
    with st.expander("📋 Table Schema", expanded=False):
        st.text(bot.table_schema)

    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., Show me all overdue invoices from this month",
        height=100
    )

    # Sample questions
    st.subheader("💡 Try These Sample Questions")
    samples = [
        "Show me all overdue invoices",
        "What is the total amount of unpaid invoices?",
        "Which vendor has the highest invoice total?",
        "How many invoices were created last month?",
        "Show me invoices over $1000 that are pending"
    ]

    cols = st.columns(len(samples))
    for i, sample in enumerate(samples):
        if cols[i].button(f"💭 Query {i+1}", key=f"sample_{i}"):
            query = sample
            st.rerun()

    # Execute query
    if st.button("🔍 Execute Query", type="primary") and query:
        with st.spinner("🤖 Processing your question..."):
            result = bot.query(query)

        if result["success"]:
            st.success("✅ Query executed successfully!")

            # Show explanation
            st.subheader("🤖 AI Explanation")
            st.write(result["explanation"])

            # Show SQL
            with st.expander("🔍 Generated SQL Query"):
                st.code(result["sql"], language="sql")

            # Show results
            if result["data"] is not None and not result["data"].empty:
                st.subheader("📊 Results")
                st.dataframe(result["data"], use_container_width=True)

                # Download option
                csv = result["data"].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )

                # Basic analytics
                if len(result["data"]) > 0:
                    st.subheader("📈 Quick Stats")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Rows", len(result["data"]))

                    numeric_cols = result["data"].select_dtypes(
                        include=['number']).columns
                    if len(numeric_cols) > 0:
                        with col2:
                            first_numeric = numeric_cols[0]
                            total = result["data"][first_numeric].sum()
                            st.metric(
                                f"Total {first_numeric}", f"{total:,.2f}")

                        with col3:
                            avg = result["data"][first_numeric].mean()
                            st.metric(
                                f"Average {first_numeric}", f"{avg:,.2f}")
            else:
                st.info("No results returned for this query.")

        else:
            st.error(f"❌ Error: {result['error']}")

            # Error help
            st.subheader("🔧 Need Help?")
            if "api key" in result['error'].lower():
                st.warning("Check your GOOGLE_API_KEY in the .env file")
            elif "authentication" in result['error'].lower():
                st.warning(
                    "Make sure your Google Cloud credentials are set up for BigQuery access")
            elif "not found" in result['error'].lower():
                st.warning(
                    "Check that your project ID, dataset, and table names are correct")
            else:
                st.info(
                    "Try rephrasing your question or use one of the sample queries above")

# Environment setup helper


def create_env_file():
    """Helper function to create .env file"""
    env_content = """# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# BigQuery Configuration (optional - can be set in UI)
GCP_PROJECT_ID=your-project-id
DATASET_ID=your-dataset
TABLE_ID=invoices
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✅ .env file created! Please update it with your actual API key.")


if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating template...")
        create_env_file()

    # For Streamlit
    main()

"""
SETUP WITH .env FILE:

1. Install packages:
   pip install google-generativeai google-cloud-bigquery streamlit pandas python-dotenv

2. Create .env file in your project root:
   GOOGLE_API_KEY=your_actual_api_key_here

3. Get your API key from:
   https://makersuite.google.com/app/apikey

4. Set up BigQuery credentials (still needed for BigQuery access):
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

5. Run:
   streamlit run app.py

KEY DIFFERENCES WITH .env APPROACH:
✅ Simpler API setup - just need GOOGLE_API_KEY
✅ No complex Vertex AI configuration
✅ Direct Gemini API access
✅ Easier to deploy and share
✅ Better for development/testing

AUTHENTICATION NEEDED:
- GOOGLE_API_KEY (in .env) → For Gemini API calls
- Service Account (GCP credentials) → For BigQuery access

This is actually EASIER than the Vertex AI approach!
"""
