import google.generativeai as genai
from google.cloud import bigquery
import pandas as pd
import json
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
import re
import base64
from PIL import Image
import io
import random
import os
from firebase_functions import https_fn
from firebase_functions.options import CorsOptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedMultiModalInvoiceAgent:
    def __init__(self, gemini_api_key: str, gcp_project_id: str, dataset_id: str = "expense_tracking"):
        """Initialize Enhanced Multi-Modal Invoice Processing Agent"""
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configure BigQuery
        self.client = bigquery.Client(project=gcp_project_id)
        self.project_id = gcp_project_id
        self.dataset_id = dataset_id
        
        # Define expense categories with detailed mappings
        self.categories = [
            "rent", "property_tax", "electricity", "gas", "groceries", 
            "restaurants_dining", "ride_sharing", "parking_tolls", 
            "medicine", "insurance", "clothing", "electronics"
        ]
        
        # Natural language category mappings
        self.category_synonyms = {
            "food": ["groceries", "restaurants_dining"],
            "dining": ["restaurants_dining"],
            "eating": ["restaurants_dining", "groceries"],
            "shopping": ["groceries", "clothing", "electronics"],
            "utilities": ["electricity", "gas"],
            "power": ["electricity"],
            "transport": ["ride_sharing", "parking_tolls"],
            "travel": ["ride_sharing", "parking_tolls"],
            "medical": ["medicine", "insurance"],
            "health": ["medicine", "insurance"],
            "clothes": ["clothing"],
            "tech": ["electronics"],
            "gadgets": ["electronics"],
            "housing": ["rent", "property_tax"],
            "home": ["rent", "property_tax", "electricity", "gas"]
        }
        
        # Time period mappings
        self.time_patterns = {
            r'today|this day': 1,
            r'yesterday': 1,
            r'this week|last week|past week': 7,
            r'this month|last month|past month': 30,
            r'this quarter|last quarter|past quarter': 90,
            r'this year|last year|past year|1 year': 365,
            r'6 months|half year': 180,
            r'3 months': 90,
            r'2 months': 60,
            r'2 weeks|fortnight': 14,
            r'30 days': 30,
            r'90 days': 90,
            r'180 days': 180,
            r'365 days': 365
        }
        
        # Initialize dataset and tables
        self._create_dataset()
        self._create_expense_tables()
    
    def _create_dataset(self):
        """Create BigQuery dataset if it doesn't exist"""
        dataset_ref = f"{self.project_id}.{self.dataset_id}"
        
        try:
            self.client.get_dataset(dataset_ref)
            print(f"✅ Dataset {self.dataset_id} already exists")
        except Exception:
            try:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "Dataset for expense tracking and invoice processing"
                dataset = self.client.create_dataset(dataset, timeout=30)
                print(f"✅ Created dataset {self.dataset_id}")
            except Exception as e:
                print(f"❌ Error creating dataset: {e}")
                raise e
    
    def _create_expense_tables(self):
        """Create expense tracking tables"""
        main_table_ref = f"{self.project_id}.{self.dataset_id}.expenses"
        main_schema = [
            bigquery.SchemaField("expense_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("vendor_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("amount", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("payment_method", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("location", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("tags", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        try:
            table = bigquery.Table(main_table_ref, schema=main_schema)
            self.client.create_table(table, exists_ok=True)
            print(f"✅ Main expenses table ready")
        except Exception as e:
            print(f"❌ Error creating main table: {e}")
    
    def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        """Enhanced intent and entity extraction using AI"""
        
        analysis_prompt = f"""
        Analyze this user query and extract structured information:
        Query: "{user_query}"
        
        Available categories: {', '.join(self.categories)}
        Category synonyms: {json.dumps(self.category_synonyms, indent=2)}
        
        Extract and return JSON:
        {{
            "intent": "spending_query|expense_analysis|return_policy|insurance_expiry|budget_analysis|vendor_analysis|comparison|general_question",
            "categories": ["list of relevant categories"],
            "time_period": {{
                "type": "days|weeks|months|years|specific_date",
                "value": 30,
                "text": "original time reference"
            }},
            "amount_filters": {{
                "min_amount": null,
                "max_amount": null,
                "comparison": "greater_than|less_than|between|exact"
            }},
            "vendors": ["specific vendor names if mentioned"],
            "locations": ["specific locations if mentioned"],
            "query_type": "total|average|count|trend|comparison|list|summary",
            "specific_questions": ["what specific information is being requested"],
            "context_needed": ["what additional context might be helpful"],
            "confidence": 0.95
        }}
        
        Rules:
        - Map synonyms to actual categories (e.g., "food" -> ["groceries", "restaurants_dining"])
        - Extract time periods flexibly (e.g., "last month", "past 3 months", "this year")
        - Identify specific vendors mentioned
        - Determine what type of analysis is needed
        - High confidence for clear queries, lower for ambiguous ones
        """
        
        try:
            response = self.vision_model.generate_content(analysis_prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_intent_extraction(user_query)
        except Exception as e:
            print(f"Error in intent extraction: {e}")
            return self._fallback_intent_extraction(user_query)
    
    def _fallback_intent_extraction(self, user_query: str) -> Dict[str, Any]:
        """Fallback intent extraction using rule-based approach"""
        query_lower = user_query.lower()
        
        # Extract categories
        categories = []
        for category in self.categories:
            if category.replace('_', ' ') in query_lower or category in query_lower:
                categories.append(category)
        
        # Check synonyms
        for synonym, cats in self.category_synonyms.items():
            if synonym in query_lower:
                categories.extend(cats)
        
        # Extract time period
        time_period = {"type": "days", "value": 30, "text": "last month"}
        for pattern, days in self.time_patterns.items():
            if re.search(pattern, query_lower):
                time_period = {"type": "days", "value": days, "text": pattern}
                break
        
        # Determine intent
        intent = "spending_query"
        if any(word in query_lower for word in ['return', 'exchange', 'refund']):
            intent = "return_policy"
        elif any(word in query_lower for word in ['insurance', 'expire', 'expiry']):
            intent = "insurance_expiry"
        elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            intent = "comparison"
        elif any(word in query_lower for word in ['trend', 'pattern', 'over time']):
            intent = "expense_analysis"
        
        return {
            "intent": intent,
            "categories": list(set(categories)),
            "time_period": time_period,
            "amount_filters": {"min_amount": None, "max_amount": None, "comparison": None},
            "vendors": [],
            "locations": [],
            "query_type": "summary",
            "specific_questions": [user_query],
            "context_needed": [],
            "confidence": 0.7
        }
    
    def process_open_ended_query(self, user_query: str, user_id: str = "USER-001") -> str:
        """Process any open-ended natural language query about expenses"""
        
        # Extract intent and entities
        extracted_info = self.extract_intent_and_entities(user_query)
        
        try:
            # Route to appropriate handler based on intent
            if extracted_info["intent"] == "spending_query":
                return self._handle_spending_query(user_query, extracted_info, user_id)
            elif extracted_info["intent"] == "return_policy":
                return self._handle_return_policy_query(user_query, extracted_info, user_id)
            elif extracted_info["intent"] == "insurance_expiry":
                return self._handle_insurance_expiry_query(user_query, extracted_info, user_id)
            elif extracted_info["intent"] == "expense_analysis":
                return self._handle_expense_analysis_query(user_query, extracted_info, user_id)
            elif extracted_info["intent"] == "comparison":
                return self._handle_comparison_query(user_query, extracted_info, user_id)
            elif extracted_info["intent"] == "vendor_analysis":
                return self._handle_vendor_analysis_query(user_query, extracted_info, user_id)
            else:
                return self._handle_general_query(user_query, extracted_info, user_id)
                
        except Exception as e:
            return f"I encountered an error while processing your question: {str(e)}. Let me try a different approach."
    
    def _handle_spending_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle spending-related queries"""
        categories = extracted_info.get("categories", [])
        time_period = extracted_info.get("time_period", {})
        days = time_period.get("value", 30)
        
        # Build SQL query based on extracted information
        where_conditions = ["user_id = @user_id"]
        query_params = [bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
        
        # Add category filter
        if categories:
            category_placeholders = []
            for i, category in enumerate(categories):
                param_name = f"category_{i}"
                category_placeholders.append(f"@{param_name}")
                query_params.append(bigquery.ScalarQueryParameter(param_name, "STRING", category))
            where_conditions.append(f"category IN ({','.join(category_placeholders)})")
        
        # Add time filter
        if days:
            where_conditions.append("date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)")
            query_params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
        
        # Determine what information to return based on query type
        query_type = extracted_info.get("query_type", "summary")
        
        if query_type in ["total", "summary"]:
            sql_query = f"""
            SELECT 
                category,
                COUNT(*) as transaction_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(date) as first_date,
                MAX(date) as last_date,
                STRING_AGG(DISTINCT vendor_name, ', ' LIMIT 5) as top_vendors
            FROM `{self.project_id}.{self.dataset_id}.expenses`
            WHERE {' AND '.join(where_conditions)}
            GROUP BY category
            ORDER BY total_amount DESC
            """
        elif query_type == "list":
            sql_query = f"""
            SELECT 
                date,
                category,
                vendor_name,
                amount,
                description
            FROM `{self.project_id}.{self.dataset_id}.expenses`
            WHERE {' AND '.join(where_conditions)}
            ORDER BY date DESC, amount DESC
            LIMIT 20
            """
        else:
            # Default to summary
            sql_query = f"""
            SELECT 
                SUM(amount) as total_spent,
                COUNT(*) as total_transactions,
                AVG(amount) as avg_transaction,
                MIN(date) as first_purchase,
                MAX(date) as last_purchase
            FROM `{self.project_id}.{self.dataset_id}.expenses`
            WHERE {' AND '.join(where_conditions)}
            """
        
        try:
            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            df = self.client.query(sql_query, job_config=job_config).to_dataframe()
            
            if df.empty:
                return self._generate_no_data_response(user_query, extracted_info)
            
            return self._generate_spending_response(user_query, extracted_info, df)
            
        except Exception as e:
            return f"I couldn't retrieve your spending data. Error: {str(e)}"
    
    def _generate_spending_response(self, user_query: str, extracted_info: Dict, df: pd.DataFrame) -> str:
        """Generate natural language response for spending queries"""
        categories = extracted_info.get("categories", [])
        time_period = extracted_info.get("time_period", {})
        period_text = time_period.get("text", "recent period")
        
        # Generate response using AI
        response_prompt = f"""
        User asked: "{user_query}"
        
        Query results:
        {df.to_json(orient='records')}
        
        Generate a natural, conversational response that:
        1. Directly answers the user's question
        2. Provides specific amounts in ₹ format
        3. Includes relevant insights
        4. Uses a friendly, helpful tone
        5. Mentions the time period clearly
        6. Highlights key findings
        
        Keep it concise but informative. Use emojis appropriately.
        """
        
        try:
            ai_response = self.vision_model.generate_content(response_prompt)
            return ai_response.text
        except:
            # Fallback to manual response generation
            if len(df) == 1 and 'total_spent' in df.columns:
                row = df.iloc[0]
                return f"💰 You spent ₹{row['total_spent']:,.2f} across {row['total_transactions']} transactions in the {period_text}. Your average transaction was ₹{row['avg_transaction']:,.2f}."
            else:
                total = df['total_amount'].sum() if 'total_amount' in df.columns else 0
                transactions = df['transaction_count'].sum() if 'transaction_count' in df.columns else len(df)
                return f"💰 Total spending: ₹{total:,.2f} across {transactions} transactions in the {period_text}."
    
    def _generate_no_data_response(self, user_query: str, extracted_info: Dict) -> str:
        """Generate response when no data is found"""
        categories = extracted_info.get("categories", [])
        time_period = extracted_info.get("time_period", {})
        
        if categories:
            category_text = ", ".join(categories)
            return f"🔍 I couldn't find any expenses for {category_text} in the specified time period. You might want to check a different time range or category."
        else:
            return f"🔍 I couldn't find any expenses matching your query. You might want to try a different time period or be more specific about what you're looking for."
    
    def _handle_general_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle general queries that don't fit specific patterns"""
        return self._handle_spending_query(user_query, extracted_info, user_id)
    
    # Placeholder methods - implement these based on your needs
    def _handle_return_policy_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        return "Return policy feature coming soon!"
    
    def _handle_insurance_expiry_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        return "Insurance tracking feature coming soon!"
    
    def _handle_expense_analysis_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        return "Expense analysis feature coming soon!"
    
    def _handle_comparison_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        return "Comparison feature coming soon!"
    
    def _handle_vendor_analysis_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        return "Vendor analysis feature coming soon!"

# Global agent instance
agent = None

def get_agent():
    """Initialize agent if not already done"""
    global agent
    if agent is None:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        gcp_project_id = os.environ.get('GCP_PROJECT_ID', 'chitgenie')
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        agent = EnhancedMultiModalInvoiceAgent(gemini_api_key, gcp_project_id)
    return agent

@https_fn.on_request(cors=CorsOptions(cors_origins=True, cors_methods=["GET", "POST"]))
def process_expense_query(req):
    """Firebase Function to process expense queries"""
    try:
        if req.method == 'OPTIONS':
            return {'status': 'ok'}
        
        if req.method != 'POST':
            return {'error': 'Only POST method allowed'}, 405
        
        data = req.get_json()
        if not data or 'query' not in data:
            return {'error': 'Query is required'}, 400
        
        user_query = data['query']
        user_id = data.get('user_id', 'USER-001')
        
        # Get agent instance
        expense_agent = get_agent()
        
        # Process the query
        response = expense_agent.process_open_ended_query(user_query, user_id)
        
        return {
            'success': True,
            'response': response,
            'query': user_query
        }
        
    except Exception as e:
        print(f"Error processing expense query: {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500

@https_fn.on_request(cors=CorsOptions(cors_origins=True, cors_methods=["GET"]))
def health_check(req):
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}