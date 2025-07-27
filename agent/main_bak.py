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
    
    def _handle_return_policy_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle return policy and eligibility queries"""
        categories = extracted_info.get("categories", ["clothing"])  # Default to clothing
        time_period = extracted_info.get("time_period", {"value": 30})
        days = min(time_period.get("value", 30), 45)  # Limit to reasonable return window
        
        query = f"""
        SELECT 
            vendor_name,
            category,
            amount,
            date,
            description,
            DATE_DIFF(CURRENT_DATE(), date, DAY) as days_since_purchase
        FROM `{self.project_id}.{self.dataset_id}.expenses`
        WHERE user_id = @user_id 
        AND category IN UNNEST(@categories)
        AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
        ORDER BY date DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ArrayQueryParameter("categories", "STRING", categories if categories else ["clothing"]),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )
        
        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
            
            if df.empty:
                return f"🔍 I couldn't find any recent purchases in {', '.join(categories) if categories else 'the specified categories'} within the last {days} days."
            
            return self._generate_return_policy_response(user_query, df)
            
        except Exception as e:
            return f"I couldn't check your return eligibility. Error: {str(e)}"
    
    def _handle_insurance_expiry_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle insurance expiry and renewal queries"""
        query = f"""
        SELECT 
            vendor_name,
            amount,
            date as last_payment,
            description,
            DATE_DIFF(CURRENT_DATE(), date, DAY) as days_since_payment
        FROM `{self.project_id}.{self.dataset_id}.expenses`
        WHERE user_id = @user_id 
        AND category = 'insurance'
        ORDER BY date DESC
        LIMIT 5
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            ]
        )
        
        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
            
            if df.empty:
                return "🔍 I couldn't find any insurance payments in your records."
            
            return self._generate_insurance_response(user_query, df)
            
        except Exception as e:
            return f"I couldn't check your insurance status. Error: {str(e)}"
    
    def _handle_expense_analysis_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle expense analysis and trend queries"""
        categories = extracted_info.get("categories", [])
        time_period = extracted_info.get("time_period", {"value": 180})  # Default 6 months for trends
        days = time_period.get("value", 180)
        
        # Monthly trend analysis
        where_conditions = ["user_id = @user_id"]
        query_params = [bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
        
        if categories:
            where_conditions.append("category IN UNNEST(@categories)")
            query_params.append(bigquery.ArrayQueryParameter("categories", "STRING", categories))
        
        where_conditions.append("date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)")
        query_params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
        
        query = f"""
        SELECT 
            DATE_TRUNC(date, MONTH) as month,
            category,
            SUM(amount) as monthly_total,
            COUNT(*) as transaction_count,
            AVG(amount) as avg_transaction
        FROM `{self.project_id}.{self.dataset_id}.expenses`
        WHERE {' AND '.join(where_conditions)}
        GROUP BY month, category
        ORDER BY month DESC, monthly_total DESC
        """
        
        try:
            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            df = self.client.query(query, job_config=job_config).to_dataframe()
            
            if df.empty:
                return "🔍 I couldn't find enough data for trend analysis."
            
            return self._generate_analysis_response(user_query, extracted_info, df)
            
        except Exception as e:
            return f"I couldn't perform the expense analysis. Error: {str(e)}"
    
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
    
    def _generate_return_policy_response(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate response for return policy queries"""
        response = "🛍️ **Return Eligibility Check:**\n\n"
        
        # Define return policies by vendor
        return_policies = {
            'zara': 30, 'h&m': 30, 'uniqlo': 30,
            'lifestyle': 15, 'westside': 15, 'pantaloons': 7,
            'amazon': 30, 'flipkart': 10, 'myntra': 30
        }
        
        eligible_items = []
        expired_items = []
        
        for _, item in df.iterrows():
            vendor = item['vendor_name'].lower()
            days_since = item['days_since_purchase']
            
            # Determine return window
            return_window = 30  # default
            for vendor_key, window in return_policies.items():
                if vendor_key in vendor:
                    return_window = window
                    break
            
            item_info = {
                'vendor': item['vendor_name'],
                'amount': item['amount'],
                'date': item['date'],
                'days_since': days_since,
                'return_window': return_window,
                'remaining_days': max(0, return_window - days_since)
            }
            
            if days_since <= return_window:
                eligible_items.append(item_info)
            else:
                expired_items.append(item_info)
        
        if eligible_items:
            response += "✅ **Items you can still return:**\n"
            for item in eligible_items[:5]:  # Limit to 5 items
                response += f"• {item['vendor']} - ₹{item['amount']:,.2f} ({item['remaining_days']} days left)\n"
        
        if expired_items:
            response += f"\n❌ **Return window expired:**\n"
            for item in expired_items[:3]:  # Limit to 3 items
                response += f"• {item['vendor']} - ₹{item['amount']:,.2f} (expired {item['days_since'] - item['return_window']} days ago)\n"
        
        response += "\n💡 **Tip:** Keep receipts and original packaging for easier returns!"
        return response
    
    def _generate_insurance_response(self, user_query: str, df: pd.DataFrame) -> str:
        """Generate response for insurance queries"""
        if df.empty:
            return "🔍 No insurance payments found in your records."
        
        latest_payment = df.iloc[0]
        days_since = latest_payment['days_since_payment']
        
        # Estimate next payment (assuming annual payments)
        estimated_next_due = 365 - days_since
        
        response = f"🛡️ **Insurance Status:**\n\n"
        response += f"📋 **Last Payment:** ₹{latest_payment['amount']:,.2f} to {latest_payment['vendor_name']}\n"
        response += f"📅 **Payment Date:** {latest_payment['last_payment']}\n"
        response += f"⏰ **Days Since Payment:** {days_since} days\n\n"
        
        if estimated_next_due < 0:
            response += f"⚠️ **ALERT:** Your insurance appears to be overdue by {abs(estimated_next_due)} days!"
        elif estimated_next_due < 30:
            response += f"⚠️ **REMINDER:** Your insurance renewal is due in approximately {estimated_next_due} days."
        elif estimated_next_due < 90:
            response += f"📌 **UPCOMING:** Your insurance renewal is due in approximately {estimated_next_due} days."
        else:
            response += f"✅ **STATUS:** Your insurance appears current. Next renewal in approximately {estimated_next_due} days."
        
        return response
    
    def _generate_analysis_response(self, user_query: str, extracted_info: Dict, df: pd.DataFrame) -> str:
        """Generate response for expense analysis queries"""
        if df.empty:
            return "📊 I couldn't find enough data for trend analysis."
        
        # Use AI to generate insights
        analysis_prompt = f"""
        User asked: "{user_query}"
        
        Monthly expense data:
        {df.to_json(orient='records')}
        
        Generate an insightful analysis that includes:
        1. Overall spending trends
        2. Month-to-month changes
        3. Highest and lowest spending periods
        4. Average monthly spending
        5. Any patterns or anomalies
        6. Actionable insights
        
        Use ₹ for currency, include emojis, and keep it conversational but informative.
        """
        
        try:
            ai_response = self.vision_model.generate_content(analysis_prompt)
            return ai_response.text
        except:
            # Fallback analysis
            total_spending = df['monthly_total'].sum()
            avg_monthly = df['monthly_total'].mean()
            highest_month = df.loc[df['monthly_total'].idxmax()]
            
            return f"""📊 **Expense Analysis:**

💰 **Total Spending:** ₹{total_spending:,.2f}
📈 **Average Monthly:** ₹{avg_monthly:,.2f}
🔝 **Highest Month:** {highest_month['month']} (₹{highest_month['monthly_total']:,.2f})
📊 **Total Transactions:** {df['transaction_count'].sum()}

💡 Your spending patterns show {'consistent' if df['monthly_total'].std() < avg_monthly * 0.3 else 'variable'} behavior over this period."""
    
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
    
    def _handle_comparison_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle comparison queries between time periods or categories"""
        # This would implement comparison logic
        return "📊 Comparison feature coming soon! For now, try asking about specific time periods separately."
    
    def _handle_vendor_analysis_query(self, user_query: str, extracted_info: Dict, user_id: str) -> str:
        """Handle vendor-specific analysis queries"""
        # This would implement vendor analysis
        return "🏪 Vendor analysis feature coming soon! Try asking about spending by category instead."
    
    # Include all other methods from the original class
    def generate_dummy_data(self, user_id: str = "USER-001", months: int = 6):
        """Generate realistic dummy data for all categories"""
        expenses_data = []
        expense_id = 1
        
        # Define realistic spending patterns for each category
        category_patterns = {
            "rent": {"base_amount": 25000, "variance": 0, "frequency": "monthly"},
            "property_tax": {"base_amount": 15000, "variance": 0, "frequency": "quarterly"},
            "electricity": {"base_amount": 3500, "variance": 800, "frequency": "monthly"},
            "gas": {"base_amount": 1200, "variance": 300, "frequency": "monthly"},
            "groceries": {"base_amount": 8000, "variance": 2000, "frequency": "weekly"},
            "restaurants_dining": {"base_amount": 1500, "variance": 800, "frequency": "weekly"},
            "ride_sharing": {"base_amount": 800, "variance": 400, "frequency": "weekly"},
            "parking_tolls": {"base_amount": 200, "variance": 100, "frequency": "weekly"},
            "medicine": {"base_amount": 1000, "variance": 500, "frequency": "monthly"},
            "insurance": {"base_amount": 5000, "variance": 0, "frequency": "monthly"},
            "clothing": {"base_amount": 3000, "variance": 1500, "frequency": "monthly"},
            "electronics": {"base_amount": 8000, "variance": 5000, "frequency": "monthly"}
        }
        
        vendors = {
            "rent": ["ABC Property Management", "XYZ Apartments", "Metro Housing"],
            "property_tax": ["Bengaluru Municipal Corporation", "BBMP Tax Office"],
            "electricity": ["BESCOM", "Karnataka Power Corporation"],
            "gas": ["Indane Gas", "HP Gas", "Bharat Gas"],
            "groceries": ["Big Bazaar", "Reliance Fresh", "More Supermarket", "D-Mart"],
            "restaurants_dining": ["Dominos", "Subway", "KFC", "Cafe Coffee Day", "Barbeque Nation"],
            "ride_sharing": ["Uber", "Ola Cabs", "Rapido"],
            "parking_tolls": ["Electronic City Toll", "Mall Parking", "Airport Parking"],
            "medicine": ["Apollo Pharmacy", "MedPlus", "Netmeds", "1mg"],
            "insurance": ["LIC", "HDFC ERGO", "ICICI Lombard"],
            "clothing": ["Westside", "Pantaloons", "Lifestyle", "Zara"],
            "electronics": ["Croma", "Reliance Digital", "Amazon", "Flipkart"]
        }
        
        start_date = datetime.now() - timedelta(days=months * 30)
        
        for category, pattern in category_patterns.items():
            category_vendors = vendors[category]
            
            # Generate expenses based on frequency
            current_date = start_date
            while current_date <= datetime.now():
                if pattern["frequency"] == "weekly":
                    # 1-3 transactions per week
                    transactions = random.randint(1, 3)
                    for _ in range(transactions):
                        if current_date <= datetime.now():
                            amount = pattern["base_amount"] + random.randint(-pattern["variance"], pattern["variance"])
                            amount = max(amount, 50)  # Minimum amount
                            
                            expense = {
                                "expense_id": f"EXP-{expense_id:06d}",
                                "user_id": user_id,
                                "category": category,
                                "vendor_name": random.choice(category_vendors),
                                "amount": float(amount),
                                "date": current_date.date().isoformat(),
                                "description": self._generate_description(category, amount),
                                "payment_method": random.choice(["credit_card", "debit_card", "upi", "cash"]),
                                "location": "Bengaluru, Karnataka",
                                "tags": f"{category}_expense",
                                "created_at": current_date.isoformat()
                            }
                            expenses_data.append(expense)
                            expense_id += 1
                    
                    current_date += timedelta(days=7)
                
                elif pattern["frequency"] == "monthly":
                    amount = pattern["base_amount"] + random.randint(-pattern["variance"], pattern["variance"])
                    amount = max(amount, 100)
                    
                    expense = {
                        "expense_id": f"EXP-{expense_id:06d}",
                        "user_id": user_id,
                        "category": category,
                        "vendor_name": random.choice(category_vendors),
                        "amount": float(amount),
                        "date": current_date.date().isoformat(),
                        "description": self._generate_description(category, amount),
                        "payment_method": random.choice(["credit_card", "debit_card", "upi", "bank_transfer"]),
                        "location": "Bengaluru, Karnataka",
                        "tags": f"{category}_expense",
                        "created_at": current_date.isoformat()
                    }
                    expenses_data.append(expense)
                    expense_id += 1
                    
                    current_date += timedelta(days=30)
                
                elif pattern["frequency"] == "quarterly":
                    amount = pattern["base_amount"] + random.randint(-pattern["variance"], pattern["variance"])
                    
                    expense = {
                        "expense_id": f"EXP-{expense_id:06d}",
                        "user_id": user_id,
                        "category": category,
                        "vendor_name": random.choice(category_vendors),
                        "amount": float(amount),
                        "date": current_date.date().isoformat(),
                        "description": self._generate_description(category, amount),
                        "payment_method": "bank_transfer",
                        "location": "Bengaluru, Karnataka",
                        "tags": f"{category}_expense",
                        "created_at": current_date.isoformat()
                    }
                    expenses_data.append(expense)
                    expense_id += 1
                    
                    current_date += timedelta(days=90)
        
        # Insert data into BigQuery
        self._insert_expenses_data(expenses_data)
        return len(expenses_data)
    
    def _generate_description(self, category: str, amount: float) -> str:
        """Generate realistic descriptions for expenses"""
        descriptions = {
            "rent": f"Monthly rent payment - ₹{amount}",
            "property_tax": f"Property tax payment - ₹{amount}",
            "electricity": f"Electricity bill - {random.randint(200, 400)} units",
            "gas": f"Gas cylinder refill - {random.randint(1, 2)} cylinders",
            "groceries": "Weekly grocery shopping",
            "restaurants_dining": f"Dining at restaurant - {random.randint(2, 4)} people",
            "ride_sharing": f"Ride booking - {random.choice(['Airport', 'Office', 'Mall', 'Home'])}",
            "parking_tolls": "Parking fees and toll charges",
            "medicine": "Medical supplies and medicines",
            "insurance": "Insurance premium payment",
            "clothing": "Clothing and accessories purchase",
            "electronics": "Electronics and gadgets purchase"
        }
        return descriptions.get(category, f"{category} expense")
    
    def _insert_expenses_data(self, expenses_data: List[dict]):
        """Insert expenses data into BigQuery"""
        table_ref = f"{self.project_id}.{self.dataset_id}.expenses"
        
        try:
            errors = self.client.insert_rows_json(table_ref, expenses_data)
            if errors:
                print(f"❌ Error inserting data: {errors}")
            else:
                print(f"✅ Successfully inserted {len(expenses_data)} expense records")
        except Exception as e:
            print(f"❌ Error inserting expenses: {e}")
    
    def process_invoice_image(self, image_path: str = None, image_data: bytes = None) -> dict:
        """Process invoice image and extract information"""
        try:
            if image_path:
                image = Image.open(image_path)
            elif image_data:
                image = Image.open(io.BytesIO(image_data))
            else:
                return {"error": "No image provided"}
            
            prompt = f"""
            Analyze this invoice/receipt image and extract the following information:
            
            Categories to classify into: {', '.join(self.categories)}
            
            Extract and return JSON format:
            {{
                "category": "one of the predefined categories",
                "vendor_name": "name of the store/service provider",
                "amount": 1234.56,
                "date": "YYYY-MM-DD",
                "description": "brief description of items/services",
                "location": "city, state if available",
                "payment_method": "cash/card/upi if visible",
                "confidence": 0.95,
                "extracted_text": "key text from invoice",
                "items": ["list of items if visible"]
            }}
            
            Rules:
            - Map vendor names to appropriate categories (e.g., "Big Bazaar" -> "groceries")
            - If category is unclear, use best judgment based on vendor/items
            - Extract total amount, not individual item prices
            - Format date as YYYY-MM-DD
            - Confidence should be 0.0 to 1.0
            """
            
            response = self.vision_model.generate_content([prompt, image])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"error": "Could not parse invoice data"}
                
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
    
    def add_expense_from_image(self, image_path: str, user_id: str = "USER-001") -> dict:
        """Add expense to database from processed image"""
        # Process the image
        invoice_data = self.process_invoice_image(image_path=image_path)
        
        if "error" in invoice_data:
            return {"success": False, "message": invoice_data["error"]}
        
        # Create expense record
        expense_id = f"EXP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        expense = {
            "expense_id": expense_id,
            "user_id": user_id,
            "category": invoice_data.get("category", "unknown"),
            "vendor_name": invoice_data.get("vendor_name", "Unknown Vendor"),
            "amount": float(invoice_data.get("amount", 0)),
            "date": invoice_data.get("date", datetime.now().date().isoformat()),
            "description": invoice_data.get("description", "Expense from invoice"),
            "payment_method": invoice_data.get("payment_method", "unknown"),
            "location": invoice_data.get("location", "Unknown"),
            "tags": f"{invoice_data.get('category', 'unknown')}_invoice",
            "created_at": datetime.now().isoformat()
        }
        
        # Insert into BigQuery
        self._insert_expenses_data([expense])
        
        return {
            "success": True,
            "expense_id": expense_id,
            "extracted_data": invoice_data,
            "stored_expense": expense
        }


def main():
    """Main function to initialize and run the Enhanced Multi-Modal Invoice Agent"""
    # Configuration - Replace with your actual values
    GEMINI_API_KEY = "AIzaSyCuI8HwBRLXudqEeOSoksvfMlvyuC314XY"  # Replace with your Gemini API key
    GCP_PROJECT_ID = "chitgenie"       # Replace with your GCP project ID
    
    print("🚀 Enhanced Multi-Modal Invoice Processing Agent")
    print("=" * 60)
    
    try:
        # Initialize the agent
        print("🔧 Initializing agent...")
        agent = EnhancedMultiModalInvoiceAgent(GEMINI_API_KEY, GCP_PROJECT_ID)
        print("✅ Agent initialized successfully!")
        
        print("\n🤖 Enhanced Multi-Modal Invoice Processing Agent")
        print("💬 Ask me anything about your expenses in natural language!")
        print("\n📝 Example questions you can ask:")
        print("• How much did I spend on groceries last month?")
        print("• Is my insurance about to expire?")
        print("• How much did I pay for electricity in the last year?")
        print("• Can I return the clothes I bought last week?")
        print("• What are my spending trends for food this year?")
        print("• Show me all my dining expenses from the past 3 months")
        print("• Which vendor did I spend the most money on?")
        print("• Compare my grocery spending this month vs last month")
        print("• Process my receipt image: receipt.jpg")
        print("• Generate sample data (to populate the database)")
        print("\n⚡ Special commands:")
        print("• 'generate data' - Create sample expense data")
        print("• 'help' - Show available categories and examples")
        print("• 'quit' or 'exit' - Exit the application")
        print("=" * 60)
        
        # Main conversation loop
        while True:
            try:
                user_input = input("\n💭 Ask me anything: ").strip()
                
                # Handle exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("👋 Thank you for using the Enhanced Invoice Agent! Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    print("🤔 Please ask me something about your expenses!")
                    continue
                
                # Handle help command
                if user_input.lower() in ['help', 'categories']:
                    print("\n📋 Available expense categories:")
                    for i, category in enumerate(agent.categories, 1):
                        print(f"  {i:2d}. {category.replace('_', ' ').title()}")
                    
                    print("\n🎯 Natural language examples:")
                    examples = [
                        "How much did I spend on food this month?",
                        "Show me my electricity bills for the past 6 months",
                        "What's my biggest expense category?",
                        "Can I still return items I bought recently?",
                        "When is my insurance due for renewal?",
                        "How much do I spend on average per month?",
                        "Show me all transactions above ₹5000",
                        "What are my spending patterns for clothing?"
                    ]
                    for example in examples:
                        print(f"  • {example}")
                    continue
                
                # Handle data generation
                if user_input.lower() in ['generate data', 'create data', 'sample data', 'dummy data']:
                    print("🔄 Generating realistic expense data...")
                    try:
                        count = agent.generate_dummy_data(months=6)
                        print(f"✅ Successfully generated {count} expense records across all categories!")
                        print("💡 Now you can ask questions about your expenses!")
                    except Exception as e:
                        print(f"❌ Error generating data: {e}")
                    continue
                
                # Handle image processing
                if any(keyword in user_input.lower() for keyword in ['process', 'image', 'receipt', 'invoice', '.jpg', '.png', '.jpeg']):
                    # Extract image path from input
                    words = user_input.split()
                    image_path = None
                    for word in words:
                        if any(ext in word.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                            image_path = word
                            break
                    
                    if image_path:
                        print(f"📸 Processing invoice image: {image_path}")
                        result = agent.add_expense_from_image(image_path)
                        
                        if result["success"]:
                            data = result["extracted_data"]
                            print(f"✅ Invoice processed successfully!")
                            print(f"  📂 Category: {data.get('category', 'Unknown')}")
                            print(f"  🏪 Vendor: {data.get('vendor_name', 'Unknown')}")
                            print(f"  💰 Amount: ₹{data.get('amount', 0):,.2f}")
                            print(f"  📅 Date: {data.get('date', 'Unknown')}")
                            print(f"  🎯 Confidence: {data.get('confidence', 0):.0%}")
                            print(f"  💾 Expense ID: {result['expense_id']}")
                        else:
                            print(f"❌ Error processing image: {result['message']}")
                    else:
                        print("❌ Please specify an image file path")
                        print("   Example: 'process receipt.jpg' or 'analyze my invoice image.png'")
                    continue
                
                # Process natural language query
                print("🤔 Analyzing your question...")
                
                # Use the enhanced open-ended query processor
                response = agent.process_open_ended_query(user_input)
                
                print(f"\n🤖 {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                print("💡 Try rephrasing your question or check your input.")
                
    except Exception as e:
        print(f"\n❌ Failed to initialize the agent: {e}")
        print("\n🔧 Setup checklist:")
        print("1. ✓ Replace GEMINI_API_KEY with your actual Gemini API key")
        print("2. ✓ Replace GCP_PROJECT_ID with your actual GCP project ID")
        print("3. ✓ Enable BigQuery API in your GCP project")
        print("4. ✓ Set up authentication: gcloud auth application-default login")
        print("5. ✓ Install required packages: pip install google-generativeai google-cloud-bigquery pandas pillow")
        print("\n📚 For setup help, visit:")
        print("   • Gemini API: https://ai.google.dev/")
        print("   • GCP BigQuery: https://cloud.google.com/bigquery")


if __name__ == "__main__":
    main()