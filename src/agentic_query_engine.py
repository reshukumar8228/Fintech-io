"""
Agentic Query Planner & Execution Engine for FinTech Analytics.
TransOrg AgentIQ Datathon - Track 1.

Translates natural language questions into structured query specifications,
executes dynamic data aggregation against actual project datasets, validates results,
synthesizes concise factual answers, generates dynamic Plotly chart specifications,
and produces execution trace metadata for Developer Mode auditability.
"""

import os
import re
import json
import logging
import sqlite3
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional Google Generative AI integration
HAS_GENAI = False
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class DatasetSchemaRegistry:
    """Registry of dataset metadata, allowed fields, dimensions, and metrics."""
    
    TABLES = {
        "fact_unified_analytics": "Denormalized payment transactions, chargebacks, customer KYC, and merchant profiles (20,000 rows).",
        "fact_transactions": "Core payment transaction events (txn_id, amount, status, timestamp, utr, date, etc.).",
        "fact_chargebacks": "Disputes & fraud complaints (complaint_id, disputed_amount, reason_category, severity, etc.).",
        "dim_customers": "Customer master & KYC profiles (user_id, full_name, kyc_status, risk_segment, city, state, monthly_income, etc.).",
        "dim_merchants": "Merchant master registry (merchant_id, merchant_name, merchant_category, declared_avg_ticket_size, city, state, status, etc.)"
    }

    ALLOWED_METRICS = {
        "amount": "Total transactional monetary value / sales / revenue in INR",
        "sales": "Alias for amount",
        "revenue": "Alias for amount",
        "disputed_amount": "Total monetary value of disputed / chargeback claims",
        "txn_count": "Total count of transactions",
        "count": "Total count of records / transactions",
        "dispute_count": "Count of disputed transactions / chargebacks",
        "failed_count": "Count of failed transactions",
        "success_count": "Count of successful transactions",
        "dispute_rate": "Ratio of chargebacks to total transactions",
        "failure_rate": "Ratio of failed transactions to total transactions",
        "avg_amount": "Average transaction ticket size / value",
        "monthly_income": "Customer declared monthly income",
        "declared_avg_ticket_size": "Merchant declared ticket size",
        "reporting_delay_days": "Days elapsed between transaction and dispute filing",
        "risk_score": "Merchant risk score (0-100)"
    }

    ALLOWED_DIMENSIONS = {
        "merchant_category": "Merchant business category (e.g. Grocery Stores, Apparel & Clothing, Restaurants & Dining, etc.)",
        "category": "Alias for merchant_category",
        "state": "Geographic state (e.g. Maharashtra, Uttar Pradesh, West Bengal, Punjab, Delhi, Karnataka, Tamil Nadu, Rajasthan, Telangana)",
        "merchant_state": "Merchant operating state",
        "customer_state": "Customer residential state",
        "region": "Mapped to state or city",
        "city": "Geographic city (e.g. Mumbai, Delhi, Bengaluru, Kolkata, Chennai, Hyderabad, Jaipur, Lucknow, Ludhiana, Jalandhar)",
        "merchant_city": "Merchant operating city",
        "customer_city": "Customer residential city",
        "kyc_status": "Customer KYC clearance tier (VERIFIED, PENDING, REJECTED, UNREGISTERED)",
        "risk_segment": "Customer risk classification (LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN)",
        "status": "Transaction clearance status (SUCCESS, FAILED, PENDING)",
        "txn_status": "Alias for status",
        "reason_category": "Chargeback reason category (Unauthorized / Fraud, Goods / Services Not Delivered, Duplicate / Technical Debit, etc.)",
        "severity": "Dispute SLA urgency level (CRITICAL, HIGH, MEDIUM, LOW)",
        "resolution_status": "Dispute operational state (CLOSED, REFUNDED, REJECTED, UNDER_REVIEW, OPEN)",
        "channel": "Dispute filing channel (Mobile App, Net Banking, Branch, Call Center, IVR)",
        "merchant_status": "Merchant status (ACTIVE, SUSPENDED, ON_HOLD, INACTIVE, UNKNOWN)",
        "business_type": "Corporate entity structure (Proprietorship, Partnership, Private Ltd, Public Ltd)",
        "occupation": "Customer occupation (Salaried, Self Employed, Business, Student)",
        "date": "Transaction calendar date (YYYY-MM-DD)",
        "year_month": "Monthly cohort period (e.g. 2026-01)",
        "hour": "Hour of transaction (0-23)",
        "day_name": "Day of week (Monday-Sunday)",
        "quarter": "Financial Quarter (Q1, Q2, Q3, Q4)",
        "merchant_name": "Merchant trade name",
        "merchant_id": "Merchant ID",
        "user_id": "Customer User ID",
        "full_name": "Customer full name"
    }

    QUARTER_DATES = {
        "Q1": ("2026-01-01", "2026-03-31"),
        "Q2": ("2026-04-01", "2026-06-30"),
        "Q3": ("2026-07-01", "2026-09-30"),
        "Q4": ("2026-10-01", "2026-12-31")
    }

    CANONICAL_CATEGORIES = [
        "Grocery Stores", "Apparel & Clothing", "Restaurants & Dining", "Department Stores",
        "Pharmacies & Healthcare", "Hotels & Lodging", "Transportation", "Telecommunication Services",
        "Books & Stationery", "Miscellaneous Retail"
    ]

    CANONICAL_STATES = [
        "Maharashtra", "Uttar Pradesh", "West Bengal", "Punjab", "Delhi",
        "Karnataka", "Tamil Nadu", "Rajasthan", "Telangana"
    ]


class AgenticQueryEngine:
    """
    Dataset-aware Agentic Query Engine.
    Handles Natural Language -> Query Planning -> Data Execution -> Answer Synthesis -> Dynamic Chart Specification.
    Includes full Execution Trace auditing for Developer Mode.
    """

    def __init__(self, data_dir: str = "data_cleaned"):
        self.data_dir = Path(data_dir)
        self.load_datasets()
        
        # User Custom API Config
        self.custom_api_key = None
        self.custom_provider = "gemini" # "gemini" or "openai"
        self.custom_model = "gemini-1.5-flash"
        
        self.init_llm_client()

    def set_custom_api_config(self, api_key: Optional[str] = None, provider: str = "gemini", model_name: Optional[str] = None):
        """Allows users to configure their own custom API key, provider, and model name."""
        if api_key:
            self.custom_api_key = api_key.strip()
        if provider:
            self.custom_provider = provider.strip().lower()
        if model_name:
            self.custom_model = model_name.strip()
            
        logger.info(f"Updated custom API config: provider={self.custom_provider}, model={self.custom_model}, key_provided={bool(self.custom_api_key)}")

    def load_datasets(self):
        """Loads analytics data frames and SQLite connection."""
        self.df_unified = pd.read_parquet(self.data_dir / "fact_unified_analytics.parquet")
        self.df_cust = pd.read_parquet(self.data_dir / "dim_customers.parquet")
        self.df_mch = pd.read_parquet(self.data_dir / "dim_merchants.parquet")
        self.df_txn = pd.read_parquet(self.data_dir / "fact_transactions.parquet")
        self.df_cb = pd.read_parquet(self.data_dir / "fact_chargebacks.parquet")

        # Standardize date types
        self.df_unified['timestamp'] = pd.to_datetime(self.df_unified['timestamp'])
        if 'date' not in self.df_unified.columns:
            self.df_unified['date'] = self.df_unified['timestamp'].dt.date.astype(str)
        self.df_unified['quarter'] = self.df_unified['timestamp'].dt.to_period('Q').astype(str).map(
            lambda q: f"Q{q[-1]}" if len(q) > 0 else "Q1"
        )

        db_path = self.data_dir / "fintech_analytics.db"
        self.db_path = db_path if db_path.exists() else None

    def init_llm_client(self):
        """Initializes default Gemini API client if key is available in environment."""
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_model = None
        if HAS_GENAI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.llm_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Successfully initialized Gemini LLM for Agentic Query Planner.")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini LLM client: {e}")
                self.llm_model = None

    def plan_query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Translates a natural language question into a structured Query Specification JSON.
        Prefers custom API key/model if provided, then environment LLM, then heuristic parser.
        """
        prompt = self._build_planner_prompt(question, history)

        # 1. Custom User API Key Execution
        if self.custom_api_key:
            try:
                raw_json_str = self._call_custom_llm(prompt)
                query_spec = self._parse_json_from_response(raw_json_str)
                if self._validate_query_spec(query_spec):
                    query_spec["_llm_used"] = f"Custom {self.custom_provider.title()} ({self.custom_model})"
                    return query_spec
            except Exception as e:
                logger.warning(f"Custom LLM API call failed: {e}. Falling back to standard LLM / Heuristic.")

        # 2. Standard Environment LLM Execution
        if self.llm_model:
            try:
                response = self.llm_model.generate_content(prompt)
                raw_text = response.text.strip()
                query_spec = self._parse_json_from_response(raw_text)
                if self._validate_query_spec(query_spec):
                    query_spec["_llm_used"] = "Default Gemini (Environment Key)"
                    return query_spec
            except Exception as e:
                logger.warning(f"Default LLM query planning failed: {e}. Falling back to heuristic planner.")

        # 3. Fallback Smart Heuristic Planner
        spec = self._heuristic_query_planner(question, history)
        spec["_llm_used"] = "Heuristic Pattern Engine"
        return spec

    def _call_custom_llm(self, prompt: str) -> str:
        """Makes direct HTTP REST API calls to Gemini or OpenAI using custom API Key and Model."""
        if "openai" in self.custom_provider or "gpt" in self.custom_model.lower():
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.custom_api_key}"
            }
            body = {
                "model": self.custom_model or "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a FinTech dataset query planner. Respond strictly with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=12) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        else:
            # Google Gemini REST API Call
            model = self.custom_model or "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.custom_api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1}
            }
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=12) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result["candidates"][0]["content"]["parts"][0]["text"]

    def _parse_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from model raw output text."""
        raw_text = text.strip()
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(1)
        else:
            first_brace = raw_text.find('{')
            last_brace = raw_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                raw_text = raw_text[first_brace:last_brace+1]
        return json.loads(raw_text)

    def _build_planner_prompt(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        history_str = ""
        if history:
            recent = history[-4:]
            history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent])

        return f"""You are an expert AI Data Agent for a FinTech Payments & Chargebacks dataset.
Your task is to analyze the user's natural language question and convert it into a structured JSON Query Plan.

### DATASET SCHEMAS & METADATA:
- Main Table: `fact_unified_analytics`
  - Metrics: `amount` (sales/revenue), `disputed_amount`, `txn_id` (count), `complaint_id` (dispute count), `reporting_delay_days`, `monthly_income`, `declared_avg_ticket_size`
  - Dimensions:
    - `merchant_category` (Grocery Stores, Apparel & Clothing, Restaurants & Dining, Department Stores, Pharmacies & Healthcare, Hotels & Lodging, Transportation, Telecommunication Services, Books & Stationery, Miscellaneous Retail)
    - `state` / `merchant_state` / `customer_state` (Maharashtra, Uttar Pradesh, West Bengal, Punjab, Delhi, Karnataka, Tamil Nadu, Rajasthan, Telangana)
    - `city` / `merchant_city` / `customer_city` (Mumbai, Delhi, Bengaluru, Kolkata, Chennai, Hyderabad, Jaipur, Lucknow, Ludhiana, Jalandhar)
    - `kyc_status` (VERIFIED, PENDING, REJECTED, UNREGISTERED)
    - `risk_segment` (LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN)
    - `status` (SUCCESS, FAILED, PENDING)
    - `reason_category` (Unauthorized / Fraud, Goods / Services Not Delivered, Duplicate / Technical Debit, etc.)
    - `severity` (CRITICAL, HIGH, MEDIUM, LOW)
    - `quarter` (Q1, Q2, Q3, Q4)
    - `year_month` (e.g. 2026-01)
    - `date` (YYYY-MM-DD)
    - `merchant_name`, `full_name`

### USER CONVERSATION HISTORY:
{history_str}

### CURRENT QUESTION:
"{question}"

### REQUIREMENTS:
Return ONLY a valid JSON object adhering to this structure:
{{
  "intent": "aggregation" | "comparison" | "trend" | "scatter" | "ranking" | "discovery" | "clarification" | "unanswerable",
  "metric": "amount" | "disputed_amount" | "txn_count" | "dispute_count" | "failure_rate" | "dispute_rate" | "monthly_income" | "reporting_delay_days",
  "aggregation": "sum" | "mean" | "count" | "min" | "max",
  "dimensions": ["dimension_field_name"],
  "filters": {{
    "quarter": "Q1|Q2|Q3|Q4",
    "status": "SUCCESS|FAILED|PENDING",
    "merchant_category": "category_name",
    "state": "state_name",
    "kyc_status": "status_name"
  }},
  "sort_order": "desc" | "asc",
  "limit": 10,
  "visualization": {{
    "type": "bar" | "line" | "scatter" | "pie" | "table" | "none",
    "x_field": "field_name",
    "y_field": "field_name",
    "title": "Clear descriptive chart title",
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label"
  }},
  "explanation": "Short note if clarification needed or data unavailable"
}}

Rules:
1. If question asks about KYC verification ("How much percentage KYC verified"), dimension MUST be ["kyc_status"], visualization MUST be "bar" or "pie", metric MUST be "txn_count".
2. If question asks to compare status ("successful, failed, pending"), dimension MUST be ["status"], visualization MUST be "bar" or "pie", metric MUST be "txn_count".
3. If question is about sales/revenue over time (monthly, daily), visualization type MUST be 'line'.
4. If question is comparing categories/regions/states/products, visualization type MUST be 'bar'.
5. If question asks for correlation/relationship between two numeric fields (e.g. income vs spend/amount), visualization type MUST be 'scatter'.
6. Handle typos gracefully (e.g., "comapre" -> "compare").
7. If question asks about data not present in dataset (e.g. stock prices, weather, Tokyo sales), set intent to 'unanswerable'.
"""

    def _validate_query_spec(self, spec: Dict[str, Any]) -> bool:
        if not isinstance(spec, dict) or "intent" not in spec:
            return False
        return True

    def _heuristic_query_planner(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Fuzzy deterministic query planner with typo handling and natural language intent matching."""
        raw_q = question.lower().strip()
        # Clean typos
        q = re.sub(r'\bcomapre\b', 'compare', raw_q)
        q = re.sub(r'\btransaction\b', 'transactions', q)

        # Check for unanswerable question
        unsupported_keywords = ['tokyo', 'london', 'new york', 'bitcoin', 'crypto', 'stock', 'weather', 'employee salary', 'inventory']
        if any(w in q for w in unsupported_keywords):
            return {
                "intent": "unanswerable",
                "explanation": f"The requested information is not available in the project's FinTech transaction and dispute dataset. Available data covers Indian UPI transactions, merchant categories, customer KYC, and chargebacks."
            }

        # Check for dataset discovery questions
        if any(w in q for w in ['what data is available', 'available data', 'what can i ask', 'fields are available', 'dataset schema', 'available fields']):
            return {
                "intent": "discovery",
                "explanation": "This project contains 5 core tables: fact_unified_analytics, fact_transactions, fact_chargebacks, dim_customers, and dim_merchants. Available fields include transaction amounts, status, timestamps, UTR validity, merchant categories, merchant states/cities, customer KYC status, risk segments, monthly income, dispute reasons, severity SLAs, and reporting delay days."
            }

        # Check for ambiguous single-word queries
        if q in ['compare', 'show metrics', 'performance', 'details', 'help', 'sales']:
            return {
                "intent": "clarification",
                "explanation": "Could you please specify which metric or dimension you'd like to analyze? For example: 'Compare Q3 sales by region', 'Show monthly sales trends', or 'Top 5 merchant categories by revenue'."
            }

        filters = {}
        # Quarter Filters
        for q_key in ['q1', 'q2', 'q3', 'q4']:
            if q_key in q:
                filters['quarter'] = q_key.upper()
                break

        # Category Filters
        for cat in DatasetSchemaRegistry.CANONICAL_CATEGORIES:
            if cat.lower() in q:
                filters['merchant_category'] = cat
                break

        # State Filters
        for st_name in DatasetSchemaRegistry.CANONICAL_STATES:
            if st_name.lower() in q:
                filters['state'] = st_name
                break

        # 1. KYC Verification Queries (e.g. "How much percentage KYC verified")
        if 'kyc' in q or 'verified' in q:
            return {
                "intent": "comparison",
                "metric": "txn_count",
                "aggregation": "count",
                "dimensions": ["kyc_status"],
                "filters": filters,
                "sort_order": "desc",
                "limit": 10,
                "visualization": {
                    "type": "bar",
                    "x_field": "kyc_status",
                    "y_field": "txn_count",
                    "title": "Customer KYC Status Breakdown & Completion",
                    "x_label": "KYC Status",
                    "y_label": "Customer Count"
                }
            }

        # 2. Transaction Status Comparison (e.g. "Compare successful, failed, and pending")
        if any(w in q for w in ['successful', 'failed', 'pending', 'status', 'success rate', 'failure rate']) or ('success' in q and 'failed' in q):
            return {
                "intent": "comparison",
                "metric": "txn_count",
                "aggregation": "count",
                "dimensions": ["status"],
                "filters": filters,
                "sort_order": "desc",
                "limit": 10,
                "visualization": {
                    "type": "bar",
                    "x_field": "status",
                    "y_field": "txn_count",
                    "title": "Transaction Clearance Status Breakdown",
                    "x_label": "Transaction Status",
                    "y_label": "Transaction Count"
                }
            }

        # Metric Determination
        metric = 'amount'
        if any(w in q for w in ['chargeback', 'dispute', 'disputed amount']):
            metric = 'disputed_amount'
        elif any(w in q for w in ['transaction count', 'number of transactions', 'volume count', 'txn count', 'how many transactions']):
            metric = 'txn_count'
        elif any(w in q for w in ['income', 'monthly income']):
            metric = 'monthly_income'
        elif any(w in q for w in ['delay', 'reporting delay']):
            metric = 'reporting_delay_days'

        intent = "aggregation"
        chart_type = "table"
        dimensions = []
        x_field = ""
        y_field = metric

        # 3. Scatter Plot / Relationship
        if any(w in q for w in ['vs', 'scatter', 'relationship', 'correlation', 'income vs', 'spend vs']):
            intent = "scatter"
            chart_type = "scatter"
            x_field = 'monthly_income' if 'income' in q else 'reporting_delay_days'
            y_field = 'amount'

        # 4. Time Series / Trend (Line chart)
        elif any(w in q for w in ['trend', 'monthly', 'daily', 'over time', 'by month', 'by date', 'history']):
            intent = "trend"
            chart_type = "line"
            if 'monthly' in q or 'month' in q:
                dimensions = ['year_month']
                x_field = 'year_month'
            else:
                dimensions = ['date']
                x_field = 'date'

        # 5. Category / Regional Comparison (Bar chart)
        elif any(w in q for w in ['by region', 'by state', 'by category', 'by severity', 'by reason', 'compare', 'region', 'state']):
            intent = "comparison"
            chart_type = "bar"
            if 'region' in q or 'state' in q:
                dimensions = ['customer_state']
                x_field = 'customer_state'
            elif 'category' in q:
                dimensions = ['merchant_category']
                x_field = 'merchant_category'
            elif 'severity' in q:
                dimensions = ['severity']
                x_field = 'severity'
            elif 'reason' in q:
                dimensions = ['reason_category']
                x_field = 'reason_category'
            else:
                dimensions = ['customer_state'] if 'region' in q else ['merchant_category']
                x_field = dimensions[0]

        # 6. Ranking / Top N (Bar chart)
        elif any(w in q for w in ['top', 'highest', 'best', 'worst', 'leading', 'rank']):
            intent = "ranking"
            chart_type = "bar"
            if 'user' in q or 'customer' in q:
                dimensions = ['full_name']
                x_field = 'full_name'
            else:
                dimensions = ['merchant_category']
                x_field = 'merchant_category'

        # 7. Simple Factual Question / Single KPI
        elif any(w in q for w in ['total', 'what is', 'average', 'summary', 'how many', 'overall', 'revenue', 'sales']):
            intent = "aggregation"
            chart_type = "table"

        # Check follow-up context
        if history and len(history) > 0 and ('line chart' in q or 'bar chart' in q or 'scatter' in q):
            if 'line' in q:
                chart_type = 'line'
                dimensions = ['date'] if not dimensions else dimensions
                x_field = dimensions[0] if dimensions else 'date'
            elif 'bar' in q:
                chart_type = 'bar'

        limit_val = 10
        top_match = re.search(r'top\s*(\d+)', q)
        if top_match:
            limit_val = int(top_match.group(1))

        return {
            "intent": intent,
            "metric": metric,
            "aggregation": "mean" if "average" in q or "avg" in q else "sum",
            "dimensions": dimensions,
            "filters": filters,
            "sort_order": "desc",
            "limit": limit_val,
            "visualization": {
                "type": chart_type,
                "x_field": x_field,
                "y_field": y_field,
                "title": f"{question.strip().capitalize()}",
                "x_label": x_field.replace("_", " ").title() if x_field else "",
                "y_label": y_field.replace("_", " ").title()
            }
        }

    def execute_query(self, query_spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes the query plan against the actual pandas dataset.
        Calculates exact metrics, aggregations, ratios, and percentage shares safely.
        """
        intent = query_spec.get("intent")
        if intent in ["unanswerable", "clarification", "discovery"]:
            return pd.DataFrame(), {}

        df = self.df_unified.copy()

        # 1. Apply Filters
        filters = query_spec.get("filters", {})
        if filters:
            if "quarter" in filters and filters["quarter"] in DatasetSchemaRegistry.QUARTER_DATES:
                q_start, q_end = DatasetSchemaRegistry.QUARTER_DATES[filters["quarter"]]
                df = df[(df['date'] >= q_start) & (df['date'] <= q_end)]

            if "status" in filters and filters["status"]:
                df = df[df['status'] == filters["status"].upper()]

            if "merchant_category" in filters and filters["merchant_category"]:
                df = df[df['merchant_category'].str.lower() == filters["merchant_category"].lower()]

            if "state" in filters and filters["state"]:
                st_val = filters["state"].lower()
                df = df[
                    (df['customer_state'].str.lower() == st_val) |
                    (df['merchant_state'].str.lower() == st_val)
                ]

            if "kyc_status" in filters and filters["kyc_status"]:
                df = df[df['kyc_status'].str.upper() == filters["kyc_status"].upper()]

        if df.empty:
            return pd.DataFrame(), {"is_empty": True}

        # 2. Handle Scatter Plot Intent
        if intent == "scatter" or query_spec.get("visualization", {}).get("type") == "scatter":
            vis = query_spec.get("visualization", {})
            x_col = vis.get("x_field", "monthly_income")
            y_col = vis.get("y_field", "amount")

            if x_col not in df.columns:
                x_col = "monthly_income" if "monthly_income" in df.columns else "amount"
            if y_col not in df.columns:
                y_col = "amount"

            scatter_df = df[[x_col, y_col, "merchant_category", "kyc_status"]].dropna().head(300)
            return scatter_df, {
                "count": len(scatter_df),
                "corr": float(scatter_df[x_col].corr(scatter_df[y_col])) if len(scatter_df) > 1 else 0.0
            }

        # 3. Handle Aggregation & Group-by
        dimensions = query_spec.get("dimensions", [])
        metric = query_spec.get("metric", "amount")
        agg_func = query_spec.get("aggregation", "sum")
        limit_val = query_spec.get("limit", 10)

        # Standardize metric field mapping
        if metric in ["sales", "revenue", "amount"]:
            target_metric = "amount"
        elif metric in ["disputed_amount", "disputes"]:
            target_metric = "disputed_amount"
        elif metric in ["monthly_income"]:
            target_metric = "monthly_income"
        elif metric in ["reporting_delay_days"]:
            target_metric = "reporting_delay_days"
        elif metric in ["txn_count", "count"]:
            target_metric = "txn_id"
            agg_func = "count"
        else:
            target_metric = "amount"

        clean_dims = []
        for d in dimensions:
            if d in ["region", "state"]:
                clean_dims.append("customer_state" if "customer_state" in df.columns else "merchant_state")
            elif d in ["category"]:
                clean_dims.append("merchant_category")
            elif d in df.columns:
                clean_dims.append(d)

        if not clean_dims:
            # Overall single-row KPI summary
            tot_val = float(df[target_metric].agg(agg_func)) if target_metric in df.columns else 0.0
            tot_txns = len(df)
            tot_cb = int(df['is_disputed'].sum())
            res_df = pd.DataFrame([{
                "metric": target_metric,
                "total_value": round(tot_val, 2),
                "total_transactions": tot_txns,
                "total_disputes": tot_cb,
                "dispute_rate": round(tot_cb / tot_txns, 4) if tot_txns > 0 else 0.0
            }])
            return res_df, {"total_value": tot_val, "total_transactions": tot_txns}

        # Execute Group By
        group_df = df.groupby(clean_dims).agg(
            total_value=(target_metric, agg_func),
            txn_count=('txn_id', 'count'),
            dispute_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum')
        ).reset_index()

        total_all_txns = len(df)
        group_df['pct_share'] = np.where(total_all_txns > 0, (group_df['txn_count'] / total_all_txns) * 100, 0.0)
        group_df['dispute_rate'] = np.where(group_df['txn_count'] > 0, group_df['dispute_count'] / group_df['txn_count'], 0.0)

        # Standardize value column name
        metric_col_name = "txn_count" if agg_func == "count" else (f"total_{target_metric}" if agg_func == "sum" else f"avg_{target_metric}")
        if metric_col_name != "txn_count":
            group_df = group_df.rename(columns={"total_value": metric_col_name})

        # Sort and limit
        sort_col = metric_col_name if metric_col_name in group_df.columns else group_df.columns[1]
        sort_order = query_spec.get("sort_order", "desc") == "desc"
        res_df = group_df.sort_values(sort_col, ascending=not sort_order).head(limit_val)

        return res_df, {
            "total_all_txns": total_all_txns,
            "top_entity": str(res_df.iloc[0][clean_dims[0]]) if not res_df.empty else "N/A",
            "top_val": float(res_df.iloc[0][sort_col]) if not res_df.empty else 0.0,
            "total_entities": len(group_df)
        }

    def answer_query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Main entry point for processing natural language questions.
        1. Plans query.
        2. Executes dataset query.
        3. Formats answer + dynamic chart spec + execution trace metadata.
        """
        query_spec = self.plan_query(question, history)
        intent = query_spec.get("intent")
        llm_engine_name = query_spec.get("_llm_used", "Heuristic Engine")

        # Step 2: Handle special non-data intents
        if intent == "unanswerable":
            trace = self._build_execution_trace(query_spec, pd.DataFrame(), llm_engine_name)
            return {
                "title": "Information Not Available in Project Dataset",
                "answer": query_spec.get("explanation", "The requested information is not available in the project's datasets."),
                "text": query_spec.get("explanation", "The requested information is not available in the project's datasets."),
                "data": pd.DataFrame(),
                "visualization": None,
                "chart_type": "none",
                "execution_trace": trace,
                "metadata": {"intent": intent}
            }

        if intent == "discovery":
            disc_df = pd.DataFrame(list(DatasetSchemaRegistry.TABLES.items()), columns=["Table Name", "Description"])
            trace = self._build_execution_trace(query_spec, disc_df, llm_engine_name)
            return {
                "title": "Available Project Datasets & Schema Overview",
                "answer": query_spec.get("explanation", "This project contains 5 core tables: fact_unified_analytics, fact_transactions, fact_chargebacks, dim_customers, and dim_merchants."),
                "text": query_spec.get("explanation", "This project contains 5 core tables: fact_unified_analytics, fact_transactions, fact_chargebacks, dim_customers, and dim_merchants."),
                "data": disc_df,
                "visualization": {"type": "table"},
                "chart_type": "table",
                "execution_trace": trace,
                "metadata": {"intent": intent}
            }

        if intent == "clarification":
            trace = self._build_execution_trace(query_spec, pd.DataFrame(), llm_engine_name)
            return {
                "title": "Clarification Needed",
                "answer": query_spec.get("explanation", "Could you please specify which metric or dimension you would like to analyze?"),
                "text": query_spec.get("explanation", "Could you please specify which metric or dimension you would like to analyze?"),
                "data": pd.DataFrame(),
                "visualization": None,
                "chart_type": "none",
                "execution_trace": trace,
                "metadata": {"intent": intent}
            }

        # Step 3: Execute Data Query
        df_res, stats = self.execute_query(query_spec)

        if df_res.empty:
            filters_str = json.dumps(query_spec.get("filters", {}))
            trace = self._build_execution_trace(query_spec, pd.DataFrame(), llm_engine_name)
            return {
                "title": "No Matching Records Found",
                "answer": f"No transactions or records matched the specified filter criteria ({filters_str}) in the dataset.",
                "text": f"No transactions or records matched the specified filter criteria ({filters_str}) in the dataset.",
                "data": pd.DataFrame(),
                "visualization": None,
                "chart_type": "none",
                "execution_trace": trace,
                "metadata": {"intent": intent, "filters": query_spec.get("filters")}
            }

        # Step 4: Synthesize Concise Direct Answer
        vis = query_spec.get("visualization", {})
        chart_type = vis.get("type", "bar")
        x_col = vis.get("x_field", "")
        y_col = vis.get("y_field", "total_amount")

        if x_col not in df_res.columns and len(df_res.columns) > 0:
            x_col = df_res.columns[0]
            vis["x_field"] = x_col

        numeric_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
        if y_col not in df_res.columns and numeric_cols:
            y_col = numeric_cols[0]
            vis["y_field"] = y_col

        answer_text = self._synthesize_answer(question, query_spec, df_res, stats, x_col, y_col)

        # Step 5: Construct Final Visualization Specification & Execution Trace
        vis_spec = {
            "type": chart_type,
            "title": vis.get("title") or f"{question.strip().capitalize()}",
            "xField": x_col,
            "yField": y_col,
            "xLabel": vis.get("x_label") or (x_col.replace("_", " ").title() if isinstance(x_col, str) else ""),
            "yLabel": vis.get("y_label") or (y_col.replace("_", " ").title() if isinstance(y_col, str) else "")
        }

        trace = self._build_execution_trace(query_spec, df_res, llm_engine_name)

        return {
            "title": vis_spec["title"],
            "answer": answer_text,
            "text": answer_text,
            "data": df_res,
            "visualization": vis_spec,
            "chart_type": chart_type,
            "x": x_col,
            "y": y_col,
            "y_label": vis_spec["yLabel"],
            "execution_trace": trace,
            "metadata": {
                "intent": intent,
                "filters": query_spec.get("filters", {}),
                "row_count": len(df_res)
            }
        }

    def _build_execution_trace(self, spec: Dict[str, Any], df: pd.DataFrame, planner_engine: str) -> Dict[str, Any]:
        """Builds Query Execution Trace object for Developer Mode audit panel."""
        vis = spec.get("visualization") or {}
        dims = spec.get("dimensions") or []
        filters = spec.get("filters") or {}

        return {
            "planner_engine": planner_engine,
            "target_dataset": "fact_unified_analytics",
            "requested_metric": spec.get("metric", "amount"),
            "aggregation_function": spec.get("aggregation", "sum").upper(),
            "grouping_dimensions": dims if dims else ["None (Overall KPI)"],
            "filters_applied": filters if filters else {"None": "No active filters"},
            "selected_chart_type": vis.get("type", "table").upper(),
            "rows_returned": len(df),
            "x_axis_field": vis.get("x_field", "N/A"),
            "y_axis_field": vis.get("y_field", "N/A")
        }

    def _synthesize_answer(self, question: str, spec: Dict[str, Any], df: pd.DataFrame, stats: Dict[str, Any], x_col: str, y_col: str) -> str:
        """Builds clean, human-readable answers with exact figures, percentages, and zero repetitive phrasing."""
        intent = spec.get("intent")
        filters = spec.get("filters", {})
        filter_desc = ""
        if "quarter" in filters:
            filter_desc = f" in {filters['quarter']}"

        # 1. Scatter Plot
        if intent == "scatter":
            corr = stats.get("corr", 0.0)
            return f"Analysis across **{len(df):,}** records shows a correlation coefficient of **{corr:.2f}** between {x_col.replace('_', ' ')} and {y_col.replace('_', ' ')}."

        # 2. KYC Breakdown
        if x_col == "kyc_status" and "pct_share" in df.columns:
            parts = []
            for _, r in df.iterrows():
                parts.append(f"**{r['kyc_status']}**: {r['txn_count']:,} ({r['pct_share']:.1f}%)")
            return f"Out of **{stats.get('total_all_txns', len(df)):,}** total customer records{filter_desc}: " + ", ".join(parts) + "."

        # 3. Transaction Status Breakdown
        if x_col == "status" and "pct_share" in df.columns:
            parts = []
            for _, r in df.iterrows():
                parts.append(f"**{r['status']}**: {r['txn_count']:,} ({r['pct_share']:.2f}%)")
            return f"Out of **{stats.get('total_all_txns', len(df)):,}** total transactions{filter_desc}: " + ", ".join(parts) + "."

        # 4. Regional or Categorical Breakdown
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            top_row = df.iloc[0]
            top_name = top_row[x_col]
            top_val = top_row[y_col]
            val_str = f"₹{top_val:,.2f}" if ("amount" in str(y_col) or "income" in str(y_col) or "value" in str(y_col)) else f"{top_val:,.2f}" if isinstance(top_val, float) else f"{top_val:,}"

            if len(df) > 1:
                second_row = df.iloc[1]
                sec_name = second_row[x_col]
                sec_val = second_row[y_col]
                sec_val_str = f"₹{sec_val:,.2f}" if ("amount" in str(y_col) or "income" in str(y_col) or "value" in str(y_col)) else f"{sec_val:,.2f}" if isinstance(sec_val, float) else f"{sec_val:,}"
                return f"**{top_name}** recorded the highest {y_col.replace('_', ' ')}{filter_desc} at **{val_str}**, followed by **{sec_name}** at **{sec_val_str}**."
            else:
                return f"**{top_name}** recorded a total {y_col.replace('_', ' ')}{filter_desc} of **{val_str}**."

        # 5. Overall Single KPI
        if "total_value" in df.columns or "total_value" in stats:
            tot_val = stats.get("total_value", df.iloc[0]["total_value"] if "total_value" in df.columns else 0.0)
            tot_txns = stats.get("total_transactions", df.iloc[0]["total_transactions"] if "total_transactions" in df.columns else len(df))
            return f"The total transaction amount{filter_desc} is **₹{tot_val:,.2f}** across **{tot_txns:,}** transactions."

        return f"Retrieved **{len(df):,}** matching records from the FinTech dataset."
