"""
TransOrg AgentIQ Datathon - Track 1: FinTech & BFSI
UPI Fraud Ring & Merchant Analytics Dashboard with Graph-First AI Agent.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AgentIQ | UPI Fraud Ring & Merchant Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analytics_engine import AnalyticsEngine
from src.graph_agent import FraudGraphEngine, AgentIQAssistant

# Custom CSS for Premium Design & Visual Polish
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #8892b0;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem 1.0rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #3a7bd5;
    }
    
    .kpi-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #00f2fe;
        margin-top: 0.3rem;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-critical {
        background-color: #ff4b4b22;
        color: #ff4b4b;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ff4b4b66;
    }
    
    .badge-high {
        background-color: #ffa50022;
        color: #ffa500;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ffa50066;
    }
    
    .badge-success {
        background-color: #00cc8822;
        color: #00cc88;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #00cc8866;
    }
    
    .insight-card {
        background: rgba(0, 210, 255, 0.05);
        border-left: 4px solid #00d2ff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }

    div[data-testid="stButton"] > button {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.04);
        color: #e0e6ed;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.45rem 0.65rem;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stButton"] > button:hover {
        border-color: #00d2ff;
        background: rgba(0, 210, 255, 0.1);
        color: #00f2fe;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing Analytics Engine & Knowledge Graph...")
def get_engines():
    engine = AnalyticsEngine("data_cleaned")
    graph_engine = FraudGraphEngine("data_cleaned")
    agent = AgentIQAssistant(engine, graph_engine)
    return engine, graph_engine, agent

engine, graph_engine, agent = get_engines()

# ==========================================
# SIDEBAR FILTERS (DYNAMIC MULTIDIMENSIONAL SLICING)
# ==========================================
st.sidebar.title("🎛️ Filter & Slice Mart")
st.sidebar.markdown("Filter all KPIs, charts, and metrics dynamically.")

# 1. Date Range
min_date = engine.df_txn['timestamp'].min().date()
max_date = engine.df_txn['timestamp'].max().date()

def reset_filters():
    st.session_state["filter_date_range"] = (min_date, max_date)
    st.session_state["filter_categories"] = []
    st.session_state["filter_kyc"] = []
    st.session_state["filter_statuses"] = []
    st.session_state["filter_risk"] = []

selected_date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="filter_date_range"
)

start_date = selected_date_range[0] if len(selected_date_range) > 0 else min_date
end_date = selected_date_range[1] if len(selected_date_range) > 1 else max_date

# 2. Merchant Categories
all_categories = sorted(list(engine.df_unified['merchant_category'].dropna().unique()))
selected_categories = st.sidebar.multiselect(
    "Merchant Categories",
    options=all_categories,
    default=[],
    key="filter_categories"
)

# 3. KYC Statuses
all_kyc = sorted(list(engine.df_unified['kyc_status'].dropna().unique()))
selected_kyc = st.sidebar.multiselect(
    "Customer KYC Status",
    options=all_kyc,
    default=[],
    key="filter_kyc"
)

# 4. Transaction Status
all_statuses = ['SUCCESS', 'FAILED', 'PENDING']
selected_statuses = st.sidebar.multiselect(
    "Transaction Status",
    options=all_statuses,
    default=[],
    key="filter_statuses"
)

# 5. Customer Risk Segment
all_risk = sorted(list(engine.df_unified['risk_segment'].dropna().unique()))
selected_risk = st.sidebar.multiselect(
    "Customer Risk Tier",
    options=all_risk,
    default=[],
    key="filter_risk"
)

# Apply dynamic filtering
filtered_df = engine.filter_data(
    start_date=start_date,
    end_date=end_date,
    categories=selected_categories if selected_categories else None,
    kyc_statuses=selected_kyc if selected_kyc else None,
    txn_statuses=selected_statuses if selected_statuses else None,
    risk_segments=selected_risk if selected_risk else None
)

kpis = engine.get_summary_kpis(filtered_df)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Slice**: {len(filtered_df):,} / {len(engine.df_unified):,} txns")
st.sidebar.button("🔄 Reset All Filters", on_click=reset_filters, use_container_width=True)

# ==========================================
# HEADER & TOP KPIS
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<div class='main-title'>🛡️ AgentIQ FinTech Intelligence Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Track 1: UPI Fraud Ring Detection, Merchant Risk Scoring & Dispute Analytics</div>", unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style='text-align: right; padding-top: 10px;'>
        <span class='badge-success'>System Online</span> &bull; 
        <span class='badge-critical'>Graph Active</span>
    </div>
    """, unsafe_allow_html=True)

# Top KPI Metric Row
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Total Volume</div>
        <div class='kpi-val'>₹{kpis['total_transaction_amount']/1e7:.2f} Cr</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Transactions</div>
        <div class='kpi-val'>{kpis['total_transaction_count']:,}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Avg Ticket Size</div>
        <div class='kpi-val'>₹{kpis['average_transaction_value']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Dispute Rate</div>
        <div class='kpi-val' style='color:#ff7675;'>{kpis['chargeback_to_transaction_ratio']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Failure Rate</div>
        <div class='kpi-val' style='color:#fab1a0;'>{kpis['failed_transaction_rate']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
with k6:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>KYC Verified</div>
        <div class='kpi-val' style='color:#55efc4;'>{kpis['kyc_completion_rate']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "📊 Executive Overview",
    "🏪 Merchant & Category Risk",
    "👤 Customer & KYC 360",
    "⚠️ Disputes & Chargebacks",
    "🕸️ Fraud Ring Graph Visualizer",
    "🤖 AgentIQ AI Copilot"
])


# Tabs initialized for subsequent tab feature additions
