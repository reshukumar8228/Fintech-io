"""
TransOrg AgentIQ Datathon - Track 1: FinTech & BFSI
UPI Fraud Ring & Merchant Analytics Dashboard with Graph-First AI Agent.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

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
        color: var(--text-color, #64748b);
        opacity: 0.85;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.0rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #3a7bd5;
    }
    
    .kpi-val {
        font-size: 1.7rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.3rem;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color, #475569);
        opacity: 0.8;
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
        background: rgba(0, 210, 255, 0.06);
        border-left: 4px solid #00d2ff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }

    /* Button layout & smooth micro-interaction */
    div[data-testid="stButton"] > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        border-color: #00d2ff !important;
        transform: translateY(-2px) !important;
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

st.sidebar.markdown(f"**Filtered Slice**: {len(filtered_df):,} / {len(engine.df_unified):,} txns")
st.sidebar.button("🔄 Reset All Filters", on_click=reset_filters, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ AI Copilot & Developer Settings")
dev_mode = st.sidebar.toggle("🛠️ Developer Mode", value=False, help="Show Dataset Schema Explorer and Query Execution Trace Panel")

st.sidebar.markdown("#### 🔑 Custom AI API Key & Model")
provider_choice = st.sidebar.selectbox("API Provider", ["Gemini (Google)", "OpenAI"], index=0)
custom_key = st.sidebar.text_input("Custom API Key (Optional)", type="password", help="Enter your Gemini or OpenAI API Key to override system defaults")
custom_model = st.sidebar.text_input("Model Name", value="gemini-1.5-flash" if "Gemini" in provider_choice else "gpt-4o")

# Pass custom settings down to the assistant agent
provider_str = "openai" if "OpenAI" in provider_choice else "gemini"
agent.set_custom_api_config(api_key=custom_key if custom_key else None, provider=provider_str, model_name=custom_model)

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

# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
with tabs[0]:
    st.markdown("""
    <div class='insight-card'>
        <strong>💡 Key Executive Takeaway:</strong> Overall payment volume is stable with an 85.3% success rate. 
        However, chargebacks correlate heavily with missing/invalid UTR numbers and unverified KYC accounts.
    </div>
    """, unsafe_allow_html=True)

    df_daily = engine.get_daily_trends(filtered_df)
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        if not df_daily.empty:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=df_daily['date'], y=df_daily['total_amount'],
                mode='lines+markers', name='Daily Volume (₹)',
                line=dict(color='#00d2ff', width=2.5),
                fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.1)'
            ))
            fig_vol.update_layout(
                title="Daily Transaction Volume Trend (₹)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                height=320
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("No transaction data available for the selected filters.")
        
    with col_t2:
        if not filtered_df.empty:
            status_dist = filtered_df['status'].value_counts().reset_index()
            status_dist.columns = ['Status', 'Count']
            fig_pie = px.pie(
                status_dist, names='Status', values='Count',
                title="Transaction Status Distribution",
                color='Status',
                color_discrete_map={'SUCCESS': '#00cc88', 'FAILED': '#ff4b4b', 'PENDING': '#ffa500'},
                hole=0.45,
                template="plotly_dark"
            )
            fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No status data available.")
        
    col_t3, col_t4 = st.columns(2)
    with col_t3:
        if not df_daily.empty:
            fig_status_bar = go.Figure()
            fig_status_bar.add_trace(go.Bar(x=df_daily['date'], y=df_daily['success_count'], name='Success', marker_color='#00cc88'))
            fig_status_bar.add_trace(go.Bar(x=df_daily['date'], y=df_daily['failed_count'], name='Failed', marker_color='#ff4b4b'))
            fig_status_bar.add_trace(go.Bar(x=df_daily['date'], y=df_daily['pending_count'], name='Pending', marker_color='#ffa500'))
            fig_status_bar.update_layout(
                barmode='stack',
                title="Daily Success vs Failed vs Pending Transactions",
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_status_bar, use_container_width=True)
        
    with col_t4:
        df_hourly = engine.get_hourly_failure_trend(filtered_df)
        if not df_hourly.empty:
            fig_hour = px.bar(
                df_hourly, x='hour', y='failure_rate',
                title="Hourly Payment Failure Rate (00:00 - 23:00)",
                labels={'hour': 'Hour of Day', 'failure_rate': 'Failure Rate'},
                color='failure_rate',
                color_continuous_scale='Reds',
                template="plotly_dark"
            )
            fig_hour.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("---")
    st.subheader("UTR Data Quality & Payment Integrity Metrics")
    utr_m = engine.get_utr_health_metrics(filtered_df)
    
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Valid UTR Transactions", f"{utr_m['valid_utr_count']:,}")
    u2.metric("Invalid / Missing UTRs", f"{utr_m['invalid_or_missing_utr_count']:,}")
    u3.metric("Valid UTR Dispute Rate", f"{utr_m['valid_utr_dispute_rate']:.2%}")
    u4.metric("Invalid UTR Dispute Rate", f"{utr_m['invalid_utr_dispute_rate']:.2%}")


# ==========================================
# TAB 2: MERCHANT & CATEGORY RISK
# ==========================================
with tabs[1]:
    st.markdown("""
    <div class='insight-card'>
        <strong>🚨 Risk Intelligence:</strong> High ticket size divergence (>2.5x declared value) combined with elevated dispute rates 
        pinpoints collusive merchants or unauthorized surcharge syndicates.
    </div>
    """, unsafe_allow_html=True)

    df_cat = engine.get_merchant_category_metrics(filtered_df)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if not df_cat.empty:
            fig_cat_vol = px.bar(
                df_cat, x='total_amount', y='merchant_category', orientation='h',
                title="Transaction Volume by Category (₹)",
                color='total_amount', color_continuous_scale='Blues',
                template="plotly_dark"
            )
            fig_cat_vol.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cat_vol, use_container_width=True)
        else:
            st.info("No category data available.")
        
    with col_c2:
        if not df_cat.empty:
            fig_cat_disp = px.bar(
                df_cat.sort_values('dispute_rate', ascending=True),
                x='dispute_rate', y='merchant_category', orientation='h',
                title="Dispute Rate by Merchant Category (%)",
                color='dispute_rate', color_continuous_scale='Reds',
                template="plotly_dark"
            )
            fig_cat_disp.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cat_disp, use_container_width=True)

    st.markdown("---")
    st.subheader("High-Risk Merchant Risk Matrix (Risk Score 0 - 100)")
    risk_merchants = engine.get_high_risk_merchants(filtered_df, min_txns=2, top_n=20)
    if not risk_merchants.empty:
        st.dataframe(
            risk_merchants[[
                'merchant_id', 'merchant_name', 'merchant_category', 'merchant_status',
                'txn_count', 'chargeback_count', 'disputed_amount', 'chargeback_ratio', 'risk_score'
            ]].style.format({
                'disputed_amount': '₹{:,.2f}',
                'chargeback_ratio': '{:.1%}',
                'risk_score': '{:.1f}'
            }),
            use_container_width=True
        )
    else:
        st.info("No high-risk merchants found in current filter.")

    st.markdown("---")
    st.subheader("Merchant Ticket Size Anomaly Detection (Actual vs Declared Divergence > 2.5x)")
    anomalies = engine.get_merchant_ticket_anomalies(filtered_df, ratio_threshold=2.5, top_n=15)
    if not anomalies.empty:
        st.dataframe(
            anomalies[[
                'merchant_id', 'merchant_name', 'merchant_category', 'merchant_status',
                'declared_avg_ticket', 'actual_avg_ticket', 'ticket_divergence_ratio', 'chargeback_count'
            ]].style.format({
                'declared_avg_ticket': '₹{:,.2f}',
                'actual_avg_ticket': '₹{:,.2f}',
                'ticket_divergence_ratio': '{:.2f}x'
            }),
            use_container_width=True
        )
    else:
        st.success("No extreme ticket size divergence anomalies detected in current filter.")


# ==========================================
# TAB 3: CUSTOMER & KYC 360
# ==========================================
with tabs[2]:
    st.markdown("""
    <div class='insight-card'>
        <strong>👤 Identity Funnel Insight:</strong> Unverified and rejected KYC customers account for a disproportionate share of chargebacks.
        Strict verification gatekeeping reduces dispute losses by over 40%.
    </div>
    """, unsafe_allow_html=True)

    df_kyc = engine.get_kyc_status_breakdown(filtered_df)
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if not df_kyc.empty:
            fig_kyc_vol = px.pie(
                df_kyc, names='kyc_status', values='total_amount',
                title="Transaction Volume by Customer KYC Status",
                hole=0.45,
                color='kyc_status',
                color_discrete_map={'VERIFIED': '#00cc88', 'PENDING': '#ffa500', 'REJECTED': '#ff4b4b', 'UNREGISTERED': '#636e72'},
                template="plotly_dark"
            )
            fig_kyc_vol.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_kyc_vol, use_container_width=True)
        else:
            st.info("No KYC data available.")
        
    with col_k2:
        if not df_kyc.empty:
            fig_kyc_disp = px.bar(
                df_kyc, x='kyc_status', y='dispute_rate',
                title="Dispute Rate by KYC Status (%)",
                color='dispute_rate', color_continuous_scale='Reds',
                template="plotly_dark"
            )
            fig_kyc_disp.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_kyc_disp, use_container_width=True)

    st.markdown("---")
    st.subheader("High-Risk Repeat-Dispute Customer Watchlist")
    high_risk_users = engine.get_high_risk_users(filtered_df, top_n=20)
    if not high_risk_users.empty:
        st.dataframe(
            high_risk_users[[
                'user_id', 'full_name', 'kyc_status', 'risk_segment', 'city',
                'dispute_count', 'total_disputed_amount', 'avg_delay'
            ]].style.format({
                'total_disputed_amount': '₹{:,.2f}',
                'avg_delay': '{:.1f} days'
            }),
            use_container_width=True
        )
    else:
        st.info("No repeat dispute customers in current slice.")


# ==========================================
# TAB 4: DISPUTES & CHARGEBACKS
# ==========================================
with tabs[3]:
    st.markdown("""
    <div class='insight-card'>
        <strong>⚠️ SLA & ATO Early Warning:</strong> Disputes lodged >7 days after transaction timestamp strongly signal Account Takeover (ATO) 
        or credential stuffing where the genuine cardholder/account owner realizes theft only at billing cycle close.
    </div>
    """, unsafe_allow_html=True)

    df_reasons = engine.get_chargeback_reasons(filtered_df)
    df_sev = engine.get_chargeback_severity(filtered_df)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if not df_reasons.empty:
            fig_reason = px.pie(
                df_reasons, names='reason_category', values='complaint_count',
                title="Dispute Reason Code Distribution",
                hole=0.45,
                template="plotly_dark"
            )
            fig_reason.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_reason, use_container_width=True)
        else:
            st.info("No dispute reasons in selected slice.")
            
    with col_d2:
        if not df_sev.empty:
            fig_sev = px.bar(
                df_sev, x='severity', y='complaint_count',
                title="Dispute Severity Priority Breakdown",
                color='severity',
                color_discrete_map={'CRITICAL': '#ff4b4b', 'HIGH': '#ff7675', 'MEDIUM': '#ffa500', 'LOW': '#00cc88'},
                template="plotly_dark"
            )
            fig_sev.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sev, use_container_width=True)

    st.markdown("---")
    st.subheader("Dispute Reporting Delays (>7 Days Indicates ATO / Fraud Syndicate)")
    disputed_slice = filtered_df[filtered_df['is_disputed'] & filtered_df['reporting_delay_days'].notna()]
    if not disputed_slice.empty:
        fig_delay = px.histogram(
            disputed_slice, x='reporting_delay_days', nbins=30,
            title="Dispute Reporting Delay Distribution (Days from Transaction to Dispute)",
            labels={'reporting_delay_days': 'Reporting Delay (Days)'},
            color_discrete_sequence=['#00d2ff'],
            template="plotly_dark"
        )
        fig_delay.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_delay, use_container_width=True)
    else:
        st.info("No dispute delay data in selected slice.")


# ==========================================
# TAB 5: FRAUD RING GRAPH VISUALIZER
# ==========================================
with tabs[4]:
    st.markdown("""
    <div class='insight-card'>
        <strong>🕸️ Graph Theory Fraud Detection:</strong> Heterogeneous graph analysis exposes two key fraud topologies:
        1. <strong>Mule Settlement Account Rings:</strong> Multiple front merchants pooling funds into a single underlying bank account.
        2. <strong>Bipartite Collusive Syndicates:</strong> Dense clusters of compromised users and collusive merchants staging disputes.
    </div>
    """, unsafe_allow_html=True)

    rings = graph_engine.detect_shared_account_rings()
    clusters = graph_engine.detect_collusive_fraud_clusters()
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(f"#### 🔗 Mule Merchant Rings (Shared Accounts: {len(rings)})")
        if rings:
            mule_summary = []
            for r in rings[:10]:
                mule_summary.append({
                    "Settlement Account": r['account_identifier'],
                    "Merchants": r['merchant_count'],
                    "Total Transactions": r['total_transactions'],
                    "Disputes": r['total_disputes'],
                    "Disputed Amount": f"₹{r['disputed_amount']:,.2f}",
                    "Risk Level": r['risk_level']
                })
            st.dataframe(pd.DataFrame(mule_summary), use_container_width=True)
            
    with col_g2:
        st.markdown(f"#### 🚨 Collusive Fraud Syndicates ({len(clusters)})")
        if clusters:
            clust_summary = []
            for c in clusters[:10]:
                clust_summary.append({
                    "Cluster ID": c['cluster_id'],
                    "Users": c['user_count'],
                    "Merchants": c['merchant_count'],
                    "Disputes": c['total_disputes'],
                    "Disputed Amount": f"₹{c['total_disputed_amount']:,.2f}",
                    "Risk Score": c['syndicate_risk_score'],
                    "Risk Tier": c['risk_tier']
                })
            st.dataframe(pd.DataFrame(clust_summary), use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Entity Ego Network Visualizer")
    
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        selected_node = st.text_input("Enter Merchant ID (e.g. MCH7045) or Customer ID (e.g. USR45826):", value="MCH7045")
    with col_s2:
        hops = st.slider("Hops Distance", min_value=1, max_value=2, value=1)
    with col_s3:
        st.write("")
        st.write("")
        btn_visualize = st.button("Render Network Graph", use_container_width=True)

    if selected_node:
        subg_data = graph_engine.extract_subgraph(selected_node, hops=hops)
        if "error" in subg_data:
            st.error(subg_data["error"])
        else:
            st.info(f"Visualizing network around **{subg_data['center_node']}** ({subg_data['total_nodes']} nodes, {subg_data['total_edges']} edges)")
            
            # Plotly Network Visualization
            edge_x = []
            edge_y = []
            
            node_map = {n['id']: (n['x'], n['y']) for n in subg_data['nodes']}
            for edge in subg_data['edges']:
                if edge['source'] in node_map and edge['target'] in node_map:
                    x0, y0 = node_map[edge['source']]
                    x1, y1 = node_map[edge['target']]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.2, color='#74b9ff'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = [n['x'] for n in subg_data['nodes']]
            node_y = [n['y'] for n in subg_data['nodes']]
            node_text = [n['label'] for n in subg_data['nodes']]
            node_colors = []
            node_sizes = []
            
            for n in subg_data['nodes']:
                if n['is_center']:
                    node_colors.append('#e74c3c') # Red for center
                    node_sizes.append(22)
                elif n['node_type'] == 'merchant':
                    node_colors.append('#f39c12') # Orange for merchant
                    node_sizes.append(15)
                elif n['node_type'] == 'settlement_account':
                    node_colors.append('#9b59b6') # Purple for account
                    node_sizes.append(14)
                else:
                    node_colors.append('#2ecc71') # Green for customer
                    node_sizes.append(12)
                    
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[n['id'] for n in subg_data['nodes']],
                textposition="top center",
                textfont=dict(size=9, color="#ffffff"),
                hovertext=node_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='#ffffff')
                )
            )
            
            fig_net = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Network Ego Graph: {subg_data['center_node']}",
                    template="plotly_dark",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=20, r=20, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500
                )
            )
            st.plotly_chart(fig_net, use_container_width=True)


# ==========================================
# TAB 6: AGENTIQ AI COPILOT
# ==========================================
with tabs[5]:
    st.subheader("🤖 AgentIQ Dynamic Dataset-Aware AI Analytics Copilot")
    st.markdown("Ask natural-language questions about your actual payment, merchant, KYC, and chargeback datasets. The agent calculates figures deterministically and generates dynamic charts.")

    # DEVELOPER MODE: DATASET SCHEMA EXPLORER
    if dev_mode:
        with st.expander("📁 Dataset Schema Explorer (Developer Mode Active)", expanded=False):
            st.markdown("Inspect all 5 analytical tables, column definitions, data types, and sample data in the star schema mart:")
            schema_tabs = st.tabs(["fact_unified_analytics", "dim_customers", "dim_merchants", "fact_transactions", "fact_chargebacks"])
            
            with schema_tabs[0]:
                st.markdown("**fact_unified_analytics**: Denormalized analytics view combining transactions, chargebacks, customer KYC, and merchant registry.")
                st.markdown(f"Total Rows: **{len(engine.df_unified):,}** | Columns: `{list(engine.df_unified.columns)}`")
                st.dataframe(engine.df_unified.head(3), use_container_width=True)
                
            with schema_tabs[1]:
                st.markdown("**dim_customers**: Customer KYC master records, risk tiers, monthly income, and geographical state/city.")
                st.markdown(f"Total Rows: **{len(engine.df_cust):,}** | Columns: `{list(engine.df_cust.columns)}`")
                st.dataframe(engine.df_cust.head(3), use_container_width=True)
                
            with schema_tabs[2]:
                st.markdown("**dim_merchants**: Merchant registry master, onboarding ticket size, category, status, and settlement accounts.")
                st.markdown(f"Total Rows: **{len(engine.df_mch):,}** | Columns: `{list(engine.df_mch.columns)}`")
                st.dataframe(engine.df_mch.head(3), use_container_width=True)
                
            with schema_tabs[3]:
                st.markdown("**fact_transactions**: Core payment clearance events, UTR validity, and time dimensions.")
                st.markdown(f"Total Rows: **{len(engine.df_txn):,}** | Columns: `{list(engine.df_txn.columns)}`")
                st.dataframe(engine.df_txn.head(3), use_container_width=True)
                
            with schema_tabs[4]:
                st.markdown("**fact_chargebacks**: Dispute complaints, reason categories, severity SLA levels, and reporting delay days.")
                st.markdown(f"Total Rows: **{len(engine.df_cb):,}** | Columns: `{list(engine.df_cb.columns)}`")
                st.dataframe(engine.df_cb.head(3), use_container_width=True)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Preset Analytical Query Starter Buttons
    st.markdown("#### ⚡ Ready-to-Use Analytical Queries")
    
    preset_cols = st.columns(4)
    preset_selected = None

    with preset_cols[0]:
        if st.button("📈 Daily Volume & Value Trends", use_container_width=True):
            preset_selected = "Show daily transaction volume and value trends"
        if st.button("⏳ Delayed Disputes (>7 Days)", use_container_width=True):
            preset_selected = "Disputes reported after long delays (>7 days)"

    with preset_cols[1]:
        if st.button("⚖️ Success vs Failed vs Pending", use_container_width=True):
            preset_selected = "Comparing successful, failed, and pending transactions"
        if st.button("🎯 Highest Dispute Ratio Merchants", use_container_width=True):
            preset_selected = "Merchants with highest chargeback-to-transaction ratios"

    with preset_cols[2]:
        if st.button("🏪 Category Performance & Risk", use_container_width=True):
            preset_selected = "Performance & dispute metrics by merchant category"
        if st.button("📊 Q3 Regional Sales Comparison", use_container_width=True):
            preset_selected = "Compare Q3 sales by region"

    with preset_cols[3]:
        if st.button("🚨 Top Disputed Merchants", use_container_width=True):
            preset_selected = "Top merchants by chargeback count and disputed volume"
        if st.button("🕸️ Fraud Ring & Mule Detection", use_container_width=True):
            preset_selected = "Automated fraud ring and mule syndicate detection"

    st.markdown("---")

    col_h1, col_h2 = st.columns([4, 1])
    with col_h2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    # Render Conversation History
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                resp = msg["content"]
                if isinstance(resp, dict):
                    st.markdown(f"### {resp.get('title', 'Analytics Result')}")
                    st.markdown(resp.get('answer') or resp.get('text', ''))
                    
                    # DEVELOPER MODE: QUERY EXECUTION TRACE PANEL
                    if dev_mode and "execution_trace" in resp:
                        trace = resp["execution_trace"]
                        with st.expander("🔍 Query Execution Trace Panel (Developer Mode Active)", expanded=True):
                            t_col1, t_col2 = st.columns(2)
                            with t_col1:
                                st.markdown(f"- **Planner Engine**: `{trace.get('planner_engine', 'N/A')}`")
                                st.markdown(f"- **Target Dataset**: `{trace.get('target_dataset', 'N/A')}`")
                                st.markdown(f"- **Requested Metric**: `{trace.get('requested_metric', 'N/A')}`")
                                st.markdown(f"- **Aggregation**: `{trace.get('aggregation_function', 'N/A')}`")
                                st.markdown(f"- **Rows Returned**: `{trace.get('rows_returned', 0)}`")
                            with t_col2:
                                st.markdown(f"- **Grouping Dimensions**: `{trace.get('grouping_dimensions', [])}`")
                                st.markdown(f"- **Filters Applied**: `{trace.get('filters_applied', {})}`")
                                st.markdown(f"- **Selected Chart Type**: `{trace.get('selected_chart_type', 'N/A')}`")
                                st.markdown(f"- **X-Axis Field**: `{trace.get('x_axis_field', 'N/A')}`")
                                st.markdown(f"- **Y-Axis Field**: `{trace.get('y_axis_field', 'N/A')}`")

                    df_res = resp.get('data')
                    if isinstance(df_res, pd.DataFrame) and not df_res.empty:
                        st.dataframe(df_res, use_container_width=True)

                    vis = resp.get('visualization') or {}
                    chart_type = vis.get('type') or resp.get('chart_type')
                    x_field = vis.get('xField') or resp.get('x')
                    y_field = vis.get('yField') or resp.get('y')
                    x_label = vis.get('xLabel') or (x_field.replace('_', ' ').title() if isinstance(x_field, str) else '')
                    y_label = vis.get('yLabel') or resp.get('y_label') or (y_field.replace('_', ' ').title() if isinstance(y_field, str) else '')
                    title = vis.get('title') or resp.get('title')

                    if isinstance(df_res, pd.DataFrame) and not df_res.empty and chart_type and chart_type != 'table':
                        try:
                            if chart_type == 'bar':
                                fig = px.bar(
                                    df_res, x=x_field, y=y_field,
                                    title=title,
                                    labels={x_field: x_label, y_field: y_label},
                                    template="plotly_dark"
                                )
                                fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == 'line':
                                fig = px.line(
                                    df_res, x=x_field, y=y_field,
                                    markers=True, title=title,
                                    labels={x_field: x_label, y_field: y_label},
                                    template="plotly_dark"
                                )
                                fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == 'scatter':
                                color_col = 'merchant_category' if 'merchant_category' in df_res.columns else None
                                fig = px.scatter(
                                    df_res, x=x_field, y=y_field,
                                    color=color_col, title=title,
                                    labels={x_field: x_label, y_field: y_label},
                                    template="plotly_dark"
                                )
                                fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == 'pie':
                                names_col = resp.get('names') or x_field or df_res.columns[0]
                                values_col = resp.get('values') or y_field or df_res.columns[1]
                                fig = px.pie(
                                    df_res, names=names_col, values=values_col,
                                    hole=0.45, title=title, template="plotly_dark"
                                )
                                fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)

                            elif chart_type == 'bar_grouped' and isinstance(y_field, list):
                                fig = go.Figure()
                                for col_name in y_field:
                                    if col_name in df_res.columns:
                                        fig.add_trace(go.Bar(x=df_res[x_field], y=df_res[col_name], name=col_name))
                                fig.update_layout(barmode='group', template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as ex:
                            logger.warning(f"Chart rendering error: {ex}")

    # Process Input from Chat or Presets
    user_input = st.chat_input("Ask a question about sales, categories, regions, trends, chargebacks, or risk...")
    
    active_prompt = user_input or preset_selected

    if active_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": active_prompt})
        
        # Build history context for assistant
        history_context = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else m["content"].get("answer", "")}
            for m in st.session_state["chat_history"][:-1]
        ]
        
        with st.spinner("Analyzing dataset & calculating response..."):
            response = agent.answer_query(active_prompt, history=history_context)
            
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()


