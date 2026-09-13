"""
Graph-First AI Agent & UPI Fraud Ring Detection Engine.
TransOrg AgentIQ Datathon.

Constructs heterogeneous payment graphs, detects collusive fraud syndicates,
mule merchant accounts, and provides an NLP query engine (AgentIQ Assistant).
"""

import re
import json
import logging
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class FraudGraphEngine:
    def __init__(self, data_dir: str = "data_cleaned"):
        self.data_dir = Path(data_dir)
        self.load_data()
        self.build_graph()

    def load_data(self):
        """Loads cleaned transaction, merchant, kyc, and chargeback data."""
        self.df_txn = pd.read_parquet(self.data_dir / "fact_transactions.parquet")
        self.df_cb = pd.read_parquet(self.data_dir / "fact_chargebacks.parquet")
        self.df_cust = pd.read_parquet(self.data_dir / "dim_customers.parquet")
        self.df_mch = pd.read_parquet(self.data_dir / "dim_merchants.parquet")
        self.df_unified = pd.read_parquet(self.data_dir / "fact_unified_analytics.parquet")

    def build_graph(self):
        """
        Constructs a heterogeneous graph:
        - Nodes: Customer (User), Merchant, Settlement Account
        - Edges:
            - User -> Merchant (TRANSACTED_WITH / DISPUTED)
            - Merchant -> Settlement Account (SETTLES_TO)
        """
        logger.info("Constructing heterogeneous payment network graph...")
        self.G = nx.Graph()
        
        # 1. Add Merchant Nodes
        for _, row in self.df_mch.iterrows():
            m_id = row['merchant_id']
            if pd.notna(m_id):
                self.G.add_node(
                    m_id,
                    node_type='merchant',
                    name=row.get('merchant_name', 'Unknown'),
                    category=row.get('merchant_category', 'Retail'),
                    status=row.get('merchant_status', 'ACTIVE'),
                    city=row.get('city', 'Unknown'),
                    state=row.get('state', 'Unknown')
                )
                
                # Add Settlement Account Node & Edge
                settle_acc = row.get('settlement_account')
                if pd.notna(settle_acc) and str(settle_acc).strip() not in ['', 'nan', 'None', 'NA']:
                    acc_node = f"ACC_{str(settle_acc).strip()}"
                    self.G.add_node(acc_node, node_type='settlement_account', account_id=settle_acc)
                    self.G.add_edge(m_id, acc_node, relation='SETTLES_TO')

        # 2. Add Customer Nodes
        for _, row in self.df_cust.iterrows():
            u_id = row['user_id']
            if pd.notna(u_id):
                self.G.add_node(
                    u_id,
                    node_type='customer',
                    name=row.get('full_name', 'Customer'),
                    kyc_status=row.get('kyc_status', 'UNREGISTERED'),
                    risk_segment=row.get('risk_segment', 'UNKNOWN'),
                    city=row.get('city', 'Unknown'),
                    state=row.get('state', 'Unknown')
                )

        # 3. Aggregate User-Merchant Transactions & Add Edges
        edge_agg = self.df_unified.groupby(['user_id', 'merchant_id']).agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            dispute_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum'),
            failed_count=('status', lambda s: (s == 'FAILED').sum())
        ).reset_index()

        for _, row in edge_agg.iterrows():
            u_id = row['user_id']
            m_id = row['merchant_id']
            if pd.notna(u_id) and pd.notna(m_id):
                # Ensure nodes exist
                if not self.G.has_node(u_id):
                    self.G.add_node(u_id, node_type='customer', name=f"User {u_id}", kyc_status='UNREGISTERED', risk_segment='UNKNOWN')
                if not self.G.has_node(m_id):
                    self.G.add_node(m_id, node_type='merchant', name=f"Merchant {m_id}", category='Retail', status='ACTIVE')
                
                self.G.add_edge(
                    u_id,
                    m_id,
                    relation='TRANSACTED_WITH',
                    txn_count=int(row['txn_count']),
                    total_amount=float(row['total_amount']),
                    dispute_count=int(row['dispute_count']),
                    disputed_amount=float(row['disputed_amount']),
                    failed_count=int(row['failed_count']),
                    has_dispute=bool(row['dispute_count'] > 0)
                )

        logger.info(f"Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

    # --- Fraud Ring Detection Algorithms ---

    def detect_shared_account_rings(self) -> List[Dict[str, Any]]:
        """
        Detects mule merchant rings where multiple distinct merchants share the same settlement account.
        """
        rings = []
        settle_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'settlement_account']
        
        for acc in settle_nodes:
            neighbors = [n for n in self.G.neighbors(acc) if self.G.nodes[n].get('node_type') == 'merchant']
            if len(neighbors) > 1:
                total_txns = 0
                total_disputes = 0
                disputed_amt = 0.0
                merchant_details = []
                
                for m in neighbors:
                    m_data = self.G.nodes[m]
                    m_txns = self.df_unified[self.df_unified['merchant_id'] == m]
                    m_cb_cnt = m_txns['is_disputed'].sum()
                    m_disp_amt = m_txns['disputed_amount'].sum()
                    
                    total_txns += len(m_txns)
                    total_disputes += m_cb_cnt
                    disputed_amt += m_disp_amt
                    
                    merchant_details.append({
                        "merchant_id": m,
                        "name": m_data.get('name'),
                        "category": m_data.get('category'),
                        "status": m_data.get('status'),
                        "txn_count": len(m_txns),
                        "dispute_count": int(m_cb_cnt)
                    })
                    
                rings.append({
                    "ring_type": "Mule / Shared Settlement Account Ring",
                    "account_identifier": acc.replace("ACC_", ""),
                    "merchant_count": len(neighbors),
                    "merchants": merchant_details,
                    "total_transactions": total_txns,
                    "total_disputes": total_disputes,
                    "disputed_amount": round(disputed_amt, 2),
                    "risk_level": "CRITICAL" if total_disputes > 0 else "HIGH"
                })
                
        return sorted(rings, key=lambda r: (r['total_disputes'], r['merchant_count']), reverse=True)

    def detect_collusive_fraud_clusters(self, min_disputes: int = 2) -> List[Dict[str, Any]]:
        """
        Detects connected bipartite subgraphs of users and merchants with dense dispute and failure activity.
        """
        dispute_edges = [
            (u, v, d) for u, v, d in self.G.edges(data=True)
            if d.get('dispute_count', 0) >= 1
        ]
        
        G_disp = nx.Graph()
        for u, v, d in dispute_edges:
            G_disp.add_node(u, **self.G.nodes[u])
            G_disp.add_node(v, **self.G.nodes[v])
            G_disp.add_edge(u, v, **d)
            
        components = list(nx.connected_components(G_disp))
        clusters = []
        
        for i, comp in enumerate(components):
            subg = G_disp.subgraph(comp)
            users = [n for n in comp if self.G.nodes[n].get('node_type') == 'customer']
            merchants = [n for n in comp if self.G.nodes[n].get('node_type') == 'merchant']
            
            tot_disputes = sum(d.get('dispute_count', 0) for _, _, d in subg.edges(data=True))
            tot_disp_amt = sum(d.get('disputed_amount', 0.0) for _, _, d in subg.edges(data=True))
            
            if tot_disputes >= min_disputes or len(users) >= 2 or len(merchants) >= 2:
                unverified_users = sum(1 for u in users if self.G.nodes[u].get('kyc_status') in ['REJECTED', 'PENDING', 'UNREGISTERED'])
                suspended_mch = sum(1 for m in merchants if self.G.nodes[m].get('status') in ['SUSPENDED', 'ON_HOLD', 'INACTIVE'])
                
                risk_score = min(100.0, (tot_disputes * 15.0) + (unverified_users * 10.0) + (suspended_mch * 15.0) + (len(comp) * 5.0))
                
                clusters.append({
                    "cluster_id": f"RING-{i+1:03d}",
                    "node_count": len(comp),
                    "user_count": len(users),
                    "merchant_count": len(merchants),
                    "users": users,
                    "merchants": merchants,
                    "total_disputes": tot_disputes,
                    "total_disputed_amount": round(tot_disp_amt, 2),
                    "unverified_user_count": unverified_users,
                    "suspended_merchant_count": suspended_mch,
                    "syndicate_risk_score": round(risk_score, 1),
                    "risk_tier": "CRITICAL" if risk_score >= 70 else "HIGH" if risk_score >= 40 else "MEDIUM"
                })
                
        return sorted(clusters, key=lambda c: c['syndicate_risk_score'], reverse=True)

    def extract_subgraph(self, center_node: str, hops: int = 2, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Extracts 1-hop or 2-hop ego network around a specific merchant or customer node.
        Returns node list, edge list, and layout attributes formatted for Plotly visualization.
        """
        center_node = center_node.strip().upper()
        if not self.G.has_node(center_node):
            digits = re.sub(r'\D', '', center_node)
            if 'USR' in center_node or len(digits) == 5:
                center_node = f"USR{int(digits):05d}"
            elif 'MCH' in center_node or len(digits) == 4:
                center_node = f"MCH{int(digits):04d}"
                
        if not self.G.has_node(center_node):
            return {"error": f"Node {center_node} not found in transaction network."}
            
        subg = nx.ego_graph(self.G, center_node, radius=hops)
        if subg.number_of_nodes() > max_nodes:
            nodes_by_degree = sorted(subg.nodes(), key=lambda n: subg.degree(n), reverse=True)[:max_nodes]
            if center_node not in nodes_by_degree:
                nodes_by_degree.append(center_node)
            subg = subg.subgraph(nodes_by_degree)
            
        pos = nx.spring_layout(subg, seed=42, k=0.5)
        
        nodes_data = []
        for node in subg.nodes():
            nd = self.G.nodes[node]
            ntype = nd.get('node_type', 'unknown')
            nodes_data.append({
                "id": node,
                "label": f"{nd.get('name', node)} ({node})",
                "node_type": ntype,
                "x": float(pos[node][0]),
                "y": float(pos[node][1]),
                "degree": subg.degree(node),
                "is_center": (node == center_node),
                "details": nd
            })
            
        edges_data = []
        for u, v, d in subg.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "relation": d.get('relation', 'CONNECTED_TO'),
                "dispute_count": d.get('dispute_count', 0),
                "disputed_amount": d.get('disputed_amount', 0.0),
                "total_amount": d.get('total_amount', 0.0),
                "txn_count": d.get('txn_count', 1),
                "is_disputed": bool(d.get('dispute_count', 0) > 0)
            })
            
        return {
            "center_node": center_node,
            "total_nodes": len(nodes_data),
            "total_edges": len(edges_data),
            "nodes": nodes_data,
            "edges": edges_data
        }


# --- AgentIQ Conversational Natural Language Assistant ---

class AgentIQAssistant:
    def __init__(self, analytics_engine, graph_engine: FraudGraphEngine):
        self.engine = analytics_engine
        self.graph = graph_engine
        
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Processes natural language queries and returns structured answers,
        markdown summaries, data tables, and chart metadata.
        """
        q = query.strip().lower()
        
        # Query 1: Daily transaction volume and value trends
        if any(w in q for w in [
            'daily transaction volume', 'volume trend', 'daily volume', 
            'transaction volume trend', 'daily trend', 'volume and value trends'
        ]):
            df = self.engine.get_daily_trends()
            total_vol = df['total_amount'].sum()
            avg_daily_vol = df['total_amount'].mean()
            peak_day = df.loc[df['total_amount'].idxmax()]
            
            return {
                "title": "Daily Transaction Volume & Value Trends",
                "text": f"Across the analyzed period, total transaction volume reached **₹{total_vol:,.2f}** across **{df['txn_count'].sum():,}** transactions, with an average daily volume of **₹{avg_daily_vol:,.2f}**. Peak transaction volume occurred on **{peak_day['date'].strftime('%Y-%m-%d')}** with **₹{peak_day['total_amount']:,.2f}** across **{peak_day['txn_count']:,}** transactions.",
                "data": df[['date', 'txn_count', 'total_amount', 'avg_amount', 'dispute_rate']],
                "chart_type": "line",
                "x": "date",
                "y": "total_amount",
                "y_label": "Transaction Volume (INR)"
            }

        # Query 2: Comparing successful, failed, and pending transactions
        elif any(w in q for w in [
            'compare successful vs failed', 'success vs failed', 'failed vs success', 
            'successful vs failed by day', 'failure trend by day', 'comparing successful, failed, and pending',
            'comparing successful', 'successful, failed, and pending transactions'
        ]):
            df = self.engine.get_daily_trends()
            tot_success = df['success_count'].sum()
            tot_failed = df['failed_count'].sum()
            tot_pending = df['pending_count'].sum()
            total = tot_success + tot_failed + tot_pending
            
            return {
                "title": "Daily Successful vs Failed vs Pending Transactions",
                "text": f"Out of **{total:,}** total transactions:\n- **Successful**: {tot_success:,} ({tot_success/total:.2%})\n- **Failed**: {tot_failed:,} ({tot_failed/total:.2%})\n- **Pending**: {tot_pending:,} ({tot_pending/total:.2%})",
                "data": df[['date', 'success_count', 'failed_count', 'pending_count']],
                "chart_type": "bar_grouped",
                "x": "date",
                "y": ["success_count", "failed_count", "pending_count"]
            }

        # Query 3: Performance & dispute metrics by merchant category
        elif any(w in q for w in [
            'by merchant category', 'category amount', 'transaction amount by category', 
            'merchant category performance', 'top merchant categories', 'performance & dispute metrics by merchant category',
            'metrics by merchant category'
        ]):
            df = self.engine.get_merchant_category_metrics()
            top_cat = df.iloc[0]
            
            return {
                "title": "Performance & Dispute Metrics by Merchant Category",
                "text": f"The top category by transaction volume is **{top_cat['merchant_category']}** generating **₹{top_cat['total_amount']:,.2f}** across **{top_cat['txn_count']:,}** transactions with a dispute rate of **{top_cat['dispute_rate']:.2%}**.",
                "data": df[['merchant_category', 'txn_count', 'total_amount', 'avg_ticket_size', 'chargeback_count', 'dispute_rate', 'failure_rate']],
                "chart_type": "bar",
                "x": "merchant_category",
                "y": "total_amount",
                "y_label": "Total Amount (INR)"
            }

        # Query 4: Top merchants by chargeback count and disputed volume
        elif any(w in q for w in [
            'highest chargeback count', 'top merchants by chargeback count', 'merchant with highest chargeback', 
            'merchants by chargeback count', 'top merchants by chargeback count and disputed volume',
            'chargeback count and disputed volume'
        ]):
            df = self.engine.get_top_merchants_by_chargebacks(top_n=10)
            top_mch = df.iloc[0]
            
            return {
                "title": "Top Merchants by Chargeback Count & Disputed Volume",
                "text": f"The merchant with the highest chargeback count is **{top_mch['merchant_name']} ({top_mch['merchant_id']})** with **{top_mch['chargeback_count']}** chargebacks totaling **₹{top_mch['disputed_amount']:,.2f}** (Dispute ratio: **{top_mch['chargeback_ratio']:.2%}**).",
                "data": df[['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status', 'txn_count', 'chargeback_count', 'disputed_amount', 'chargeback_ratio']],
                "chart_type": "bar",
                "x": "merchant_name",
                "y": "chargeback_count",
                "y_label": "Chargeback Count"
            }

        # Query 5: Merchant category with highest disputed amount
        elif any(w in q for w in [
            'highest disputed amount', 'category has the highest disputed amount', 
            'disputed amount by category', 'dispute rate by merchant category'
        ]):
            df = self.engine.get_merchant_category_metrics().sort_values('disputed_amount', ascending=False)
            top_disp = df.iloc[0]
            
            return {
                "title": "Disputed Amount by Merchant Category",
                "text": f"The category with the highest disputed volume is **{top_disp['merchant_category']}** with **₹{top_disp['disputed_amount']:,.2f}** in chargebacks ({top_disp['chargeback_count']:,} complaints, {top_disp['dispute_rate']:.2%} dispute rate).",
                "data": df[['merchant_category', 'disputed_amount', 'chargeback_count', 'dispute_rate', 'dispute_volume_share']],
                "chart_type": "bar",
                "x": "merchant_category",
                "y": "disputed_amount",
                "y_label": "Disputed Amount (INR)"
            }

        # Query 6: Chargeback reason code and severity SLA distributions
        elif any(w in q for w in [
            'chargeback reason', 'reason distribution', 'dispute reason distribution', 
            'chargeback reason distribution', 'chargeback reason code and severity',
            'severity sla distributions', 'chargeback reason code and severity sla distributions'
        ]):
            df = self.engine.get_chargeback_reasons()
            top_r = df.iloc[0]
            
            return {
                "title": "Chargeback Reason Code Distribution",
                "text": f"The leading reason for chargebacks is **{top_r['reason_category']}** representing **{top_r['share_of_disputes']:.2%}** of all disputes (**{top_r['complaint_count']:,}** cases totaling **₹{top_r['total_disputed_amount']:,.2f}**).",
                "data": df[['reason_category', 'complaint_count', 'total_disputed_amount', 'avg_disputed_amount', 'share_of_disputes']],
                "chart_type": "pie",
                "names": "reason_category",
                "values": "complaint_count"
            }

        # Query 7: High-risk repeat-dispute customers
        elif any(w in q for w in [
            'top 10 users', 'users by disputed amount', 'top users by disputed amount', 
            'high-risk users', 'users with repeated disputes', 'high-risk repeat-dispute customers',
            'repeat-dispute customers'
        ]):
            df = self.engine.get_high_risk_users(top_n=10)
            top_u = df.iloc[0]
            
            return {
                "title": "High-Risk Repeat-Dispute Customers",
                "text": f"The highest disputed user is **{top_u['full_name']} ({top_u['user_id']})** with **{top_u['dispute_count']}** disputes totaling **₹{top_u['total_disputed_amount']:,.2f}** (KYC Status: **{top_u['kyc_status']}**, Risk: **{top_u['risk_segment']}**).",
                "data": df[['user_id', 'full_name', 'kyc_status', 'risk_segment', 'city', 'dispute_count', 'total_disputed_amount', 'avg_delay']],
                "chart_type": "bar",
                "x": "full_name",
                "y": "total_disputed_amount",
                "y_label": "Disputed Amount (INR)"
            }

        # Query 8: Average transaction value (ATV) trends over time
        elif any(w in q for w in [
            'average transaction value', 'atv trend', 'avg transaction value trend', 
            'atv over time', 'average transaction value (atv) trends over time', 'atv trends over time'
        ]):
            df = self.engine.get_daily_trends()
            overall_atv = self.engine.df_txn['amount'].mean()
            
            return {
                "title": "Average Transaction Value (ATV) Trend Over Time",
                "text": f"The overall Average Transaction Value (ATV) across all settled payments is **₹{overall_atv:,.2f}**. Daily ATV remains stable across observed billing cycles.",
                "data": df[['date', 'avg_amount', 'txn_count', 'total_amount']],
                "chart_type": "line",
                "x": "date",
                "y": "avg_amount",
                "y_label": "Average Ticket Size (INR)"
            }

        # Query 9: Transaction volume breakdown by customer KYC status
        elif any(w in q for w in [
            'kyc status has the highest', 'kyc status transaction amount', 'transaction amount by kyc', 
            'kyc status distribution', 'transaction volume breakdown by customer kyc status', 'volume breakdown by customer kyc status'
        ]):
            df = self.engine.get_kyc_status_breakdown()
            top_kyc = df.iloc[0]
            
            return {
                "title": "Transaction Volume Breakdown by Customer KYC Status",
                "text": f"The KYC tier with the highest transaction volume is **{top_kyc['kyc_status']}** accounting for **₹{top_kyc['total_amount']:,.2f}** ({top_kyc['txn_count']:,} txns, {top_kyc['dispute_rate']:.2%} dispute rate).",
                "data": df[['kyc_status', 'txn_count', 'total_amount', 'avg_amount', 'chargeback_count', 'dispute_rate']],
                "chart_type": "bar",
                "x": "kyc_status",
                "y": "total_amount",
                "y_label": "Total Amount (INR)"
            }

        # Query 10: Compare chargebacks by severity level
        elif any(w in q for w in [
            'severity level', 'chargebacks by severity', 'dispute severity', 'severity distribution'
        ]):
            df = self.engine.get_chargeback_severity()
            
            return {
                "title": "Chargebacks by Severity SLA Level",
                "text": "Disputes broken down by SLA priority tier:\n" + "\n".join([f"- **{r['severity']}**: {r['complaint_count']:,} cases ({r['share_of_complaints']:.2%}), Total: ₹{r['total_disputed_amount']:,.2f}" for _, r in df.iterrows()]),
                "data": df[['severity', 'complaint_count', 'total_disputed_amount', 'avg_disputed_amount', 'share_of_complaints']],
                "chart_type": "bar",
                "x": "severity",
                "y": "complaint_count",
                "y_label": "Number of Chargebacks"
            }

        # Query 11: Disputes reported after long delays (>7 days)
        elif any(w in q for w in [
            'after 7 days', 'disputes reported after', 'reporting delay', 'delayed disputes', 
            'disputes reported after long delays', 'disputes reported after long delays (>7 days)',
            'long delays'
        ]):
            df_cb = self.engine.df_cb
            delayed = df_cb[df_cb['reporting_delay_days'] > 7]
            tot_cb = len(df_cb)
            
            return {
                "title": "Disputes Reported After Long Delays (>7 Days)",
                "text": f"A total of **{len(delayed):,}** chargebacks (**{len(delayed)/tot_cb:.2%}** of all disputes) were reported **more than 7 days** after the transaction occurred. The average delay in this cohort is **{delayed['reporting_delay_days'].mean():.1f} days**, highly indicative of delayed fraud realization, credential stuffing, or Account Takeover (ATO).",
                "data": delayed[['complaint_id', 'txn_id', 'user_id', 'merchant_id', 'disputed_amount', 'reason_category', 'severity', 'reporting_delay_days']].head(15),
                "chart_type": "table"
            }

        # Query 12: Merchants with highest chargeback-to-transaction ratios
        elif any(w in q for w in [
            'highest chargeback-to-transaction ratio', 'highest chargeback-to-transaction ratios',
            'highest chargeback ratio', 'dispute ratio merchant', 'high-risk merchants',
            'merchants with highest chargeback-to-transaction ratios'
        ]):
            df = self.engine.get_high_risk_merchants(min_txns=3, top_n=10)
            top_ratio_mch = df.iloc[0]
            
            return {
                "title": "Merchants with Highest Chargeback-to-Transaction Ratio",
                "text": f"The merchant with the highest chargeback ratio is **{top_ratio_mch['merchant_name']} ({top_ratio_mch['merchant_id']})** with a **{top_ratio_mch['chargeback_ratio']:.2%}** dispute rate (**{top_ratio_mch['chargeback_count']}** disputes across **{top_ratio_mch['txn_count']}** txns). Risk Score: **{top_ratio_mch['risk_score']} / 100**.",
                "data": df[['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status', 'txn_count', 'chargeback_count', 'chargeback_ratio', 'risk_score']],
                "chart_type": "bar",
                "x": "merchant_name",
                "y": "chargeback_ratio",
                "y_label": "Chargeback Ratio"
            }

        # Query 13: Automated fraud ring and mule syndicate detection
        elif any(w in q for w in [
            'fraud ring', 'mule', 'syndicate', 'shared account', 'clusters', 
            'detect mule accounts', 'automated fraud ring', 'automated fraud ring and mule syndicate detection',
            'mule syndicate detection'
        ]):
            rings = self.graph.detect_shared_account_rings()
            clusters = self.graph.detect_collusive_fraud_clusters()
            
            cluster_df = pd.DataFrame(clusters)
            top_cluster = clusters[0] if clusters else None
            
            summary_txt = f"Detected **{len(rings)} Mule / Shared Settlement Account Rings** and **{len(clusters)} Collusive Fraud Syndicates** in the UPI transaction graph."
            if top_cluster:
                summary_txt += f"\n\n**Top Syndicate**: {top_cluster['cluster_id']} contains **{top_cluster['user_count']} users** and **{top_cluster['merchant_count']} merchants** responsible for **{top_cluster['total_disputes']} disputes** (Disputed: **₹{top_cluster['total_disputed_amount']:,.2f}**, Risk Score: **{top_cluster['syndicate_risk_score']}**)."
                
            return {
                "title": "Automated Fraud Ring & Mule Syndicate Detection",
                "text": summary_txt,
                "data": cluster_df[['cluster_id', 'node_count', 'user_count', 'merchant_count', 'total_disputes', 'total_disputed_amount', 'syndicate_risk_score', 'risk_tier']].head(10) if not cluster_df.empty else pd.DataFrame(),
                "chart_type": "table"
            }

        # Query 14: Ticket size anomalies / Spike detection
        elif any(w in q for w in ['spike', 'ticket size anomaly', 'anomalies', 'ticket divergence']):
            anomalies = self.engine.get_merchant_ticket_anomalies(ratio_threshold=2.5, top_n=10)
            if not anomalies.empty:
                top_anom = anomalies.iloc[0]
                return {
                    "title": "Merchant Ticket Size Divergence Anomalies",
                    "text": f"Identified **{len(anomalies)}** merchants whose actual average transaction amount exceeds their declared onboarding ticket size by >250%. Top divergence: **{top_anom['merchant_name']} ({top_anom['merchant_id']})** with actual avg ₹{top_anom['actual_avg_ticket']:,.2f} vs declared ₹{top_anom['declared_avg_ticket']:,.2f} ({top_anom['ticket_divergence_ratio']:.1f}x divergence).",
                    "data": anomalies[['merchant_id', 'merchant_name', 'merchant_category', 'declared_avg_ticket', 'actual_avg_ticket', 'ticket_divergence_ratio', 'chargeback_count']],
                    "chart_type": "bar",
                    "x": "merchant_name",
                    "y": "ticket_divergence_ratio",
                    "y_label": "Ticket Divergence Multiple (x)"
                }
            else:
                return {
                    "title": "Merchant Ticket Size Anomalies",
                    "text": "No extreme ticket size divergence anomalies detected above the 2.5x threshold.",
                    "data": pd.DataFrame(),
                    "chart_type": "table"
                }

        # Fallback / General Search
        else:
            kpis = self.engine.get_summary_kpis()
            return {
                "title": f"FinTech Analytics Overview for '{query}'",
                "text": f"Found **{kpis['total_transaction_count']:,}** total transactions valued at **₹{kpis['total_transaction_amount']:,.2f}**. Success rate is **{kpis['success_transaction_rate']:.2%}**, Failed rate is **{kpis['failed_transaction_rate']:.2%}**, with **{kpis['chargeback_count']:,}** total chargebacks (Overall dispute rate: **{kpis['chargeback_to_transaction_ratio']:.2%}**).\n\n*Tip: Click any of the ready-to-use preset buttons above for instant deep-dive analytics!*",
                "data": pd.DataFrame([kpis]),
                "chart_type": "table"
            }
