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
