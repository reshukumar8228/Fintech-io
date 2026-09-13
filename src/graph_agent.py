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
