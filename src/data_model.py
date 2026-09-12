"""
Data Model and Star-Schema Construction for FinTech Analytics.
TransOrg AgentIQ Datathon.
"""

import sqlite3
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def build_relational_model(
    upi_clean: pd.DataFrame,
    kyc_clean: pd.DataFrame,
    mch_clean: pd.DataFrame,
    cb_clean: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds the relational Star Schema data model:
    - Dim Customer
    - Dim Merchant
    - Fact Transaction
    - Fact Chargeback
    - Fact Unified Analytics (Denormalized Analytics Mart)
    """
    logger.info("Building relational data model...")
    
    # 1. Dim Customers
    dim_customers = kyc_clean.copy()
    
    # 2. Dim Merchants
    dim_merchants = mch_clean.copy()
    
    # 3. Fact Transactions
    fact_transactions = upi_clean.copy()
    fact_transactions['date'] = fact_transactions['timestamp'].dt.date.astype(str)
    fact_transactions['year_month'] = fact_transactions['timestamp'].dt.to_period('M').astype(str)
    fact_transactions['hour'] = fact_transactions['timestamp'].dt.hour
    fact_transactions['day_name'] = fact_transactions['timestamp'].dt.day_name()
    
    # 4. Fact Chargebacks
    fact_chargebacks = cb_clean.copy()
    
    # 5. Consolidated Unified Analytics Mart (Star Schema Join)
    logger.info("Generating denormalized Unified Analytics Mart...")
    
    # Merge transactions with dispute details
    cb_subset = fact_chargebacks[[
        'complaint_id', 'txn_id', 'disputed_amount', 'reason_category', 
        'severity', 'resolution_status', 'channel', 'reporting_delay_days', 'reported_timestamp'
    ]].drop_duplicates(subset=['txn_id'])
    
    unified = pd.merge(
        fact_transactions,
        cb_subset,
        on='txn_id',
        how='left'
    )
    
    unified['is_disputed'] = unified['complaint_id'].notna()
    unified['disputed_amount'] = unified['disputed_amount'].fillna(0.0)
    
    # Merge with Merchant Attributes
    mch_subset = dim_merchants[[
        'merchant_id', 'merchant_name', 'merchant_category', 'business_type', 
        'city', 'state', 'merchant_status', 'declared_avg_ticket_size'
    ]].rename(columns={'city': 'merchant_city', 'state': 'merchant_state'})
    
    unified = pd.merge(
        unified,
        mch_subset,
        on='merchant_id',
        how='left'
    )
    unified['merchant_category'] = unified['merchant_category'].fillna('Miscellaneous Retail')
    unified['merchant_status'] = unified['merchant_status'].fillna('UNKNOWN')
    
    # Merge with Customer Attributes
    kyc_subset = dim_customers[[
        'user_id', 'full_name', 'city', 'state', 'monthly_income', 
        'occupation', 'kyc_status', 'risk_segment', 'is_valid_pan', 'is_valid_aadhaar'
    ]].rename(columns={'city': 'customer_city', 'state': 'customer_state'})
    
    unified = pd.merge(
        unified,
        kyc_subset,
        on='user_id',
        how='left'
    )
    unified['kyc_status'] = unified['kyc_status'].fillna('UNREGISTERED')
    unified['risk_segment'] = unified['risk_segment'].fillna('UNKNOWN')
    
    logger.info(f"Unified data mart generated with {len(unified)} records.")
    return dim_customers, dim_merchants, fact_transactions, fact_chargebacks, unified

