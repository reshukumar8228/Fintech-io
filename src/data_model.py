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

def export_data_model(
    dim_customers: pd.DataFrame,
    dim_merchants: pd.DataFrame,
    fact_transactions: pd.DataFrame,
    fact_chargebacks: pd.DataFrame,
    unified: pd.DataFrame,
    output_dir: str = "data_cleaned",
    db_name: str = "fintech_analytics.db"
):
    """
    Exports cleaned datasets to Parquet, CSV, and SQLite with indexes.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    tables = {
        'dim_customers': dim_customers,
        'dim_merchants': dim_merchants,
        'fact_transactions': fact_transactions,
        'fact_chargebacks': fact_chargebacks,
        'fact_unified_analytics': unified
    }
    
    for name, df in tables.items():
        csv_file = out_path / f"{name}.csv"
        parquet_file = out_path / f"{name}.parquet"
        df.to_csv(csv_file, index=False)
        
        df_pq = df.copy()
        for col in df_pq.select_dtypes(include=['period[M]']).columns:
            df_pq[col] = df_pq[col].astype(str)
        df_pq.to_parquet(parquet_file, index=False)
        logger.info(f"Exported {name} -> {csv_file.name} & {parquet_file.name} ({len(df)} rows)")
        
    # Export to SQLite DB
    db_path = out_path / db_name
    conn = sqlite3.connect(db_path)
    try:
        for name, df in tables.items():
            df_sql = df.copy()
            for col in df_sql.select_dtypes(include=['datetime64[ns]', 'period[M]']).columns:
                df_sql[col] = df_sql[col].astype(str)
            df_sql.to_sql(name, conn, if_exists='replace', index=False)
            
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_id ON fact_transactions(txn_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_user ON fact_transactions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_mch ON fact_transactions(merchant_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cb_txn ON fact_chargebacks(txn_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cb_user ON fact_chargebacks(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cb_mch ON fact_chargebacks(merchant_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_user ON dim_customers(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mch_mch ON dim_merchants(merchant_id);")
        conn.commit()
        logger.info(f"Exported SQLite database to {db_path} with indexes.")
    finally:
        conn.close()
