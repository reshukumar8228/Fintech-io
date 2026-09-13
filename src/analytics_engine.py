"""
Business and Risk Analytics Engine for Track 1: FinTech & BFSI - UPI Fraud Ring & Merchant Analytics.
TransOrg AgentIQ Datathon.

This module computes all business metrics, risk scores, anomaly metrics, and merchant/customer 360 profiles.
Supports dynamic multidimensional slicing (Date Range, Categories, KYC status, Transaction status, Risk segments).
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    def __init__(self, data_dir: str = "data_cleaned"):
        self.data_dir = Path(data_dir)
        self.load_data()
        
    def load_data(self):
        """Loads cleaned data files."""
        self.df_txn = pd.read_parquet(self.data_dir / "fact_transactions.parquet")
        self.df_cb = pd.read_parquet(self.data_dir / "fact_chargebacks.parquet")
        self.df_cust = pd.read_parquet(self.data_dir / "dim_customers.parquet")
        self.df_mch = pd.read_parquet(self.data_dir / "dim_merchants.parquet")
        self.df_unified = pd.read_parquet(self.data_dir / "fact_unified_analytics.parquet")
        
        # Ensure timestamp types
        self.df_txn['timestamp'] = pd.to_datetime(self.df_txn['timestamp'])
        self.df_cb['reported_timestamp'] = pd.to_datetime(self.df_cb['reported_timestamp'])
        self.df_unified['timestamp'] = pd.to_datetime(self.df_unified['timestamp'])
        if 'date' not in self.df_unified.columns:
            self.df_unified['date'] = self.df_unified['timestamp'].dt.date.astype(str)

    def filter_data(
        self,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        categories: Optional[List[str]] = None,
        kyc_statuses: Optional[List[str]] = None,
        txn_statuses: Optional[List[str]] = None,
        risk_segments: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Dynamically slices fact_unified_analytics by date range, merchant category,
        KYC status, transaction status, and customer risk segment.
        """
        df = self.df_unified.copy()
        
        if start_date is not None:
            s_ts = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= s_ts]
            
        if end_date is not None:
            e_ts = pd.to_datetime(end_date)
            # Include full end-of-day for dates/timestamps with 00:00:00
            if hasattr(end_date, 'time'):
                if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
                    e_ts = e_ts + pd.Timedelta(days=1, microseconds=-1)
            else:
                e_ts = e_ts + pd.Timedelta(days=1, microseconds=-1)
            df = df[df['timestamp'] <= e_ts]
            
        if categories and len(categories) > 0:
            df = df[df['merchant_category'].isin(categories)]
            
        if kyc_statuses and len(kyc_statuses) > 0:
            df = df[df['kyc_status'].isin(kyc_statuses)]
            
        if txn_statuses and len(txn_statuses) > 0:
            df = df[df['status'].isin(txn_statuses)]
            
        if risk_segments and len(risk_segments) > 0:
            df = df[df['risk_segment'].isin(risk_segments)]
            
        return df

    # --- Core Business Metrics ---

    def get_summary_kpis(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculates core executive business KPIs from unified mart or slice."""
        data = self.df_unified if df is None else df
        total_txns = len(data)
        
        if total_txns == 0:
            return {
                "total_transaction_count": 0,
                "total_transaction_amount": 0.0,
                "average_transaction_value": 0.0,
                "success_transaction_rate": 0.0,
                "failed_transaction_rate": 0.0,
                "pending_transaction_rate": 0.0,
                "chargeback_count": 0,
                "chargeback_amount": 0.0,
                "chargeback_to_transaction_ratio": 0.0,
                "chargeback_to_volume_ratio": 0.0,
                "kyc_completion_rate": 0.0,
                "kyc_rejection_rate": 0.0,
                "total_kyc_users": 0,
                "average_dispute_reporting_delay_days": 0.0,
                "disputes_over_7_days_count": 0,
                "disputes_over_7_days_pct": 0.0
            }
            
        total_vol = float(data['amount'].sum())
        atv = float(data['amount'].mean()) if total_txns > 0 else 0.0
        
        status_counts = data['status'].value_counts()
        success_cnt = int(status_counts.get('SUCCESS', 0))
        failed_cnt = int(status_counts.get('FAILED', 0))
        pending_cnt = int(status_counts.get('PENDING', 0))
        
        failed_rate = (failed_cnt / total_txns) if total_txns > 0 else 0.0
        pending_rate = (pending_cnt / total_txns) if total_txns > 0 else 0.0
        success_rate = (success_cnt / total_txns) if total_txns > 0 else 0.0
        
        # Chargebacks in this slice
        disputed_subset = data[data['is_disputed']]
        cb_count = len(disputed_subset)
        cb_amount = float(disputed_subset['disputed_amount'].sum())
        cb_ratio = (cb_count / total_txns) if total_txns > 0 else 0.0
        cb_vol_ratio = (cb_amount / total_vol) if total_vol > 0 else 0.0
        
        # KYC Metrics from slice customers
        user_ids = data['user_id'].dropna().unique()
        matched_cust = self.df_cust[self.df_cust['user_id'].isin(user_ids)]
        kyc_total = len(matched_cust)
        kyc_status_counts = matched_cust['kyc_status'].value_counts()
        kyc_completed = int(kyc_status_counts.get('VERIFIED', 0))
        kyc_rejected = int(kyc_status_counts.get('REJECTED', 0))
        
        kyc_comp_rate = (kyc_completed / kyc_total) if kyc_total > 0 else 0.0
        kyc_rej_rate = (kyc_rejected / kyc_total) if kyc_total > 0 else 0.0
        
        # Dispute reporting delay
        delays = disputed_subset['reporting_delay_days'].dropna()
        avg_delay = float(delays.mean()) if len(delays) > 0 else 0.0
        delay_gt_7d = int((delays > 7).sum())
        delay_gt_7d_pct = (delay_gt_7d / cb_count) if cb_count > 0 else 0.0

        return {
            "total_transaction_count": total_txns,
            "total_transaction_amount": round(total_vol, 2),
            "average_transaction_value": round(atv, 2),
            "success_transaction_rate": round(success_rate, 4),
            "failed_transaction_rate": round(failed_rate, 4),
            "pending_transaction_rate": round(pending_rate, 4),
            "chargeback_count": cb_count,
            "chargeback_amount": round(cb_amount, 2),
            "chargeback_to_transaction_ratio": round(cb_ratio, 4),
            "chargeback_to_volume_ratio": round(cb_vol_ratio, 4),
            "kyc_completion_rate": round(kyc_comp_rate, 4),
            "kyc_rejection_rate": round(kyc_rej_rate, 4),
            "total_kyc_users": kyc_total,
            "average_dispute_reporting_delay_days": round(avg_delay, 2),
            "disputes_over_7_days_count": delay_gt_7d,
            "disputes_over_7_days_pct": round(delay_gt_7d_pct, 4)
        }

    # --- Trend Analytics ---

    def get_daily_trends(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Returns daily transaction volume, count, ATV, and success/fail/pending/dispute splits."""
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame(columns=['date', 'txn_count', 'total_amount', 'avg_amount', 'success_count', 'failed_count', 'pending_count', 'chargeback_count', 'disputed_amount', 'dispute_rate'])
            
        data = data.copy()
        if 'date' not in data.columns:
            data['date'] = data['timestamp'].dt.date.astype(str)
            
        daily = data.groupby('date').agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_amount=('amount', 'mean'),
            success_count=('status', lambda s: (s == 'SUCCESS').sum()),
            failed_count=('status', lambda s: (s == 'FAILED').sum()),
            pending_count=('status', lambda s: (s == 'PENDING').sum()),
            chargeback_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum')
        ).reset_index()
        
        daily['dispute_rate'] = np.where(daily['txn_count'] > 0, daily['chargeback_count'] / daily['txn_count'], 0.0)
        daily['date'] = pd.to_datetime(daily['date'])
        return daily.sort_values('date')

    def get_hourly_failure_trend(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Returns failure rates and volume by hour of day (0-23)."""
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame(columns=['hour', 'total_txns', 'failed_txns', 'success_txns', 'pending_txns', 'avg_amount', 'failure_rate'])
            
        data = data.copy()
        if 'hour' not in data.columns:
            data['hour'] = data['timestamp'].dt.hour
            
        hourly = data.groupby('hour').agg(
            total_txns=('txn_id', 'count'),
            failed_txns=('status', lambda s: (s == 'FAILED').sum()),
            success_txns=('status', lambda s: (s == 'SUCCESS').sum()),
            pending_txns=('status', lambda s: (s == 'PENDING').sum()),
            avg_amount=('amount', 'mean')
        ).reset_index()
        
        hourly['failure_rate'] = np.where(hourly['total_txns'] > 0, hourly['failed_txns'] / hourly['total_txns'], 0.0)
        return hourly.sort_values('hour')

    # --- Merchant & Category Performance ---
