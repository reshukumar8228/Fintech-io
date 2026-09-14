"""
Business and Risk Analytics Engine for Track 1: FinTech & BFSI - UPI Fraud Ring & Merchant Analytics.
TransOrg AgentIQ Datathon.

This module computes all business metrics, risk scores, anomaly metrics, and merchant/customer 360 profiles.
Supports dynamic multidimensional slicing (Date Range, Categories, KYC status, Transaction status, Risk segments).
# Filter validation and zero-division resilience optimizations applied.
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

    def get_merchant_category_metrics(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Computes performance, volume, dispute rate, and failure rate by merchant category."""
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame(columns=['merchant_category', 'txn_count', 'total_amount', 'avg_ticket_size', 'chargeback_count', 'disputed_amount', 'failed_count', 'dispute_rate', 'dispute_volume_share', 'failure_rate'])
            
        cat_summary = data.groupby('merchant_category').agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_ticket_size=('amount', 'mean'),
            chargeback_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum'),
            failed_count=('status', lambda s: (s == 'FAILED').sum())
        ).reset_index()
        
        cat_summary['dispute_rate'] = np.where(cat_summary['txn_count'] > 0, cat_summary['chargeback_count'] / cat_summary['txn_count'], 0.0)
        cat_summary['dispute_volume_share'] = np.where(cat_summary['total_amount'] > 0, cat_summary['disputed_amount'] / cat_summary['total_amount'], 0.0)
        cat_summary['failure_rate'] = np.where(cat_summary['txn_count'] > 0, cat_summary['failed_count'] / cat_summary['txn_count'], 0.0)
        return cat_summary.sort_values('total_amount', ascending=False)

    def get_top_merchants_by_chargebacks(self, df: Optional[pd.DataFrame] = None, top_n: int = 15) -> pd.DataFrame:
        """Returns top merchants by chargeback count, disputed amount, and dispute ratio."""
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame(columns=['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status', 'txn_count', 'total_amount', 'chargeback_count', 'disputed_amount', 'chargeback_ratio'])
            
        mch_perf = data.groupby(['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status']).agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            chargeback_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum')
        ).reset_index()
        
        mch_perf['chargeback_ratio'] = np.where(mch_perf['txn_count'] > 0, mch_perf['chargeback_count'] / mch_perf['txn_count'], 0.0)
        return mch_perf.sort_values('chargeback_count', ascending=False).head(top_n)

    def get_high_risk_merchants(self, df: Optional[pd.DataFrame] = None, min_txns: int = 3, top_n: int = 20) -> pd.DataFrame:
        """
        Identifies high-risk merchants using a calibrated multi-factor risk model:
        - Factor 1: Chargeback ratio (0-40 pts)
        - Factor 2: Total Disputed Volume (0-30 pts)
        - Factor 3: Declared Ticket Size Discrepancy (0-15 pts)
        - Factor 4: Suspended / Inactive merchant status (0-15 pts)
        """
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame(columns=['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status', 'txn_count', 'total_amount', 'chargeback_count', 'disputed_amount', 'chargeback_ratio', 'risk_score'])
            
        mch_perf = data.groupby(['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status']).agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            chargeback_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum'),
            declared_avg_ticket=('declared_avg_ticket_size', 'first')
        ).reset_index()
        
        mch_perf['chargeback_ratio'] = np.where(mch_perf['txn_count'] > 0, mch_perf['chargeback_count'] / mch_perf['txn_count'], 0.0)
        mch_perf['actual_avg_ticket'] = np.where(mch_perf['txn_count'] > 0, mch_perf['total_amount'] / mch_perf['txn_count'], 0.0)
        
        max_cb_amt = mch_perf['disputed_amount'].max() or 1.0
        
        def calc_risk(row):
            score = min(row['chargeback_ratio'] * 200, 40.0)
            score += min((row['disputed_amount'] / max_cb_amt) * 30, 30.0)
            if pd.notna(row['declared_avg_ticket']) and row['declared_avg_ticket'] > 0:
                ticket_ratio = row['actual_avg_ticket'] / row['declared_avg_ticket']
                if ticket_ratio > 2.0 or ticket_ratio < 0.5:
                    score += 15.0
            if row['merchant_status'] in ['SUSPENDED', 'ON_HOLD', 'INACTIVE']:
                score += 15.0
            return round(min(score, 100.0), 1)
            
        mch_perf['risk_score'] = mch_perf.apply(calc_risk, axis=1)
        filtered = mch_perf[mch_perf['txn_count'] >= min_txns]
        return filtered.sort_values('risk_score', ascending=False).head(top_n)

    def get_merchant_ticket_anomalies(self, df: Optional[pd.DataFrame] = None, ratio_threshold: float = 2.5, top_n: int = 15) -> pd.DataFrame:
        """
        Detects merchants where actual transaction average exceeds declared ticket size by >250%.
        This is a classic indicator of account compromise, unauthorized billing, or high-risk category evasion.
        """
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame()
            
        mch_stats = data.groupby(['merchant_id', 'merchant_name', 'merchant_category', 'merchant_status']).agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            declared_avg_ticket=('declared_avg_ticket_size', 'first'),
            chargeback_count=('is_disputed', 'sum')
        ).reset_index()
        
        mch_stats = mch_stats[mch_stats['declared_avg_ticket'].notna() & (mch_stats['declared_avg_ticket'] > 0) & (mch_stats['txn_count'] >= 3)]
        mch_stats['actual_avg_ticket'] = mch_stats['total_amount'] / mch_stats['txn_count']
        mch_stats['ticket_divergence_ratio'] = mch_stats['actual_avg_ticket'] / mch_stats['declared_avg_ticket']
        
        anomalies = mch_stats[mch_stats['ticket_divergence_ratio'] >= ratio_threshold].copy()
        return anomalies.sort_values('ticket_divergence_ratio', ascending=False).head(top_n)

    # --- Customer & KYC Analytics ---

    def get_kyc_status_breakdown(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Returns KYC status volume and dispute rates."""
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame(columns=['kyc_status', 'txn_count', 'total_amount', 'avg_amount', 'chargeback_count', 'disputed_amount', 'failed_count', 'dispute_rate', 'failure_rate'])
            
        kyc_df = data.groupby('kyc_status').agg(
            txn_count=('txn_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_amount=('amount', 'mean'),
            chargeback_count=('is_disputed', 'sum'),
            disputed_amount=('disputed_amount', 'sum'),
            failed_count=('status', lambda s: (s == 'FAILED').sum())
        ).reset_index()
        
        kyc_df['dispute_rate'] = np.where(kyc_df['txn_count'] > 0, kyc_df['chargeback_count'] / kyc_df['txn_count'], 0.0)
        kyc_df['failure_rate'] = np.where(kyc_df['txn_count'] > 0, kyc_df['failed_count'] / kyc_df['txn_count'], 0.0)
        return kyc_df.sort_values('total_amount', ascending=False)

    def get_high_risk_users(self, df: Optional[pd.DataFrame] = None, top_n: int = 20) -> pd.DataFrame:
        """Identifies customers with repeat chargebacks and high dispute values."""
        data = self.df_unified if df is None else df
        if data.empty:
            return pd.DataFrame()
            
        disputed = data[data['is_disputed']]
        if disputed.empty:
            return pd.DataFrame()
            
        user_cb = disputed.groupby(['user_id']).agg(
            dispute_count=('complaint_id', 'count'),
            total_disputed_amount=('disputed_amount', 'sum'),
            avg_delay=('reporting_delay_days', 'mean'),
            severity_critical=('severity', lambda s: (s == 'CRITICAL').sum())
        ).reset_index()
        
        # Merge with KYC info
        user_cb = pd.merge(user_cb, self.df_cust, on='user_id', how='left')
        user_cb['kyc_status'] = user_cb['kyc_status'].fillna('UNREGISTERED')
        user_cb['risk_segment'] = user_cb['risk_segment'].fillna('UNKNOWN')
        user_cb['full_name'] = user_cb['full_name'].fillna(user_cb['user_id'])
        
        return user_cb.sort_values(['dispute_count', 'total_disputed_amount'], ascending=False).head(top_n)

    # --- Dispute & Chargeback Analytics ---

    def get_chargeback_reasons(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Returns dispute distribution by canonical reason."""
        data = self.df_unified if df is None else df
        disputed = data[data['is_disputed'] & data['reason_category'].notna()]
        if disputed.empty:
            return pd.DataFrame(columns=['reason_category', 'complaint_count', 'total_disputed_amount', 'avg_disputed_amount', 'avg_delay_days', 'share_of_disputes'])
            
        reasons = disputed.groupby('reason_category').agg(
            complaint_count=('complaint_id', 'count'),
            total_disputed_amount=('disputed_amount', 'sum'),
            avg_disputed_amount=('disputed_amount', 'mean'),
            avg_delay_days=('reporting_delay_days', 'mean')
        ).reset_index()
        total_disp = len(disputed)
        reasons['share_of_disputes'] = np.where(total_disp > 0, reasons['complaint_count'] / total_disp, 0.0)
        return reasons.sort_values('complaint_count', ascending=False)

    def get_chargeback_severity(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Returns dispute breakdown by severity level."""
        data = self.df_unified if df is None else df
        disputed = data[data['is_disputed'] & data['severity'].notna()]
        if disputed.empty:
            return pd.DataFrame(columns=['severity', 'complaint_count', 'total_disputed_amount', 'avg_disputed_amount', 'share_of_complaints'])
            
        sev = disputed.groupby('severity').agg(
            complaint_count=('complaint_id', 'count'),
            total_disputed_amount=('disputed_amount', 'sum'),
            avg_disputed_amount=('disputed_amount', 'mean')
        ).reset_index()
        total_disp = len(disputed)
        sev['share_of_complaints'] = np.where(total_disp > 0, sev['complaint_count'] / total_disp, 0.0)
        
        sev_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        sev['order'] = sev['severity'].map(sev_order).fillna(4)
        return sev.sort_values('order').drop(columns=['order'])

    def get_utr_health_metrics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyzes missing and invalid UTR correlation with failed and disputed transactions."""
        data = self.df_unified if df is None else df
        total = len(data)
        if total == 0:
            return {
                "total_transactions": 0,
                "valid_utr_count": 0,
                "invalid_or_missing_utr_count": 0,
                "valid_utr_failure_rate": 0.0,
                "invalid_utr_failure_rate": 0.0,
                "valid_utr_dispute_rate": 0.0,
                "invalid_utr_dispute_rate": 0.0,
            }
            
        invalid_utr = data[~data['is_valid_utr'] | data['utr'].isna()]
        valid_utr = data[data['is_valid_utr'] & data['utr'].notna()]
        
        return {
            "total_transactions": total,
            "valid_utr_count": len(valid_utr),
            "invalid_or_missing_utr_count": len(invalid_utr),
            "valid_utr_failure_rate": round(float((valid_utr['status'] == 'FAILED').mean()), 4) if len(valid_utr) > 0 else 0.0,
            "invalid_utr_failure_rate": round(float((invalid_utr['status'] == 'FAILED').mean()), 4) if len(invalid_utr) > 0 else 0.0,
            "valid_utr_dispute_rate": round(float(valid_utr['is_disputed'].mean()), 4) if len(valid_utr) > 0 else 0.0,
            "invalid_utr_dispute_rate": round(float(invalid_utr['is_disputed'].mean()), 4) if len(invalid_utr) > 0 else 0.0,
        }

    def detect_merchant_transaction_spikes(self, z_thresh: float = 2.5) -> pd.DataFrame:
        """
        Detects merchants experiencing sudden transaction count or volume spikes followed by disputes.
        """
        mch_daily = self.df_unified.groupby(['merchant_id', 'merchant_name', 'date']).agg(
            daily_txns=('txn_id', 'count'),
            daily_vol=('amount', 'sum'),
            daily_disputes=('is_disputed', 'sum')
        ).reset_index()
        
        results = []
        for mch_id, group in mch_daily.groupby('merchant_id'):
            if len(group) >= 3:
                mean_txns = group['daily_txns'].mean()
                std_txns = group['daily_txns'].std()
                if std_txns and std_txns > 0:
                    group = group.copy()
                    group['z_score'] = (group['daily_txns'] - mean_txns) / std_txns
                    spikes = group[group['z_score'] >= z_thresh]
                    if not spikes.empty:
                        results.append(spikes)
                        
        if results:
            spike_df = pd.concat(results, ignore_index=True)
            return spike_df.sort_values('z_score', ascending=False)
        return pd.DataFrame(columns=['merchant_id', 'merchant_name', 'date', 'daily_txns', 'daily_vol', 'daily_disputes', 'z_score'])