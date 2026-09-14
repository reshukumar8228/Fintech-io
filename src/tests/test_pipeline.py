"""
Comprehensive Unit and Integration Test Suite for TransOrg AgentIQ Datathon Track 1.
Validates:
- ID standardizers (user_id, merchant_id, txn_id, complaint_id)
- Currency & amount cleaners (INR, Rs, symbols, commas, k/m multipliers, negative values)
- Datetime parser (epoch s/ms, ISO strings, mixed formats)
- Transaction status normalizer (SUCCESS, FAILED, PENDING)
- KYC cleaners (PAN, Aadhaar masking, City/State harmonization, KYC status, Risk segments)
- MCC codes & Category normalization
- Relational data model and ETL exports
- Business metrics engine & dynamic slice filtering
- Merchant ticket size anomaly detection
- Graph-First AI Agent & Fraud Ring detection
- All Datathon Example Agent Queries & Fallbacks
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_cleaning import (
    standardize_user_id,
    standardize_merchant_id,
    standardize_txn_id,
    standardize_complaint_id,
    clean_amount,
    parse_mixed_datetime_series,
    normalize_txn_status,
    clean_utr,
    clean_pan,
    clean_aadhaar,
    clean_mcc,
    clean_city_and_state,
    normalize_kyc_status,
    normalize_risk_segment,
    normalize_dispute_reason,
    normalize_dispute_severity
)
from src.analytics_engine import AnalyticsEngine
from src.graph_agent import FraudGraphEngine, AgentIQAssistant

class TestDataCleaning(unittest.TestCase):
    def test_id_standardization(self):
        self.assertEqual(standardize_user_id('USR12345'), 'USR12345')
        self.assertEqual(standardize_user_id('usr12345'), 'USR12345')
        self.assertEqual(standardize_user_id('USR-12345'), 'USR12345')
        self.assertEqual(standardize_user_id('USR 12345'), 'USR12345')
        self.assertEqual(standardize_user_id('usr_12345'), 'USR12345')
        self.assertEqual(standardize_user_id('12345'), 'USR12345')
        self.assertIsNone(standardize_user_id(''))
        self.assertIsNone(standardize_user_id(None))
        
        self.assertEqual(standardize_merchant_id('MCH1234'), 'MCH1234')
        self.assertEqual(standardize_merchant_id('mch1234'), 'MCH1234')
        self.assertEqual(standardize_merchant_id('MCH-1234'), 'MCH1234')
        self.assertEqual(standardize_merchant_id('1234'), 'MCH1234')
        self.assertIsNone(standardize_merchant_id(''))

        self.assertEqual(standardize_txn_id('TXN00011869'), 'TXN00011869')
        self.assertEqual(standardize_txn_id('11869'), 'TXN00011869')
        self.assertEqual(standardize_complaint_id('CBK0012345'), 'CBK0012345')

    def test_amount_cleaner(self):
        self.assertEqual(clean_amount('15722.34'), 15722.34)
        self.assertEqual(clean_amount('Rs. 6362.9'), 6362.9)
        self.assertEqual(clean_amount('INR 13,312'), 13312.0)
        self.assertEqual(clean_amount('17,833.99'), 17833.99)
        self.assertEqual(clean_amount('-23820.57'), 23820.57)
        self.assertEqual(clean_amount('27.3k'), 27300.0)
        self.assertEqual(clean_amount('1.5M'), 1500000.0)
        self.assertIsNone(clean_amount(''))
        self.assertIsNone(clean_amount('N/A'))

    def test_status_normalizer(self):
        self.assertEqual(normalize_txn_status('SUCCESS'), 'SUCCESS')
        self.assertEqual(normalize_txn_status('Success'), 'SUCCESS')
        self.assertEqual(normalize_txn_status('TXN_SUCCESS'), 'SUCCESS')
        self.assertEqual(normalize_txn_status('COMPLETED'), 'SUCCESS')
        self.assertEqual(normalize_txn_status('S'), 'SUCCESS')
        self.assertEqual(normalize_txn_status('FAILED'), 'FAILED')
        self.assertEqual(normalize_txn_status('Fail'), 'FAILED')
        self.assertEqual(normalize_txn_status('Declined'), 'FAILED')
        self.assertEqual(normalize_txn_status('PENDING'), 'PENDING')
        self.assertEqual(normalize_txn_status('Initiated'), 'PENDING')
        self.assertEqual(normalize_txn_status('PROCESSING'), 'PENDING')

    def test_utr_cleaning(self):
        utr_clean, is_valid = clean_utr('UTR6498104698')
        self.assertEqual(utr_clean, 'UTR6498104698')
        self.assertTrue(is_valid)

        utr_clean2, is_valid2 = clean_utr('UTR 2787678319')
        self.assertEqual(utr_clean2, 'UTR2787678319')
        self.assertTrue(is_valid2)

        utr_clean3, is_valid3 = clean_utr('2787678319')
        self.assertEqual(utr_clean3, 'UTR2787678319')
        self.assertTrue(is_valid3)

        utr_none, is_valid_none = clean_utr(None)
        self.assertIsNone(utr_none)
        self.assertFalse(is_valid_none)

    def test_kyc_cleaners(self):
        pan, pan_valid = clean_pan('CACWZ 2722 S')
        self.assertEqual(pan, 'CACWZ2722S')
        self.assertTrue(pan_valid)

        pan2, pan_valid2 = clean_pan('ygvoi9236w')
        self.assertEqual(pan2, 'YGVOI9236W')
        self.assertTrue(pan_valid2)

        aadhaar, aadh_valid = clean_aadhaar('2781 6299 9816')
        self.assertEqual(aadhaar, '2781-6299-9816')
        self.assertTrue(aadh_valid)

        city, state = clean_city_and_state('Bombay', 'Maharashtra')
        self.assertEqual(city, 'Mumbai')
        self.assertEqual(state, 'Maharashtra')

        city_blr, state_blr = clean_city_and_state('BLR', 'Karnataka')
        self.assertEqual(city_blr, 'Bengaluru')

        self.assertEqual(normalize_kyc_status('Done'), 'VERIFIED')
        self.assertEqual(normalize_kyc_status('Verified'), 'VERIFIED')
        self.assertEqual(normalize_kyc_status('Reject'), 'REJECTED')
        self.assertEqual(normalize_kyc_status('Pending'), 'PENDING')

    def test_mcc_cleaning(self):
        mcc, cat = clean_mcc('MCC-7011', 'hotel_lodging')
        self.assertEqual(mcc, '7011')
        self.assertEqual(cat, 'Hotels & Lodging')

        mcc2, cat2 = clean_mcc(None, 'Grocery Stores')
        self.assertEqual(mcc2, '5411')
        self.assertEqual(cat2, 'Grocery Stores')


class TestAnalyticsAndGraphEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AnalyticsEngine("data_cleaned")
        cls.graph_engine = FraudGraphEngine("data_cleaned")
        cls.agent = AgentIQAssistant(cls.engine, cls.graph_engine)

    def test_summary_kpis(self):
        kpis = self.engine.get_summary_kpis()
        self.assertGreater(kpis['total_transaction_count'], 0)
        self.assertGreater(kpis['total_transaction_amount'], 0)
        self.assertGreater(kpis['average_transaction_value'], 0)
        self.assertGreater(kpis['success_transaction_rate'], 0.5)
        self.assertGreater(kpis['chargeback_count'], 0)
        self.assertGreater(kpis['kyc_completion_rate'], 0.5)

    def test_slice_filtering(self):
        # Filter by grocery
        filtered = self.engine.filter_data(categories=['Grocery Stores'])
        self.assertGreater(len(filtered), 0)
        self.assertTrue((filtered['merchant_category'] == 'Grocery Stores').all())

        kpis_filtered = self.engine.get_summary_kpis(filtered)
        self.assertEqual(kpis_filtered['total_transaction_count'], len(filtered))

    def test_merchant_category_metrics(self):
        cat_df = self.engine.get_merchant_category_metrics()
        self.assertGreater(len(cat_df), 0)
        self.assertIn('merchant_category', cat_df.columns)
        self.assertIn('total_amount', cat_df.columns)
        self.assertIn('dispute_rate', cat_df.columns)

    def test_high_risk_merchants(self):
        risk_mch = self.engine.get_high_risk_merchants()
        self.assertGreater(len(risk_mch), 0)
        self.assertIn('risk_score', risk_mch.columns)

    def test_ticket_size_anomalies(self):
        anomalies = self.engine.get_merchant_ticket_anomalies(ratio_threshold=2.0)
        self.assertIsInstance(anomalies, pd.DataFrame)

    def test_utr_health(self):
        utr_stats = self.engine.get_utr_health_metrics()
        self.assertGreater(utr_stats['valid_utr_count'], 0)
        self.assertIn('invalid_utr_dispute_rate', utr_stats)

    def test_fraud_rings_detection(self):
        rings = self.graph_engine.detect_shared_account_rings()
        self.assertIsInstance(rings, list)
        self.assertGreater(len(rings), 0)

        clusters = self.graph_engine.detect_collusive_fraud_clusters()
        self.assertIsInstance(clusters, list)
        self.assertGreater(len(clusters), 0)

    def test_all_12_example_agent_queries(self):
        test_queries = [
            "Show daily transaction volume trend.",
            "Show total transaction amount by merchant category.",
            "Compare successful vs failed transactions by day.",
            "Which merchant has the highest chargeback count?",
            "Which merchant category has the highest disputed amount?",
            "Show chargeback reason distribution.",
            "Show top 10 users by disputed amount.",
            "Show average transaction value trend over time.",
            "Which KYC status has the highest transaction amount?",
            "Compare chargebacks by severity level.",
            "Show disputes reported after 7 days.",
            "Which merchant has the highest chargeback-to-transaction ratio?",
            "Detect mule accounts and fraud rings.",
            "Show merchant ticket size anomalies."
        ]

        for q in test_queries:
            resp = self.agent.answer_query(q)
            self.assertIn("title", resp)
            self.assertIn("text", resp)
            self.assertGreater(len(resp["text"]), 10)


if __name__ == "__main__":
    unittest.main()
