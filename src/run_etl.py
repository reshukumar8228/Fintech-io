"""
Main ETL Pipeline Runner for TransOrg AgentIQ Datathon (Track 1).
Executes end-to-end data extraction, cleaning, relational modeling, and artifact persistence.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import (
    clean_upi_transactions,
    clean_kyc_records,
    clean_merchants_master,
    clean_chargebacks
)
from src.data_model import build_relational_model, export_data_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ETL_Pipeline")

def run_etl(raw_data_dir: str = "track1_fintech_dataset_files", output_dir: str = "data_cleaned"):
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STARTING TRACK 1 FINTECH & BFSI DATA PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load Raw Datasets
    logger.info("Step 1: Loading raw datasets...")
    raw_upi = pd.read_csv(raw_path / "track1_upi_transactions.csv")
    raw_kyc = pd.read_csv(raw_path / "track1_kyc_records.csv")
    raw_mch = pd.read_csv(raw_path / "track1_merchants_master.csv")
    with open(raw_path / "track1_chargebacks.json", "r", encoding="utf-8") as f:
        raw_cb = json.load(f)
        
    logger.info(f"Loaded Raw Shapes: UPI={raw_upi.shape}, KYC={raw_kyc.shape}, Merchants={raw_mch.shape}, Chargebacks={len(raw_cb)}")
    
    # 2. Clean Datasets
    logger.info("Step 2: Cleaning and normalizing datasets...")
    clean_upi = clean_upi_transactions(raw_upi)
    clean_kyc = clean_kyc_records(raw_kyc)
    clean_mch = clean_merchants_master(raw_mch)
    clean_cb = clean_chargebacks(raw_cb, upi_df=clean_upi)
    
    logger.info(f"Cleaned Shapes: UPI={clean_upi.shape}, KYC={clean_kyc.shape}, Merchants={clean_mch.shape}, Chargebacks={clean_cb.shape}")
    
    # 3. Build Star Schema & Consolidated Mart
    logger.info("Step 3: Constructing Star Schema and Unified Analytics Mart...")
    dim_cust, dim_mch, fact_txn, fact_cb, unified = build_relational_model(
        clean_upi, clean_kyc, clean_mch, clean_cb
    )
    
    # 4. Export Artifacts
    logger.info("Step 4: Persisting Parquet, CSV, and SQLite datasets...")
    export_data_model(dim_cust, dim_mch, fact_txn, fact_cb, unified, output_dir=str(out_path))
    
    # 5. Data Health & Summary Statistics
    logger.info("=" * 60)
    logger.info("ETL PIPELINE VALIDATION & HEALTH REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Transactions: {len(fact_txn):,}")
    logger.info(f"Total Transaction Volume: INR {fact_txn['amount'].sum():,.2f}")
    logger.info(f"Success Rate: {(fact_txn['status'] == 'SUCCESS').mean():.2%}")
    logger.info(f"Failed Rate: {(fact_txn['status'] == 'FAILED').mean():.2%}")
    logger.info(f"Pending Rate: {(fact_txn['status'] == 'PENDING').mean():.2%}")
    logger.info(f"Total Chargebacks: {len(fact_cb):,}")
    logger.info(f"Total Disputed Volume: INR {fact_cb['disputed_amount'].sum():,.2f}")
    logger.info(f"Overall Dispute Rate: {len(fact_cb) / len(fact_txn):.2%}")
    logger.info(f"Valid KYC Customers: {len(dim_cust):,}")
    logger.info(f"Valid Merchants: {len(dim_mch):,}")
    logger.info("=" * 60)
    logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_etl()
