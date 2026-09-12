"""
Data Cleaning and Standardization Engine for Track 1: FinTech & BFSI - UPI Fraud Ring & Merchant Analytics.
TransOrg AgentIQ Datathon.
"""

import re
import json
import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical City to State & Standard Name Mapping
CITY_MAPPING = {
    'BOMBAY': ('Mumbai', 'Maharashtra'),
    'MUMBAI': ('Mumbai', 'Maharashtra'),
    'DELHI': ('Delhi', 'Delhi'),
    'DILLI': ('Delhi', 'Delhi'),
    'NEW DELHI': ('Delhi', 'Delhi'),
    'BLR': ('Bengaluru', 'Karnataka'),
    'BANGALORE': ('Bengaluru', 'Karnataka'),
    'BENGALURU': ('Bengaluru', 'Karnataka'),
    'KOLKATA': ('Kolkata', 'West Bengal'),
    'CALCUTTA': ('Kolkata', 'West Bengal'),
    'CHENNAI': ('Chennai', 'Tamil Nadu'),
    'MADRAS': ('Chennai', 'Tamil Nadu'),
    'HYDERABAD': ('Hyderabad', 'Telangana'),
    'HYD': ('Hyderabad', 'Telangana'),
    'JAIPUR': ('Jaipur', 'Rajasthan'),
    'JPR': ('Jaipur', 'Rajasthan'),
    'LUCKNOW': ('Lucknow', 'Uttar Pradesh'),
    'LKO': ('Lucknow', 'Uttar Pradesh'),
    'AMRITSAR': ('Amritsar', 'Punjab'),
    'ASR': ('Amritsar', 'Punjab'),
    'JALANDHAR': ('Jalandhar', 'Punjab'),
    'JALANDAR': ('Jalandhar', 'Punjab'),
    'LUDHIANA': ('Ludhiana', 'Punjab'),
    'PUNE': ('Pune', 'Maharashtra'),
}

# MCC to Canonical Category Mapping
MCC_CATEGORY_MAP = {
    '5411': 'Grocery Stores',
    '4131': 'Transportation',
    '5812': 'Restaurants & Dining',
    '5912': 'Pharmacies & Healthcare',
    '7011': 'Hotels & Lodging',
    '5699': 'Apparel & Clothing',
    '5311': 'Department Stores',
    '5999': 'Miscellaneous Retail',
    '4814': 'Telecommunication Services',
    '5942': 'Books & Stationery'
}

CATEGORY_TO_MCC_MAP = {
    'grocery': '5411', 'groceries': '5411', 'grocery store': '5411', 'grocery stores': '5411', 'kirana': '5411', 'supermarket': '5411',
    'restaurant': '5812', 'restaurants': '5812', 'eating place': '5812', 'food': '5812', 'food services': '5812', 'dining': '5812',
    'medical': '5912', 'medical store': '5912', 'pharmacy': '5912', 'pharmacies': '5912', 'chemist': '5912', 'healthcare': '5912',
    'hotel': '7011', 'hotels': '7011', 'hotel lodging': '7011', 'hotel_lodging': '7011', 'hospitality': '7011',
    'transport': '4131', 'transportation': '4131', 'bus/taxi': '4131', 'travel': '4131', 'transprt': '4131',
    'apparel': '5699', 'clothing': '5699', 'cloths': '5699', 'garments': '5699', 'fashion': '5699',
    'department store': '5311', 'department stores': '5311', 'dept_store': '5311', 'dept store': '5311',
    'telecom': '4814', 'mobile recharge': '4814', 'phone service': '4814',
    'book store': '5942', 'books': '5942', 'books stationery': '5942', 'books_stationery': '5942', 'stationery': '5942',
    'retail': '5999', 'retail other': '5999', 'misc retail': '5999', 'miscellaneous': '5999', 'other': '5999'
}

# --- Core ID Normalizers ---

def standardize_user_id(val: Any) -> Optional[str]:
    """Standardize user_id variations to canonical USR{5-digits}."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'null', 'na']:
        return None
    digits = re.sub(r'\D', '', s)
    if digits:
        return f"USR{int(digits):05d}"
    return None

def standardize_merchant_id(val: Any) -> Optional[str]:
    """Standardize merchant_id variations to canonical MCH{4-digits}."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'null', 'na']:
        return None
    digits = re.sub(r'\D', '', s)
    if digits:
        return f"MCH{int(digits):04d}"
    return None

def standardize_txn_id(val: Any) -> Optional[str]:
    """Standardize transaction_id variations to canonical TXN{8-digits}."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'null', 'na']:
        return None
    digits = re.sub(r'\D', '', s)
    if digits:
        return f"TXN{int(digits):08d}"
    return None

def standardize_complaint_id(val: Any) -> Optional[str]:
    """Standardize complaint_id variations to CBK{7-digits}."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'null', 'na']:
        return None
    digits = re.sub(r'\D', '', s)
    if digits:
        return f"CBK{int(digits):07d}"
    return None

# --- Currency and Amount Cleaners ---

def clean_amount(val: Any) -> Optional[float]:
    """Clean and normalize transaction/disputed amounts."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'null', 'na', '']:
        return None
    
    s = re.sub(r'(?i)\b(inr|rs)\b', '', s)
    s = re.sub(r'[\u20b9$Rs,]', '', s, flags=re.IGNORECASE)
    s = s.strip().strip('.').strip()
    
    if not s:
        return None
        
    multiplier = 1.0
    if s.lower().endswith('k'):
        multiplier = 1000.0
        s = s[:-1].strip()
    elif s.lower().endswith('m'):
        multiplier = 1000000.0
        s = s[:-1].strip()
        
    try:
        amt = float(s) * multiplier
        return round(abs(amt), 2)
    except Exception:
        return None

# --- Robust Datetime Parser ---

def parse_mixed_datetime_series(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp series containing epoch and string dates."""
    s = series.astype(str).str.strip()
    s = s.replace(['', 'nan', 'None', 'NaT', 'null', 'NULL', 'None'], np.nan)
    
    is_num = s.str.match(r'^-?\d+(\.\d+)?$').fillna(False)
    result = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    
    if is_num.any():
        num_vals = pd.to_numeric(s[is_num], errors='coerce')
        sec_mask = (num_vals > -2e9) & (num_vals < 2e9)
        ms_mask = (num_vals >= 2e9) | (num_vals <= -2e9)
        
        result.loc[is_num & sec_mask] = pd.to_datetime(num_vals[sec_mask], unit='s', errors='coerce')
        result.loc[is_num & ms_mask] = pd.to_datetime(num_vals[ms_mask], unit='ms', errors='coerce')
        
    str_mask = ~is_num & s.notna()
    if str_mask.any():
        result.loc[str_mask] = pd.to_datetime(s[str_mask], format='mixed', errors='coerce')
        
    return result

# --- Transaction Status Normalizer ---

def normalize_txn_status(val: Any) -> str:
    """Normalize transaction status to canonical: SUCCESS, FAILED, PENDING."""
    if pd.isna(val):
        return 'UNKNOWN'
    s = str(val).strip().upper()
    
    success_synonyms = {'SUCCESS', 'TXN_SUCCESS', 'COMPLETED', 'S', 'SUCCESSFUL', 'PAID', 'SETTLED'}
    failed_synonyms = {'FAILED', 'TXN_FAILED', 'FAIL', 'DECLINED', 'F', 'FAILURE', 'REJECTED', 'CANCELLED'}
    pending_synonyms = {'PENDING', 'PROCESSING', 'INITIATED', 'IN_PROGRESS', 'P', 'QUEUED'}
    
    if s in success_synonyms:
        return 'SUCCESS'
    elif s in failed_synonyms:
        return 'FAILED'
    elif s in pending_synonyms:
        return 'PENDING'
    return 'UNKNOWN'

# --- UTR Cleaners & Validators ---

def clean_utr(val: Any) -> Tuple[Optional[str], bool]:
    """Clean UTR number, standardizing spaces and hyphens."""
    if pd.isna(val):
        return (None, False)
    s = str(val).strip().upper()
    if not s or s in ['NAN', 'NONE', 'NULL', 'NA', 'UNKNOWN', '-']:
        return (None, False)
    
    cleaned = re.sub(r'[\s\-_]', '', s)
    if re.match(r'^UTR\d{10,12}$', cleaned):
        return (cleaned, True)
    elif re.match(r'^\d{10,12}$', cleaned):
        return (f"UTR{cleaned}", True)
    else:
        return (cleaned, False)

# --- KYC Fields Cleaners ---

def clean_pan(val: Any) -> Tuple[Optional[str], bool]:
    """Clean and validate Indian PAN numbers."""
    if pd.isna(val):
        return (None, False)
    s = str(val).strip().upper()
    s = re.sub(r'[\s\-_]', '', s)
    if not s or s in ['NAN', 'NONE', 'NULL', 'NA']:
        return (None, False)
    
    if re.match(r'^[A-Z]{5}\d{4}[A-Z]$', s):
        return (s, True)
    return (s, False)

def clean_aadhaar(val: Any) -> Tuple[Optional[str], bool]:
    """Clean and validate Aadhaar numbers."""
    if pd.isna(val):
        return (None, False)
    s = str(val).strip().upper()
    s = re.sub(r'[\s]', '', s)
    if not s or s in ['NAN', 'NONE', 'NULL', 'NA']:
        return (None, False)
    
    if re.match(r'^X{4}-?X{4}-?\d{4}$', s):
        digits = s[-4:]
        return (f"XXXX-XXXX-{digits}", True)
    
    digits_only = re.sub(r'\D', '', s)
    if len(digits_only) == 12:
        return (f"{digits_only[:4]}-{digits_only[4:8]}-{digits_only[8:]}", True)
    elif len(digits_only) == 13:
        return (f"{digits_only[:4]}-{digits_only[4:8]}-{digits_only[8:12]}", False)
    return (s, False)

def clean_city_and_state(city_val: Any, state_val: Any) -> Tuple[str, str]:
    """Standardize City and State names."""
    c_str = str(city_val).strip().upper() if pd.notna(city_val) else ''
    s_str = str(state_val).strip() if pd.notna(state_val) else ''
    
    if c_str in CITY_MAPPING:
        std_city, std_state = CITY_MAPPING[c_str]
        return std_city, std_state
    
    cleaned_city = str(city_val).strip().title() if pd.notna(city_val) else 'Unknown'
    cleaned_state = str(state_val).strip().title() if pd.notna(state_val) else 'Unknown'
    return cleaned_city, cleaned_state

def normalize_kyc_status(val: Any) -> str:
    """Normalize KYC status to VERIFIED, PENDING, REJECTED."""
    if pd.isna(val):
        return 'UNKNOWN'
    s = str(val).strip().upper()
    if s in {'VERIFIED', 'DONE', 'APPROVED', 'V', 'KYC_DONE', 'SUCCESS'}:
        return 'VERIFIED'
    elif s in {'PENDING', 'IN_PROGRESS', 'UNDER REVIEW', 'P', 'INITIATED'}:
        return 'PENDING'
    elif s in {'REJECTED', 'REJECT', 'FAILED', 'R', 'DECLINED'}:
        return 'REJECTED'
    return 'UNKNOWN'

def normalize_risk_segment(val: Any) -> str:
    """Normalize user risk segment to LOW, MEDIUM, HIGH, UNKNOWN."""
    if pd.isna(val):
        return 'UNKNOWN'
    s = str(val).strip().upper()
    if s in {'LOW', 'MEDIUM', 'HIGH'}:
        return s
    return 'UNKNOWN'

# --- Merchant Master Cleaners ---

def clean_mcc(mcc_val: Any, category_val: Any) -> Tuple[str, str]:
    """Normalize MCC code to 4 digits and map to canonical category."""
    mcc_clean = None
    if pd.notna(mcc_val):
        mcc_s = str(mcc_val).strip()
        mcc_digits = re.sub(r'\D', '', mcc_s)
        if '.' in mcc_s:
            try:
                mcc_digits = str(int(float(mcc_s)))
            except:
                pass
        if len(mcc_digits) == 4:
            mcc_clean = mcc_digits
        elif len(mcc_digits) == 5 and mcc_digits.startswith('0'):
            mcc_clean = mcc_digits[1:]
    
    cat_clean = None
    if pd.notna(category_val):
        cat_norm = str(category_val).strip().lower().replace('_', ' ').replace('-', ' ')
        cat_norm = re.sub(r'\s+', ' ', cat_norm)
        if cat_norm in CATEGORY_TO_MCC_MAP:
            inferred_mcc = CATEGORY_TO_MCC_MAP[cat_norm]
            if not mcc_clean:
                mcc_clean = inferred_mcc
            cat_clean = MCC_CATEGORY_MAP.get(inferred_mcc, cat_norm.title())
        else:
            cat_clean = cat_norm.title()
            
    if mcc_clean and mcc_clean in MCC_CATEGORY_MAP:
        cat_clean = MCC_CATEGORY_MAP[mcc_clean]
    elif not mcc_clean:
        mcc_clean = '5999'
        cat_clean = cat_clean or 'Miscellaneous Retail'
        
    return mcc_clean, cat_clean or 'Miscellaneous Retail'

def normalize_business_type(val: Any) -> str:
    """Normalize business legal structure."""
    if pd.isna(val):
        return 'Individual'
    s = str(val).strip().upper().replace('-', ' ').replace('_', ' ')
    if 'PVT' in s or 'PRIVATE' in s:
        return 'Private Limited'
    elif 'SOLE' in s or 'PROPRIETOR' in s:
        return 'Sole Proprietorship'
    elif 'PARTNER' in s:
        return 'Partnership'
    elif 'LLC' in s or 'LIMITED' in s:
        return 'LLC / Corporate'
    elif 'INDIVIDUAL' in s:
        return 'Individual'
    return 'Individual'

def normalize_merchant_status(val: Any) -> str:
    """Normalize merchant status."""
    if pd.isna(val):
        return 'UNKNOWN'
    s = str(val).strip().upper()
    if s in {'ACTIVE', 'LIVE', 'ENABLED', 'A'}:
        return 'ACTIVE'
    elif s in {'INACTIVE', 'I', 'DISABLED', 'CLOSED'}:
        return 'INACTIVE'
    elif s in {'SUSPENDED', 'BLOCKED', 'S'}:
        return 'SUSPENDED'
    elif s in {'HOLD', 'ON_HOLD', 'PENDING'}:
        return 'ON_HOLD'
    return 'UNKNOWN'

# --- Chargebacks JSON Cleaners ---

def normalize_dispute_reason(val: Any) -> str:
    """Standardize chargeback reason codes."""
    if pd.isna(val):
        return 'Other Dispute'
    s = str(val).strip().lower()
    
    fraud_keywords = ['login compromised', 'unauthorised', 'unauthorized', 'fraud', 'ato', 
                      'takeover', 'hacked', 'scam', 'unauth', 'not done by me', 'suspicious']
    delivery_keywords = ['not delivered', 'no service', 'service issue', 'service failed', 
                         'not provided', 'delivery issue', 'item not received']
    tech_keywords = ['extra amount', 'double debit', 'charged twice', 'wrong amount', 
                     'dup_debit', 'incorrect amount', 'duplicate debit', 'amount mismatch']
    
    for kw in fraud_keywords:
        if kw in s:
            return 'Unauthorized / Fraud / ATO'
    for kw in delivery_keywords:
        if kw in s:
            return 'Goods / Services Not Delivered'
    for kw in tech_keywords:
        if kw in s:
            return 'Duplicate / Technical Debit'
    return 'Customer Dispute'

def normalize_dispute_severity(val: Any) -> str:
    """Normalize dispute severity to CRITICAL, HIGH, MEDIUM, LOW."""
    if pd.isna(val):
        return 'MEDIUM'
    s = str(val).strip().upper()
    if s in {'CRITICAL', 'CRIT', 'P1'}:
        return 'CRITICAL'
    elif s in {'HIGH', 'H', 'P2'}:
        return 'HIGH'
    elif s in {'MEDIUM', 'M', 'P3'}:
        return 'MEDIUM'
    elif s in {'LOW', 'L', 'P4'}:
        return 'LOW'
    return 'MEDIUM'

def normalize_resolution_status(val: Any) -> str:
    """Normalize dispute resolution status."""
    if pd.isna(val):
        return 'OPEN'
    s = str(val).strip().upper()
    if s in {'CLOSED'}:
        return 'CLOSED'
    elif s in {'RESOLVED'}:
        return 'RESOLVED'
    elif s in {'IN PROGRESS', 'IN_PROGRESS', 'WIP', 'PENDING BANK', 'PENDING_BANK'}:
        return 'IN_PROGRESS'
    elif s in {'OPEN'}:
        return 'OPEN'
    elif s in {'REJECTED'}:
        return 'REJECTED'
    return 'OPEN'

def normalize_channel(val: Any) -> str:
    """Normalize complaint reporting channel."""
    if pd.isna(val):
        return 'Other'
    s = str(val).strip().title()
    if 'Ivr' in s:
        return 'IVR'
    elif 'Call Center' in s:
        return 'Call Center'
    elif 'App' in s:
        return 'Mobile App'
    elif 'Branch' in s:
        return 'Branch'
    elif 'Chatbot' in s:
        return 'Chatbot'
    elif 'Email' in s:
        return 'Email'
    return s

# --- Full Pipeline Cleaning Functions ---

def clean_upi_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize UPI transactions dataset."""
    logger.info("Cleaning UPI Transactions dataset...")
    df = df.copy()
    
    df['txn_id'] = df['txn_id'].apply(standardize_txn_id)
    df['user_id'] = df['user_id'].apply(standardize_user_id)
    df['merchant_id'] = df['merchant_id'].apply(standardize_merchant_id)
    df['amount'] = df['amount'].apply(clean_amount)
    df['timestamp'] = parse_mixed_datetime_series(df['timestamp'])
    df['status'] = df['status'].apply(normalize_txn_status)
    
    utr_results = df['utr'].apply(clean_utr)
    df['utr'] = [res[0] for res in utr_results]
    df['is_valid_utr'] = [res[1] for res in utr_results]
    
    df['mcc'] = df['mcc'].apply(lambda x: clean_mcc(x, None)[0])
    
    # Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=['txn_id', 'user_id', 'merchant_id', 'timestamp', 'amount'], keep='first')
    dedup_count = initial_len - len(df)
    if dedup_count > 0:
        logger.info(f"Removed {dedup_count} exact duplicate transaction rows.")
        
    return df

def clean_kyc_records(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize Customer KYC Master dataset."""
    logger.info("Cleaning Customer KYC dataset...")
    df = df.copy()
    
    df['user_id'] = df['user_id'].apply(standardize_user_id)
    df['full_name'] = df['full_name'].astype(str).str.strip().str.title()
    
    pan_res = df['pan'].apply(clean_pan)
    df['pan'] = [r[0] for r in pan_res]
    df['is_valid_pan'] = [r[1] for r in pan_res]
    
    aadhaar_res = df['aadhaar'].apply(clean_aadhaar)
    df['aadhaar'] = [r[0] for r in aadhaar_res]
    df['is_valid_aadhaar'] = [r[1] for r in aadhaar_res]
    
    df['date_of_birth'] = parse_mixed_datetime_series(df['date_of_birth'])
    df['signup_timestamp'] = parse_mixed_datetime_series(df['signup_timestamp'])
    
    city_state_res = [clean_city_and_state(c, s) for c, s in zip(df['city'], df['state'])]
    df['city'] = [cs[0] for cs in city_state_res]
    df['state'] = [cs[1] for cs in city_state_res]
    
    df['monthly_income'] = df['monthly_income'].apply(clean_amount)
    
    df['occupation'] = df['occupation'].astype(str).str.strip().str.title()
    df['occupation'] = df['occupation'].replace({'Nan': 'Unspecified', 'None': 'Unspecified', '': 'Unspecified'})
    
    df['kyc_status'] = df['kyc_status'].apply(normalize_kyc_status)
    df['risk_segment'] = df['risk_segment'].apply(normalize_risk_segment)
    
    df = df.drop_duplicates(subset=['user_id'], keep='first')
    return df

def clean_merchants_master(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize Merchant Master dataset."""
    logger.info("Cleaning Merchant Master dataset...")
    df = df.copy()
    
    df['merchant_id'] = df['merchant_id'].apply(standardize_merchant_id)
    df['merchant_name'] = df['merchant_name'].astype(str).str.strip().str.title()
    
    mcc_cat_res = [clean_mcc(m, c) for m, c in zip(df['mcc'], df['merchant_category'])]
    df['mcc'] = [mc[0] for mc in mcc_cat_res]
    df['merchant_category'] = [mc[1] for mc in mcc_cat_res]
    
    df['business_type'] = df['business_type'].apply(normalize_business_type)
    df['merchant_status'] = df['merchant_status'].apply(normalize_merchant_status)
    
    city_state_res = [clean_city_and_state(c, s) for c, s in zip(df['city'], df['state'])]
    df['city'] = [cs[0] for cs in city_state_res]
    df['state'] = [cs[1] for cs in city_state_res]
    
    df['onboarding_date'] = parse_mixed_datetime_series(df['onboarding_date'])
    df['declared_avg_ticket_size'] = df['declared_avg_ticket_size'].apply(clean_amount)
    
    df['settlement_account'] = df['settlement_account'].astype(str).str.strip()
    df['settlement_account'] = df['settlement_account'].replace({'nan': None, 'None': None, 'NA': None, '': None})
    
    df = df.drop_duplicates(subset=['merchant_id'], keep='first')
    return df

def clean_chargebacks(json_data: Union[list, str, pd.DataFrame], upi_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Clean and standardize Chargebacks dataset."""
    logger.info("Cleaning Chargebacks dataset...")
    if isinstance(json_data, str):
        with open(json_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif isinstance(json_data, list):
        df = pd.DataFrame(json_data)
    else:
        df = json_data.copy()
        
    df['complaint_id'] = df['complaint_id'].apply(standardize_complaint_id)
    df['txn_id'] = df['txn_id'].apply(standardize_txn_id)
    df['user_id'] = df['user_id'].apply(standardize_user_id)
    df['merchant_id'] = df['merchant_id'].apply(standardize_merchant_id)
    
    df['transaction_timestamp'] = parse_mixed_datetime_series(df['transaction_timestamp'])
    df['reported_timestamp'] = parse_mixed_datetime_series(df['reported_timestamp'])
    df['bank_response_timestamp'] = parse_mixed_datetime_series(df['bank_response_timestamp'])
    
    df['disputed_amount'] = df['disputed_amount'].apply(clean_amount)
    
    # Impute missing disputed amounts from UPI transactions where available
    if upi_df is not None:
        txn_amt_map = upi_df.dropna(subset=['txn_id', 'amount']).drop_duplicates(subset=['txn_id']).set_index('txn_id')['amount'].to_dict()
        df['disputed_amount'] = df.apply(
            lambda r: txn_amt_map.get(r['txn_id'], r['disputed_amount']) if pd.isna(r['disputed_amount']) else r['disputed_amount'],
            axis=1
        )
        
    df['reason_category'] = df['reason_code'].apply(normalize_dispute_reason)
    df['severity'] = df['severity'].apply(normalize_dispute_severity)
    df['resolution_status'] = df['resolution_status'].apply(normalize_resolution_status)
    df['channel'] = df['channel'].apply(normalize_channel)
    
    df['reporting_delay_days'] = (df['reported_timestamp'] - df['transaction_timestamp']).dt.total_seconds() / (24 * 3600)
    df['reporting_delay_days'] = df['reporting_delay_days'].apply(lambda x: max(0.0, round(x, 2)) if pd.notna(x) else None)
    
    df = df.drop_duplicates(subset=['complaint_id'], keep='first')
    return df
