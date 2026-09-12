# 📊 Data Cleaning & Rescue Proof Report

**Track 1: FinTech & BFSI — UPI Fraud Ring & Merchant Analytics**  
**TransOrg AgentIQ Datathon**

---

## 🎯 Executive Summary of Data Cleaning Strategy

In digital payments analytics, arbitrary row deletion causes catastrophic data leakage, undercounts transaction volume, and conceals coordinated fraud rings. 

Our data engineering engine adheres strictly to the **Data Rescue Principle**:
1. **Zero Unjustified Data Loss:** No rows were dropped for having messy formatting, mixed datetime representations, currency symbols, or missing non-critical attributes.
2. **Deterministic Canonical Normalization:** Robust regular expressions, dictionary mappings, and datetime coercers transformed chaotic raw strings into strict relational types.
3. **Audit Trail & Quality Flagging:** Data anomalies (e.g. invalid UTR formats, unverified PANs) were flagged as diagnostic columns rather than discarded, enabling downstream fraud correlation analytics.

---

## 📈 Raw vs. Cleaned Record Count Tracking

| Dataset Entity | Raw Source File | Raw Records | Cleaned Output Table | Cleaned Records | Delta / Resolution Strategy |
|---|---|:---:|---|:---:|---|
| **UPI Transactions** | `track1_upi_transactions.csv` | **20,400** | `fact_transactions` | **20,000** | **-400 duplicate rows** (removed 100% exact duplicate transaction entries). |
| **Customer KYC** | `track1_kyc_records.csv` | **36,400** | `dim_customers` | **28,920** | **-7,480 repeat submissions** (merged repeated customer KYC verification attempts into unique master profiles, preserving verified status). |
| **Merchant Master** | `track1_merchants_master.csv` | **6,210** | `dim_merchants` | **4,343** | **-1,867 multi-registrations** (deduplicated repeated onboarding entries to unique merchant master). |
| **Disputes & Chargebacks** | `track1_chargebacks.json` | **2,884** | `fact_chargebacks` | **2,800** | **-84 duplicate complaint logs** (resolved repeat submissions and linked valid disputes to transactions). |
| **Unified Analytics Mart** | *(Star Schema Join)* | **20,400** | `fact_unified_analytics` | **20,000** | Full 1-to-1 coverage of all unique payment events enriched with dispute, KYC, and merchant dimensions. |

---

## 🔍 Before vs. After Column Profiling & Missing Values

### 1. UPI Transactions (`track1_upi_transactions.csv`)

| Column Name | Raw Missing Count | Cleaned Missing Count | Cleaning Treatment & Validation |
|---|:---:|:---:|---|
| `txn_id` | 0 | 0 | Standardized prefixes to canonical `TXN{8-digits}` (e.g., `11869` -> `TXN00011869`). |
| `user_id` | 0 | 0 | Standardized mixed formats (`usr12345`, `USR-12345`, `USR 12345`) to `USR{5-digits}`. |
| `merchant_id` | 0 | 0 | Standardized mixed formats (`mch1234`, `MCH-1234`, `1234`) to `MCH{4-digits}`. |
| `amount` | 0 | 0 | Stripped `₹`, `Rs.`, `INR`, commas, converted `27.3k` to `27300.0`, handled negative amounts via absolute value. |
| `timestamp` | 0 | 0 | Parsed mixed formats: Unix epoch seconds, epoch milliseconds, ISO-8601 strings, and 12-hour AM/PM strings. |
| `status` | 0 | 0 | Mapped 10+ raw synonyms (`TXN_SUCCESS`, `COMPLETED`, `S`, `Fail`, `Declined`, `Processing`) to `SUCCESS`, `FAILED`, `PENDING`. |
| `utr` | 1,024 | 0 (Flagged) | Standardized spaces/hyphens; retained invalid UTRs with `is_valid_utr=False` to uncover dispute correlations. |

### 2. Customer KYC Records (`track1_kyc_records.csv`)

| Column Name | Raw Missing Count | Cleaned Missing Count | Cleaning Treatment & Validation |
|---|:---:|:---:|---|
| `user_id` | 0 | 0 | Canonical `USR{5-digits}` standardization. |
| `full_name` | 0 | 0 | Cleansed whitespace and normalized casing. |
| `pan` | 1,210 | 1,210 | Uppercased, stripped spaces/hyphens, validated with regex `^[A-Z]{5}\d{4}[A-Z]$` (`is_valid_pan`). |
| `aadhaar` | 1,540 | 1,540 | Cleaned to masked format `XXXX-XXXX-NNNN` for PII protection (`is_valid_aadhaar`). |
| `monthly_income` | 2,100 | 0 (Imputed) | Imputed missing values with median income by occupation tier. |
| `city` / `state` | 0 | 0 | Normalized using 25+ canonical aliases (e.g. `BOMBAY` -> `Mumbai`, `BLR` -> `Bengaluru`, `DILLI` -> `Delhi`). |
| `kyc_status` | 0 | 0 | Standardized to `VERIFIED`, `PENDING`, `REJECTED`, or `UNREGISTERED`. |
| `risk_segment` | 0 | 0 | Standardized to `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`. |

### 3. Merchant Master (`track1_merchants_master.csv`)

| Column Name | Raw Missing Count | Cleaned Missing Count | Cleaning Treatment & Validation |
|---|:---:|:---:|---|
| `merchant_id` | 0 | 0 | Canonical `MCH{4-digits}` standardization. |
| `mcc_code` | 420 | 0 (Imputed) | Derived missing MCC codes from merchant category names using ISO 18245 lookup. |
| `merchant_category`| 310 | 0 (Imputed) | Derived missing category names from MCC codes. |
| `declared_avg_ticket_size` | 512 | 0 (Imputed)| Imputed missing ticket sizes from median ticket size of the respective merchant category. |
| `settlement_account` | 240 | 240 | Cleaned account identifiers; used to construct knowledge graph settlement edges. |
| `merchant_status` | 0 | 0 | Standardized to `ACTIVE`, `SUSPENDED`, `ON_HOLD`, `INACTIVE`. |

### 4. Disputes & Chargebacks (`track1_chargebacks.json`)

| Column Name | Raw Missing Count | Cleaned Missing Count | Cleaning Treatment & Validation |
|---|:---:|:---:|---|
| `complaint_id` | 0 | 0 | Standardized to `CBK{7-digits}`. |
| `txn_id` | 0 | 0 | Standardized and verified against `fact_transactions`. |
| `disputed_amount` | 120 | 0 (Imputed) | Imputed missing dispute amounts from the linked transaction `amount`. |
| `reason_category` | 0 | 0 | Grouped into `Unauthorized / Fraud`, `Goods Not Delivered`, `Duplicate Billing`, `Technical Failure`. |
| `severity` | 0 | 0 | Mapped to `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`. |
| `reporting_delay_days` | 0 | 0 | Computed as `(reported_timestamp - txn_timestamp).days`. |

---

## 🧪 Verification & Pipeline Execution Metrics

When running `python src/run_etl.py`, the following verification metrics are generated:

```
Total Cleaned Transactions: 20,000
Total Transaction Volume: INR 249,772,508.82
Average Transaction Value (ATV): INR 12,488.63
Payment Success Rate: 85.27%
Payment Failed Rate: 9.78%
Payment Pending Rate: 4.96%
Total Cleaned Chargebacks: 2,800
Total Disputed Volume: INR 10,073,050.05
Overall Dispute Rate: 14.00%
Cleaned Customer KYC Profiles: 28,920
Cleaned Merchant Masters: 4,343
SQLite Database Status: Persisted and indexed (fintech_analytics.db)
Unit & Integration Tests: 14 / 14 Passed
```
