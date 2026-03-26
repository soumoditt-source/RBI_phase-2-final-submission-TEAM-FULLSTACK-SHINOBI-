"""
RBI NFPC Phase 2 — Schema Configuration (REAL — verified from disk scan)
==========================================================================
VERIFIED COLUMNS (from schema_discovery.json — DO NOT INVENT NEW ONES):

transactions/batch-*/part_*.parquet (396 parts, ~1M rows each):
  transaction_id, account_id, transaction_timestamp, mcc_code,
  channel, amount, txn_type (C/D), counterparty_id

transactions_additional/batch-*/part_*.parquet (311 parts, ~1.4M rows):
  transaction_id, mnemonic_code, latitude, longitude, ip_address,
  balance_after_transaction, part_transaction_type,
  atm_deposit_channel_code, transaction_sub_type

Reference tables (all small, load in Pandas):
  accounts.parquet           — 160,153 rows
  accounts-additional.parquet — 160,153 rows  (just account_id + scheme_code)
  customers.parquet          — 159,416 rows
  customer_account_linkage   — 160,153 rows
  demographics.parquet       — 159,416 rows  🔴 PII + RED HERRING
  branch.parquet             — 9,000 rows
  product_details.parquet    — 159,416 rows
  train_labels.parquet       — 96,091 rows   🔴 LEAKY cols: mule_flag_date / alert_reason / flagged_by_branch
  test_accounts.parquet      — 64,062 rows   (account_id only)
"""

from __future__ import annotations
import pyarrow as pa

# ─────────────────────────────────────────────────────────────────────────────
# Optimised dtypes for transaction stream  (reduces memory ~70%)
# ─────────────────────────────────────────────────────────────────────────────
TXN_ARROW_SCHEMA = pa.schema([
    ("transaction_id",         pa.string()),
    ("account_id",             pa.string()),
    ("transaction_timestamp",  pa.string()),   # parse to datetime after load
    ("mcc_code",               pa.int32()),    # int32 vs int64 =  50% saving
    ("channel",                pa.string()),
    ("amount",                 pa.float32()),  # float32 vs float64 = 50% saving
    ("txn_type",               pa.string()),   # "C" or "D" only
    ("counterparty_id",        pa.string()),
])

TXNA_ARROW_SCHEMA = pa.schema([
    ("transaction_id",           pa.string()),
    ("mnemonic_code",            pa.string()),
    ("latitude",                 pa.float32()),
    ("longitude",                pa.float32()),
    ("ip_address",               pa.string()),
    ("balance_after_transaction",pa.float32()),
    ("part_transaction_type",    pa.string()),
    ("atm_deposit_channel_code", pa.string()),
    ("transaction_sub_type",     pa.string()),
])

# Columns that are LEAKY (post-event ground truth from labels table)
LEAKY_COLUMNS = frozenset([
    "freeze_date", "unfreeze_date",
    "mule_flag_date", "alert_reason", "flagged_by_branch",
])

# Columns that are PII — must be blocked from feature space
PII_COLUMNS = frozenset([
    "name", "address", "phone_number",
])

# Demographic red-herrings — may correlate but forensically unsound
DEMOGRAPHIC_RED_HERRINGS = frozenset([
    "gender", "nri_flag", "date_of_birth",
])

# ALL columns to drop before any feature engineering or modelling
DROP_BEFORE_FEATURES = LEAKY_COLUMNS | PII_COLUMNS | DEMOGRAPHIC_RED_HERRINGS

# ─────────────────────────────────────────────────────────────────────────────
# Pandas read_parquet dtype overrides (for reference tables)
# ─────────────────────────────────────────────────────────────────────────────
TRANSACTION_DTYPES: dict = {
    "mcc_code": "int32",
    "amount":   "float32",
}

ACCOUNT_DERIVED_DTYPES: dict = {
    "avg_balance":           "float32",
    "monthly_avg_balance":   "float32",
    "quarterly_avg_balance": "float32",
    "daily_avg_balance":     "float32",
    "branch_code":           "int32",
    "num_chequebooks":       "int16",
}
