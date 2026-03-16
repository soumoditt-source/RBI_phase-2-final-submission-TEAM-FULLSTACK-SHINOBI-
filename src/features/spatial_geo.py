"""
RBI NFPC Phase 2 — Spatial & Contextual Feature Extractor
===========================================================
REAL SCHEMA (verified from disk):

transactions_additional cols used:
  transaction_id, latitude, longitude, ip_address,
  balance_after_transaction, transaction_sub_type

accounts cols used:
  account_id, kyc_compliant, avg_balance, monthly_avg_balance,
  quarterly_avg_balance, num_chequebooks, scheme_code,
  product_family, cheque_allowed, cheque_availed, rural_branch

customers cols used:
  pan_available, mobile_banking_flag, internet_banking_flag,
  atm_card_flag, demat_flag, credit_card_flag, fastag_flag

product_details cols used:
  loan_sum, loan_count, cc_sum, cc_count, od_sum, od_count

ZERO invented columns — every feature is derived via forensic math from real cols.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

log = logging.getLogger("NFPC.Spatial")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM    = 6371.0
GEO_EPS_KM         = 20.0            # DBSCAN radius — 20 km = same "region"
GEO_MIN_SAMPLES     = 2              # minimum txns per cluster
NEAR_ZERO_BALANCE   = 100.0          # ₹100 threshold for near-zero drain detection


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon pairs."""
    r = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlam       = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class SpatialFeatureExtractor:
    """
    Extracts geo-drift and contextual features from the REAL dataset.

    call order:
      1. extract_geo_features(txna_df)   — from transactions_additional
      2. extract_contextual_features(master_df)  — from account master
    """

    # ──────────────────────────────────────────────────────────────────────
    # Geo-features from transactions_additional
    # ──────────────────────────────────────────────────────────────────────

    def extract_geo_features(self, txna_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-account spatial features from the transactions_additional table.

        Parameters
        ----------
        txna_df : pd.DataFrame
            Must contain: account_id (joined via transaction_id),
            latitude, longitude, ip_address, balance_after_transaction,
            transaction_sub_type, transaction_timestamp (if available)

        Returns
        -------
        pd.DataFrame with per-account spatial features.
        """
        log.info("Geo feature extraction: %d rows…", len(txna_df))
        rows = []

        for acc_id, grp in txna_df.groupby("account_id"):
            # ── Geo-clustering (only rows with valid coords) ───────────────
            geo_valid = grp.dropna(subset=["latitude", "longitude"])
            if len(geo_valid) >= GEO_MIN_SAMPLES:
                coords_rad = np.radians(geo_valid[["latitude", "longitude"]].values)
                eps_rad    = GEO_EPS_KM / EARTH_RADIUS_KM
                labels     = DBSCAN(
                    eps=eps_rad, min_samples=GEO_MIN_SAMPLES, metric="haversine", n_jobs=1
                ).fit_predict(coords_rad)
                n_clusters    = len(set(labels)) - (1 if -1 in labels else 0)
                outlier_ratio = float((labels == -1).mean())
            else:
                n_clusters    = 0
                outlier_ratio = 0.0

            # ── Max sustained geo-drift (km) between consecutive geo txns ─
            max_drift_km = 0.0
            if len(geo_valid) >= 2:
                lats = geo_valid["latitude"].values
                lons = geo_valid["longitude"].values
                dists = [
                    _haversine_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
                    for i in range(len(lats) - 1)
                ]
                max_drift_km = float(max(dists)) if dists else 0.0

            # ── balance_after_transaction features ─────────────────────────
            bal = grp["balance_after_transaction"].dropna()
            bal_near_zero_ratio = float(
                (bal < NEAR_ZERO_BALANCE).sum() / max(len(bal), 1)
            )
            bal_min    = float(bal.min()) if not bal.empty else 0.0
            bal_cv     = float(bal.std() / (bal.mean() + 1e-6)) if len(bal) > 1 else 0.0

            # ── Unique IP count ────────────────────────────────────────────
            unique_ips = grp["ip_address"].dropna().nunique()

            # ── Sub-type signals ───────────────────────────────────────────
            sub_types    = grp["transaction_sub_type"].dropna()
            sub_type_div = sub_types.nunique()  # diversity of sub-types

            # ── Last known exact coordinates (Google Earth mapping) ────────
            last_lat = float(geo_valid["latitude"].iloc[-1]) if not geo_valid.empty else 0.0
            last_lon = float(geo_valid["longitude"].iloc[-1]) if not geo_valid.empty else 0.0

            rows.append(
                {
                    "account_id":           acc_id,
                    "geo_cluster_count":    n_clusters,
                    "geo_outlier_ratio":    outlier_ratio,
                    "geo_max_drift_km":     max_drift_km,
                    "balance_near_zero_ratio": bal_near_zero_ratio,
                    "balance_min":          bal_min,
                    "balance_cv":           bal_cv,
                    "unique_ip_count":      unique_ips,
                    "txn_sub_type_div":     sub_type_div,
                    "last_lat":             last_lat,
                    "last_lon":             last_lon,
                }
            )

        result = pd.DataFrame(rows)
        log.info("Geo features: %s", result.shape)
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Contextual features from account master  (accounts + customers + products)
    # ──────────────────────────────────────────────────────────────────────

    def extract_contextual_features(self, master: pd.DataFrame) -> pd.DataFrame:
        """
        Derive forensically meaningful features from the account master table.
        Uses ONLY columns that exist in the real dataset.

        Parameters
        ----------
        master : pd.DataFrame
            Merged: accounts + accounts-additional + customers + product_details + branch.
            Must NOT contain freeze_date / PII (pipeline.py strips them).
        """
        log.info("Contextual feature extraction: %d accounts…", len(master))
        feats = master[["account_id"]].copy()

        # ── KYC compliance ─────────────────────────────────────────────────
        if "kyc_compliant" in master.columns:
            feats["kyc_lapse"] = (master["kyc_compliant"].astype(str).str.strip().str.upper() != "Y").astype("int8")
        else:
            feats["kyc_lapse"] = 0

        # ── Balance features ───────────────────────────────────────────────
        for col in ["avg_balance", "monthly_avg_balance", "quarterly_avg_balance", "daily_avg_balance"]:
            if col in master.columns:
                feats[col] = master[col].fillna(0).astype("float32")

        # ── Balance zero / near-zero (no transaction activity) ─────────────
        if "avg_balance" in master.columns:
            feats["low_avg_balance"] = (master["avg_balance"].fillna(0) < 1000.0).astype("int8")

        # ── Cheque activity ────────────────────────────────────────────────
        if "cheque_allowed" in master.columns and "cheque_availed" in master.columns:
            allowed  = master["cheque_allowed"].astype(str).str.upper() == "Y"
            availed  = master["cheque_availed"].astype(str).str.upper() == "Y"
            feats["cheque_avail_ratio"] = (availed & allowed).astype("int8")
        if "num_chequebooks" in master.columns:
            feats["num_chequebooks"] = master["num_chequebooks"].fillna(0).astype("int16")

        # ── Digital access score (mobile + internet + ATM + demat + credit card + fastag) ──
        digital_cols = [
            "mobile_banking_flag", "internet_banking_flag",
            "atm_card_flag", "demat_flag", "credit_card_flag", "fastag_flag",
        ]
        present_digital = [c for c in digital_cols if c in master.columns]
        if present_digital:
            digital_df = master[present_digital].apply(
                lambda col: (col.astype(str).str.upper() == "Y").astype("int8")
            )
            feats["digital_access_score"] = digital_df.sum(axis=1).astype("int8")

        # ── Product diversity score ────────────────────────────────────────
        product_count_cols = [c for c in ["loan_count", "cc_count", "od_count", "ka_count", "sa_count"] if c in master.columns]
        if product_count_cols:
            feats["multi_product_count"] = master[product_count_cols].fillna(0).sum(axis=1).astype("int16")

        # ── PAN + Aadhaar document availability ────────────────────────────
        if "pan_available" in master.columns:
            feats["pan_available"] = (master["pan_available"].astype(str).str.upper() == "Y").astype("int8")
        if "aadhaar_available" in master.columns:
            feats["aadhaar_available"] = (master["aadhaar_available"].fillna("N").astype(str).str.upper() == "Y").astype("int8")

        # ── Rural branch indicator ─────────────────────────────────────────
        if "rural_branch" in master.columns:
            feats["rural_branch"] = (master["rural_branch"].astype(str).str.upper() == "Y").astype("int8")

        # ── scheme_code risk encoding ──────────────────────────────────────
        # PMJDY and Jan Dhan accounts are commonly exploited
        if "scheme_code" in master.columns:
            risky_schemes = {"BSBD", "PMJDY", "JANSEFALOAN", "SB-PMJDY"}
            feats["risky_scheme"] = master["scheme_code"].isin(risky_schemes).astype("int8")

        # ── product_family from accounts ───────────────────────────────────
        if "product_family" in master.columns:
            feats["is_savings_account"] = (master["product_family"].astype(str).str.upper() == "SB").astype("int8")

        # ── Account age in days ────────────────────────────────────────────
        if "account_opening_date" in master.columns:
            today = pd.Timestamp("2024-12-31")   # Dataset end approx
            acc_date = pd.to_datetime(master["account_opening_date"], errors="coerce")
            feats["account_age_days"] = ((today - acc_date).dt.days.fillna(0).clip(lower=0)).astype("int32")
            feats["new_account"] = (feats["account_age_days"] < 90).astype("int8")

        # ── Loan + OD exposure ─────────────────────────────────────────────
        for col in ["loan_sum", "od_sum", "sa_sum"]:
            if col in master.columns:
                feats[col] = master[col].fillna(0).astype("float32")

        log.info("Contextual features shape: %s", feats.shape)
        return feats


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.data.pipeline import DataPipeline
    p = DataPipeline()

    print("\n=== Testing Contextual Feature Extraction ===")
    master = p.build_account_master(split="train")
    ext = SpatialFeatureExtractor()
    ctx = ext.extract_contextual_features(master)
    print(ctx.head(5).to_string())
    print(f"\nContextual features shape: {ctx.shape}")
    print(f"Columns: {list(ctx.columns)}")
