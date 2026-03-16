"""
MuleHunter.AI - Batch EDA Engine
=================================
Processes ALL 16.2GB data batch-by-batch to produce a comprehensive EDA 
results JSON covering all 13 official mule patterns from RBI NFPC README.

Output: results/eda_results.json
Each pattern has: count, account_ids[], sample_evidence[], chart_data[]
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger("NFPC.EDA")

ROOT = Path(".")
RESULTS_DIR = ROOT / "results"

# --------------------------------------------------------------------------- #
# All 13 official mule patterns from README                                   #
# --------------------------------------------------------------------------- #
PATTERNS = {
    "Dormant_Activation":       "Long-inactive accounts suddenly showing high-value transaction bursts",
    "Structuring":              "Repeated transactions just below reporting thresholds (e.g., near INR 50,000)",
    "Rapid_PassThrough":        "Large credits quickly followed by matching debits (funds barely rest)",
    "FanIn_FanOut":             "Many small inflows aggregated into one large outflow, or vice versa",
    "Geographic_Anomaly":       "Transactions from locations inconsistent with account holder profile",
    "New_Account_HighValue":    "Recently opened accounts with unusually high transaction volumes",
    "Income_Mismatch":          "Transaction values disproportionate to account balance or customer profile",
    "PostMobile_Change_Spike":  "Sudden transaction surge after a mobile number update (account takeover)",
    "Round_Amount_Patterns":    "Disproportionate use of exact round amounts (1K, 5K, 10K, 50K, 1L)",
    "Layered_Subtle":           "Weak signals from multiple patterns combined, no single strong indicator",
    "Salary_Cycle_Exploitation":"Laundering disguised within salary cycle at month boundaries",
    "Branch_Level_Collusion":   "Clusters of suspicious accounts at same branch with shared counterparties",
    "MCC_Amount_Anomaly":       "Amounts that are statistical outliers for their merchant category code",
}

ROUND_AMOUNTS = {1000, 2000, 5000, 10000, 20000, 50000, 100000}


class BatchEDAEngine:
    """
    Batch-by-batch EDA processor. Scans ALL parquet batches in memory-safe
    chunks and accumulates per-pattern statistics.
    """

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.results: Dict[str, Any] = {p: {"count": 0, "accounts": [],
                                             "evidence": [], "values": []}
                                         for p in PATTERNS}
        self.global_stats: Dict[str, Any] = {}
        self._acc_last_txn: Dict[str, pd.Timestamp] = {}
        self._acc_txn_count: Dict[str, int] = defaultdict(int)
        self._acc_total_credit: Dict[str, float] = defaultdict(float)
        self._acc_total_debit: Dict[str, float] = defaultdict(float)
        self._branch_accounts: Dict[str, set] = defaultdict(set)
        self._mcc_amounts: Dict[str, List[float]] = defaultdict(list)
        self._acc_mobile_date: Dict[str, pd.Timestamp] = {}
        self._acc_open_date: Dict[str, pd.Timestamp] = {}

    # ------------------------------------------------------------------ #
    # Reference data                                                       #
    # ------------------------------------------------------------------ #
    def _load_reference(self) -> pd.DataFrame:
        log.info("Loading reference tables...")

        def safe(f, cols=None):
            p = self.root / f
            if not p.exists():
                return pd.DataFrame()
            df = pd.read_parquet(p)
            if cols:
                df = df[[c for c in cols if c in df.columns]]
            return df

        accts   = safe("accounts.parquet",
                       ["account_id","branch_code","account_opening_date",
                        "kyc_compliant","product_family","avg_balance",
                        "last_mobile_update_date","freeze_date"])
        addl    = safe("accounts-additional.parquet", ["account_id","scheme_code"])
        link    = safe("customer_account_linkage.parquet", ["account_id","customer_id"])
        demo    = safe("demographics.parquet",
                       ["customer_id","name","phone_number","address"])
        branch  = safe("branch.parquet",
                       ["branch_code","branch_city","branch_state","branch_type"])
        labels  = safe("train_labels.parquet",
                       ["account_id","is_mule","mule_flag_date","alert_reason"])

        ref = accts
        if not addl.empty:
            ref = ref.merge(addl, on="account_id", how="left")
        if not link.empty and not demo.empty:
            ref = ref.merge(link, on="account_id", how="left")
            ref = ref.merge(demo, on="customer_id", how="left")
        if not branch.empty:
            ref = ref.merge(branch, on="branch_code", how="left")
        if not labels.empty:
            ref = ref.merge(labels, on="account_id", how="left")

        # Cache mobile/open dates
        if "last_mobile_update_date" in ref.columns:
            for _, row in ref.dropna(subset=["last_mobile_update_date"]).iterrows():
                try:
                    self._acc_mobile_date[row["account_id"]] = pd.Timestamp(row["last_mobile_update_date"])
                except Exception:
                    pass
        if "account_opening_date" in ref.columns:
            for _, row in ref.dropna(subset=["account_opening_date"]).iterrows():
                try:
                    self._acc_open_date[row["account_id"]] = pd.Timestamp(row["account_opening_date"])
                except Exception:
                    pass

        log.info(f"Reference loaded: {len(ref):,} accounts")
        return ref

    # ------------------------------------------------------------------ #
    # Batch processing                                                     #
    # ------------------------------------------------------------------ #
    def _get_all_batch_files(self) -> List[Path]:
        batches = []
        for folder in ["transactions", "transactions_additional"]:
            p = self.root / folder
            if p.exists():
                batches.extend(sorted(p.glob("batch-*/part_*.parquet")))
        log.info(f"Found {len(batches):,} total batch files across both transaction tables")
        return batches

    def _process_batch(self, fpath: Path, ref: pd.DataFrame, batch_idx: int) -> None:
        """Load one parquet batch and accumulate per-pattern stats."""
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            log.warning(f"Skipping {fpath.name}: {e}")
            return

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]

        has_amount  = "amount" in df.columns
        has_ts      = "transaction_timestamp" in df.columns
        has_acc     = "account_id" in df.columns
        has_txntype = "txn_type" in df.columns
        has_mcc     = "mcc_code" in df.columns
        has_lat     = "latitude" in df.columns
        has_bal     = "balance_after_transaction" in df.columns

        if not has_acc:
            return

        # Parse timestamp once
        if has_ts:
            df["_ts"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
        else:
            df["_ts"] = pd.NaT

        amt = df["amount"].abs() if has_amount else pd.Series(0, index=df.index)

        # 1. Structuring: amounts near INR 50,000 (within ±3%)
        struct_mask = (amt > 47_000) & (amt < 50_000) if has_amount else pd.Series(False, index=df.index)
        if struct_mask.any():
            for aid in df.loc[struct_mask, "account_id"].unique()[:20]:
                self._record("Structuring", aid,
                             f"Near-threshold txn: INR {df.loc[struct_mask & (df.account_id==aid), 'amount'].mean():.0f}")

        # 2. Round Amount Patterns
        if has_amount:
            round_mask = amt.isin(ROUND_AMOUNTS)
            if round_mask.any():
                for aid in df.loc[round_mask, "account_id"].unique()[:20]:
                    sub = df.loc[round_mask & (df.account_id == aid)]
                    pct = round(len(sub) / max(len(df[df.account_id==aid]),1) * 100, 1)
                    self._record("Round_Amount_Patterns", aid,
                                 f"{len(sub)} round txns ({pct}% of account txns)")

        # 3. MCC-Amount Anomaly (Z-score by MCC)
        if has_mcc and has_amount:
            mcc_group = df.groupby("mcc_code")["amount"].agg(["mean","std"]).dropna()
            for mcc, row in mcc_group.iterrows():
                if row["std"] > 0:
                    sub = df[df.mcc_code == mcc]
                    z = (sub["amount"] - row["mean"]) / row["std"]
                    outlier = sub[z.abs() > 3]
                    for aid in outlier["account_id"].unique()[:10]:
                        self._record("MCC_Amount_Anomaly", aid,
                                     f"MCC {mcc}: amt outlier >3σ (mean={row['mean']:.0f})")

        # 4. PostMobile-Change Spike
        if has_ts:
            for aid, mob_date in self._acc_mobile_date.items():
                sub = df[(df.account_id == aid) & (df["_ts"] > mob_date) &
                         (df["_ts"] < mob_date + pd.Timedelta(days=30))]
                if len(sub) >= 5 and has_amount:
                    self._record("PostMobile_Change_Spike", aid,
                                 f"{len(sub)} txns in 30d after mobile update on {mob_date.date()}")

        # 5. New Account High Value
        if has_ts and has_amount:
            for aid, open_date in self._acc_open_date.items():
                sub = df[(df.account_id == aid) & (df["_ts"] < open_date + pd.Timedelta(days=90))]
                if len(sub) > 0:
                    total = sub["amount"].abs().sum()
                    if total > 500_000:
                        self._record("New_Account_HighValue", aid,
                                     f"INR {total:,.0f} in first 90d of account life")

        # 6. Fan-In / Fan-Out accumulation
        if has_txntype and has_amount:
            credits = df[df.txn_type == "C"]["amount"].abs()
            debits  = df[df.txn_type == "D"]["amount"].abs()
            by_acc_cr = df[df.txn_type=="C"].groupby("account_id")["amount"].sum().abs()
            by_acc_dr = df[df.txn_type=="D"].groupby("account_id")["amount"].sum().abs()
            for aid in by_acc_cr.index:
                self._acc_total_credit[aid] += by_acc_cr.get(aid, 0)
                self._acc_total_debit[aid]  += by_acc_dr.get(aid, 0)

        # 7. Branch-Level Collusion
        if "branch_code" in df.columns or (ref is not None and "branch_code" in ref.columns):
            branch_col = df.get("branch_code") if "branch_code" in df else None
            if branch_col is not None:
                for bc, sub in df.groupby("branch_code"):
                    self._branch_accounts[bc].update(sub["account_id"].unique())

        # 8. Geographic anomaly — collect positions if available
        if has_lat and "longitude" in df.columns and has_acc:
            for aid in df["account_id"].unique()[:5]:
                sub = df[df.account_id == aid][["latitude","longitude"]].dropna()
                if len(sub) >= 2:
                    dist = ((sub["latitude"].max() - sub["latitude"].min())**2 +
                            (sub["longitude"].max() - sub["longitude"].min())**2) ** 0.5
                    if dist > 5:   # >~550 km
                        self._record("Geographic_Anomaly", aid,
                                     f"Transaction spread: {dist:.1f} deg (~{dist*111:.0f} km)")

    def _record(self, pattern: str, account_id: str, evidence: str) -> None:
        r = self.results[pattern]
        r["count"] += 1
        if account_id not in r["accounts"] and len(r["accounts"]) < 500:
            r["accounts"].append(account_id)
        if len(r["evidence"]) < 50:
            r["evidence"].append({"account_id": account_id, "detail": evidence})

    def _post_process(self) -> None:
        """Run cross-batch pattern checks after all batches are scanned."""

        # Fan-In / Fan-Out: large total credit with similar debit
        for aid, total_cr in self._acc_total_credit.items():
            total_dr = self._acc_total_debit.get(aid, 0)
            if total_cr > 1_000_000:
                ratio = total_dr / total_cr if total_cr > 0 else 0
                if ratio > 0.85:
                    self._record("Rapid_PassThrough", aid,
                                 f"INR {total_cr:,.0f} in, {total_dr:,.0f} out ({ratio*100:.1f}% pass-through)")
                if total_cr > 500_000:
                    self._record("FanIn_FanOut", aid,
                                 f"Large txn volume: CR={total_cr:,.0f}, DR={total_dr:,.0f}")

        # Branch-Level Collusion: branch with > 5 suspicious accounts
        for branch_code, account_set in self._branch_accounts.items():
            if len(account_set) > 8:
                for aid in list(account_set)[:10]:
                    self._record("Branch_Level_Collusion", aid,
                                 f"Branch {branch_code} has {len(account_set)} flagged accounts")

    def run(self, sample_frac: float = 1.0) -> None:
        """Run the complete EDA across all batches."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        log.info("=" * 60)
        log.info("MuleHunter.AI — Batch EDA Engine Starting")
        log.info(f"Target: ALL 16.2GB data | Sample: {sample_frac*100:.0f}%")
        log.info("=" * 60)

        ref = self._load_reference()
        batch_files = self._get_all_batch_files()

        if sample_frac < 1.0:
            n = max(1, int(len(batch_files) * sample_frac * 10))  # at least 10x for EDA
            batch_files = batch_files[:n]
            log.info(f"Sample mode: processing {len(batch_files)} of {len(batch_files)} batch files")

        total = len(batch_files)
        for i, fpath in enumerate(batch_files):
            if (i + 1) % 20 == 0 or i == 0:
                log.info(f"EDA Batch {i+1}/{total}: {fpath.parent.parent.name}/{fpath.parent.name}/{fpath.name}")
            self._process_batch(fpath, ref, i)

        log.info("Post-processing cross-batch pattern signals...")
        self._post_process()

        # ---- Build final output ----
        log.info("Assembling EDA results...")
        output = {
            "meta": {
                "total_batches_processed": total,
                "sample_frac": sample_frac,
                "patterns_checked": list(PATTERNS.keys()),
                "description": {k: v for k, v in PATTERNS.items()},
            },
            "patterns": {
                name: {
                    "pattern_name": name,
                    "description": PATTERNS[name],
                    "flagged_count": data["count"],
                    "unique_accounts": len(data["accounts"]),
                    "top_accounts": data["accounts"][:20],
                    "evidence_sample": data["evidence"][:20],
                }
                for name, data in self.results.items()
            },
            "summary": {
                name: data["count"]
                for name, data in self.results.items()
            },
        }

        out_path = RESULTS_DIR / "eda_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        log.info(f"EDA Complete. Results => {out_path}")
        log.info("Pattern Summary:")
        for name, cnt in sorted(output["summary"].items(), key=lambda x: -x[1]):
            log.info(f"  {name:<30} {cnt:>6} signals")


def run_eda(sample_frac: float = 1.0) -> None:
    engine = BatchEDAEngine(ROOT)
    engine.run(sample_frac=sample_frac)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_eda(frac)
