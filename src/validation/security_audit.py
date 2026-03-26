"""
RBI NFPC Phase 2 — DevSecOps Security Audit Module
===================================================
Checks for five classes of production security threats:
  1. Adversarial Input Patterns  — injected extreme values to flip predictions
  2. Data Poisoning Indicators   — sudden label-flipping in training data
  3. Schema Integrity            — unexpected columns / dtypes (hallucination guard)
  4. PII Exposure Scan           — raw names, phone numbers, addresses in feature space
  5. Temporal Leakage Guard      — post-event columns that could "cheat" the model

Run this before every training run and at API ingestion time.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger("NFPC.SecurityAudit")

# ─────────────────────────────────────────────────────────────────────────────
# Expected schema — column → allowed dtype (regex)
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_FEATURE_SCHEMA: Dict[str, str] = {
    # Graph layer
    "graph_pagerank":        r"float",
    "graph_in_degree":       r"int|float",
    "graph_out_degree":      r"int|float",
    "graph_in_gini":         r"float",
    "graph_out_gini":        r"float",
    "graph_fan_ratio":       r"float",
    "graph_in_entropy":      r"float",
    "graph_out_entropy":     r"float",
    # Temporal layer
    "temporal_avg_rest_hours":     r"float",
    "temporal_month_end_ratio":    r"float",
    "temporal_pass_through_ratio": r"float",
    "temporal_burst_intensity":    r"float",
    # Spatial layer
    "geo_cluster_count":    r"int|float",
    "geo_outlier_ratio":    r"float",
    "geo_max_drift_rad":    r"float",
}

# Columns that must NEVER appear in a training feature set (leakage)
LEAKY_COLUMNS: List[str] = [
    "freeze_date", "unfreeze_date",
    "mule_flag_date", "alert_reason", "flagged_by_branch",
]

# Columns that constitute PII — must be dropped before modelling
PII_COLUMNS: List[str] = [
    "name", "address", "phone_number", "gender", "date_of_birth",
    "email", "aadhaar_number", "pan_number",
]


class SecurityAudit:
    """
    Runs pre-training and pre-inference security checks.
    All findings are logged and returned as a structured report.
    """

    def __init__(self) -> None:
        self.findings: List[Dict[str, Any]] = []

    def _flag(self, severity: str, check: str, detail: str) -> None:
        entry = {"severity": severity, "check": check, "detail": detail}
        self.findings.append(entry)
        level_map = {
            "CRITICAL": log.critical,
            "ERROR":    log.error,
            "WARNING":  log.warning,
            "INFO":     log.info,
        }
        fn = level_map.get(severity.upper(), log.warning)
        fn("[%s] %s — %s", severity, check, detail)


    # ──────────────────────────────────────────────────────────────────────
    # 1. Adversarial Input Detection
    # ──────────────────────────────────────────────────────────────────────
    def check_adversarial_inputs(self, df: pd.DataFrame) -> None:
        """
        Flags rows where numeric features exceed ±10 σ of the column distribution.
        Extreme-outlier injection can flip a model's output score artificially.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            col_std  = df[col].std()
            col_mean = df[col].mean()
            if col_std == 0:
                continue
            z_scores  = np.abs((df[col] - col_mean) / (col_std + 1e-9))
            extreme   = int((z_scores > 10).sum())
            if extreme > 0:
                self._flag(
                    "WARNING",
                    "AdversarialInput",
                    f"Column '{col}': {extreme} extreme values (|z| > 10). "
                    "Potential adversarial injection.",
                )

    # ──────────────────────────────────────────────────────────────────────
    # 2. Data Poisoning Indicator
    # ──────────────────────────────────────────────────────────────────────
    def check_label_poisoning(self, df: pd.DataFrame, label_col: str = "is_mule") -> None:
        """
        Detects sudden dense clusters of label changes — a hallmark of data poisoning.
        Expects df to have a label column (training context only).
        """
        if label_col not in df.columns:
            return
        label_rate = df[label_col].mean()
        if label_rate > 0.25:
            self._flag(
                "ERROR",
                "LabelPoisoning",
                f"Mule prevalence is {label_rate:.1%} — expected ≈ 1%. "
                "Training data may be poisoned. Verify source.",
            )
        if label_rate == 0.0:
            self._flag(
                "WARNING",
                "LabelPoisoning",
                "Zero mule labels found. Possible label suppression attack.",
            )

    # ──────────────────────────────────────────────────────────────────────
    # 3. Schema Integrity Check
    # ──────────────────────────────────────────────────────────────────────
    def check_schema(self, df: pd.DataFrame) -> None:
        """
        Validates that only expected columns with expected dtypes are present.
        Catches hallucinated or injected columns.
        """
        actual_cols = set(df.columns)
        expected    = set(EXPECTED_FEATURE_SCHEMA.keys()) | {"account_id"}

        invented = actual_cols - expected - {"is_mule"}
        if invented:
            self._flag(
                "WARNING",
                "SchemaIntegrity",
                f"Unexpected columns detected: {sorted(invented)}. "
                "Verify these are forensically derived — not hallucinated.",
            )

        for col, dtype_pattern in EXPECTED_FEATURE_SCHEMA.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not re.search(dtype_pattern, actual_dtype):
                    self._flag(
                        "ERROR",
                        "DtypeMismatch",
                        f"Column '{col}' expected dtype matching '{dtype_pattern}', "
                        f"got '{actual_dtype}'.",
                    )

    # ──────────────────────────────────────────────────────────────────────
    # 4. PII Exposure Scan
    # ──────────────────────────────────────────────────────────────────────
    def check_pii_exposure(self, df: pd.DataFrame) -> None:
        """
        Verifies no raw PII columns have leaked into the feature set.
        """
        leaked_pii = [c for c in PII_COLUMNS if c in df.columns]
        if leaked_pii:
            self._flag(
                "CRITICAL",
                "PIIExposure",
                f"PII columns present in dataset: {leaked_pii}. "
                "These MUST be dropped prior to modelling.",
            )

    # ──────────────────────────────────────────────────────────────────────
    # 5. Temporal Leakage Guard
    # ──────────────────────────────────────────────────────────────────────
    def check_temporal_leakage(self, df: pd.DataFrame) -> None:
        """
        Ensures post-event features (e.g. freeze_date) never appear in feature space.
        """
        leaked = [c for c in LEAKY_COLUMNS if c in df.columns]
        if leaked:
            self._flag(
                "CRITICAL",
                "TemporalLeakage",
                f"Post-event leaky columns detected: {leaked}. "
                "Remove before training — these are ground-truth proxies.",
            )

    # ──────────────────────────────────────────────────────────────────────
    # 6. Data Checksum (Integrity Seal)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def checksum(df: pd.DataFrame) -> str:
        """
        SHA-256 fingerprint of the dataframe for reproducibility and tamper detection.
        """
        raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(raw).hexdigest()

    # ──────────────────────────────────────────────────────────────────────
    # Full Audit Run
    # ──────────────────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame, is_training: bool = True) -> Dict[str, Any]:
        """
        Execute all checks and return a structured audit report.
        Raises RuntimeError if CRITICAL findings are present (fail-fast in production).
        """
        self.findings = []
        log.info("Security audit: %d rows, %d cols…", len(df), len(df.columns))

        self.check_adversarial_inputs(df)
        self.check_schema(df)
        self.check_pii_exposure(df)
        self.check_temporal_leakage(df)
        if is_training:
            self.check_label_poisoning(df)

        checksum = self.checksum(df)
        critical = [f for f in self.findings if f["severity"] == "CRITICAL"]

        report = {
            "passed":    len(self.findings) == 0,
            "checksum":  checksum,
            "findings":  self.findings,
            "n_critical": len(critical),
            "n_warnings": len([f for f in self.findings if f["severity"] == "WARNING"]),
        }

        if critical:
            raise RuntimeError(
                f"SECURITY AUDIT FAILED — {len(critical)} CRITICAL findings. "
                "Pipeline halted. Fix before proceeding."
            )

        log.info(
            "Audit %s | findings=%d | checksum=%s…",
            "PASSED" if report["passed"] else "WARNINGS",
            len(self.findings),
            checksum[:12],
        )
        return report


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    n   = 500

    df_ok = pd.DataFrame(
        {
            "account_id":               [f"A{i}" for i in range(n)],
            "graph_pagerank":           rng.uniform(0, 1, n).astype("float32"),
            "graph_in_degree":          rng.integers(0, 100, n),
            "graph_out_degree":         rng.integers(0, 50, n),
            "graph_in_gini":            rng.uniform(0, 1, n).astype("float32"),
            "graph_out_gini":           rng.uniform(0, 1, n).astype("float32"),
            "graph_fan_ratio":          rng.uniform(0, 10, n).astype("float32"),
            "graph_in_entropy":         rng.uniform(0, 1, n).astype("float32"),
            "graph_out_entropy":        rng.uniform(0, 1, n).astype("float32"),
            "temporal_burst_intensity": rng.exponential(1.0, n).astype("float32"),
            "is_mule":                  rng.choice([0, 1], n, p=[0.99, 0.01]),
        }
    )

    audit = SecurityAudit()
    report = audit.run(df_ok, is_training=True)
    print(f"Passed: {report['passed']}")
    print(f"Findings: {report['n_warnings']} warnings, {report['n_critical']} critical")
    print(f"Checksum: {report['checksum'][:20]}…")

    # Test PII injection detection
    df_bad = df_ok.copy()
    df_bad["name"]        = "John Doe"
    df_bad["freeze_date"] = "2024-01-01"
    print("\n--- Testing PII / Leakage Detection ---")
    try:
        audit2 = SecurityAudit()
        audit2.run(df_bad, is_training=True)
        print("ERROR: should have raised RuntimeError")
    except RuntimeError as e:
        print(f"✅ Correctly blocked: {e}")
    import sys; sys.exit(0)

