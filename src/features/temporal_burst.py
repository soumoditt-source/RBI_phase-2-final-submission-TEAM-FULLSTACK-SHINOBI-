"""
RBI NFPC Phase 2 — Temporal Forensic Engine (Centered Burst + Multi-Signal Change-Point)
==========================================================================================
Three independent "trigger" mechanisms:
  1. Velocity-Ratio Spike   — rolling 7-day volume vs 90-day baseline
  2. Dormancy-to-Burst      — silence ratio before a burst window
  3. Structuring Probe      — density of transactions in the ₹45,000–55,000 band

Each trigger independently produces (suspicious_start, suspicious_end).
The *union* of all triggered windows is the final reported window.
This guarantees: every timestamp is backed by a real forensic trigger.

Schema columns used (no invented fields):
  account_id, transaction_timestamp, amount, txn_type
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("NFPC.Temporal")


# ─────────────────────────────────────────────────────────────────────────────
# Constants (forensic thresholds — documented in report for defensibility)
# ─────────────────────────────────────────────────────────────────────────────
VELOCITY_SPIKE_SIGMA   = 3.0          # z-score threshold to flag a burst day
DORMANCY_SILENCE_DAYS  = 30           # days with near-zero activity = dormant
DORMANCY_BURST_RATIO   = 5.0          # burst volume / pre-burst baseline
STRUCTURING_BAND_LOW   = 45_000.0     # ₹ — just below ₹50 K reporting threshold
STRUCTURING_BAND_HIGH  = 55_000.0     # ₹
STRUCTURING_MIN_COUNT  = 3            # repeated hits in a 30-day window = flag
BURST_WINDOW_DAYS      = 14           # centred window half-width in days
PASS_THROUGH_HOURS     = 24.0         # credits matched to debits within 24 hours


class TemporalBurstEngine:
    """
    Stateless per-account temporal analyser.
    Call `extract_features(df)` with the full transaction dataframe.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Internal Trigger 1: Velocity-Ratio Spike (Z-score)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _velocity_trigger(
        daily_vol: pd.Series,
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, float]]:
        """
        Flags a day where the rolling 7-day sum is > VELOCITY_SPIKE_SIGMA std-devs
        above the 90-day rolling mean.
        """
        roll7   = daily_vol.rolling(7,  min_periods=1).sum()
        roll90  = daily_vol.rolling(90, min_periods=1).mean()
        roll90s = daily_vol.rolling(90, min_periods=1).std().fillna(0)

        z = (roll7 - roll90) / (roll90s + 1e-6)
        if z.max() < VELOCITY_SPIKE_SIGMA:
            return None

        peak = z.idxmax()
        half = pd.Timedelta(days=BURST_WINDOW_DAYS // 2)
        start = peak - half
        end   = peak + half
        return start, end, float(z.max())

    # ──────────────────────────────────────────────────────────────────────
    # Internal Trigger 2: Dormancy-to-Burst Detector
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _dormancy_trigger(
        daily_vol: pd.Series,
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, float]]:
        """
        Scans for a silence period of >= DORMANCY_SILENCE_DAYS followed by a
        sudden volume burst exceeding DORMANCY_BURST_RATIO × pre-burst baseline.
        """
        silence_mask = daily_vol < (daily_vol.mean() * 0.05)   # ≤ 5% of mean
        in_silence   = False
        silence_start_idx = None

        for i, (ts, is_quiet) in enumerate(silence_mask.items()):
            if is_quiet:
                if not in_silence:
                    in_silence        = True
                    silence_start_idx = i
            else:
                if in_silence:
                    silence_days = i - silence_start_idx
                    if silence_days >= DORMANCY_SILENCE_DAYS:
                        # Check burst ratio
                        pre  = daily_vol.iloc[max(0, silence_start_idx - 30):silence_start_idx]
                        post = daily_vol.iloc[i:min(len(daily_vol), i + 30)]
                        baseline = pre.mean() + 1e-6
                        burst_ratio = float(post.mean() / baseline)
                        if burst_ratio >= DORMANCY_BURST_RATIO:
                            burst_start = daily_vol.index[i]
                            burst_end   = daily_vol.index[min(i + BURST_WINDOW_DAYS, len(daily_vol) - 1)]
                            return burst_start, burst_end, burst_ratio
                in_silence = False

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Internal Trigger 3: Structuring Probe (₹45 K–₹55 K Band)
    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    # Internal Trigger 4: Change-Point Detection (Mule Transition)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _changepoint_trigger(
        daily_vol: pd.Series,
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, float]]:
        """
        Uses a sliding variance ratio to detect the exact moment an account 
        switches its behavior profile.
        """
        if len(daily_vol) < 60: # Need at least 2 months of data
            return None
            
        # Cumulative sum of variance
        v_ratio = daily_vol.rolling(30).std() / (daily_vol.expanding().std() + 1e-6)
        
        if v_ratio.max() > 2.5: # 250% variance surge
            peak_ts = v_ratio.idxmax()
            # Forensic window: 7 days before to 21 days after the shift
            return peak_ts - pd.Timedelta(days=7), peak_ts + pd.Timedelta(days=21), float(v_ratio.max())
            
        return None

    @staticmethod
    def _structuring_trigger(
        group: pd.DataFrame,
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, float]]:
        """
        Looks for repeated transactions in the structuring band (₹45 K–₹55 K)
        within any rolling 30-day window.
        """
        credits = group[group["txn_type"] == "C"].copy()
        band    = credits[
            credits["amount"].between(STRUCTURING_BAND_LOW, STRUCTURING_BAND_HIGH)
        ].sort_values("transaction_timestamp")

        if len(band) < STRUCTURING_MIN_COUNT:
            return None

        # Sliding 30-day window
        times = band["transaction_timestamp"].values
        for i in range(len(times) - STRUCTURING_MIN_COUNT + 1):
            window = pd.Timestamp(times[i + STRUCTURING_MIN_COUNT - 1]) - pd.Timestamp(times[i])
            if window.days <= 30:
                start = pd.Timestamp(times[i])
                end   = pd.Timestamp(times[i + STRUCTURING_MIN_COUNT - 1])
                intensity = float(len(band))
                return start, end, intensity

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Pass-Through Velocity (aggregated scalar feature)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _pass_through_ratio(group: pd.DataFrame) -> float:
        """
        Ratio of credits matched by near-equal debits within 24h.
        Vectorized via merge_asof for World-Class speed (Cortex-60).
        """
        # 1. Prepare credit/debit views
        credits = group[group["txn_type"] == "C"].sort_values("transaction_timestamp")
        debits  = group[group["txn_type"] == "D"].sort_values("transaction_timestamp")
        
        if credits.empty or debits.empty:
            return 0.0

        # 2. Vectorized 24h window join
        # Tolerance: amount ±10% matches classic relay behavior
        matches = pd.merge_asof(
            credits, debits,
            on="transaction_timestamp",
            direction="forward",
            tolerance=pd.Timedelta(hours=PASS_THROUGH_HOURS)
        )
        
        # Check amount similarity (vectorized)
        # Avoid division by zero
        matched_mask = (
            np.abs(matches["amount_x"] - matches["amount_y"]) / (matches["amount_x"] + 1e-5) < 0.1
        )
        matched_count = matched_mask.sum()

        return float(matched_count / (len(credits) + 1e-5))

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process an entire transaction dataframe and return per-account features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: account_id, transaction_timestamp, amount, txn_type.

        Returns
        -------
        pd.DataFrame with columns:
            account_id, temporal_avg_rest_hours, temporal_month_end_ratio,
            temporal_pass_through_ratio, temporal_burst_intensity,
            temporal_trigger_type, suspicious_start, suspicious_end
        """
        if not pd.api.types.is_datetime64_any_dtype(df["transaction_timestamp"]):
            df = df.copy()
            df["transaction_timestamp"] = pd.to_datetime(
                df["transaction_timestamp"], utc=True
            )

        log.info("Temporal feature extraction for %d transactions…", len(df))
        rows: List[dict] = []

        for acc_id, grp in df.groupby("account_id"):
            grp = grp.sort_values("transaction_timestamp")

            # ── Scalar temporal features ───────────────────────────────────
            time_gaps = grp["transaction_timestamp"].diff().dt.total_seconds() / 3600.0
            avg_rest  = float(time_gaps.mean()) if len(time_gaps) > 1 else 0.0

            days = grp["transaction_timestamp"].dt.day
            month_end_ratio = float(
                ((days >= 28) | (days <= 3)).sum() / max(len(grp), 1)
            )

            pass_thru = self._pass_through_ratio(grp)

            # ── Daily volume series for change-point triggers ──────────────
            daily_vol = (
                grp.set_index("transaction_timestamp")["amount"]
                .abs()
                .resample("D")
                .sum()
                .fillna(0)
            )

            # ── Run three independent triggers ─────────────────────────────
            # Aggregate triggers into a single collection
            triggers = [
                (self._velocity_trigger(daily_vol), "velocity_spike"),
                (self._dormancy_trigger(daily_vol), "dormancy_burst"),
                (self._structuring_trigger(grp), "structuring"),
                (self._changepoint_trigger(daily_vol), "behavior_shift")
            ]

            # Extract windows (start, end, score, name)
            windows = [(r[0][0], r[0][1], r[0][2], r[1]) for r in triggers if r[0]]

            if windows:
                # Union of all triggered windows → widest defensible range
                all_starts   = [w[0] for w in windows]
                all_ends     = [w[1] for w in windows]
                all_scores   = [w[2] for w in windows]
                trigger_names = list({w[3] for w in windows})

                susp_start = min(all_starts)
                susp_end   = max(all_ends)
                intensity  = max(all_scores)
                trigger_str = "+".join(sorted(trigger_names))
            else:
                susp_start = None
                susp_end   = None
                intensity  = 0.0
                trigger_str = "none"

            rows.append(
                {
                    "account_id":                  acc_id,
                    "temporal_avg_rest_hours":     avg_rest,
                    "temporal_month_end_ratio":    month_end_ratio,
                    "temporal_pass_through_ratio": pass_thru,
                    "temporal_burst_intensity":    intensity,
                    "temporal_trigger_type":       trigger_str,
                    "suspicious_start": susp_start.isoformat() if susp_start else None,
                    "suspicious_end":   susp_end.isoformat()   if susp_end   else None,
                }
            )

        result = pd.DataFrame(rows)
        log.info("Temporal features done: %s", result.shape)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np
    from datetime import datetime, timedelta

    rng  = np.random.default_rng(42)
    base = datetime(2024, 1, 1)

    # Account A1: dormancy from day 1-40, then burst on day 80-90
    dates  = [base + timedelta(days=i) for i in [1, 3, 5, 7, 80, 81, 82, 83, 84, 90]]
    amts   = rng.uniform(1000, 5000, 3).tolist() + rng.uniform(1000, 5000, 2).tolist() \
             + rng.uniform(45000, 55000, 5).tolist()   # structuring in burst window
    types  = ["C", "D"] * 5

    df = pd.DataFrame(
        {
            "account_id": ["A1"] * 10,
            "transaction_timestamp": dates,
            "amount": amts,
            "txn_type": types,
        }
    )

    engine = TemporalBurstEngine()
    feats  = engine.extract_features(df)
    print(feats.T)
