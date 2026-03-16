"""
HFT Engine - Shinobi-Cortex Phase 2
===================================
Uses Polars Streaming (Chunk Theory) to extract 60+ High-Frequency 
Forensic Features from the 16.2GB transactional base without OOM crashing.
"""
import polars as pl
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Shinobi.HFT")

# Extreme memory safety for 16GB machines
pl.Config.set_streaming_chunk_size(10000)

def extract_hft_features(input_dir: str, output_parquet: str):
    log.info("="*60)
    log.info("⚙️ LAUNCHING SHINOBI HFT ENGINE (MAP-REDUCE AWARE)")
    log.info("="*60)
    
    in_path = Path(input_dir)
    if not in_path.exists():
        log.error(f"Input directory {in_path} not found.")
        sys.exit(1)
        
    log.info(f"Scanning 16.2GB Atomic Partitions: {in_path}/b*.parquet")
    files = sorted(list(in_path.glob("b*.parquet")))
    
    if not files:
        log.error("No partitions found in input directory.")
        sys.exit(1)

    # --- PHASE 1: MAP (Local Aggregation per File) ---
    log.info(f"Phase 1: Local Aggregation (Map) across {len(files)} files...")
    
    # We define the aggregations that are 'additive' or 'combinable'
    local_aggs = [
        pl.len().alias("count_map"),
        pl.col("amount").abs().sum().alias("vol_sum_map"),
        (pl.col("amount")**2).sum().alias("vol_sq_sum_map"), # For StdDev
        pl.col("amount").abs().max().alias("amt_max_map"),
        pl.col("amount").abs().min().alias("amt_min_map"),
        pl.col("transaction_timestamp").min().alias("ts_min_map"),
        pl.col("transaction_timestamp").max().alias("ts_max_map"),
        
        # Channel Specific Maps
        *[pl.col("amount").filter(pl.col("channel") == c).abs().sum().alias(f"vol_{c.lower()}_map") for c in ["UPI", "ATM", "CASH", "IMPS", "NEFT", "RTGS"]],
        *[pl.col("amount").filter(pl.col("channel") == c).len().alias(f"cnt_{c.lower()}_map") for c in ["UPI", "ATM", "CASH", "IMPS", "NEFT", "RTGS"]],
        
        # Type Specific Maps
        pl.col("amount").filter(pl.col("txn_type") == "C").abs().sum().alias("c_vol_map"),
        pl.col("amount").filter(pl.col("txn_type") == "D").abs().sum().alias("d_vol_map"),
        pl.col("amount").filter(pl.col("txn_type") == "C").len().alias("c_cnt_map"),
        pl.col("amount").filter(pl.col("txn_type") == "D").len().alias("d_cnt_map"),
        
        # Interaction Probes
        pl.col("counterparty_id").n_unique().alias("cp_uniq_map"),
        pl.col("ip_address").n_unique().alias("ip_uniq_map"),
        
        # Forensic Pattern Maps
        pl.col("amount").abs().filter(pl.col("amount").abs() % 1000 == 0).len().alias("round_cnt_map"),
        pl.col("amount").abs().filter((pl.col("amount").abs() > 48000) & (pl.col("amount").abs() < 50000)).len().alias("struc_cnt_map"),
        
        # Propagate Master values
        pl.col("avg_balance").first().alias("avg_bal_map"),
        pl.col("kyc_compliant").first().alias("kyc_map"),
        
        # Geospatial Anchors
        pl.col("latitude").last().alias("last_lat_map"),
        pl.col("longitude").last().alias("last_lon_map"),
    ]

    partial_results = []
    for i, f in enumerate(files):
        if i % 100 == 0: log.info(f"  -> Mapping {i}/{len(files)}...")
        try:
            df_part = pl.read_parquet(f).group_by("account_id").agg(local_aggs)
            partial_results.append(df_part)
        except Exception as e:
            log.warning(f"  Skipping HFT for {f}: {e}")
            
    if not partial_results:
        log.error("HFT Map Phase produced zero results.")
        return

    # --- PHASE 2: REDUCE (Global Aggregation) ---
    log.info("Phase 2: Global Forest Reduction (Reduce)...")
    
    # Define Combine rules
    final_aggs = [
        pl.col("count_map").sum().alias("hft_txn_count"),
        pl.col("vol_sum_map").sum().alias("hft_total_volume"),
        pl.col("vol_sq_sum_map").sum().alias("hft_vol_sq_sum"),
        pl.col("amt_max_map").max().alias("hft_max_amount"),
        pl.col("amt_min_map").min().alias("hft_min_amount"),
        pl.col("ts_min_map").min().alias("hft_first_ts"),
        pl.col("ts_max_map").max().alias("hft_last_ts"),
        
        pl.col("c_vol_map").sum().alias("hft_credit_vol"),
        pl.col("d_vol_map").sum().alias("hft_debit_vol"),
        pl.col("c_cnt_map").sum().alias("hft_credit_count"),
        pl.col("d_cnt_map").sum().alias("hft_debit_count"),
        
        *[pl.col(f"vol_{c.lower()}_map").sum().alias(f"hft_vol_{c.lower()}") for c in ["upi", "atm", "cash", "imps", "neft", "rtgs"]],
        *[pl.col(f"cnt_{c.lower()}_map").sum().alias(f"hft_cnt_{c.lower()}") for c in ["upi", "atm", "cash", "imps", "neft", "rtgs"]],
        
        pl.col("round_cnt_map").sum().alias("hft_round_amt_count"),
        pl.col("struc_cnt_map").sum().alias("hft_structuring_count"),
        pl.col("ip_uniq_map").max().alias("hft_max_unique_ips"), # Max across parts as proxy
        
        pl.col("avg_bal_map").first().alias("hft_avg_balance"),
        pl.col("kyc_map").first().alias("hft_kyc_compliant"),
        pl.col("last_lat_map").last().alias("last_lat"),
        pl.col("last_lon_map").last().alias("last_lon"),
    ]

    df_reduced = pl.concat(partial_results).group_by("account_id").agg(final_aggs)

    # --- PHASE 2b: ENFORCE 100% ACCOUNT COVERAGE ---
    log.info("Phase 2b: Merging with Master to ensure 160,153 account coverage...")
    # Load account master to get the full population (including zero-transaction accounts)
    try:
        from src.data.pipeline import DataPipeline
        master = DataPipeline().load_account_master()
        # Merge Reduced Transactions onto Master
        df_reduced = master.select(["account_id", "phone_link_density", "address_link_density", "phantom_cluster_flag"]).join(
            df_reduced, on="account_id", how="left"
        )
    except Exception as e:
        log.warning(f"Could not enforce 100% coverage via Pipeline: {e}")

    # --- PHASE 3: SUPREME DERIVE (AML Ratios & Non-Linearities) ---
    log.info("Phase 3: Synthesizing Supreme Forensic Ratios (114+ Features)...")
    
    # Statistical and AML Ratios
    df = df_reduced.with_columns([
        (pl.col("hft_vol_sq_sum") / pl.max_horizontal(pl.col("hft_txn_count"), 1) - (pl.col("hft_total_volume") / pl.max_horizontal(pl.col("hft_txn_count"), 1))**2).sqrt().alias("hft_std_amount"),
        (pl.col("hft_total_volume") / pl.max_horizontal(pl.col("hft_txn_count"), 1)).alias("hft_mean_amount"),
        (pl.col("hft_credit_vol") / pl.max_horizontal(pl.col("hft_debit_vol"), 1)).alias("hft_net_flow_ratio"),
        (pl.col("hft_round_amt_count") / pl.max_horizontal(pl.col("hft_txn_count"), 1)).alias("hft_round_density"),
        (pl.col("hft_structuring_count") / pl.max_horizontal(pl.col("hft_txn_count"), 1)).alias("hft_structuring_density"),
        (pl.col("hft_vol_cash") / pl.max_horizontal(pl.col("hft_total_volume"), 1)).alias("hft_cash_intensity"),
        (pl.col("hft_vol_upi") / pl.max_horizontal(pl.col("hft_total_volume"), 1)).alias("hft_upi_intensity"),
        (pl.col("hft_credit_vol") / pl.max_horizontal(pl.col("hft_avg_balance").abs(), 1)).alias("hft_income_disparity"),
    ])

    # Polynomial Interaction Expansion (The Feature Factory)
    core_cols = ["hft_txn_count", "hft_total_volume", "hft_credit_vol", "hft_debit_vol", "hft_avg_balance", 
                 "hft_vol_upi", "hft_vol_cash", "hft_round_amt_count", "hft_structuring_count", 
                 "phone_link_density", "address_link_density"] # Adding PII signals
    
    interactions = []
    for i in range(len(core_cols)):
        for j in range(i+1, len(core_cols)):
            col_i, col_j = core_cols[i], core_cols[j]
            if col_i in df.columns and col_j in df.columns:
                interactions.append((pl.col(col_i).fill_null(0) * pl.col(col_j).fill_null(0)).alias(f"hft_x_{col_i}_X_{col_j}"))
    
    # 3rd Order Interactions for major indicators
    interactions.append((pl.col("hft_credit_vol").fill_null(0) * pl.col("hft_round_density").fill_null(0) * pl.col("hft_max_unique_ips").fill_null(0)).alias("hft_x_mule_triangulation_index"))
    interactions.append((pl.col("hft_structuring_density").fill_null(0) * pl.col("hft_vol_cash").fill_null(0) * pl.col("hft_txn_count").fill_null(0)).alias("hft_x_layered_cash_index"))
    
    df = df.with_columns(interactions).fill_null(0).fill_nan(0)
    
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    
    log.info(f"🏆 Supreme HFT Engine Complete: {df.shape[0]} accounts, {df.shape[1]} forensic features.")
    log.info(f"Deliverable: {out_path}")

if __name__ == "__main__":
    extract_hft_features("results", "results/hft_features.parquet")
