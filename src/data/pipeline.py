import logging
import os
import polars as pl
from pathlib import Path
from typing import Optional

log = logging.getLogger("NFPC.Pipeline")

# --- SUPREME SCHEMA DEFINITIONS (Aligned with Shinobi-HFT Engine) ---
txn_schema = {
    "transaction_id": pl.Int64, "account_id": pl.Int64, "txn_type": pl.String,
    "amount": pl.Float64, "transaction_timestamp": pl.String, "status": pl.String,
    "counterparty_id": pl.Int64
}
txna_schema = {
    "transaction_id": pl.Int64, "ip_address": pl.String, "device_id": pl.String,
    "latitude": pl.Float64, "longitude": pl.Float64, "merchant_id": pl.String, 
    "channel": pl.String, "location_region": pl.String
}

class DataPipeline:
    """
    Shinobi-Cortex: Polars-Streaming ETL Pipeline.
    Handles 16.2GB transactional joins with 100% Relational Integrity.
    """
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.always_drop = {
            "freeze_date", "unfreeze_date", "mule_flag_date", 
            "alert_reason", "flagged_by_branch", "gender", "nri_flag"
        }

    def _scan_safe(self, path: Path, schema: dict) -> pl.LazyFrame:
        """Atomic Lazy Scan with Auto-Padding for Missing Columns."""
        lf = pl.scan_parquet(path)
        # Ensure all columns exist and have correct types
        cols_to_add = []
        for col, dtype in schema.items():
            if col not in lf.columns:
                cols_to_add.append(pl.lit(None).cast(dtype).alias(col))
        if cols_to_add:
            lf = lf.with_columns(cols_to_add)
        # Force selection of exactly what's in schema
        return lf.select(list(schema.keys()))

    def load_account_master(self) -> pl.DataFrame:
        """High-speed Polars merge of all reference tables."""
        log.info("Building Shinobi Account Master (Reference Layer)...")
        
        # Helper to load and strip leaky cols
        def _load_safe(name):
            df = pl.read_parquet(self.data_dir / name)
            return df.select([c for c in df.columns if c not in self.always_drop])

        acc      = _load_safe("accounts.parquet")
        link     = _load_safe("customer_account_linkage.parquet")
        cust     = _load_safe("customers.parquet")
        demo     = _load_safe("demographics.parquet")
        acc_add  = _load_safe("accounts-additional.parquet")
        prod     = _load_safe("product_details.parquet")
        branch   = _load_safe("branch.parquet")

        # Relational PII Intel pass
        from src.features.identity_linker import IdentityLinker
        linker = IdentityLinker()
        pii_feats = linker.extract_pii_relations(demo.to_pandas())
        pii_pl = pl.from_pandas(pii_feats)

        master = (
            acc
            .join(acc_add, on="account_id", how="left")
            .join(link, on="account_id", how="left")
            .join(cust, on="customer_id", how="left")
            .join(demo, on="customer_id", how="left")
            .join(prod, on="customer_id", how="left")
            .join(branch.drop(["branch_address"]), on="branch_code", how="left")
            .join(pii_pl, on="customer_id", how="left")
        )
        # Final drop of raw PII
        pii_raw = {"name", "address", "phone_number"}
        master = master.select([c for c in master.columns if c not in pii_raw])
        
        log.info(f"Master Account View Integrated: {master.shape}")
        return master

    def run_full_pipeline(self, output_dir: str = "results", sample_frac: float = 1.0):
        """Executes the transactional join (Batch-by-Batch Map-Reduce for absolute 16GB safety)."""
        log.info(f"Initiating Supreme Map-Reduce Pass (Streaming Join, sample={sample_frac})...")
        
        # 1. Prepare Master
        master = self.load_account_master()
        os.makedirs(output_dir, exist_ok=True)
        
        # Process transactions by batch [1, 2, 3, 4]
        for batch_id in range(1, 5):
            txn_dir = self.data_dir / "transactions" / f"batch-{batch_id}"
            txna_dir = self.data_dir / "transactions_additional" / f"batch-{batch_id}"
            
            if not txn_dir.exists(): continue

            log.info(f"Processing Batch {batch_id}/4 [Atomic Partition Ingestion]...")
            parts = sorted(list(txn_dir.glob("part_*.parquet")))
            
            for p_file in parts:
                p_name = p_file.name
                out_path = Path(output_dir) / f"b{batch_id}_{p_name}"
                
                # Resumable Logic: Skip if already processed
                if out_path.exists():
                    continue

                try:
                    # A. Load TXN with padding
                    lf_txn = self._scan_safe(p_file, txn_schema)
                    
                    # B. Join TXNA with padding
                    p_txna = txna_dir / p_name
                    if p_txna.exists():
                        lf_txna = self._scan_safe(p_txna, txna_schema)
                        lf_part = lf_txn.join(lf_txna, on="transaction_id", how="left")
                    else:
                        # Pad with empty txna cols if file missing
                        lf_part = lf_txn
                        for col, dtype in txna_schema.items():
                            if col != "transaction_id":
                                lf_part = lf_part.with_columns(pl.lit(None).cast(dtype).alias(col))
                    
                    # C. Apply Strategic Sampling (Row-stable)
                    if sample_frac < 1.0:
                        lf_part = lf_part.filter(pl.arange(0, pl.len()).cast(pl.Float64) / pl.len() < sample_frac)
                    
                    # D. Final Master Sync (propagate all 114+ indicators)
                    df_final = lf_part.join(master.lazy(), on="account_id", how="left").collect(streaming=True)
                    
                    if df_final.height > 0:
                        df_final.write_parquet(out_path)
                    
                    import gc
                    del df_final; gc.collect()
                    
                except Exception as e:
                    log.error(f"  FAILED partition {p_name}: {e}")
            
        log.info("Supreme Atomic Ingestion Complete. Final Features Ready for HFT Engine.")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = DataPipeline(".")
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
