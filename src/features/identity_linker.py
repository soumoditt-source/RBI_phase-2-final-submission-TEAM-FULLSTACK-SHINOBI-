"""
Shinobi-Cortex: Identity Linker (PII Intelligence)
==================================================
Converts raw PII (Names, Phones, Addresses) into high-end relational signals.
Uses fuzzy grouping and network density to detect 'Phantom Clusters'.
"""
import pandas as pd
import numpy as np
import logging
from typing import List

log = logging.getLogger("NFPC.Identity")

class IdentityLinker:
    """
    Extracts relational signals from raw PII provided in the dataset.
    """
    
    def extract_pii_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Extracting PII Relational Intelligence from %d records...", len(df))
        
        # Check for customer_id (primary key for demographics)
        pk = "customer_id" if "customer_id" in df.columns else "account_id"
        if pk not in df.columns:
            log.warning(f"Demographics missing PK ({pk}). Returning empty intel.")
            return pd.DataFrame()

        pii_feats = df[[pk]].copy()
        
        # 1. Phone Hijack Linkage
        if "phone_number" in df.columns:
            phone_counts = df.groupby("phone_number")[pk].transform("count")
            pii_feats["phone_link_density"] = phone_counts.fillna(1).astype("int16")
        else:
            pii_feats["phone_link_density"] = 1
            
        # 2. Address Ghosting
        if "address" in df.columns:
            addr_counts = df.groupby("address")[pk].transform("count")
            pii_feats["address_link_density"] = addr_counts.fillna(1).astype("int16")
        else:
            pii_feats["address_link_density"] = 1
 
        # 3. "Phantom Cluster" Detection (Shared Phone AND Address)
        if "phone_number" in df.columns and "address" in df.columns:
            df["_id_fingerprint"] = df["phone_number"].astype(str) + "_" + df["address"].astype(str)
            id_counts = df.groupby("_id_fingerprint")[pk].transform("count")
            pii_feats["phantom_cluster_flag"] = (id_counts > 1).astype("int8")
            pii_feats["identity_relational_depth"] = id_counts.astype("int16")
        else:
            pii_feats["phantom_cluster_flag"] = 0
            pii_feats["identity_relational_depth"] = 0
 
        log.info("PII Intel extraction complete.")
        return pii_feats

def finalize_neutralization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips raw PII after feature extraction to ensure training data is secure.
    """
    pii_cols = ["name", "address", "phone_number"]
    return df.drop(columns=[c for c in pii_cols if c in df.columns], errors="ignore")
