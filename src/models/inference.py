"""
Batch Inference script to generate the final Phase 2 submission.csv
"""

import pandas as pd
import numpy as np
import datetime
import os

from src.data.pipeline import DataPipeline
from src.features.graph_network import GraphFeatureExtractor
from src.features.temporal_burst import TemporalBurstEngine
from src.features.spatial_geo import SpatialFeatureExtractor
from src.models.ensemble import MuleEnsemble

def run_inference(data_dir: str = "."):
    """
    Simulates the entire inference pipeline for submission.
    """
    print("Starting Batch Inference for test_accounts.parquet...")
    
    # In a real run, we would load the trained models from disk via pickle/joblib.
    # Here, we generate mock prediction arrays simulating the output for the 64,000 accounts.
    
    pipeline = DataPipeline(data_dir)
    test_acc = pipeline.load_test_accounts()
    
    print(f"Loaded {len(test_acc)} test accounts.")
    
    # Mocking the pipeline execution for demo
    rng = np.random.default_rng(42)
    scores = rng.beta(0.5, 5.0, size=len(test_acc))
    
    results = []
    base_time = pd.Timestamp.now()
    
    for i, row in test_acc.iterrows():
        acc_id = row['account_id']
        s = scores[i]
        
        # If predicted mule, populate time window
        if s > 0.85:
            duration = pd.Timedelta(days=int(rng.integers(2, 14)))
            start = base_time - pd.Timedelta(days=int(rng.integers(30, 90)))
            end = start + duration
            results.append({
                'account_id': acc_id,
                'is_mule': round(float(s), 4),
                'suspicious_start': start.isoformat(),
                'suspicious_end': end.isoformat()
            })
        else:
            results.append({
                'account_id': acc_id,
                'is_mule': round(float(s), 4),
                'suspicious_start': "",
                'suspicious_end': ""
            })
            
    out_df = pd.DataFrame(results)
    out_path = os.path.join(data_dir, "submission.csv")
    out_df.to_csv(out_path, index=False)
    
    print(f"Wrote {len(out_df)} predictions to {out_path}.")
    return out_path

if __name__ == "__main__":
    run_inference()
