# ⚙️ TECHNICAL ARCHITECTURE: National Fraud Prevention Challenge 2026 Phase 2

**Team**: FullStack Shinobi  

This document outlines the elite-grade technological stack and mathematical framework built to conquer the RBI National Fraud Prevention Challenge.

---

## 1. The 17.2GB Ingestion Engine (Polars Map-Reduce)

Traditional Pandas dataframes crash instantly on 17.2GB of highly-skewed transaction data. We engineerd a **Streaming Map-Reduce Pipeline** using `polars` and `pyarrow`.

- **Chunked Scanning**: Transactions are scanned in 5-million row chunks.
- **Identity Linkage (The PII Cross-Reference)**: We parse `demographics.parquet` to build an `IdentityLinker`. If multiple seemingly un-connected accounts share an encrypted `phone_number` or `address` hash, they are tagged as a **Phantom Cluster**.
- **The Output**: Exactly 160,153 accounts (96k Train, 64k Test) are deterministically parsed with **zero lost records**.

## 2. High-Frequency Trading (HFT) Feature Extraction

We extracted 54 distinct mathematical vectors from the raw ledger, including:

- **`hft_structuring_density`**: The ratio of small, rapid bursts of transactions just below PAN/KYC reporting thresholds (₹49,000).
- **`hft_round_density`**: Detection of unnaturally perfectly-rounded transfer amounts (e.g., ₹50,000.00 exact).
- **`hft_income_disparity`**: Comparing historic average balance to sudden, catastrophic inward and outward flows (Burn-and-Churn accounts).
- **Velocity Vectors**: Temporal measurement of funds entering and exiting an account within < 5 minutes (indicating a pure transit / smurf role).

## 3. Machine Learning Ensemble (The "AML Brain")

The competition is evaluated on AUC-ROC and F1 score against a severe class imbalance (few real mules vs. massive normal accounts), deliberately seeded with Red Herring patterns.

- **Resampling**: We utilize **SMOTE-NC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous) to mathematically balance the training set before the models ever see it, preventing minority-class starvation.
- **The 7-Model Blend**:
  1. XGBoost (Extreme Gradient Boosting)
  2. LightGBM (For hyper-fast, leaf-wise tree splits on 160k rows)
  3. CatBoost (Superior handling of categorical encodings without one-hot explosion)
  4. Random Forest (For low-variance, generalized boundaries)
  5. Extra Trees (For aggressive feature randomization)
  6. Logistic Regression (ElasticNet baseline)
  7. Gaussian Naive Bayes (Temporal distribution baseline)
- **Venn-Abers Calibration**: The ensemble outputs are heavily calibrated. Instead of raw probability (which tree models distort), we output mathematically bound calibration probabilities ensuring the judges see true certainty, not just model confidence.

## 4. Visual Topology & Geo-Spatial Intelligence

The Streamlit FIU (Financial Intelligence Unit) dashboard is not just for show; it is an analyst-grade forensic tool.

- **The Money Chain (`rebuild_money_chain_v3.py`)**:
  - Pure Polars aggregation scanning all 54,648 account pairs.
  - Top 150 Hubs identified by flow-volume Centrality (`In_Volume + Out_Volume`).
  - Automatically tags accounts as `MULE_HUB` (Red), `COLLECTOR` (Orange), or `SMURFER` (Yellow) based on graph centrality percentile.
- **Geo-Intelligence (PyDeck)**:
  - Utilizes 3D WebGL via PyDeck to map the geospatial transfer arcs (latitude/longitude extracted from `recent_transactions`).
  - Features 65-degree pitch fly-overs and "Cinematic Glow" color blending to track money leaving rural/tier-3 cities and converging on coastal/metropolitan exit nodes.

---

### Conclusion

This architecture is built for survival and scale. It respects the 17.2GB data constraint, leverages cutting-edge graph topology (bypassing buggy Python wrappers for pure Polars logic), and produces a calibrated, red-herring-resistant probability score for every required test account.
