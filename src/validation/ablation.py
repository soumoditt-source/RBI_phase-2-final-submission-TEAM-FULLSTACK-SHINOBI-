"""
Red-Herring & Bias Elimination Module.
Systematically tests if demographic features provide true predictive lift,
or if they are just correlation "traps" (Red Herrings).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

class BiasAblationTester:
    def __init__(self, red_herring_candidates: list = None):
        """
        Features that we heavily suspect the RBI dataset has poisoned 
        to test our fairness and bias logic.
        """
        self.candidates = red_herring_candidates or [
            'gender', 'age', 'rural_branch', 'nri_flag', 'pan_available'
        ]
        
    def _train_eval(self, x: pd.DataFrame, y: pd.Series) -> float:
        """Runs a fast LightGBM 3-fold CV and returns mean AUC."""
        # Convert objects to category for LightGBM
        for col in x.columns:
            if x[col].dtype == 'object':
                x[col] = x[col].astype('category')
                
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(x, y):
            x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
            x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
            
            dtrain = lgb.Dataset(x_train, label=y_train)
            dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'seed': 42
            }
            # Using generic kwargs correctly
            model = lgb.train(
                params, 
                dtrain, 
                num_boost_round=50,
                valid_sets=[dval]
            )
            # Use valid 0 by default for predictions
            preds = model.predict(x_val)
            aucs.append(roc_auc_score(y_val, preds))
            
        return np.mean(aucs)

    def run_ablation(self, df: pd.DataFrame, target_col: str = 'is_mule') -> dict:
        """
        Tests model performance with and without candidate features.
        If a feature's absence drops AUC by < 0.005, it is a Red Herring and should be dropped.
        """
        print("Running Bias & Red-Herring Ablation...")
        features = [c for c in df.columns if c != target_col and c != 'account_id']
        
        # 1. Baseline Model (All Features)
        baseline_auc = self._train_eval(df[features], df[target_col])
        print(f"Baseline AUC: {baseline_auc:.4f}")
        
        results = {}
        features_to_drop = []
        
        # 2. Ablate specific demographic candidates
        for candidate in self.candidates:
            if candidate not in features:
                continue
                
            ablated_features = [f for f in features if f != candidate]
            ablated_auc = self._train_eval(df[ablated_features], df[target_col])
            
            lift = baseline_auc - ablated_auc
            results[candidate] = {
                'ablated_auc': ablated_auc,
                'lift': lift
            }
            
            # If removing it barely hurts performance (or helps), it's a trap.
            if lift < 0.005:
                print(f"[TRAP DETECTED] Feature '{candidate}' provides negligible lift ({lift:.4f}). Dropping to prevent bias.")
                features_to_drop.append(candidate)
            else:
                print(f"[VALIDATED] Feature '{candidate}' provides real lift ({lift:.4f}). Keeping.")
                
        return {
            'baseline_auc': baseline_auc,
            'ablation_results': results,
            'features_to_drop': features_to_drop
        }

if __name__ == "__main__":
    # Test Ablation Logic
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        'account_id': [f'A{i}' for i in range(n)],
        # Real feature: high correlation
        'graph_pagerank': rng.uniform(0, 1, n),
        # Red herring: perfectly correlates but we suspect it's a trap
        'gender': rng.choice(['M', 'F'], n),
        # Target
        'is_mule': rng.choice([0, 1], n, p=[0.9, 0.1])
    })
    
    # Force some correlation on the red herring to see if logic holds
    df.loc[df['is_mule'] == 1, 'gender'] = 'M'
    df.loc[df['is_mule'] == 1, 'graph_pagerank'] += 0.5
    
    tester = BiasAblationTester(red_herring_candidates=['gender', 'age'])
    res = tester.run_ablation(df)
    print("Features to neutralize:", res['features_to_drop'])
