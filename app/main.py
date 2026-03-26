"""
FastAPI Backend Endpoint for Mule Detection.
High-throughput REST API simulating 100ms inference for Production.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from datetime import datetime
import uvicorn

app = FastAPI(title="RBI MuleHunter.AI Phase 2 API")

class TransactionInput(BaseModel):
    account_id: str
    amount: float
    counterparty_id: str
    channel: str

class PredictionResponse(BaseModel):
    account_id: str
    is_mule_prob: float
    bounds_lower: float
    bounds_upper: float
    suspicious_start: str
    suspicious_end: str
    flagged: bool

@app.get("/")
def health_check():
    return {"status": "ok", "version": "v1.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict_mule(txn: TransactionInput):
    """
    Real-time inference endpoint for incoming transactions.
    """
    # Simulate an ultra-low latency response based on the Triple-Layer features
    prob = random.uniform(0.01, 0.99)
    flagged = prob > 0.85
    
    # Venn-Abers calibration bounds simulated
    lower = max(0.0, prob - random.uniform(0.02, 0.05))
    upper = min(1.0, prob + random.uniform(0.02, 0.05))
    
    now = datetime.now()
    if flagged:
        # Simulate centered burst window
        start = (now - pd.Timedelta(days=random.randint(10, 20))).isoformat()
        end = now.isoformat()
    else:
        start, end = "", ""
        
    return PredictionResponse(
        account_id=txn.account_id,
        is_mule_prob=round(prob, 4),
        bounds_lower=round(lower, 4),
        bounds_upper=round(upper, 4),
        suspicious_start=start,
        suspicious_end=end,
        flagged=flagged
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
