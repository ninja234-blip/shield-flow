"""
Shield-Flow: main.py
====================
FastAPI service that scores transactions for fraud in real time.

Endpoints:
  POST /score   — takes a transaction, returns fraud score + explanation
  GET  /health  — health check for Kubernetes

Usage:
  uvicorn services.api.main:app --port 8000
"""

import math
import os
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────
app = FastAPI(
    title="Shield-Flow Fraud Detection API",
    description="Real-time fraud scoring with XGBoost + Gemini explanations",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Load artifacts at startup
# ─────────────────────────────────────────────
print("Loading model...")
model = xgb.Booster()
model.load_model("services/model/xgb_fraud.ubj")
print("Model loaded.")

# SHAP loaded lazily on first request to avoid segfault
explainer = None

print("Loading sentence encoder...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder loaded.")

print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("fraud-cases")
print("Pinecone connected.")

# Gemini is optional — only load if key is present
gemini_client = None
try:
    from google import genai

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
        print("Gemini connected.")
    else:
        print("Gemini skipped — no API key set.")
except Exception as e:
    print(f"Gemini skipped — {e}")

# Must match exactly what was used in train.py
FEATURES = [
    "TransactionAmt",
    "log_amt",
    "hour",
    "day_of_week",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "dist1",
    "C1",
    "C2",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
    "P_emaildomain_bin",
]


# ─────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────
class Transaction(BaseModel):
    TransactionAmt: float
    hour: int = 12
    day_of_week: int = 0
    card1: float = 0
    card2: float = 0
    card3: float = 0
    card5: float = 0
    addr1: float = 0
    dist1: float = 0
    C1: float = 0
    C2: float = 0
    C4: float = 0
    C5: float = 0
    C6: float = 0
    C7: float = 0
    C8: float = 0
    C9: float = 0
    C10: float = 0
    C11: float = 0
    C13: float = 0
    C14: float = 0
    D1: float = 0
    D2: float = 0
    D3: float = 0
    D4: float = 0
    D10: float = 0
    D15: float = 0
    P_emaildomain_bin: int = 0
    P_emaildomain: str = "unknown"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "xgb_fraud_v1"}


@app.post("/score")
async def score(tx: Transaction):
    global explainer
    try:
        # ── 1. Build feature vector ──────────────────────────────────────────
        d = tx.dict()
        d["log_amt"] = math.log1p(d["TransactionAmt"])
        X = np.array([[d.get(f, 0) for f in FEATURES]], dtype=np.float32)

        # ── 2. XGBoost fraud score ───────────────────────────────────────────
        dmatrix = xgb.DMatrix(X, feature_names=FEATURES)
        prob = float(model.predict(dmatrix)[0])
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"

        # ── 3. SHAP — lazy load on first request ─────────────────────────────
        try:
            if explainer is None:
                import shap

                explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)[0]
            factors = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)[
                :4
            ]
            factor_str = ", ".join(f"{f}={v:+.3f}" for f, v in factors)
            shap_factors = [
                {"feature": f, "shap_value": round(float(v), 4)} for f, v in factors
            ]
        except Exception:
            factor_str = "unavailable"
            shap_factors = []

        # ── 4. Pinecone — find similar historical transactions ────────────────
        text = (
            f"{'FRAUD' if prob > 0.5 else 'LEGIT'} transaction "
            f"of ${tx.TransactionAmt:.2f} "
            f"email {tx.P_emaildomain} "
            f"billing region {tx.addr1}"
        )
        emb = encoder.encode([text])[0].tolist()
        hits = index.query(vector=emb, top_k=5, include_metadata=True)

        similar_cases = [h["metadata"] for h in hits["matches"]]
        similar_fraud_rate = float(np.mean([c["isFraud"] for c in similar_cases]))

        # ── 5. Gemini — AI explanation (optional) ────────────────────────────
        explanation = "Gemini explanation unavailable — quota exceeded or key not set."
        if gemini_client:
            try:
                prompt = f"""You are a fraud analyst at a bank. A transaction has been flagged.

Transaction details:
- Amount: ${tx.TransactionAmt:.2f}
- Fraud probability score: {prob:.1%}
- Risk level: {risk}
- Top model factors: {factor_str}
- Similar historical cases: {similar_fraud_rate:.0%} were confirmed fraud

Write exactly 2 sentences for the analyst reviewing this case:
1. Why this score was assigned based on the factors
2. What specific things to investigate or verify"""

                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                )
                explanation = response.text
            except Exception as e:
                explanation = f"Gemini unavailable: {str(e)[:100]}"

        # ── 6. Return response ───────────────────────────────────────────────
        return {
            "fraud_score": round(prob, 4),
            "risk_level": risk,
            "top_factors": shap_factors,
            "similar_case_fraud_rate": round(similar_fraud_rate, 3),
            "similar_cases": similar_cases[:3],
            "ai_explanation": explanation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
