"""
Inference engine for Credit Card Fraud dataset.
Combines XGBoost + FAISS RAG + LoRA/Groq LLM.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import faiss

from utils.cc_data_utils import (
    preprocess_creditcard, build_cc_transaction_summary, compute_recall_at_k
)
from utils.cc_llm_chain import explain_cc_transaction, get_lora_status

CC_ARTIFACTS_DIR = "models/creditcard/"
CC_VS_DIR        = "vectorstore/creditcard/"


class CCFraudEngine:
    def __init__(self):
        self.model        = None
        self.feature_cols = None
        self.metrics      = None
        self.faiss_index  = None
        self.embedder     = None
        self.vs_metadata  = None
        self._ready       = False

    def load_all(self):
        model_path = os.path.join(CC_ARTIFACTS_DIR, "xgb_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Credit Card model not found. Run: python train_creditcard.py data/creditcard.csv"
            )

        self.model        = joblib.load(model_path)
        self.feature_cols = joblib.load(os.path.join(CC_ARTIFACTS_DIR, "feature_cols.pkl"))
        with open(os.path.join(CC_ARTIFACTS_DIR, "metrics.json")) as f:
            self.metrics = json.load(f)

        # Load vector store
        try:
            vs_index_path = os.path.join(CC_VS_DIR, "fraud_index.faiss")
            if os.path.exists(vs_index_path):
                self.faiss_index = faiss.read_index(vs_index_path)
                self.embedder    = joblib.load(os.path.join(CC_VS_DIR, "embedder.pkl"))
                with open(os.path.join(CC_VS_DIR, "metadata.json")) as f:
                    self.vs_metadata = json.load(f)
                print("✅ CC vector store loaded")
        except Exception as e:
            print(f"⚠️  CC vector store unavailable: {e}")

        self._ready = True
        print("✅ Credit Card model loaded")

    def predict_single(self, row_dict: dict) -> dict:
        df = pd.DataFrame([row_dict])
        df, _, _ = preprocess_creditcard(df, fit=False)

        # Align features
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_cols]

        fraud_prob = float(self.model.predict_proba(df)[0][1])
        prediction = int(fraud_prob >= 0.5)
        risk_level = (
            "HIGH"   if fraud_prob >= 0.7
            else "MEDIUM" if fraud_prob >= 0.4
            else "LOW"
        )
        return {"fraud_prob": fraud_prob, "prediction": prediction, "risk_level": risk_level}

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        X, _, _ = preprocess_creditcard(df, fit=False)

        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols]

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        risk  = pd.cut(probs, bins=[-0.01, 0.4, 0.7, 1.01], labels=["LOW", "MEDIUM", "HIGH"])

        result = df_raw.copy()
        result["fraud_probability"] = probs
        result["prediction"]        = preds
        result["risk_level"]        = risk
        return result

    def retrieve_similar(self, summary: str, k: int = 5) -> list:
        if not self.faiss_index or not self.embedder:
            return []
        vec = self.embedder.encode([summary], convert_to_numpy=True).astype("float32")
        distances, indices = self.faiss_index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            item = self.vs_metadata[idx].copy()
            item["distance"] = float(dist)
            results.append(item)
        return results

    def full_analysis(self, row_dict: dict, use_llm: bool = True, k_similar: int = 5) -> dict:
        row_series = pd.Series(row_dict)
        if "Time" in row_dict:
            row_series["hour_of_day"] = int((row_dict["Time"] // 3600) % 24)

        txn_summary = build_cc_transaction_summary(row_series)
        ml_result   = self.predict_single(row_dict)
        fraud_prob  = ml_result["fraud_prob"]

        similar_cases = self.retrieve_similar(txn_summary, k=k_similar)

        llm_explanation = None
        llm_source      = None
        if use_llm:
            try:
                llm_explanation, llm_source = explain_cc_transaction(
                    txn_summary, similar_cases, fraud_prob
                )
            except Exception as e:
                llm_explanation = f"⚠️ LLM explanation unavailable: {e}"
                llm_source = "error"

        lora_status = get_lora_status()

        return {
            "transaction_summary": txn_summary,
            "fraud_probability":   fraud_prob,
            "prediction":          ml_result["prediction"],
            "risk_level":          ml_result["risk_level"],
            "similar_cases":       similar_cases,
            "llm_explanation":     llm_explanation,
            "llm_source":          llm_source,
            "lora_status":         lora_status,
        }


_cc_engine = None

def get_cc_engine() -> CCFraudEngine:
    global _cc_engine
    if _cc_engine is None:
        _cc_engine = CCFraudEngine()
    return _cc_engine
