"""
Inference engine: combines XGBoost predictions + FAISS RAG + Groq LLM explanation.
Used by the Streamlit UI.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Optional

from utils.data_utils import preprocess, get_feature_matrix, build_transaction_summary
from utils.vectorstore_utils import load_vectorstore, retrieve_similar
from utils.llm_chain import explain_transaction, generate_risk_summary

ARTIFACTS_DIR   = "models/"
VECTORSTORE_DIR = "vectorstore/"


class FraudDetectionEngine:
    def __init__(self):
        self.model         = None
        self.feature_cols  = None
        self.metrics       = None
        self.faiss_index   = None
        self.embedder      = None
        self.vs_metadata   = None
        self._model_loaded = False
        self._vs_loaded    = False

    def load_model(self):
        """Load XGBoost model and feature metadata."""
        model_path   = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
        feat_path    = os.path.join(ARTIFACTS_DIR, "feature_cols.pkl")
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run train_model.py first."
            )

        self.model        = joblib.load(model_path)
        self.feature_cols = joblib.load(feat_path)
        with open(metrics_path) as f:
            self.metrics = json.load(f)

        self._model_loaded = True
        print("✅ ML model loaded")

    def load_vectorstore(self):
        """Load FAISS vector store for RAG."""
        try:
            self.faiss_index, self.embedder, self.vs_metadata = load_vectorstore()
            self._vs_loaded = True
            print("✅ Vector store loaded")
        except Exception as e:
            print(f"⚠️  Vector store not available: {e}")

    def load_all(self):
        self.load_model()
        self.load_vectorstore()

    def predict_single(self, row_dict: dict) -> dict:
        """
        Predict fraud probability for a single transaction dict.
        Returns: {fraud_prob, prediction, risk_level, features_used}
        """
        df = pd.DataFrame([row_dict])
        df = preprocess(df, fit=False, artifacts_dir=ARTIFACTS_DIR)

        # Align feature columns
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = -999
        df = df[self.feature_cols]

        fraud_prob   = float(self.model.predict_proba(df)[0][1])
        prediction   = int(fraud_prob >= 0.5)
        risk_level   = (
            "HIGH"   if fraud_prob >= 0.7
            else "MEDIUM" if fraud_prob >= 0.4
            else "LOW"
        )

        return {
            "fraud_prob": fraud_prob,
            "prediction": prediction,
            "risk_level": risk_level,
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud probability for a DataFrame of raw transactions."""
        df = df_raw.copy()
        df_proc = preprocess(df, fit=False, artifacts_dir=ARTIFACTS_DIR)

        for col in self.feature_cols:
            if col not in df_proc.columns:
                df_proc[col] = -999
        df_proc = df_proc[self.feature_cols]

        probs = self.model.predict_proba(df_proc)[:, 1]
        preds = (probs >= 0.5).astype(int)
        risk  = pd.cut(
            probs,
            bins=[-0.01, 0.4, 0.7, 1.01],
            labels=["LOW", "MEDIUM", "HIGH"],
        )

        result = df_raw.copy()
        result["fraud_probability"] = probs
        result["prediction"]        = preds
        result["risk_level"]        = risk
        return result

    def full_analysis(
        self,
        row_dict: dict,
        use_llm: bool = True,
        k_similar: int = 5,
    ) -> dict:
        """
        Full pipeline: ML prediction + RAG retrieval + LLM explanation.

        Args:
            row_dict: Raw transaction dict (before preprocessing)
            use_llm:  Whether to call the LLM for explanation
            k_similar: Number of similar cases to retrieve

        Returns dict with all analysis results.
        """
        # 1. Build human-readable summary (uses raw data)
        row_series = pd.Series(row_dict)
        if "TransactionDT" in row_dict:
            row_series["hour"] = (row_dict["TransactionDT"] // 3600) % 24
            row_series["day"]  = (row_dict["TransactionDT"] // 86400) % 7

        txn_summary = build_transaction_summary(row_series)

        # 2. ML prediction
        ml_result = self.predict_single(row_dict)
        fraud_prob = ml_result["fraud_prob"]

        # 3. RAG retrieval
        similar_cases = []
        if self._vs_loaded:
            similar_cases = retrieve_similar(
                txn_summary, self.faiss_index, self.embedder, self.vs_metadata, k=k_similar
            )

        # 4. LLM explanation
        llm_explanation = None
        if use_llm:
            try:
                llm_explanation = explain_transaction(txn_summary, similar_cases, fraud_prob)
            except Exception as e:
                llm_explanation = f"⚠️ LLM explanation unavailable: {e}"

        return {
            "transaction_summary": txn_summary,
            "fraud_probability":   fraud_prob,
            "prediction":          ml_result["prediction"],
            "risk_level":          ml_result["risk_level"],
            "similar_cases":       similar_cases,
            "llm_explanation":     llm_explanation,
        }


# Singleton for Streamlit (cached)
_engine = None

def get_engine() -> FraudDetectionEngine:
    global _engine
    if _engine is None:
        _engine = FraudDetectionEngine()
    return _engine
