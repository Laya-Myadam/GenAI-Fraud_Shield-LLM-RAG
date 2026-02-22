"""
Preprocessing for Credit Card Fraud Detection dataset (284K records).
Features: Time, V1-V28 (PCA), Amount, Class
Simple single-CSV pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

CC_ARTIFACTS_DIR = "models/creditcard/"


def load_creditcard(csv_path: str) -> pd.DataFrame:
    print("📂 Loading Credit Card Fraud dataset...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} transactions | Fraud rate: {df['Class'].mean()*100:.3f}%")
    return df


def preprocess_creditcard(df: pd.DataFrame, fit: bool = True) -> tuple:
    """
    Preprocess credit card dataset.
    - Scale Amount and Time
    - Return X, y, feature_cols
    """
    os.makedirs(CC_ARTIFACTS_DIR, exist_ok=True)

    df = df.copy()

    # Feature engineering
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # Scale Amount and Time (V1-V28 are already PCA scaled)
    scale_cols = ["Amount", "Time", "log_amount"]

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        joblib.dump(scaler, os.path.join(CC_ARTIFACTS_DIR, "scaler.pkl"))
    else:
        scaler = joblib.load(os.path.join(CC_ARTIFACTS_DIR, "scaler.pkl"))
        df[scale_cols] = scaler.transform(df[scale_cols])

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"] if "Class" in df.columns else None

    if fit:
        joblib.dump(feature_cols, os.path.join(CC_ARTIFACTS_DIR, "feature_cols.pkl"))

    return X, y, feature_cols


def split_creditcard(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_cc_transaction_summary(row: pd.Series) -> str:
    """
    Build human-readable summary for RAG embedding.
    V1-V28 are PCA features — we summarize statistically.
    """
    parts = []

    if "Amount" in row:
        parts.append(f"Transaction amount: ${row['Amount']:.2f}")
    if "hour_of_day" in row:
        parts.append(f"Hour of day: {int(row['hour_of_day'])}:00")
    elif "Time" in row:
        hour = int((row["Time"] // 3600) % 24)
        parts.append(f"Hour of day: {hour}:00")

    # Summarize PCA features in buckets
    v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in row.index]
    if v_cols:
        v_vals = [row[c] for c in v_cols]
        v_mean = float(np.mean(v_vals))
        v_std  = float(np.std(v_vals))
        v_min  = float(np.min(v_vals))
        v_max  = float(np.max(v_vals))
        parts.append(f"PCA feature profile: mean={v_mean:.3f}, std={v_std:.3f}, min={v_min:.3f}, max={v_max:.3f}")

        # Flag anomalous PCA features (abs > 3)
        anomalous = [v_cols[i] for i, v in enumerate(v_vals) if abs(v) > 3]
        if anomalous:
            parts.append(f"Anomalous PCA features: {', '.join(anomalous[:5])}")

    if "Class" in row:
        label = "FRAUD" if row["Class"] == 1 else "LEGITIMATE"
        parts.append(f"Label: {label}")

    return " | ".join(parts)


def compute_recall_at_k(y_true, y_scores, k: int = 100) -> float:
    """Compute Recall@K — fraction of actual fraud in top-K scored transactions."""
    df = pd.DataFrame({"score": y_scores, "label": y_true})
    top_k = df.nlargest(k, "score")
    recall_at_k = top_k["label"].sum() / max(y_true.sum(), 1)
    return float(recall_at_k)