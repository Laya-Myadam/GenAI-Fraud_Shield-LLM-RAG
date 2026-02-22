"""
Data preprocessing for IEEE-CIS Fraud Detection dataset.
Handles loading, merging, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(transaction_path: str, identity_path: str = None) -> pd.DataFrame:
    """Load and merge transaction + identity data."""
    print("📂 Loading transaction data...")
    df_trans = pd.read_csv(transaction_path)

    if identity_path and os.path.exists(identity_path):
        print("📂 Loading identity data...")
        df_id = pd.read_csv(identity_path)
        df = df_trans.merge(df_id, on="TransactionID", how="left")
        print(f"✅ Merged dataset shape: {df.shape}")
    else:
        df = df_trans
        print(f"✅ Transaction-only dataset shape: {df.shape}")

    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def preprocess(df: pd.DataFrame, fit: bool = True, artifacts_dir: str = "models/") -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Drop high-null columns
    - Encode categoricals
    - Fill missing values
    - Feature engineering
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # Drop columns with >75% missing
    threshold = 0.75
    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > threshold].index.tolist()
    # Always keep target
    drop_cols = [c for c in drop_cols if c != "isFraud"]
    df = df.drop(columns=drop_cols)
    print(f"🗑️  Dropped {len(drop_cols)} high-null columns")

    # Feature engineering
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        df["day"]  = (df["TransactionDT"] // 86400) % 7

    if "TransactionAmt" in df.columns:
        df["log_amt"] = np.log1p(df["TransactionAmt"])
        df["amt_cents"] = df["TransactionAmt"] % 1  # cents portion

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    if fit:
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("missing")
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        joblib.dump(encoders, os.path.join(artifacts_dir, "label_encoders.pkl"))
    else:
        encoders = joblib.load(os.path.join(artifacts_dir, "label_encoders.pkl"))
        for col in cat_cols:
            if col in encoders:
                df[col] = df[col].astype(str).fillna("missing")
                # Handle unseen labels
                le = encoders[col]
                df[col] = df[col].map(lambda x: x if x in le.classes_ else "missing")
                df[col] = le.transform(df[col])

    # Fill remaining NaNs
    df = df.fillna(-999)

    # Reduce memory
    df = reduce_memory(df)

    return df


def get_feature_matrix(df: pd.DataFrame):
    """Split into features and target."""
    target = "isFraud"
    drop = ["TransactionID", target]
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols]
    y = df[target] if target in df.columns else None
    return X, y, feature_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """Train/test split maintaining fraud ratio."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_transaction_summary(row: pd.Series) -> str:
    """
    Build a human-readable transaction summary for LLM context / RAG embedding.
    """
    parts = []

    if "TransactionAmt" in row:
        parts.append(f"Transaction amount: ${row['TransactionAmt']:.2f}")
    if "ProductCD" in row:
        parts.append(f"Product code: {row['ProductCD']}")
    if "card4" in row:
        parts.append(f"Card network: {row['card4']}")
    if "card6" in row:
        parts.append(f"Card type: {row['card6']}")
    if "P_emaildomain" in row:
        parts.append(f"Purchaser email domain: {row['P_emaildomain']}")
    if "R_emaildomain" in row:
        parts.append(f"Recipient email domain: {row['R_emaildomain']}")
    if "addr1" in row:
        parts.append(f"Billing zip (encoded): {row['addr1']}")
    if "addr2" in row:
        parts.append(f"Billing country (encoded): {row['addr2']}")
    if "hour" in row:
        parts.append(f"Transaction hour: {int(row['hour'])}:00")
    if "day" in row:
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        parts.append(f"Day of week: {days[int(row['day']) % 7]}")
    if "isFraud" in row:
        label = "FRAUD" if row["isFraud"] == 1 else "LEGITIMATE"
        parts.append(f"Label: {label}")

    return " | ".join(parts)
