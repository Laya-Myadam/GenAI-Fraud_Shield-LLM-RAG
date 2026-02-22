"""
Build FAISS vector store for Credit Card Fraud dataset.
Run AFTER train_creditcard.py.

Usage:
    python build_vectorstore_cc.py data/creditcard.csv
"""

import pandas as pd
import numpy as np
import faiss
import joblib
import os
import json
import sys
from sentence_transformers import SentenceTransformer

from utils.cc_data_utils import load_creditcard, build_cc_transaction_summary

CC_VS_DIR   = "vectorstore/creditcard/"
EMBED_MODEL = "all-MiniLM-L6-v2"


def build_cc_vectorstore(csv_path: str, n_samples: int = 5000, fraud_oversample: float = 0.5):
    os.makedirs(CC_VS_DIR, exist_ok=True)

    print("=" * 60)
    print("🔨 Building Credit Card FAISS Vector Store")
    print("=" * 60)

    df = load_creditcard(csv_path)

    # Feature engineering for summaries
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # Balanced sample
    df_fraud = df[df["Class"] == 1]
    df_legit = df[df["Class"] == 0]
    n_fraud  = min(int(n_samples * fraud_oversample), len(df_fraud))
    n_legit  = min(n_samples - n_fraud, len(df_legit))

    df_sample = pd.concat([
        df_fraud.sample(n_fraud, random_state=42),
        df_legit.sample(n_legit, random_state=42),
    ]).reset_index(drop=True)

    print(f"📊 Indexing {n_fraud} fraud + {n_legit} legit = {len(df_sample)} transactions")

    # Build summaries
    print("📝 Building summaries...")
    summaries = [build_cc_transaction_summary(row) for _, row in df_sample.iterrows()]

    # Embed
    print(f"🧠 Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("⚙️  Generating embeddings...")
    embeddings = embedder.encode(summaries, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index: {index.ntotal} vectors, dim={dim}")

    # Save
    faiss.write_index(index, os.path.join(CC_VS_DIR, "fraud_index.faiss"))
    joblib.dump(embedder, os.path.join(CC_VS_DIR, "embedder.pkl"))

    metadata = [
        {
            "summary":  summaries[i],
            "label":    int(df_sample.iloc[i]["Class"]),
            "amount":   float(df_sample.iloc[i]["Amount"]),
            "hour":     int(df_sample.iloc[i]["hour_of_day"]),
        }
        for i in range(len(df_sample))
    ]
    with open(os.path.join(CC_VS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved to {CC_VS_DIR}")
    return index, embedder, metadata


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    build_cc_vectorstore(csv_path)
