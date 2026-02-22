"""
Build FAISS vector store from fraud transaction summaries.
Used for RAG — retrieve similar past fraud cases to provide LLM context.
"""

import pandas as pd
import numpy as np
import faiss
import joblib
import os
import json
from sentence_transformers import SentenceTransformer

from utils.data_utils import load_data, preprocess, build_transaction_summary

VECTORSTORE_DIR = "vectorstore/"
ARTIFACTS_DIR   = "models/"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # small, fast, runs on CPU


def build_vectorstore(
    transaction_path: str,
    identity_path: str = None,
    n_samples: int = 5000,          # how many transactions to index
    fraud_oversample: float = 0.5,  # fraction of fraud samples
):
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    print("=" * 60)
    print("🔨 Building FAISS Vector Store for RAG")
    print("=" * 60)

    # ── Load raw (pre-encoded) data for readable summaries ─────
    print("📂 Loading raw data for summaries...")
    df_trans = pd.read_csv(transaction_path)
    if identity_path and os.path.exists(identity_path):
        df_id = pd.read_csv(identity_path)
        df_raw = df_trans.merge(df_id, on="TransactionID", how="left")
    else:
        df_raw = df_trans

    # ── Sample balanced fraud/legit for indexing ───────────────
    df_fraud  = df_raw[df_raw["isFraud"] == 1]
    df_legit  = df_raw[df_raw["isFraud"] == 0]

    n_fraud = int(n_samples * fraud_oversample)
    n_legit = n_samples - n_fraud

    n_fraud = min(n_fraud, len(df_fraud))
    n_legit = min(n_legit, len(df_legit))

    df_sample = pd.concat([
        df_fraud.sample(n_fraud, random_state=42),
        df_legit.sample(n_legit, random_state=42),
    ]).reset_index(drop=True)

    print(f"📊 Indexing {n_fraud} fraud + {n_legit} legit = {len(df_sample)} transactions")

    # ── Feature engineering for summaries ─────────────────────
    if "TransactionDT" in df_sample.columns:
        df_sample["hour"] = (df_sample["TransactionDT"] // 3600) % 24
        df_sample["day"]  = (df_sample["TransactionDT"] // 86400) % 7

    # ── Build text summaries ───────────────────────────────────
    print("📝 Building transaction summaries...")
    summaries = [
        build_transaction_summary(row)
        for _, row in df_sample.iterrows()
    ]

    # ── Embed with SentenceTransformer ─────────────────────────
    print(f"🧠 Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("⚙️  Generating embeddings (this may take a minute)...")
    embeddings = embedder.encode(summaries, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # ── Build FAISS index ──────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── Save everything ────────────────────────────────────────
    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "fraud_index.faiss"))
    joblib.dump(embedder, os.path.join(VECTORSTORE_DIR, "embedder.pkl"))

    # Save metadata (summaries + labels)
    metadata = [
        {
            "summary": summaries[i],
            "label": int(df_sample.iloc[i]["isFraud"]),
            "amount": float(df_sample.iloc[i].get("TransactionAmt", 0)),
            "transaction_id": str(df_sample.iloc[i].get("TransactionID", i)),
        }
        for i in range(len(df_sample))
    ]
    with open(os.path.join(VECTORSTORE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved to {VECTORSTORE_DIR}")
    return index, embedder, metadata


def load_vectorstore():
    """Load FAISS index, embedder, and metadata."""
    index    = faiss.read_index(os.path.join(VECTORSTORE_DIR, "fraud_index.faiss"))
    embedder = joblib.load(os.path.join(VECTORSTORE_DIR, "embedder.pkl"))
    with open(os.path.join(VECTORSTORE_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return index, embedder, metadata


def retrieve_similar(query_summary: str, index, embedder, metadata, k: int = 5):
    """Retrieve k most similar transactions from FAISS."""
    query_vec = embedder.encode([query_summary], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        item = metadata[idx].copy()
        item["distance"] = float(dist)
        results.append(item)
    return results


if __name__ == "__main__":
    import sys
    t_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_transaction.csv"
    i_path = sys.argv[2] if len(sys.argv) > 2 else "data/train_identity.csv"
    build_vectorstore(t_path, i_path)
