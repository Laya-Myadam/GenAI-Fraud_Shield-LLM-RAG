"""
Standalone script to build the FAISS vector store.
Run AFTER train_model.py.

Usage:
    python build_vectorstore.py data/train_transaction.csv data/train_identity.csv
"""

import sys
from utils.vectorstore_utils import build_vectorstore

if __name__ == "__main__":
    t_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_transaction.csv"
    i_path = sys.argv[2] if len(sys.argv) > 2 else "data/train_identity.csv"
    build_vectorstore(t_path, i_path, n_samples=5000, fraud_oversample=0.5)
