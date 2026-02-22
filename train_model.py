"""
Train XGBoost fraud detection model on IEEE-CIS dataset.
Saves model + metrics for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold

from utils.data_utils import load_data, preprocess, get_feature_matrix, split_data

ARTIFACTS_DIR = "models/"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train(transaction_path: str, identity_path: str = None):
    print("=" * 60)
    print("🚀 IEEE-CIS Fraud Detection — Model Training")
    print("=" * 60)

    # ── Load & preprocess ──────────────────────────────────────
    df = load_data(transaction_path, identity_path)
    df = preprocess(df, fit=True, artifacts_dir=ARTIFACTS_DIR)
    X, y, feature_cols = get_feature_matrix(df)

    # Save feature columns for inference
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_cols.pkl"))
    print(f"📊 Features: {len(feature_cols)} | Fraud rate: {y.mean()*100:.2f}%")

    # ── Train / test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── XGBoost (handles imbalance via scale_pos_weight) ───────
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️  scale_pos_weight = {fraud_ratio:.1f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_ratio,
        use_label_encoder=False,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\n🏋️  Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    # ── Evaluate ───────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(roc_auc, 4),
        "confusion_matrix": cm,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "fraud_rate": round(float(y.mean()), 4),
        "n_features": len(feature_cols),
    }

    print("\n📈 EVALUATION RESULTS:")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ── Feature importance (top 30) ────────────────────────────
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(30).to_dict()
    metrics["top_features"] = {k: round(float(v), 6) for k, v in top_features.items()}

    # ── Save artifacts ─────────────────────────────────────────
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Model saved to {ARTIFACTS_DIR}xgb_model.pkl")
    print(f"✅ Metrics saved to {ARTIFACTS_DIR}metrics.json")
    return model, metrics


if __name__ == "__main__":
    import sys
    t_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_transaction.csv"
    i_path = sys.argv[2] if len(sys.argv) > 2 else "data/train_identity.csv"
    train(t_path, i_path)
