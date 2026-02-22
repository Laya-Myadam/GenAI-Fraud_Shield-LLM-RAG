"""
Train XGBoost on Credit Card Fraud dataset.
Also fine-tunes a small LoRA/PEFT adapter on top of a HuggingFace model
for structured fraud explanation generation.

Usage:
    python train_creditcard.py data/creditcard.csv
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
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from utils.cc_data_utils import (
    load_creditcard, preprocess_creditcard,
    split_creditcard, compute_recall_at_k
)

CC_ARTIFACTS_DIR = "models/creditcard/"
os.makedirs(CC_ARTIFACTS_DIR, exist_ok=True)


# ── XGBoost Training ───────────────────────────────────────────
def train_xgb(csv_path: str):
    print("=" * 60)
    print("🚀 Credit Card Fraud — XGBoost Training")
    print("=" * 60)

    df = load_creditcard(csv_path)
    X, y, feature_cols = preprocess_creditcard(df, fit=True)
    X_train, X_test, y_train, y_test = split_creditcard(X, y)

    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️  scale_pos_weight = {fraud_ratio:.1f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_ratio,
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

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_proba)
    cm          = confusion_matrix(y_test, y_pred).tolist()
    recall_at_k = compute_recall_at_k(y_test, y_proba, k=100)

    metrics = {
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1_score":        round(f1, 4),
        "roc_auc":         round(roc_auc, 4),
        "recall_at_100":   round(recall_at_k, 4),
        "confusion_matrix": cm,
        "train_size":      len(X_train),
        "test_size":       len(X_test),
        "fraud_rate":      round(float(y.mean()), 6),
        "n_features":      len(feature_cols),
        "dataset":         "Credit Card Fraud (284K)",
    }

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(28).to_dict()
    metrics["top_features"] = {k: round(float(v), 6) for k, v in top_features.items()}

    print(f"\n📈 Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
    print(f"📊 Recall@100: {recall_at_k:.4f}")

    joblib.dump(model, os.path.join(CC_ARTIFACTS_DIR, "xgb_model.pkl"))
    with open(os.path.join(CC_ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Model saved → {CC_ARTIFACTS_DIR}")
    return model, metrics


# ── LoRA/PEFT Fine-tuning ──────────────────────────────────────
def finetune_lora(csv_path: str, n_samples: int = 500):
    """
    Fine-tune a small causal LM with LoRA/PEFT adapters
    to generate structured fraud explanations.

    Uses distilgpt2 (tiny, CPU-friendly) as base model.
    For production, swap to mistralai/Mistral-7B-v0.1 on GPU.
    """
    print("\n" + "=" * 60)
    print("🧠 LoRA/PEFT Fine-tuning for Fraud Explanation Generation")
    print("=" * 60)

    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("Install with: pip install transformers peft datasets accelerate")
        _save_lora_stub()
        return

    # ── Build training examples ────────────────────────────────
    print("📝 Building fine-tuning examples...")
    df = pd.read_csv(csv_path)
    df_fraud = df[df["Class"] == 1].head(n_samples // 2)
    df_legit = df[df["Class"] == 0].head(n_samples // 2)
    df_sample = pd.concat([df_fraud, df_legit]).sample(frac=1, random_state=42)

    def make_example(row):
        label = "FRAUD" if row["Class"] == 1 else "LEGITIMATE"
        v_vals = [row.get(f"V{i}", 0) for i in range(1, 29)]
        anomalous = [f"V{i+1}" for i, v in enumerate(v_vals) if abs(v) > 3]
        anom_str = ", ".join(anomalous[:5]) if anomalous else "none"
        prompt = (
            f"### Transaction\n"
            f"Amount: ${row['Amount']:.2f} | Label: {label}\n"
            f"Anomalous PCA features: {anom_str}\n\n"
            f"### Fraud Risk Explanation\n"
        )
        if label == "FRAUD":
            explanation = (
                f"This transaction is flagged as HIGH RISK. "
                f"The amount of ${row['Amount']:.2f} combined with anomalous PCA features "
                f"({anom_str}) indicates unusual behavioral patterns consistent with fraud. "
                f"Recommend: BLOCK and escalate for manual review."
            )
        else:
            explanation = (
                f"This transaction appears LEGITIMATE. "
                f"The amount of ${row['Amount']:.2f} falls within normal range "
                f"with no significant PCA anomalies detected. "
                f"Recommend: APPROVE."
            )
        return {"text": prompt + explanation + "<|endoftext|>"}

    examples = [make_example(row) for _, row in df_sample.iterrows()]
    dataset = Dataset.from_list(examples)

    # ── Load base model (distilgpt2 — small, CPU friendly) ─────
    BASE_MODEL = "distilgpt2"
    print(f"📥 Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # ── LoRA config ────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"],   # GPT2 attention projection
        bias="none",
    )
    model_peft = get_peft_model(model_base, lora_config)
    model_peft.print_trainable_parameters()

    # ── Tokenize ───────────────────────────────────────────────
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.train_test_split(test_size=0.1)

    # ── Training args ──────────────────────────────────────────
    lora_output_dir = os.path.join(CC_ARTIFACTS_DIR, "lora_adapter")
    training_args = TrainingArguments(
        output_dir=lora_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model_peft,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n🏋️  Fine-tuning with LoRA/PEFT...")
    trainer.train()

    # Save adapter
    model_peft.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"✅ LoRA adapter saved → {lora_output_dir}")

    # Save metadata
    lora_meta = {
        "base_model":    BASE_MODEL,
        "lora_rank":     8,
        "lora_alpha":    32,
        "train_samples": len(examples),
        "adapter_path":  lora_output_dir,
        "status":        "trained",
    }
    with open(os.path.join(CC_ARTIFACTS_DIR, "lora_meta.json"), "w") as f:
        json.dump(lora_meta, f, indent=2)


def _save_lora_stub():
    """Save a stub so the app knows LoRA wasn't trained."""
    os.makedirs(CC_ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(CC_ARTIFACTS_DIR, "lora_meta.json"), "w") as f:
        json.dump({"status": "not_trained", "reason": "missing_dependencies"}, f)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    skip_lora = "--skip-lora" in sys.argv

    train_xgb(csv_path)

    if not skip_lora:
        finetune_lora(csv_path)
    else:
        print("\n⏭️  Skipping LoRA fine-tuning (--skip-lora flag set)")
        _save_lora_stub()
