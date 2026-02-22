"""
LLM chain for Credit Card Fraud dataset.
Uses Groq API for explanation + optionally the LoRA adapter for local generation.
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CC_ARTIFACTS_DIR = "models/creditcard/"


def get_lora_status() -> dict:
    meta_path = os.path.join(CC_ARTIFACTS_DIR, "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"status": "not_trained"}


def generate_with_lora(transaction_summary: str) -> str:
    """Generate explanation using fine-tuned LoRA adapter."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        meta = get_lora_status()
        if meta.get("status") != "trained":
            return None

        adapter_path = meta["adapter_path"]
        base_model   = meta["base_model"]

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model_base = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model_base, adapter_path)
        model.eval()

        prompt = f"### Transaction\n{transaction_summary}\n\n### Fraud Risk Explanation\n"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the explanation part
        if "### Fraud Risk Explanation" in generated:
            return generated.split("### Fraud Risk Explanation")[-1].strip()
        return generated

    except Exception as e:
        return None


CC_SYSTEM_PROMPT = """You are an expert financial fraud analyst specializing in credit card fraud detection.
You analyze transactions with PCA-anonymized features (V1-V28) plus Amount and Time.
Provide clear, structured, and actionable fraud risk assessments.
Be precise and factual — only reference data you are given."""


def build_cc_rag_prompt(transaction_summary: str, similar_cases: list, fraud_prob: float) -> str:
    fraud_cases = [c for c in similar_cases if c["label"] == 1]
    legit_cases = [c for c in similar_cases if c["label"] == 0]

    risk_level = (
        "🔴 HIGH RISK"   if fraud_prob >= 0.7
        else "🟡 MEDIUM RISK" if fraud_prob >= 0.4
        else "🟢 LOW RISK"
    )

    context_parts = ["### Similar Historical Cases (RAG Retrieved):\n"]
    if fraud_cases:
        context_parts.append("**Matched FRAUD patterns:**")
        for i, c in enumerate(fraud_cases[:3], 1):
            context_parts.append(f"  {i}. {c['summary']}")
    if legit_cases:
        context_parts.append("\n**Matched LEGITIMATE patterns:**")
        for i, c in enumerate(legit_cases[:2], 1):
            context_parts.append(f"  {i}. {c['summary']}")

    context_str = "\n".join(context_parts)

    return f"""
## Credit Card Transaction Under Review
{transaction_summary}

**ML Model Fraud Probability: {fraud_prob*100:.1f}% ({risk_level})**

{context_str}

---

Provide a structured fraud risk assessment:

### 1. 🔍 Risk Analysis
Explain what makes this transaction suspicious or normal.
Focus on Amount, Time-of-day, and which PCA features are anomalous.

### 2. 📊 Pattern Comparison
Compare to the retrieved historical cases above.
What fraud patterns match? What suggests legitimacy?

### 3. ⚠️ Key Risk Indicators
List 3-5 specific risk factors from this transaction's data.

### 4. ✅ Analyst Recommendation
- Decision: APPROVE / FLAG FOR REVIEW / BLOCK
- Confidence level (%)
- Any further verification steps

### 5. 📝 Executive Summary
One concise sentence suitable for a fraud operations dashboard.
"""


def explain_cc_transaction(
    transaction_summary: str,
    similar_cases: list,
    fraud_prob: float,
    use_lora_first: bool = True,
) -> tuple:
    """
    Generate fraud explanation.
    Tries LoRA adapter first (if trained), falls back to Groq API.
    Returns: (explanation_text, source) where source is 'lora' or 'groq'
    """
    # Try LoRA first
    if use_lora_first:
        lora_result = generate_with_lora(transaction_summary)
        if lora_result:
            return lora_result, "lora"

    # Fall back to Groq
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = build_cc_rag_prompt(transaction_summary, similar_cases, fraud_prob)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": CC_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content, "groq"
