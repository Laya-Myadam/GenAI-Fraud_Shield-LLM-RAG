"""
LangChain RAG pipeline using Groq LLM.
Given a transaction + similar cases, generates:
  - Fraud risk explanation
  - Risk summary
  - Recommended analyst action
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file")
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are an expert financial fraud analyst AI assistant. 
Your job is to analyze transaction data and provide clear, structured fraud risk assessments.
Always be precise, factual, and actionable. Format your response in clear sections.
Never fabricate data — only use what is provided."""


def build_rag_prompt(transaction_summary: str, similar_cases: list, fraud_prob: float) -> str:
    """Build the RAG-augmented prompt for the LLM."""

    # Format similar cases as context
    fraud_cases  = [c for c in similar_cases if c["label"] == 1]
    legit_cases  = [c for c in similar_cases if c["label"] == 0]

    context_parts = ["### Similar Historical Cases Retrieved:\n"]

    if fraud_cases:
        context_parts.append("**Known FRAUD cases (similar pattern):**")
        for i, c in enumerate(fraud_cases[:3], 1):
            context_parts.append(f"  {i}. {c['summary']}")

    if legit_cases:
        context_parts.append("\n**Known LEGITIMATE cases (similar pattern):**")
        for i, c in enumerate(legit_cases[:2], 1):
            context_parts.append(f"  {i}. {c['summary']}")

    context_str = "\n".join(context_parts)

    risk_level = (
        "🔴 HIGH RISK" if fraud_prob >= 0.7
        else "🟡 MEDIUM RISK" if fraud_prob >= 0.4
        else "🟢 LOW RISK"
    )

    prompt = f"""
## Transaction Under Review
{transaction_summary}

**ML Model Fraud Probability: {fraud_prob*100:.1f}% ({risk_level})**

{context_str}

---

Please provide a structured fraud risk assessment with the following sections:

### 1. 🔍 Risk Analysis
Explain WHY this transaction is or isn't suspicious based on the specific features above.
Point to specific data points (amount, email domain, card type, timing, etc.).

### 2. 📊 Pattern Comparison  
Compare this transaction to the retrieved historical fraud/legit cases.
What patterns match known fraud? What patterns suggest legitimacy?

### 3. ⚠️ Risk Factors Identified
List the top 3-5 specific risk indicators present in this transaction.

### 4. ✅ Analyst Recommendation
Provide a clear recommended action:
- APPROVE / FLAG FOR REVIEW / BLOCK
- Confidence level
- Any additional verification steps recommended

### 5. 📝 Executive Summary (1-2 sentences)
A concise summary for a fraud operations dashboard.
"""
    return prompt


def explain_transaction(
    transaction_summary: str,
    similar_cases: list,
    fraud_prob: float,
    stream: bool = False,
):
    """
    Call Groq LLM with RAG context and return fraud explanation.
    Returns a string response.
    """
    client = get_groq_client()
    prompt = build_rag_prompt(transaction_summary, similar_cases, fraud_prob)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=stream,
    )

    if stream:
        return response  # caller handles streaming
    return response.choices[0].message.content


def generate_risk_summary(transaction_summary: str, fraud_prob: float) -> str:
    """Generate a short 1-paragraph risk summary (fast, low tokens)."""
    client = get_groq_client()

    prompt = f"""Transaction: {transaction_summary}
Fraud Probability: {fraud_prob*100:.1f}%

Write a 2-3 sentence risk summary for a fraud analyst dashboard. 
Be specific about what's suspicious or normal. Keep it factual and concise."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a fraud detection assistant. Give concise, factual risk summaries."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    return response.choices[0].message.content


def batch_generate_summaries(transactions: list, fraud_probs: list) -> list:
    """Generate risk summaries for multiple transactions."""
    summaries = []
    for txn, prob in zip(transactions, fraud_probs):
        try:
            summary = generate_risk_summary(txn, prob)
        except Exception as e:
            summary = f"Summary unavailable: {e}"
        summaries.append(summary)
    return summaries
