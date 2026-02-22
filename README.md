# 🛡️ AI-Powered Fraud Detection System

A dual-dataset fraud detection platform combining XGBoost, RAG (Retrieval-Augmented Generation), and LLM-powered explanations via Groq — with a Streamlit interface for real-time transaction analysis.

---

## 📸 Screenshots

<img width="1909" height="975" alt="Screenshot 2026-02-21 185034" src="https://github.com/user-attachments/assets/880d84a4-eb6d-4827-9707-e39182b56640" />
<img width="1906" height="974" alt="Screenshot 2026-02-21 185049" src="https://github.com/user-attachments/assets/8e6ac5f0-767d-466e-b24d-882b4592b9d2" />
<img width="1907" height="977" alt="Screenshot 2026-02-21 185103" src="https://github.com/user-attachments/assets/4feb2156-7ed0-463e-bb60-16805016b600" />
<img width="1901" height="967" alt="Screenshot 2026-02-21 185131" src="https://github.com/user-attachments/assets/2fbeb40f-af35-46f3-9db4-dc0fe3b7bbd2" />
<img width="1913" height="970" alt="Screenshot 2026-02-21 185143" src="https://github.com/user-attachments/assets/1e742803-aaee-4c28-b0f5-a4ec93093ee9" />
<img width="1903" height="968" alt="Screenshot 2026-02-21 185158" src="https://github.com/user-attachments/assets/d2113461-c1cf-410b-8226-7a99d9108363" />






---

## 🧠 Architecture

| Component | IEEE-CIS Dataset | Credit Card Dataset |
|-----------|-----------------|---------------------|
| **ML Model** | XGBoost | XGBoost |
| **LLM** | Groq Llama 3.3-70b | LoRA adapter → Groq fallback |
| **RAG** | FAISS (transaction summaries) | FAISS (PCA summaries) |
| **Key Metrics** | Precision / Recall / F1 / AUC + Recall@K | Precision / Recall / F1 / AUC |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
conda install pandas numpy scikit-learn joblib matplotlib seaborn -y

pip install xgboost faiss-cpu langchain langchain-groq langchain-community \
    sentence-transformers streamlit plotly python-dotenv groq \
    peft transformers torch datasets accelerate
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and set:
# GROQ_API_KEY=gsk_xxxxxxxxxxxx
# GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Get Datasets

| Dataset | Source | Files |
|---------|--------|-------|
| **IEEE-CIS Fraud Detection** | [kaggle.com/competitions/ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | `train_transaction.csv`, `train_identity.csv` |
| **Credit Card Fraud** | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` |

Place all files in the `data/` directory.

### 4. Train Models

```bash
# IEEE-CIS
python train_model.py data/train_transaction.csv data/train_identity.csv

# Credit Card (with LoRA fine-tuning)
python train_creditcard.py data/creditcard.csv

# Credit Card (skip LoRA, use Groq directly)
python train_creditcard.py data/creditcard.csv --skip-lora
```

### 5. Build Vector Stores

```bash
python build_vectorstore.py data/train_transaction.csv data/train_identity.csv
python build_vectorstore_cc.py data/creditcard.csv
```

### 6. Launch App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                      # Streamlit application
├── train_model.py              # IEEE-CIS XGBoost training
├── train_creditcard.py         # Credit Card XGBoost + LoRA training
├── build_vectorstore.py        # FAISS vector store (IEEE-CIS)
├── build_vectorstore_cc.py     # FAISS vector store (Credit Card)
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── creditcard.csv
├── models/                     # Saved XGBoost + LoRA models
├── vectorstores/               # FAISS indexes
└── .env                        # API keys
```

---

## ✨ Features

- **Dual-dataset support** — analyze both IEEE-CIS and Credit Card transactions
- **XGBoost ML models** — high-performance gradient boosting for fraud classification
- **RAG-powered explanations** — retrieves similar historical transactions for context
- **LLM analysis** — natural language fraud explanations via Groq Llama 3.3-70b
- **LoRA fine-tuning** — optional parameter-efficient fine-tuning on Credit Card data
- **Interactive dashboard** — Streamlit UI with Plotly visualizations
- **Recall@K metric** — evaluates model performance at catching top-K risky transactions

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key |
| `GROQ_MODEL` | Model to use (default: `llama-3.3-70b-versatile`) |

---

## 📊 Evaluation Metrics

- **Precision** — fraction of flagged transactions that are truly fraudulent
- **Recall** — fraction of actual fraud cases caught
- **F1 Score** — harmonic mean of precision and recall
- **AUC-ROC** — overall discrimination ability of the model
- **Recall@K** *(IEEE-CIS only)* — recall among the top-K highest-risk predictions

---

## 🛠️ Tech Stack

`Python` · `XGBoost` · `FAISS` · `LangChain` · `Groq` · `Llama 3.3` · `LoRA / PEFT` · `Streamlit` · `Plotly` · `Sentence Transformers` · `scikit-learn` · `PyTorch`
