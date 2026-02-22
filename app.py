"""
FraudShield AI — Dual Dataset Fraud Detection
IEEE-CIS + Credit Card Fraud Detection
Streamlit UI with sidebar dataset toggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    .risk-high   { background:#da3633; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
    .risk-medium { background:#d29922; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
    .risk-low    { background:#238636; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
    .txn-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 16px 20px; margin: 8px 0;
    }
    .llm-box {
        background: #161b22; border-left: 4px solid #58a6ff;
        border-radius: 0 12px 12px 0; padding: 20px 24px;
        margin: 12px 0; font-size: 0.95rem; line-height: 1.7;
    }
    .llm-box-lora {
        background: #161b22; border-left: 4px solid #a371f7;
        border-radius: 0 12px 12px 0; padding: 20px 24px;
        margin: 12px 0; font-size: 0.95rem; line-height: 1.7;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #58a6ff;
        border-bottom: 1px solid #30363d; padding-bottom: 8px; margin: 20px 0 12px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .fraud-pill { background:#da363320; border:1px solid #da3633; border-radius:8px; padding:8px 12px; margin:4px 0; font-size:0.82rem; }
    .legit-pill { background:#23863620; border:1px solid #238636; border-radius:8px; padding:8px 12px; margin:4px 0; font-size:0.82rem; }
    .dataset-badge-ieee { background:#1f6feb; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .dataset-badge-cc   { background:#a371f7; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .lora-badge { background:#a371f720; border:1px solid #a371f7; border-radius:8px; padding:6px 12px; font-size:0.82rem; display:inline-block; }
    header[data-testid="stHeader"] { background: transparent; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ── Load Engines ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ieee_engine():
    from utils.inference import FraudDetectionEngine
    engine = FraudDetectionEngine()
    try:
        engine.load_all()
        return engine, None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_cc_engine():
    from utils.cc_inference import CCFraudEngine
    engine = CCFraudEngine()
    try:
        engine.load_all()
        return engine, None
    except FileNotFoundError as e:
        return None, str(e)


# ── Helpers ────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    cls = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(level, "risk-low")
    return f'<span class="{cls}">{level} RISK</span>'


def gauge_chart(prob: float):
    color = "#da3633" if prob >= 0.7 else "#d29922" if prob >= 0.4 else "#238636"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":  {"color": color, "thickness": 0.8},
            "bgcolor": "#21262d",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  40], "color": "rgba(35,134,54,0.08)"},
                {"range": [40, 70], "color": "rgba(210,153,34,0.08)"},
                {"range": [70,100], "color": "rgba(218,54,51,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
        title={"text": "Fraud Probability", "font": {"size": 14, "color": "#8b949e"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf3"},
    )
    return fig


def feature_importance_chart(metrics: dict, title: str = "Top Feature Importances"):
    features = metrics.get("top_features", {})
    if not features:
        return None
    top = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:15])
    fig = px.bar(
        x=list(top.values()), y=list(top.keys()), orientation="h",
        color=list(top.values()), color_continuous_scale=["#238636","#d29922","#da3633"],
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
        font={"color": "#e6edf3"}, height=420, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
    )
    return fig


def confusion_matrix_chart(cm: list, title: str = "Confusion Matrix"):
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: Legit","Pred: Fraud"], y=["True: Legit","True: Fraud"],
        text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
        colorscale=[[0,"#161b22"],[0.5,"#1f6feb"],[1,"#58a6ff"]], showscale=False,
    ))
    fig.update_layout(
        title=title, paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3"}, height=300, margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


def similar_cases_section(similar_cases: list):
    if not similar_cases:
        return
    st.markdown('<div class="section-header">🔎 Similar Historical Cases (RAG)</div>', unsafe_allow_html=True)
    fraud_cases = [c for c in similar_cases if c["label"] == 1]
    legit_cases = [c for c in similar_cases if c["label"] == 0]
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**🚨 Matched Fraud Patterns**")
        for c in fraud_cases[:3]:
            st.markdown(
                f'<div class="fraud-pill">🔴 ${c["amount"]:.2f} — {c["summary"][:110]}...</div>',
                unsafe_allow_html=True
            )
    with sc2:
        st.markdown("**✅ Matched Legit Patterns**")
        for c in legit_cases[:2]:
            st.markdown(
                f'<div class="legit-pill">🟢 ${c["amount"]:.2f} — {c["summary"][:110]}...</div>',
                unsafe_allow_html=True
            )


def llm_section(result: dict):
    if not result.get("llm_explanation"):
        return
    source = result.get("llm_source", "groq")
    if source == "lora":
        label = "🤖 AI Fraud Analysis (LoRA Fine-tuned Model)"
        box_class = "llm-box-lora"
        st.markdown('<div class="lora-badge">⚡ Generated by fine-tuned LoRA/PEFT adapter</div>', unsafe_allow_html=True)
    else:
        label = "🤖 AI Fraud Analysis (LLM + RAG)"
        box_class = "llm-box"
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="{box_class}">{result["llm_explanation"].replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("<small style='color:#8b949e'>Dual-Dataset Fraud Detection + RAG + LLM</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Active Dataset**")
    dataset = st.radio(
        "dataset",
        ["🏦 IEEE-CIS (Transaction)", "💳 Credit Card (284K)"],
        label_visibility="collapsed",
    )
    is_ieee = dataset.startswith("🏦")

    if is_ieee:
        st.markdown('<span class="dataset-badge-ieee">IEEE-CIS · XGBoost + RAG</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="dataset-badge-cc">Credit Card · LoRA/PEFT + RAG</span>', unsafe_allow_html=True)

    st.divider()
    page = st.radio(
        "Navigation",
        ["🔍 Analyze Transaction", "📁 Batch Analysis", "📊 Model Dashboard", "⚙️ Setup Guide"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**Model Settings**")
    use_llm   = st.toggle("Enable LLM Explanation", value=True)
    use_rag   = st.toggle("Enable RAG Context",     value=True)
    k_similar = st.slider("Similar cases (K)", 3, 10, 5)

    st.divider()
    if os.getenv("GROQ_API_KEY", ""):
        st.success("✅ Groq API connected")
    else:
        st.warning("⚠️ Set GROQ_API_KEY in .env")

    if not is_ieee:
        try:
            from utils.cc_llm_chain import get_lora_status
            lora = get_lora_status()
            if lora.get("status") == "trained":
                st.success("✅ LoRA adapter trained")
            else:
                st.info("ℹ️ LoRA not trained\n(Groq fallback active)")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# PAGE: Analyze Transaction
# ══════════════════════════════════════════════════════════════
if page == "🔍 Analyze Transaction":

    if is_ieee:
        st.markdown("# 🏦 IEEE-CIS Transaction Analysis")
        st.markdown("Analyze e-commerce transactions with card and identity features.")

        engine, err = load_ieee_engine()
        if err:
            st.error(f"❌ {err}")
            st.info("Run: `python train_model.py data/train_transaction.csv data/train_identity.csv`")
            st.stop()

        with st.expander("📋 Transaction Details", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                txn_amt    = st.number_input("Amount ($)", min_value=0.01, value=150.50, step=0.01)
                product_cd = st.selectbox("Product Code", ["W","H","C","S","R"])
                card4      = st.selectbox("Card Network", ["visa","mastercard","american express","discover"])
            with c2:
                card6   = st.selectbox("Card Type", ["debit","credit","debit or credit","charge card"])
                p_email = st.selectbox("Purchaser Email Domain",
                             ["gmail.com","yahoo.com","hotmail.com","anonymous.com","outlook.com","icloud.com","other"])
                r_email = st.selectbox("Recipient Email Domain",
                             ["gmail.com","yahoo.com","hotmail.com","anonymous.com","outlook.com","icloud.com","other"])
            with c3:
                addr1  = st.number_input("Billing ZIP",     min_value=0, value=299,   step=1)
                addr2  = st.number_input("Country Code",    min_value=0, value=87,    step=1)
                txn_dt = st.number_input("Timestamp (DT)",  min_value=0, value=86405, step=3600)

        if st.button("🚀 Analyze Transaction", use_container_width=True, key="ieee_btn"):
            row_dict = {
                "TransactionAmt": txn_amt, "ProductCD": product_cd,
                "card4": card4, "card6": card6,
                "P_emaildomain": p_email, "R_emaildomain": r_email,
                "addr1": addr1, "addr2": addr2, "TransactionDT": txn_dt,
            }
            with st.spinner("🔄 Running ML + RAG + LLM..."):
                result = engine.full_analysis(row_dict, use_llm=use_llm,
                                              k_similar=k_similar if use_rag else 0)
            st.divider()
            col_g, col_d = st.columns([1, 2])
            with col_g:
                st.plotly_chart(gauge_chart(result["fraud_probability"]), use_container_width=True)
                st.markdown(f"<div style='text-align:center'>{risk_badge(result['risk_level'])}</div>",
                            unsafe_allow_html=True)
            with col_d:
                st.markdown('<div class="section-header">📋 Transaction Summary</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="txn-card"><code style="font-size:0.88rem;line-height:1.8">'
                    f'{result["transaction_summary"].replace(" | ", "<br>")}</code></div>',
                    unsafe_allow_html=True
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud Prob", f"{result['fraud_probability']*100:.1f}%")
                m2.metric("Prediction", "🚨 FRAUD" if result["prediction"] else "✅ LEGIT")
                m3.metric("Risk",       result["risk_level"])
            if use_rag: similar_cases_section(result["similar_cases"])
            if use_llm: llm_section(result)

    else:
        st.markdown("# 💳 Credit Card Fraud Analysis")
        st.markdown("PCA-anonymized features (V1–V28) + Amount + Time. LoRA/PEFT-powered explanations.")

        engine, err = load_cc_engine()
        if err:
            st.error(f"❌ {err}")
            st.info("Run: `python train_creditcard.py data/creditcard.csv`")
            st.stop()

        with st.expander("📋 Transaction Details", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                amount   = st.number_input("Amount ($)", min_value=0.01, value=124.50, step=0.01)
                time_sec = st.number_input("Time (seconds from first txn)", min_value=0, value=52000, step=1000)
                st.caption(f"→ Hour of day: {int((time_sec // 3600) % 24)}:00")
            with c2:
                preset = st.selectbox("PCA Preset", [
                    "Normal transaction",
                    "Suspicious — high V4/V11/V14",
                    "Suspicious — small amount odd hour",
                    "Manual entry",
                ])

            st.markdown("**PCA Features (V1–V28)**")
            st.caption("Anonymized behavioral features from PCA transformation.")

            if preset == "Normal transaction":
                v_defaults = {f"V{i}": round(float(np.random.normal(0, 0.5)), 3) for i in range(1, 29)}
            elif preset == "Suspicious — high V4/V11/V14":
                v_defaults = {f"V{i}": round(float(np.random.normal(0, 0.5)), 3) for i in range(1, 29)}
                v_defaults.update({"V4": -5.2, "V11": -4.8, "V14": -6.1})
            elif preset == "Suspicious — small amount odd hour":
                v_defaults = {f"V{i}": round(float(np.random.normal(0, 1)), 3) for i in range(1, 29)}
                v_defaults.update({"V1": -3.5, "V3": -4.1})
            else:
                v_defaults = {f"V{i}": 0.0 for i in range(1, 29)}

            vcols = st.columns(4)
            v_vals = {}
            for idx, i in enumerate(range(1, 29)):
                with vcols[idx % 4]:
                    v_vals[f"V{i}"] = st.number_input(
                        f"V{i}", value=float(v_defaults[f"V{i}"]),
                        step=0.1, format="%.3f", key=f"v{i}"
                    )

        if st.button("🚀 Analyze Transaction", use_container_width=True, key="cc_btn"):
            row_dict = {"Amount": amount, "Time": time_sec, **v_vals}
            with st.spinner("🔄 Running ML + RAG + LLM..."):
                result = engine.full_analysis(row_dict, use_llm=use_llm,
                                              k_similar=k_similar if use_rag else 0)
            st.divider()
            col_g, col_d = st.columns([1, 2])
            with col_g:
                st.plotly_chart(gauge_chart(result["fraud_probability"]), use_container_width=True)
                st.markdown(f"<div style='text-align:center'>{risk_badge(result['risk_level'])}</div>",
                            unsafe_allow_html=True)
            with col_d:
                st.markdown('<div class="section-header">📋 Transaction Summary</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="txn-card"><code style="font-size:0.88rem;line-height:1.8">'
                    f'{result["transaction_summary"].replace(" | ", "<br>")}</code></div>',
                    unsafe_allow_html=True
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud Prob", f"{result['fraud_probability']*100:.1f}%")
                m2.metric("Prediction", "🚨 FRAUD" if result["prediction"] else "✅ LEGIT")
                m3.metric("Risk",       result["risk_level"])
                if result.get("llm_source") == "lora":
                    st.markdown('<div class="lora-badge">⚡ LoRA/PEFT adapter active</div>', unsafe_allow_html=True)
            if use_rag: similar_cases_section(result["similar_cases"])
            if use_llm: llm_section(result)


# ══════════════════════════════════════════════════════════════
# PAGE: Batch Analysis
# ══════════════════════════════════════════════════════════════
elif page == "📁 Batch Analysis":
    st.markdown(f"# 📁 Batch — {'IEEE-CIS' if is_ieee else 'Credit Card'}")

    engine, err = load_ieee_engine() if is_ieee else load_cc_engine()
    if err:
        st.error(f"❌ {err}")
        st.stop()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_raw):,} transactions")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Predictions", use_container_width=True):
            with st.spinner("Processing..."):
                results_df = engine.predict_batch(df_raw)

            total   = len(results_df)
            flagged = (results_df["prediction"] == 1).sum()
            high    = (results_df["risk_level"] == "HIGH").sum()

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total",         f"{total:,}")
            mc2.metric("Fraud Flagged", f"{flagged:,}")
            mc3.metric("High Risk",     f"{high:,}")
            mc4.metric("Fraud Rate",    f"{flagged/total*100:.3f}%")

            risk_counts = results_df["risk_level"].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={"HIGH":"#da3633","MEDIUM":"#d29922","LOW":"#238636"},
                title="Risk Distribution",
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#e6edf3"}, height=320)

            fig_hist = px.histogram(
                results_df, x="fraud_probability", nbins=50,
                title="Fraud Probability Distribution",
                color_discrete_sequence=["#a371f7" if not is_ieee else "#58a6ff"],
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
                font={"color":"#e6edf3"}, height=320,
                xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
            )

            ch1, ch2 = st.columns(2)
            with ch1: st.plotly_chart(fig_pie,  use_container_width=True)
            with ch2: st.plotly_chart(fig_hist, use_container_width=True)

            flagged_df = results_df[results_df["prediction"]==1].sort_values("fraud_probability", ascending=False)
            amt_col = "TransactionAmt" if "TransactionAmt" in flagged_df.columns else "Amount" if "Amount" in flagged_df.columns else None
            id_col  = "TransactionID"  if "TransactionID"  in flagged_df.columns else None
            display_cols = [c for c in [id_col, amt_col, "fraud_probability", "risk_level"] if c and c in flagged_df.columns]

            st.markdown("### 🔍 Flagged Transactions")
            st.dataframe(
                flagged_df[display_cols].head(100).style.background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r"),
                use_container_width=True,
            )
            st.download_button("⬇️ Download CSV", results_df.to_csv(index=False), "predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# PAGE: Model Dashboard
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.markdown("# 📊 Model Performance Dashboard")

    ieee_metrics, cc_metrics = None, None
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            ieee_metrics = json.load(f)
    if os.path.exists("models/creditcard/metrics.json"):
        with open("models/creditcard/metrics.json") as f:
            cc_metrics = json.load(f)

    if not ieee_metrics and not cc_metrics:
        st.warning("⚠️ No metrics found. Train models first.")
        st.stop()

    # Side-by-side comparison if both exist
    if ieee_metrics and cc_metrics:
        st.markdown("### 📊 Model Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<span class="dataset-badge-ieee">IEEE-CIS</span>', unsafe_allow_html=True)
            st.metric("Precision", f"{ieee_metrics['precision']*100:.1f}%")
            st.metric("Recall",    f"{ieee_metrics['recall']*100:.1f}%")
            st.metric("F1-Score",  f"{ieee_metrics['f1_score']*100:.1f}%")
            st.metric("ROC-AUC",   f"{ieee_metrics['roc_auc']:.4f}")
        with col_b:
            st.markdown('<span class="dataset-badge-cc">Credit Card</span>', unsafe_allow_html=True)
            st.metric("Precision", f"{cc_metrics['precision']*100:.1f}%")
            st.metric("Recall",    f"{cc_metrics['recall']*100:.1f}%")
            st.metric("F1-Score",  f"{cc_metrics['f1_score']*100:.1f}%")
            st.metric("ROC-AUC",   f"{cc_metrics['roc_auc']:.4f}")
            if cc_metrics.get("recall_at_100"):
                st.metric("Recall@100", f"{cc_metrics['recall_at_100']*100:.1f}%")

        # Comparison bar
        fig_cmp = go.Figure()
        metrics_list = ["precision","recall","f1_score","roc_auc"]
        labels       = ["Precision","Recall","F1","ROC-AUC"]
        fig_cmp.add_bar(name="IEEE-CIS",    x=labels, y=[ieee_metrics[m] for m in metrics_list], marker_color="#1f6feb")
        fig_cmp.add_bar(name="Credit Card", x=labels, y=[cc_metrics[m]   for m in metrics_list], marker_color="#a371f7")
        fig_cmp.update_layout(
            barmode="group", title="Metrics Comparison",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
            font={"color":"#e6edf3"}, height=350,
            xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d", range=[0,1]),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.divider()

    active = ieee_metrics if is_ieee else cc_metrics
    label  = "IEEE-CIS" if is_ieee else "Credit Card"
    if active:
        st.markdown(f"### {'🏦' if is_ieee else '💳'} {label} — Detailed")
        d1, d2 = st.columns(2)
        with d1:
            if active.get("confusion_matrix"):
                st.plotly_chart(confusion_matrix_chart(active["confusion_matrix"]), use_container_width=True)
        with d2:
            fig_fi = feature_importance_chart(active, f"Top Features — {label}")
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)
        st.divider()
        ds1, ds2, ds3, ds4 = st.columns(4)
        ds1.metric("Train Samples", f"{active.get('train_size',0):,}")
        ds2.metric("Test Samples",  f"{active.get('test_size',0):,}")
        ds3.metric("Fraud Rate",    f"{active.get('fraud_rate',0)*100:.3f}%")
        ds4.metric("Features",      f"{active.get('n_features',0)}")


# ══════════════════════════════════════════════════════════════
# PAGE: Setup Guide
# ══════════════════════════════════════════════════════════════
elif page == "⚙️ Setup Guide":
    st.markdown("# ⚙️ Setup Guide")
    st.markdown("""
## Quick Start

### 1. Install Dependencies
```bash
conda install pandas numpy scikit-learn joblib matplotlib seaborn -y
pip install xgboost faiss-cpu langchain langchain-groq langchain-community sentence-transformers streamlit plotly python-dotenv groq peft transformers torch datasets accelerate
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env → GROQ_API_KEY=gsk_xxxxxxxxxxxx
# GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Get Datasets
- **IEEE-CIS:** kaggle.com/competitions/ieee-fraud-detection → `train_transaction.csv` + `train_identity.csv`
- **Credit Card:** kaggle.com/datasets/mlg-ulb/creditcardfraud → `creditcard.csv`

Place all files in `data/`

### 4. Train Models
```bash
# IEEE-CIS
python train_model.py data/train_transaction.csv data/train_identity.csv

# Credit Card (add --skip-lora to skip fine-tuning, uses Groq instead)
python train_creditcard.py data/creditcard.csv
python train_creditcard.py data/creditcard.csv --skip-lora
```

### 5. Build Vector Stores
```bash
python build_vectorstore.py data/train_transaction.csv data/train_identity.csv
python build_vectorstore_cc.py data/creditcard.csv
```

### 6. Launch
```bash
streamlit run app.py
```

---

## Architecture

| Component | IEEE-CIS | Credit Card |
|---|---|---|
| ML Model | XGBoost | XGBoost |
| LLM | Groq Llama 3.3 | LoRA adapter → Groq fallback |
| RAG | FAISS (transaction summaries) | FAISS (PCA summaries) |
| Special metrics | Precision/Recall/F1/AUC | + Recall@K |
    """)