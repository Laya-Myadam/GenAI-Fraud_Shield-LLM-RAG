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
import requests
import datetime
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
    .signal-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px 16px; margin:4px 0; }
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


# ── External Signals ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_exchange_rate_risk(amount: float, country_code: int) -> dict:
    HIGH_RISK_CURRENCIES = {"NGN", "VES", "IRR", "MMK", "SDG", "SYP", "ZWL"}
    COUNTRY_TO_CURRENCY = {
        87: "USD", 62: "EUR", 13: "GBP", 142: "JPY",
        208: "CNY", 92: "INR", 32: "BRL", 45: "CAD",
        50: "NGN", 117: "VES", 96: "IRR",
    }
    try:
        currency = COUNTRY_TO_CURRENCY.get(country_code, "USD")
        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        data = resp.json()
        if data.get("result") == "success":
            rate = data["rates"].get(currency, 1.0)
            usd_amount = amount / rate if currency != "USD" else amount
            is_high_risk = currency in HIGH_RISK_CURRENCIES
            return {
                "currency": currency, "exchange_rate": round(rate, 4),
                "usd_equivalent": round(usd_amount, 2),
                "high_risk_currency": is_high_risk,
                "large_usd_amount": usd_amount > 5000,
                "risk_score": 3 if is_high_risk else (2 if usd_amount > 5000 else 0),
                "status": "live",
            }
    except Exception:
        pass
    return {
        "currency": "USD", "exchange_rate": 1.0, "usd_equivalent": amount,
        "high_risk_currency": False, "large_usd_amount": amount > 5000,
        "risk_score": 2 if amount > 5000 else 0, "status": "unavailable",
    }


@st.cache_data(ttl=86400)
def get_geolocation_risk(country_code: int) -> dict:
    HIGH_RISK_COUNTRIES = {50, 96, 117, 140, 178, 204, 232}
    MEDIUM_RISK_COUNTRIES = {32, 55, 76, 92, 103, 136, 159, 190}
    SANCTIONED_COUNTRIES = {96, 117, 140}
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=4)
        ip_data = resp.json()
        ip_org = ip_data.get("org", "")
        ip_city = ip_data.get("city", "Unknown")
        is_vpn = any(x in ip_org.lower() for x in ["vpn","proxy","hosting","datacenter","digitalocean","linode","aws","google cloud"])
    except Exception:
        ip_city = "Unknown"
        is_vpn = False

    is_high_risk  = country_code in HIGH_RISK_COUNTRIES
    is_sanctioned = country_code in SANCTIONED_COUNTRIES
    is_medium     = country_code in MEDIUM_RISK_COUNTRIES
    risk_score = (4 if is_sanctioned else 3 if is_high_risk else 1 if is_medium else 0) + (2 if is_vpn else 0)
    return {
        "ip_city": ip_city, "is_high_risk_country": is_high_risk,
        "is_sanctioned_country": is_sanctioned, "vpn_proxy_detected": is_vpn,
        "risk_score": min(risk_score, 5),
        "risk_tier": "HIGH" if risk_score >= 3 else "MEDIUM" if risk_score >= 1 else "LOW",
    }


def get_time_risk(transaction_dt: int) -> dict:
    now  = datetime.datetime.utcnow()
    hour = (transaction_dt % 86400) // 3600
    is_odd     = hour < 6 or hour >= 23
    is_weekend = now.weekday() >= 5
    is_holiday = now.month in [11, 12]
    is_monthend = now.day >= 28
    is_rapid   = transaction_dt < 3600
    flags, score = [], 0
    if is_odd:     flags.append(f"Off-hours transaction ({hour}:00 UTC)"); score += 2
    if is_weekend: flags.append("Weekend transaction"); score += 1
    if is_holiday: flags.append("Holiday season (elevated fraud period)"); score += 1
    if is_monthend:flags.append("Month-end (elevated fraud period)"); score += 1
    if is_rapid:   flags.append("Very early in sequence (possible test transaction)"); score += 2
    return {
        "hour_utc": hour, "is_odd_hours": is_odd, "is_weekend": is_weekend,
        "is_holiday_season": is_holiday, "is_rapid_sequence": is_rapid,
        "risk_flags": flags, "risk_score": min(score, 5),
        "risk_tier": "HIGH" if score >= 3 else "MEDIUM" if score >= 1 else "LOW",
    }


def get_fraud_pattern_indicators(amount, product_cd=None, p_email=None, r_email=None, card_type=None) -> dict:
    HIGH_RISK_EMAILS = ["anonymous.com","protonmail.com","guerrillamail.com","tempmail.com"]
    HIGH_RISK_PRODUCTS = ["H", "S"]
    flags, score = [], 0
    if any(abs(amount - r) < 0.01 for r in [100,200,500,1000,2000,5000]):
        flags.append("Round-number amount (structuring indicator)"); score += 1
    if amount < 1.0:
        flags.append("Micro-transaction (card verification pattern)"); score += 2
    if amount > 3000:
        flags.append("High-value transaction (above typical threshold)"); score += 1
    if p_email and any(h in p_email for h in HIGH_RISK_EMAILS):
        flags.append(f"High-risk purchaser email domain ({p_email})"); score += 2
    if p_email and r_email and p_email != r_email and ("anonymous" in p_email or "anonymous" in r_email):
        flags.append("Anonymous email on transaction party (identity concealment)"); score += 2
    if product_cd and product_cd in HIGH_RISK_PRODUCTS:
        flags.append(f"High-risk product category ({product_cd})"); score += 1
    if card_type and "charge card" in card_type.lower():
        flags.append("Charge card type (higher fraud association)"); score += 1
    if amount > 500 and p_email and "anonymous" in p_email:
        flags.append("High amount + anonymous email (combined risk signal)"); score += 2
    return {
        "pattern_flags": flags, "patterns_detected": len(flags),
        "risk_score": min(score, 5),
        "risk_tier": "HIGH" if score >= 4 else "MEDIUM" if score >= 2 else "LOW",
    }


def get_all_external_signals(amount, transaction_dt=86400, country_code=87,
                              product_cd=None, p_email=None, r_email=None, card_type=None) -> dict:
    fx      = get_exchange_rate_risk(amount, country_code)
    geo     = get_geolocation_risk(country_code)
    time    = get_time_risk(transaction_dt)
    pattern = get_fraud_pattern_indicators(amount, product_cd, p_email, r_email, card_type)
    total   = fx["risk_score"] + geo["risk_score"] + time["risk_score"] + pattern["risk_score"]
    return {
        "exchange_rate": fx, "geolocation": geo, "time_risk": time,
        "fraud_patterns": pattern, "total_risk_score": total,
        "normalized_score": round((total / 18) * 100, 1),
        "overall_tier": "HIGH" if total >= 7 else "MEDIUM" if total >= 3 else "LOW",
    }


# ── Helpers ────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    cls = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(level, "risk-low")
    return f'<span class="{cls}">{level} RISK</span>'


def render_external_signals(signals: dict):
    st.markdown('<div class="section-header">🌐 External Risk Signals</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    fx  = signals["exchange_rate"]
    geo = signals["geolocation"]
    tm  = signals["time_risk"]
    pat = signals["fraud_patterns"]
    tier_color = {"HIGH": "#da3633", "MEDIUM": "#d29922", "LOW": "#238636"}

    with e1:
        st.metric("💱 FX Risk", fx["currency"],
                  delta="HIGH RISK" if fx["high_risk_currency"] else "Normal",
                  delta_color="inverse" if fx["high_risk_currency"] else "off")
        st.caption(f"Rate: {fx['exchange_rate']} | ~${fx['usd_equivalent']:,.2f} USD")
    with e2:
        st.metric("🌍 Geo Risk", geo["risk_tier"],
                  delta="VPN/Proxy ⚠️" if geo["vpn_proxy_detected"] else "Clean",
                  delta_color="inverse" if geo["vpn_proxy_detected"] else "off")
        st.caption(f"Sanctioned country: {'Yes ⚠️' if geo['is_sanctioned_country'] else 'No'}")
    with e3:
        st.metric("🕐 Time Risk", tm["risk_tier"],
                  delta=f"{tm['hour_utc']}:00 UTC",
                  delta_color="inverse" if tm["is_odd_hours"] else "off")
        st.caption(f"{'Off-hours' if tm['is_odd_hours'] else 'Normal hours'} · {'Weekend' if tm['is_weekend'] else 'Weekday'}")
    with e4:
        st.metric("🚩 Pattern Flags", f"{pat['patterns_detected']} detected",
                  delta=pat["risk_tier"],
                  delta_color="inverse" if pat["risk_tier"] != "LOW" else "off")
        st.caption(f"Pattern risk score: {pat['risk_score']}/5")

    all_flags = tm["risk_flags"] + pat["pattern_flags"]
    if all_flags:
        with st.expander("📋 View All Risk Flag Details"):
            for flag in all_flags:
                st.markdown(f'<div class="fraud-pill">⚠️ {flag}</div>', unsafe_allow_html=True)

    overall_color = tier_color.get(signals["overall_tier"], "#238636")
    st.markdown(
        f'<div class="txn-card" style="border-left:4px solid {overall_color}; margin-top:12px">'
        f'<b>🌐 External Signal Score: {signals["normalized_score"]}/100</b> &nbsp;|&nbsp; '
        f'Overall External Risk: <b style="color:{overall_color}">{signals["overall_tier"]}</b>'
        f'</div>',
        unsafe_allow_html=True
    )


def gauge_chart(prob: float):
    color = "#da3633" if prob >= 0.7 else "#d29922" if prob >= 0.4 else "#238636"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":  {"color": color, "thickness": 0.8},
            "bgcolor": "#21262d", "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  40], "color": "rgba(35,134,54,0.08)"},
                {"range": [40, 70], "color": "rgba(210,153,34,0.08)"},
                {"range": [70,100], "color": "rgba(218,54,51,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": prob * 100},
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
            st.markdown(f'<div class="fraud-pill">🔴 ${c["amount"]:.2f} — {c["summary"][:110]}...</div>', unsafe_allow_html=True)
    with sc2:
        st.markdown("**✅ Matched Legit Patterns**")
        for c in legit_cases[:2]:
            st.markdown(f'<div class="legit-pill">🟢 ${c["amount"]:.2f} — {c["summary"][:110]}...</div>', unsafe_allow_html=True)


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
    use_llm      = st.toggle("Enable LLM Explanation",      value=True)
    use_rag      = st.toggle("Enable RAG Context",          value=True)
    use_external = st.toggle("Enable External Risk Signals", value=True)
    k_similar    = st.slider("Similar cases (K)", 3, 10, 5)

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
        st.markdown("Analyze payment card transactions with financial institution risk features.")

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

            if use_external:
                with st.spinner("🌐 Fetching external risk signals..."):
                    signals = get_all_external_signals(
                        amount=txn_amt, transaction_dt=txn_dt, country_code=addr2,
                        product_cd=product_cd, p_email=p_email, r_email=r_email, card_type=card6,
                    )
                render_external_signals(signals)

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

            if use_external:
                with st.spinner("🌐 Fetching external risk signals..."):
                    signals = get_all_external_signals(
                        amount=amount, transaction_dt=time_sec, country_code=87,
                    )
                render_external_signals(signals)


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
python train_model.py data/train_transaction.csv data/train_identity.csv
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
| External Signals | FX Risk · Geo Risk · Time Risk · Fraud Patterns | Same |
| Special metrics | Precision/Recall/F1/AUC | + Recall@K |
    """)