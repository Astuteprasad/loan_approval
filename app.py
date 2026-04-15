"""
LendingClub Advanced Loan Approval System — Streamlit Web App
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ — Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #162032);
        border: 1px solid #2d3f55;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .approve-banner {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .reject-banner {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .risk-score-low    { color: #10b981; font-weight: 700; }
    .risk-score-medium { color: #f59e0b; font-weight: 700; }
    .risk-score-high   { color: #ef4444; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }
    div[data-testid="stSidebarContent"] { background-color: #0d1520; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifact_dir = 'model_artifacts'
    try:
        model = joblib.load(f'{artifact_dir}/xgb_model.pkl')
        scaler = joblib.load(f'{artifact_dir}/scaler.pkl')
        le_dict = joblib.load(f'{artifact_dir}/label_encoders.pkl')
        with open(f'{artifact_dir}/config.json') as f:
            config = json.load(f)
        return model, scaler, le_dict, config
    except FileNotFoundError:
        return None, None, None, None

model, scaler, le_dict, config = load_artifacts()
THRESHOLD = config['optimal_threshold'] if config else 0.35

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.title("LoanIQ")
    st.caption("Advanced Credit Risk Assessment")
    st.divider()

    if config:
        st.metric("Model AUC", f"{config.get('model_auc', 0):.4f}")
        st.metric("PR-AUC", f"{config.get('pr_auc', 0):.4f}")
        st.metric("Decision Threshold", f"{THRESHOLD:.2f}")
    else:
        st.warning("⚠️ Model artifacts not found. Place `model_artifacts/` folder in this directory.")
        st.info("Run the Colab notebook first, download the ZIP, and extract it here.")

    st.divider()
    st.markdown("**Model:** XGBoost + LightGBM Stacking  \n"
                "**Data:** LendingClub 2007–2018  \n"
                "**Loans analysed:** ~1.8M")

# ─── MAIN HEADER ──────────────────────────────────────────────────────────────
st.title("🏦 LoanIQ — Loan Approval System")
st.markdown("Enter applicant details below to get an **instant credit decision** with explainability.")
st.divider()

# ─── INPUT FORM ──────────────────────────────────────────────────────────────
with st.form("loan_form"):
    st.subheader("📋 Loan Application")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Loan Details**")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000,
                                     value=10000, step=500)
        term = st.selectbox("Term (months)", [36, 60])
        purpose = st.selectbox("Purpose", [
            'debt_consolidation', 'credit_card', 'home_improvement',
            'other', 'major_purchase', 'medical', 'small_business',
            'car', 'vacation', 'moving', 'house', 'wedding', 'renewable_energy'
        ])
        grade = st.selectbox("LendingClub Grade", ['A','B','C','D','E','F','G'])
        int_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 12.0, 0.1)

    with col2:
        st.markdown("**Borrower Financials**")
        annual_inc = st.number_input("Annual Income ($)", min_value=10000,
                                      max_value=1000000, value=65000, step=1000)
        dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 15.0, 0.5)
        emp_length = st.selectbox("Employment Length (years)",
                                   list(range(0, 12)), index=5)
        home_ownership = st.selectbox("Home Ownership",
                                       ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        installment = st.number_input("Monthly Installment ($)", min_value=10.0,
                                       max_value=2000.0, value=300.0, step=10.0)

    with col3:
        st.markdown("**Credit History**")
        fico_score = st.slider("FICO Score", 580, 850, 700)
        revol_util = st.slider("Credit Utilisation (%)", 0.0, 100.0, 45.0, 0.5)
        revol_bal = st.number_input("Revolving Balance ($)", 0, 200000, 15000, 500)
        open_acc = st.number_input("Open Accounts", 0, 50, 8)
        total_acc = st.number_input("Total Accounts", 1, 100, 20)
        delinq_2yrs = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)
        pub_rec = st.number_input("Public Records", 0, 10, 0)
        inq_last_6mths = st.number_input("Inquiries (last 6 mo.)", 0, 20, 1)

    submitted = st.form_submit_button("🔍 Assess Application", use_container_width=True)

# ─── PREDICTION ──────────────────────────────────────────────────────────────
def build_input_df(loan_amnt, term, int_rate, installment, grade, purpose,
                    annual_inc, dti, emp_length, home_ownership,
                    fico_score, revol_util, revol_bal, open_acc, total_acc,
                    delinq_2yrs, pub_rec, inq_last_6mths):
    """Build feature-engineered input row matching training pipeline."""
    grade_map = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
    purpose_map = {
        'debt_consolidation':0,'credit_card':1,'home_improvement':2,'other':3,
        'major_purchase':4,'medical':5,'small_business':6,'car':7,
        'vacation':8,'moving':9,'house':10,'wedding':11,'renewable_energy':12
    }
    ownership_map = {'RENT':0,'OWN':1,'MORTGAGE':2,'OTHER':3}

    dti_bucket = min(int(dti // 10), 4)
    loan_to_income = loan_amnt / max(annual_inc, 1)
    payment_to_income = (installment * 12) / max(annual_inc, 1)
    high_util_flag = 1 if revol_util > 75 else 0
    has_pub_rec = 1 if pub_rec > 0 else 0

    row = {
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
        'installment': installment, 'grade': grade_map.get(grade, 2),
        'sub_grade': grade_map.get(grade, 2) * 5,
        'purpose': purpose_map.get(purpose, 3),
        'annual_inc': annual_inc, 'dti': dti, 'emp_length': emp_length,
        'home_ownership': ownership_map.get(home_ownership, 0),
        'fico_score': fico_score, 'open_acc': open_acc, 'pub_rec': pub_rec,
        'revol_bal': revol_bal, 'revol_util': revol_util, 'total_acc': total_acc,
        'inq_last_6mths': inq_last_6mths,
        'mths_since_last_delinq': 999 if delinq_2yrs == 0 else 12,
        'delinq_2yrs': delinq_2yrs,
        'pub_rec_bankruptcies': 0, 'mort_acc': 1,
        'credit_history_yrs': 10, 'dti_bucket': dti_bucket,
        'loan_to_income': loan_to_income, 'payment_to_income': payment_to_income,
        'fico_band': 2, 'high_util_flag': high_util_flag,
        'has_pub_rec': has_pub_rec, 'has_bankruptcy': 0,
        'int_rate_tier': 2, 'zip3': 100, 'addr_state': 5,
    }
    return pd.DataFrame([row])


if submitted:
    st.divider()

    if model is None:
        st.error("⚠️ Model not loaded. Please add model_artifacts/ directory and refresh.")
        st.stop()

    input_df = build_input_df(
        loan_amnt, term, int_rate, installment, grade, purpose,
        annual_inc, dti, emp_length, home_ownership,
        fico_score, revol_util, revol_bal, open_acc, total_acc,
        delinq_2yrs, pub_rec, inq_last_6mths
    )

    # Align columns to model training features
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    prob = model.predict_proba(input_df)[0][1]
    decision = "APPROVED ✅" if prob < THRESHOLD else "REJECTED ❌"
    risk_pct = f"{prob:.1%}"

    # Risk tier
    if prob < 0.2:
        risk_tier, risk_css = "Low Risk", "risk-score-low"
    elif prob < 0.4:
        risk_tier, risk_css = "Medium Risk", "risk-score-medium"
    else:
        risk_tier, risk_css = "High Risk", "risk-score-high"

    # ── DECISION BANNER ────────────────────────────────────────────────────────
    banner_class = "approve-banner" if prob < THRESHOLD else "reject-banner"
    icon = "✅" if prob < THRESHOLD else "❌"
    action = "APPROVED" if prob < THRESHOLD else "REJECTED"

    st.markdown(f"""
    <div class="{banner_class}">
        <h1 style="font-size:2.5rem; margin:0">{icon} {action}</h1>
        <p style="font-size:1.2rem; margin:0.5rem 0 0">
            Default Probability: <strong>{risk_pct}</strong> &nbsp;|&nbsp;
            <span class="{risk_css}">{risk_tier}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── METRICS ROW ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Default Probability", risk_pct)
    m2.metric("Decision Threshold", f"{THRESHOLD:.0%}")
    m3.metric("Risk Tier", risk_tier)
    m4.metric("Loan-to-Income", f"{loan_amnt/max(annual_inc,1):.2f}x")

    # ── RISK GAUGE ─────────────────────────────────────────────────────────────
    col_gauge, col_factors = st.columns([1, 1])

    with col_gauge:
        st.subheader("📊 Risk Probability")
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(['Default Probability'], [prob], color='#ef4444' if prob > THRESHOLD else '#10b981',
                height=0.4)
        ax.barh(['Default Probability'], [1], color='#1e2a3a', height=0.4)
        ax.barh(['Default Probability'], [prob], color='#ef4444' if prob > THRESHOLD else '#10b981',
                height=0.4)
        ax.axvline(THRESHOLD, color='#f59e0b', linewidth=2, linestyle='--', label=f'Threshold ({THRESHOLD:.0%})')
        ax.set_xlim(0, 1)
        ax.set_facecolor('#0f1117')
        fig.patch.set_facecolor('#0f1117')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#2d3f55')
        ax.legend(facecolor='#1e2a3a', edgecolor='#2d3f55', labelcolor='white')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_factors:
        st.subheader("⚠️ Key Risk Factors")

        flags = []
        if dti > 35: flags.append(("🔴 High DTI", f"{dti}% (>35%)"))
        if fico_score < 650: flags.append(("🔴 Low FICO", f"{fico_score}"))
        if revol_util > 75: flags.append(("🟡 High Credit Utilisation", f"{revol_util}%"))
        if delinq_2yrs > 0: flags.append(("🔴 Recent Delinquencies", f"{delinq_2yrs}"))
        if pub_rec > 0: flags.append(("🔴 Public Records", f"{pub_rec}"))
        if inq_last_6mths > 3: flags.append(("🟡 Multiple Inquiries", f"{inq_last_6mths}"))
        if loan_amnt / max(annual_inc,1) > 0.4: flags.append(("🟡 High Loan-to-Income", f"{loan_amnt/max(annual_inc,1):.2f}x"))

        if not flags:
            st.success("✅ No significant risk flags identified")
        else:
            for label, val in flags:
                st.markdown(f"**{label}** — {val}")

    # ── SHAP EXPLANATION ───────────────────────────────────────────────────────
    if st.checkbox("🔍 Show SHAP Explanation (why this decision?)"):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_df)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv[0],
                    base_values=explainer.expected_value,
                    data=input_df.values[0],
                    feature_names=list(input_df.columns)
                ), show=False
            )
            ax2 = plt.gca()
            ax2.set_facecolor('#0f1117')
            plt.gcf().patch.set_facecolor('#0f1117')
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    # ── LOAN SUMMARY ───────────────────────────────────────────────────────────
    with st.expander("📄 Application Summary"):
        summary = {
            'Loan Amount': f"${loan_amnt:,}",
            'Term': f"{term} months",
            'Interest Rate': f"{int_rate}%",
            'Annual Income': f"${annual_inc:,}",
            'Monthly Installment': f"${installment:,}",
            'DTI': f"{dti}%",
            'FICO Score': fico_score,
            'Grade': grade,
            'Purpose': purpose,
            'Default Probability': risk_pct,
            'Decision': action,
        }
        summary_df = pd.DataFrame(summary.items(), columns=['Field', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
