"""
E-commerce MSME Credit Risk Screener — Streamlit UI
Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go

from model.predict import predict
from model.explainer import explain
from llm.report_generator import generate_credit_report

st.set_page_config(
    page_title="MSME Credit Risk Screener",
    page_icon=None,
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    .risk-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.03em;
    }
    .risk-low    { background:#1a6630; color:#ffffff; border: 1px solid #1a6630; }
    .risk-medium { background:#7a5c00; color:#ffffff; border: 1px solid #7a5c00; }
    .risk-high   { background:#8b1a1a; color:#ffffff; border: 1px solid #8b1a1a; }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
        margin-bottom: 4px;
    }
    .prob-row { font-size: 0.9rem; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Input Form ──────────────────────────────────────────────────────
st.sidebar.title("MSME Input Form")
st.sidebar.markdown("Enter the business details and click **Assess Credit Risk**.")

msme_name = st.sidebar.text_input("Business Name", value="Sunrise Traders Pvt Ltd")

st.sidebar.markdown("#### GST & Financial Data")
gst_total_filings = st.sidebar.selectbox("GST Filing Type", [12, 4],
    format_func=lambda x: f"Monthly ({x} filings/year)" if x == 12 else f"Quarterly ({x} filings/year)")
gst_ontime_filings = st.sidebar.slider("On-time Filings", 0, gst_total_filings, int(gst_total_filings * 0.85))
gst_avg_delay_days = st.sidebar.slider("Average Filing Delay (days)", 0, 90, 5)
gst_annual_turnover_lakhs = st.sidebar.number_input("Annual Turnover (Rs. Lakhs)", min_value=1.0, max_value=1000.0, value=120.0, step=5.0)

st.sidebar.markdown("#### Marketplace Performance")
seller_rating = st.sidebar.slider("Seller Rating", 1.0, 5.0, 4.2, step=0.1)
order_fulfillment_rate = st.sidebar.slider("Order Fulfillment Rate", 0.50, 1.00, 0.92, step=0.01)
return_rate = st.sidebar.slider("Return Rate", 0.00, 0.40, 0.08, step=0.01)
cancellation_rate = st.sidebar.slider("Cancellation Rate", 0.00, 0.30, 0.05, step=0.01)
seller_account_age_months = st.sidebar.slider("Seller Account Age (months)", 1, 120, 36)

st.sidebar.markdown("#### Payment Behaviour")
payment_punctuality_ratio = st.sidebar.slider("Payment Punctuality Ratio", 0.0, 1.0, 0.88, step=0.01)
late_payment_count_per_month = st.sidebar.slider("Late Payments / Month", 0, 20, 2)
avg_payment_delay_days = st.sidebar.slider("Avg Payment Delay (days)", 0, 90, 7)
outstanding_liability_ratio = st.sidebar.slider("Outstanding Liability Ratio", 0.0, 1.0, 0.22, step=0.01)

assess_btn = st.sidebar.button("Assess Credit Risk", type="primary", use_container_width=True)

# ── Main Panel ────────────────────────────────────────────────────────────────
st.title("E-commerce MSME Credit Risk Screener")
st.markdown("AI-powered credit assessment using GST compliance, marketplace performance, and payment behaviour data.")

if not assess_btn:
    st.info("Complete the MSME details in the sidebar and click **Assess Credit Risk** to begin.")
    st.markdown("""
    **How it works:**
    1. Enter the MSME's financial and operational data in the sidebar
    2. The system computes a **Credit Risk Score (0–100)** using a trained Random Forest model
    3. Feature analysis identifies the key **positive and negative factors** driving the score
    4. An AI analyst (Llama 3) generates a **human-readable credit assessment report**
    """)
    st.stop()

# ── Run Assessment ─────────────────────────────────────────────────────────────
input_data = {
    "gst_total_filings": gst_total_filings,
    "gst_ontime_filings": gst_ontime_filings,
    "gst_avg_delay_days": gst_avg_delay_days,
    "gst_annual_turnover_lakhs": gst_annual_turnover_lakhs,
    "seller_rating": seller_rating,
    "order_fulfillment_rate": order_fulfillment_rate,
    "return_rate": return_rate,
    "cancellation_rate": cancellation_rate,
    "seller_account_age_months": seller_account_age_months,
    "payment_punctuality_ratio": payment_punctuality_ratio,
    "late_payment_count_per_month": late_payment_count_per_month,
    "avg_payment_delay_days": avg_payment_delay_days,
    "outstanding_liability_ratio": outstanding_liability_ratio,
}

with st.spinner("Running credit risk assessment..."):
    result = predict(input_data)
    shap_result = explain(result["engineered_df"])

credit_score = result["credit_score"]
risk_tier = result["risk_tier"]
sub_scores = result["sub_scores"]

# ── Row 1: Score + Tier ────────────────────────────────────────────────────────
st.markdown(f"### Assessment for: **{msme_name}**")

col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

tier_css = {"Low Risk": "risk-low", "Medium Risk": "risk-medium", "High Risk": "risk-high"}

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Credit Risk Score", "font": {"size": 15}},
        number={"font": {"size": 42}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickvals": [0, 20, 40, 60, 80, 100],
            },
            "bar": {"color": "#1a56db"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60],  "color": "rgba(180,40,40,0.25)"},
                {"range": [60, 80], "color": "rgba(200,160,0,0.20)"},
                {"range": [80, 100],"color": "rgba(30,140,60,0.20)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": credit_score,
            },
        },
    ))
    fig_gauge.update_layout(
        height=220,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("<p class='section-label'>Risk Tier</p>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='risk-badge {tier_css[risk_tier]}'>{risk_tier}</div>",
        unsafe_allow_html=True,
    )
    proba = result["probabilities"]
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Probability Breakdown</p>", unsafe_allow_html=True)
    for label, prob in sorted(proba.items(), key=lambda x: ["Low Risk", "Medium Risk", "High Risk"].index(x[0])):
        st.markdown(f"<div class='prob-row'>{label} &nbsp; <strong>{prob*100:.1f}%</strong></div>", unsafe_allow_html=True)

with col3:
    st.markdown("<p class='section-label'>Sub-scores</p>", unsafe_allow_html=True)
    st.metric("GST Compliance", f"{sub_scores['gst']:.1f} / 100")
    st.metric("Payment Behaviour", f"{sub_scores['payment']:.1f} / 100")

with col4:
    st.markdown("<p class='section-label'>&nbsp;</p>", unsafe_allow_html=True)
    st.metric("Marketplace Performance", f"{sub_scores['marketplace']:.1f} / 100")
    st.metric("Composite Score", f"{credit_score} / 100")

st.divider()

# ── Row 2: Factor Chart ───────────────────────────────────────────────────────
st.markdown("### Key Factors Influencing This Score")

all_factors = shap_result["positive_factors"] + shap_result["negative_factors"]
all_factors_sorted = sorted(all_factors, key=lambda x: x["shap_value"])

labels = [f["label"] for f in all_factors_sorted]
values = [f["shap_value"] for f in all_factors_sorted]
colors = ["#c0392b" if v < 0 else "#1e8449" for v in values]

fig_factors = go.Figure(go.Bar(
    x=values,
    y=labels,
    orientation="h",
    marker_color=colors,
    marker_line_width=0,
    text=[f"{v:+.4f}" for v in values],
    textposition="outside",
))
fig_factors.update_layout(
    title={
        "text": "Feature Contributions — Positive values improve creditworthiness; negative values reduce it",
        "font": {"size": 13},
        "x": 0,
    },
    xaxis_title="Contribution Score",
    yaxis_title="",
    height=max(300, len(labels) * 40),
    margin=dict(l=10, r=90, t=50, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        zeroline=True,
        zerolinecolor="rgba(150,150,150,0.6)",
        zerolinewidth=1.5,
        gridcolor="rgba(150,150,150,0.15)",
    ),
)
st.plotly_chart(fig_factors, use_container_width=True)

st.divider()

# ── Row 3: AI Report ──────────────────────────────────────────────────────────
st.markdown("### AI Credit Assessment Report")

pos_factors = shap_result["positive_factors"]
neg_factors = shap_result["negative_factors"]

with st.spinner("Generating credit assessment narrative via Llama 3..."):
    report = generate_credit_report(
        msme_name=msme_name,
        credit_score=credit_score,
        risk_tier=risk_tier,
        sub_scores=sub_scores,
        positive_factors=pos_factors,
        negative_factors=neg_factors,
    )

with st.expander("View Full Credit Assessment Report", expanded=True):
    st.text(report)

# ── Download ──────────────────────────────────────────────────────────────────
download_content = f"""E-COMMERCE MSME CREDIT RISK ASSESSMENT
Generated by: MSME Credit Risk Screener (C2_AIB0326 - Team 1)
========================================

Business Name : {msme_name}
Credit Score  : {credit_score} / 100
Risk Tier     : {risk_tier}

Sub-scores:
  GST Compliance          : {sub_scores['gst']} / 100
  Payment Behaviour       : {sub_scores['payment']} / 100
  Marketplace Performance : {sub_scores['marketplace']} / 100

Risk Probabilities:
{chr(10).join(f"  {k}: {v*100:.1f}%" for k, v in result['probabilities'].items())}

Key Positive Factors:
{chr(10).join(f"  + {f['label']} ({f['shap_value']:+.4f})" for f in pos_factors)}

Key Risk Factors:
{chr(10).join(f"  - {f['label']} ({f['shap_value']:+.4f})" for f in neg_factors)}

========================================
CREDIT ASSESSMENT REPORT
========================================
{report}
"""

st.download_button(
    label="Download Full Report (.txt)",
    data=download_content,
    file_name=f"credit_report_{msme_name.replace(' ', '_')}.txt",
    mime="text/plain",
)
