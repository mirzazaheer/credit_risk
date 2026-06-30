# Technical Documentation
## E-commerce MSME Credit Risk Screener
**C2_AIB0326 — Team 1**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Data Layer](#3-data-layer)
4. [Feature Engineering](#4-feature-engineering)
5. [Machine Learning Model](#5-machine-learning-model)
6. [Explainability Layer](#6-explainability-layer)
7. [LLM Report Generator](#7-llm-report-generator)
8. [Web Application](#8-web-application)
9. [Module Reference](#9-module-reference)
10. [Performance Metrics](#10-performance-metrics)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. System Overview

The MSME Credit Risk Screener is an AI-powered credit assessment system designed for Indian e-commerce businesses (MSMEs) that lack traditional credit histories. It uses alternative data signals — GST tax compliance, marketplace performance, and payment behaviour — to generate a credit risk score and human-readable lending recommendation.

### Problem Statement
Traditional credit scoring models rely on CIBIL scores, collateral, and audited financials. E-commerce MSMEs, especially new or informal sellers, often have none of these. This system fills that gap using signals that are native to their digital business activity.

### Key Design Decisions
- **Random Forest over Neural Networks** — interpretability and SHAP compatibility matter more than marginal accuracy gains for a financial decision tool
- **Local LLM (Llama 3 via Ollama)** — no API subscription required, data stays on-premise, suitable for sensitive financial data
- **Synthetic data for prototype** — real GST and marketplace APIs require institutional access; synthetic data with realistic cohort distributions serves as a valid proxy for demonstration and evaluation

---

## 2. Architecture

### Full System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT (Streamlit UI)                │
│  Business Name │ GST Data │ Marketplace │ Payment Behaviour  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│                   features/engineer.py                       │
│                                                             │
│  Raw Inputs ──► GST Compliance Score    (0–100)             │
│             ──► Payment Behaviour Score (0–100)             │
│             ──► Marketplace Score       (0–100)             │
│             ──► Seller Age (normalised) (0–1)               │
│             ──► Composite Credit Score  (0–100)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               RANDOM FOREST CLASSIFIER                       │
│                   model/predict.py                           │
│                                                             │
│  18 features ──► Class probabilities (Low/Medium/High)      │
│              ──► Credit Risk Score (from composite formula) │
│              ──► Risk Tier classification                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌──────────────────────┐   ┌──────────────────────────────────┐
│   EXPLAINABILITY     │   │        LLM REPORT GENERATOR       │
│  model/explainer.py  │   │     llm/report_generator.py       │
│                      │   │                                   │
│  Feature importance  │   │  Llama 3 via Ollama local server  │
│  + direction sign    │   │  Streaming HTTP to port 11434     │
│  ──► Top positive    │   │  Prompt: score + factors + tier   │
│  ──► Top negative    │   │  ──► Narrative report text        │
└──────────┬───────────┘   └──────────────────┬────────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT (Streamlit UI)                      │
│                                                             │
│  ● Credit Risk Score gauge (Plotly)                         │
│  ● Risk Tier badge (Low / Medium / High)                    │
│  ● Probability breakdown (Low / Medium / High %)            │
│  ● Sub-score metrics (GST / Payment / Marketplace)          │
│  ● Feature contribution bar chart                           │
│  ● AI Credit Assessment Report (Llama 3 narrative)          │
│  ● Downloadable .txt report                                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Dependency Map

```
app.py
  ├── model/predict.py
  │     ├── features/engineer.py
  │     └── model/rf_model.joblib  (generated by train.py)
  ├── model/explainer.py
  │     └── model/rf_model.joblib
  └── llm/report_generator.py
        └── Ollama HTTP API (localhost:11434)
```

---

## 3. Data Layer

### 3.1 Data Sources (Production Intent)

| Source | Data Type | Access |
|---|---|---|
| GST Portal (`services.gst.gov.in`) | Filing history, turnover, compliance | Institutional API |
| Udyam Registration (`udyamregistration.gov.in`) | MSME registration, category | Public API |
| Amazon / Flipkart Seller APIs | Ratings, fulfillment, returns | Seller account credentials |
| Payment gateways (Razorpay, Paytm) | Invoice settlement, late payments | Integration agreement |
| CIBIL / CRISIL / SIDBI | Credit history, defaults | Institutional license |

### 3.2 Synthetic Data Generation (`data/generate_synthetic.py`)

Since real APIs require institutional access, training data is generated synthetically using statistically realistic cohort profiles.

**Generation Strategy:** Three cohorts are defined (Low / Medium / High Risk) with distinct statistical distributions. Each cohort has 400 records, giving a balanced dataset of 1,200 total.

#### Feature Distributions by Cohort

| Feature | Low Risk | Medium Risk | High Risk |
|---|---|---|---|
| GST on-time filing ratio | Beta(9, 1.5) ≈ 85–95% | Beta(5, 3) ≈ 55–70% | Beta(2, 6) ≈ 20–40% |
| GST avg delay (days) | Exp(λ=3), capped 0–90 | Exp(λ=12) | Exp(λ=35) |
| Annual turnover (₹ Lakhs) | Uniform(80, 500) | Uniform(30, 200) | Uniform(5, 80) |
| Seller rating | Normal(4.4, 0.5), clip 1–5 | Normal(3.6, 0.5) | Normal(2.5, 0.5) |
| Order fulfillment rate | Normal(0.95, 0.08), clip 0.5–1 | Normal(0.82, 0.08) | Normal(0.65, 0.08) |
| Return rate | Normal(0.06, 0.05), clip 0–0.4 | Normal(0.16, 0.05) | Normal(0.28, 0.05) |
| Cancellation rate | Normal(0.04, 0.04), clip 0–0.3 | Normal(0.12, 0.04) | Normal(0.22, 0.04) |
| Seller account age (months) | Normal(60, 15), clip 1–120 | Normal(36, 15) | Normal(18, 15) |
| Payment punctuality ratio | Normal(0.92, 0.12), clip 0–1 | Normal(0.70, 0.12) | Normal(0.40, 0.12) |
| Late payments / month | Exp(λ=1.0), capped 0–20 | Exp(λ=4.0) | Exp(λ=9.0) |
| Avg payment delay (days) | Exp(λ=4), capped 0–90 | Exp(λ=18) | Exp(λ=40) |
| Outstanding liability ratio | Normal(0.18, 0.10), clip 0–1 | Normal(0.42, 0.10) | Normal(0.68, 0.10) |

Labels are assigned by cohort (not derived from a formula), so the model learns to distinguish genuine risk profiles rather than just reversing a scoring function.

---

## 4. Feature Engineering

**File:** `features/engineer.py`

Raw inputs are transformed into three domain sub-scores plus a composite credit score. All sub-scores are in the range 0–100.

### 4.1 GST Compliance Score

Measures how consistently and promptly the business files GST returns.

```
gst_compliance_score =
    [ 0.60 × (gst_ontime_filings / gst_total_filings)
    + 0.40 × (1 − gst_avg_delay_days / 90) ] × 100
```

| Weight | Component | Rationale |
|---|---|---|
| 60% | On-time filing ratio | Primary compliance indicator |
| 40% | Inverse of delay (normalised over 90 days) | Penalises habitual late filers |

### 4.2 Payment Behaviour Score

Measures reliability in settling financial obligations to vendors and creditors.

```
payment_behaviour_score =
    [ 0.30 × payment_punctuality_ratio
    + 0.30 × (1 − late_payment_count_per_month / 20)
    + 0.20 × (1 − avg_payment_delay_days / 90)
    + 0.20 × (1 − outstanding_liability_ratio) ] × 100
```

| Weight | Component | Rationale |
|---|---|---|
| 30% | Payment punctuality ratio | Overall on-time payment rate |
| 30% | Inverse late payment frequency | Frequency of missed deadlines |
| 20% | Inverse average delay | Severity of lateness |
| 20% | Inverse outstanding liability | Current debt burden |

### 4.3 Marketplace Performance Score

Measures the quality and reliability of the business's e-commerce operations.

```
marketplace_performance_score =
    [ 0.30 × (seller_rating / 5.0)
    + 0.25 × order_fulfillment_rate
    + 0.25 × (1 − return_rate)
    + 0.20 × (1 − cancellation_rate) ] × 100
```

| Weight | Component | Rationale |
|---|---|---|
| 30% | Normalised seller rating | Customer trust and satisfaction |
| 25% | Order fulfillment rate | Operational reliability |
| 25% | Inverse return rate | Product quality proxy |
| 20% | Inverse cancellation rate | Inventory and fulfilment stability |

### 4.4 Composite Credit Score

Final weighted combination of the three domain scores:

```
credit_score = 0.35 × gst_compliance_score
             + 0.35 × payment_behaviour_score
             + 0.30 × marketplace_performance_score
```

**Risk Tier Mapping:**

| Score Range | Risk Tier |
|---|---|
| 80 – 100 | Low Risk |
| 60 – 79 | Medium Risk |
| 0 – 59 | High Risk |

### 4.5 Additional Engineered Feature

```
seller_age_norm = min(seller_account_age_months / 120, 1.0)
```

Normalises seller tenure from months to a 0–1 scale (120 months = 10 years = maximum score).

### 4.6 Full Feature Vector (18 features)

The model receives all 13 raw inputs plus 5 engineered features:

```
Raw (13):   gst_total_filings, gst_ontime_filings, gst_avg_delay_days,
            gst_annual_turnover_lakhs, seller_rating, order_fulfillment_rate,
            return_rate, cancellation_rate, seller_account_age_months,
            payment_punctuality_ratio, late_payment_count_per_month,
            avg_payment_delay_days, outstanding_liability_ratio

Engineered (5): gst_compliance_score, payment_behaviour_score,
                marketplace_performance_score, seller_age_norm, credit_score
```

---

## 5. Machine Learning Model

**File:** `model/train.py`, `model/predict.py`

### 5.1 Algorithm: Random Forest Classifier

Random Forest was chosen over alternatives for the following reasons:

| Criterion | Random Forest | XGBoost | Logistic Regression |
|---|---|---|---|
| Interpretability | High (feature importances) | Medium | High |
| Handles mixed feature scales | Yes (no normalisation needed) | Yes | No |
| Robust to outliers | Yes | Moderate | No |
| Training speed | Fast | Fast | Very fast |
| Accuracy on tabular data | High | Higher | Lower |
| Overfitting risk | Low (ensemble) | Higher without tuning | Low |

### 5.2 Hyperparameters

```python
RandomForestClassifier(
    n_estimators  = 200,       # 200 decision trees in the ensemble
    max_depth     = None,      # Trees grow until leaves are pure
    random_state  = 42,        # Reproducibility
    class_weight  = "balanced",# Equal weight per class despite equal cohort sizes
    n_jobs        = -1,        # Use all available CPU cores
)
```

### 5.3 Training Pipeline

```
1. Load data/msme_data.csv  (1,200 records)
2. Apply engineer_features() → 18-column feature matrix
3. Train/test split: 80% train (960) / 20% test (240), stratified by label
4. Fit RandomForestClassifier
5. Evaluate on test set → classification report + confusion matrix
6. Serialize model to model/rf_model.joblib
7. Save feature column list to model/feature_columns.json
```

### 5.4 Inference Pipeline

```
1. Receive input_dict (13 raw feature values from UI)
2. Wrap in single-row DataFrame
3. Apply engineer_features() → 18-column feature row
4. Load rf_model.joblib + feature_columns.json
5. model.predict_proba(X) → [P(Low), P(Medium), P(High)]
6. credit_score taken directly from engineered feature (not from model probability)
7. risk_tier derived from score thresholds (not from argmax of probabilities)
8. Return: score, tier, probabilities, sub-scores, engineered_df
```

> **Why use the formula score rather than the model's output for the score?**
> The composite formula score is deterministic and directly interpretable. Model probabilities reflect learned patterns but are less transparent. Lenders can audit the formula; they cannot easily audit a forest of 200 trees.

### 5.5 Model Artifacts

| File | Contents | Size |
|---|---|---|
| `model/rf_model.joblib` | Serialised RandomForestClassifier | ~15–25 MB |
| `model/feature_columns.json` | Ordered list of 18 feature names | <1 KB |

---

## 6. Explainability Layer

**File:** `model/explainer.py`

### 6.1 Approach

Rather than SHAP (which requires `numba`, incompatible with Python 3.14), the explainer uses **Random Forest feature importances with direction analysis**.

### 6.2 Algorithm

For each feature `f` with model importance `I(f)`:

```
1. Get feature importance:  I(f) = model.feature_importances_[i]   (always ≥ 0)

2. Determine direction sign:
   - If feature is a "negative direction" feature (higher = worse, e.g. return_rate):
       sign = -1 if value > threshold else +1
   - Otherwise (higher = better, e.g. seller_rating):
       sign = +1 if value > threshold else -1

3. Signed importance = sign × I(f)

4. Positive factors: features with signed_importance > 0  (helping creditworthiness)
   Negative factors: features with signed_importance < 0  (hurting creditworthiness)
```

### 6.3 Negative Direction Features

Features where a higher value is worse for creditworthiness:

```python
NEGATIVE_DIRECTION_FEATURES = {
    "gst_avg_delay_days",
    "return_rate",
    "cancellation_rate",
    "late_payment_count_per_month",
    "avg_payment_delay_days",
    "outstanding_liability_ratio",
}
```

### 6.4 Neutral Thresholds

Each feature has a defined midpoint. Values above/below determine direction of impact:

| Feature | Threshold | Rationale |
|---|---|---|
| seller_rating | 3.5 | Midpoint of 1–5 scale |
| order_fulfillment_rate | 0.80 | Industry minimum acceptable level |
| return_rate | 0.15 | Industry average |
| payment_punctuality_ratio | 0.75 | Acceptable payment reliability |
| gst_avg_delay_days | 15 | Two-week grace interpretation |
| outstanding_liability_ratio | 0.40 | Moderate debt burden threshold |

### 6.5 Output

```python
{
  "positive_factors": [
      {"feature": "credit_score", "label": "Composite Credit Score",
       "shap_value": 0.274, "value": 88.4},
      ...  # top 5 by signed importance
  ],
  "negative_factors": [
      {"feature": "return_rate", "label": "Product Return Rate",
       "shap_value": -0.031, "value": 0.18},
      ...  # top 5 by signed importance
  ],
  "all_shap": {"feature_name": signed_importance, ...},
  "feature_values": {"feature_name": current_value, ...}
}
```

---

## 7. LLM Report Generator

**File:** `llm/report_generator.py`

### 7.1 Model

**Llama 3 8B** (`llama3`) running locally via **Ollama** on port `11434`.

- No API key required
- Data stays on-premise (important for financial use cases)
- Works offline after initial download
- Free to use with no rate limits

### 7.2 Communication

Uses Ollama's REST API in **streaming mode** to avoid read timeouts on long outputs:

```
POST http://localhost:11434/api/generate
{
    "model": "llama3",
    "prompt": "<credit assessment prompt>",
    "stream": true
}
```

Response is consumed line-by-line (each line is a JSON chunk). The full response is assembled from the `response` field of each chunk until `done: true`.

### 7.3 Prompt Structure

```
System context: Senior credit analyst at an Indian MSME lending institution

Input block:
  - Business name
  - Credit Risk Score (x/100)
  - Risk Tier
  - GST Compliance sub-score
  - Payment Behaviour sub-score
  - Marketplace Performance sub-score
  - Key strengths (positive factor labels)
  - Key risk factors (negative factor labels)

Output instructions:
  1. Executive Summary (2–3 sentences)
  2. Strengths Analysis (paragraph form)
  3. Risk Concerns (paragraph form)
  4. Lending Recommendation: Approve / Conditional Approval / Decline
     with reasoning and suggested loan conditions
```

### 7.4 Fallback Behaviour

If Ollama is not running or the request fails for any reason, the system silently falls back to a **rule-based template report**:

```python
if risk_tier == "Low Risk":
    recommendation = "APPROVE — strong financial discipline and operational reliability"
elif risk_tier == "Medium Risk":
    recommendation = "CONDITIONAL APPROVAL — consider additional collateral or shorter tenure"
else:
    recommendation = "DECLINE / HIGH SCRUTINY — significant risk indicators present"
```

The app is fully functional without Ollama; the AI narrative is an enhancement.

---

## 8. Web Application

**File:** `app.py`

### 8.1 Framework

Built with **Streamlit** — a Python web framework that converts Python scripts into interactive web apps without requiring HTML/CSS/JavaScript.

### 8.2 UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ SIDEBAR                    │ MAIN PANEL                      │
│                            │                                 │
│ Business Name              │ Title + subtitle                │
│                            │                                 │
│ ── GST & Financial ──      │ Assessment for: [Name]          │
│ Filing type                │                                 │
│ On-time filings            │ [Gauge]  [Risk Badge]  [Scores] │
│ Avg delay                  │          [Probability] [Metrics]│
│ Annual turnover            │                                 │
│                            │ ─────────────────────────────── │
│ ── Marketplace ──          │                                 │
│ Seller rating              │ Key Factors Influencing Score   │
│ Fulfillment rate           │ [Horizontal bar chart]          │
│ Return rate                │                                 │
│ Cancellation rate          │ ─────────────────────────────── │
│ Account age                │                                 │
│                            │ AI Credit Assessment Report     │
│ ── Payment ──              │ [Expandable text panel]         │
│ Punctuality ratio          │                                 │
│ Late payments/month        │ [Download Report button]        │
│ Payment delay              │                                 │
│ Liability ratio            │                                 │
│                            │                                 │
│ [Assess Credit Risk btn]   │                                 │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 Execution Flow

```
1. User fills sidebar form
2. User clicks "Assess Credit Risk"
3. predict(input_data) → credit_score, risk_tier, sub_scores, engineered_df
4. explain(engineered_df) → positive_factors, negative_factors
5. generate_credit_report(...) → AI narrative text
6. Render gauge, badge, metrics, chart, report, download button
```

### 8.4 Charts

| Chart | Library | Type |
|---|---|---|
| Credit Risk Score | Plotly `go.Indicator` | Gauge with colour bands |
| Feature Contributions | Plotly `go.Bar` | Horizontal bar (green/red) |

---

## 9. Module Reference

### `features/engineer.py`

| Function | Signature | Returns |
|---|---|---|
| `engineer_features` | `(df: pd.DataFrame) -> pd.DataFrame` | Input df with 5 added score columns |
| `get_feature_columns` | `() -> list[str]` | Ordered list of 18 ML feature names |
| `score_to_tier` | `(score: float) -> str` | "Low Risk" / "Medium Risk" / "High Risk" |

---

### `model/predict.py`

| Function | Signature | Returns |
|---|---|---|
| `predict` | `(input_dict: dict) -> dict` | `{credit_score, risk_tier, probabilities, sub_scores, engineered_df}` |

`input_dict` keys: all 13 raw feature names (matching CSV column names, excluding `msme_name`, `gstin`, `risk_label`).

---

### `model/explainer.py`

| Function | Signature | Returns |
|---|---|---|
| `explain` | `(engineered_df: pd.DataFrame, top_n: int = 5) -> dict` | `{positive_factors, negative_factors, all_shap, feature_values}` |

---

### `llm/report_generator.py`

| Function | Signature | Returns |
|---|---|---|
| `generate_credit_report` | `(msme_name, credit_score, risk_tier, sub_scores, positive_factors, negative_factors) -> str` | Narrative report text |
| `_rule_based_report` | (same args) | Structured template fallback |

---

### `data/generate_synthetic.py`

Run directly: `python data/generate_synthetic.py`

Outputs `data/msme_data.csv` with 1,200 records and columns:
`msme_name, gstin, gst_total_filings, gst_ontime_filings, gst_avg_delay_days, gst_annual_turnover_lakhs, seller_rating, order_fulfillment_rate, return_rate, cancellation_rate, seller_account_age_months, payment_punctuality_ratio, late_payment_count_per_month, avg_payment_delay_days, outstanding_liability_ratio, risk_label`

---

## 10. Performance Metrics

Evaluated on held-out 20% test set (240 records, 80 per class):

```
              precision    recall  f1-score   support

    Low Risk       0.99      1.00      0.99        80
 Medium Risk       0.99      1.00      0.99        80
   High Risk       1.00      0.97      0.99        80

    accuracy                           0.99       240
   macro avg       0.99      0.99      0.99       240
weighted avg       0.99      0.99      0.99       240

Confusion Matrix:
[[80  0  0]
 [ 1 78  1]
 [ 0  0 80]]
```

**Note on high accuracy:** The 99% accuracy reflects that the three cohorts have well-separated distributions in feature space (by design). Real-world performance would depend on the quality and coverage of actual MSME data.

---

## 11. Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Synthetic training data | Model learns simulated rather than real patterns | Replace with real MSME data when API access is available |
| No real-time data pipeline | Manual input only | Integrate GST Portal and marketplace APIs |
| Feature importance ≠ SHAP | Less precise than SHAP values | Upgrade when a numba-compatible Python version is used |
| Llama 3 requires 8GB+ RAM | May be slow on low-spec machines | Use `--skip-ollama` flag; or integrate a cloud LLM |
| No authentication | App is open to anyone on the network | Add login layer before production deployment |
| Fixed scoring weights | Weights were manually set, not learned | Learn optimal weights from labelled real-world data |

### Suggested Improvements for Production

1. **Live data ingestion** — Connect to GST Portal GSTIN lookup API and marketplace seller APIs
2. **Model retraining pipeline** — Periodic retraining as new repayment outcome data becomes available
3. **SHAP integration** — Switch to `shap.TreeExplainer` once Python/numba compatibility is resolved
4. **Audit trail** — Log every assessment with inputs, outputs, and timestamp for regulatory compliance
5. **Loan outcome feedback loop** — Tag assessments with eventual loan performance to improve model labels
6. **Multi-lender support** — Allow different risk threshold configurations per lender's risk appetite
7. **Regional language output** — Generate reports in Hindi or regional languages via translation layer
