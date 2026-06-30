"""
Feature-importance-based explainability for the RandomForest credit risk model.

Uses the model's feature_importances_ combined with value-vs-threshold direction
to determine whether each feature is a positive or negative contributor.
This replaces SHAP, which requires numba (incompatible with Python 3.14).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_columns.json")

_model = None
_feature_cols = None

FEATURE_LABELS = {
    "gst_total_filings": "GST Filing Count",
    "gst_ontime_filings": "On-time GST Filings",
    "gst_avg_delay_days": "GST Filing Delay (days)",
    "gst_annual_turnover_lakhs": "Annual Turnover (₹ Lakhs)",
    "seller_rating": "Seller Rating",
    "order_fulfillment_rate": "Order Fulfillment Rate",
    "return_rate": "Product Return Rate",
    "cancellation_rate": "Order Cancellation Rate",
    "seller_account_age_months": "Seller Account Age (months)",
    "payment_punctuality_ratio": "Payment Punctuality",
    "late_payment_count_per_month": "Late Payments / Month",
    "avg_payment_delay_days": "Avg Payment Delay (days)",
    "outstanding_liability_ratio": "Outstanding Liability Ratio",
    "gst_compliance_score": "GST Compliance Score",
    "payment_behaviour_score": "Payment Behaviour Score",
    "marketplace_performance_score": "Marketplace Performance Score",
    "seller_age_norm": "Seller Tenure (normalised)",
    "credit_score": "Composite Credit Score",
}

# Features where a HIGHER value means WORSE creditworthiness
NEGATIVE_DIRECTION_FEATURES = {
    "gst_avg_delay_days",
    "return_rate",
    "cancellation_rate",
    "late_payment_count_per_month",
    "avg_payment_delay_days",
    "outstanding_liability_ratio",
}


def _load():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feature_cols = json.load(f)


def _get_thresholds() -> dict:
    """Neutral midpoint for each feature (used to determine direction of impact)."""
    return {
        "gst_total_filings": 8,
        "gst_ontime_filings": 8,
        "gst_avg_delay_days": 15,
        "gst_annual_turnover_lakhs": 100,
        "seller_rating": 3.5,
        "order_fulfillment_rate": 0.80,
        "return_rate": 0.15,
        "cancellation_rate": 0.12,
        "seller_account_age_months": 24,
        "payment_punctuality_ratio": 0.75,
        "late_payment_count_per_month": 5,
        "avg_payment_delay_days": 20,
        "outstanding_liability_ratio": 0.40,
        "gst_compliance_score": 65,
        "payment_behaviour_score": 65,
        "marketplace_performance_score": 65,
        "seller_age_norm": 0.20,
        "credit_score": 70,
    }


def explain(engineered_df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Explain model output using RF feature importances and value direction.

    Args:
        engineered_df: DataFrame row already processed by engineer_features()
        top_n: number of top factors per category to return

    Returns:
        {
          "positive_factors": [{"feature", "label", "shap_value", "value"}],
          "negative_factors": [...],
          "all_shap": {feature: signed_importance},
          "feature_values": {feature: value}
        }
    """
    _load()

    importances = _model.feature_importances_  # shape: (n_features,)
    feat_vals = engineered_df[_feature_cols].iloc[0].to_dict()
    thresholds = _get_thresholds()

    factors = []
    for i, feat in enumerate(_feature_cols):
        importance = float(importances[i])
        val = feat_vals[feat]
        threshold = thresholds.get(feat, 0)

        # Determine sign: does this feature's current value help or hurt creditworthiness?
        if feat in NEGATIVE_DIRECTION_FEATURES:
            sign = -1 if val > threshold else 1
        else:
            sign = 1 if val > threshold else -1

        signed_importance = sign * importance
        factors.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "shap_value": round(signed_importance, 6),
            "value": val,
        })

    factors_sorted = sorted(factors, key=lambda x: x["shap_value"], reverse=True)
    positive = [f for f in factors_sorted if f["shap_value"] > 0][:top_n]
    negative = sorted([f for f in factors_sorted if f["shap_value"] < 0],
                      key=lambda x: x["shap_value"])[:top_n]

    return {
        "positive_factors": positive,
        "negative_factors": negative,
        "all_shap": {f["feature"]: f["shap_value"] for f in factors},
        "feature_values": feat_vals,
    }
