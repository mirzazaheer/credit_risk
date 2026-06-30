"""
Feature engineering pipeline.
Computes GST, Payment, and Marketplace sub-scores,
then derives the composite credit score used for labelling and inference.
"""

import pandas as pd
import numpy as np

_RAW_FEATURES = [
    "gst_total_filings",
    "gst_ontime_filings",
    "gst_avg_delay_days",
    "gst_annual_turnover_lakhs",
    "seller_rating",
    "order_fulfillment_rate",
    "return_rate",
    "cancellation_rate",
    "seller_account_age_months",
    "payment_punctuality_ratio",
    "late_payment_count_per_month",
    "avg_payment_delay_days",
    "outstanding_liability_ratio",
]

_ENGINEERED_FEATURES = [
    "gst_compliance_score",
    "payment_behaviour_score",
    "marketplace_performance_score",
    "seller_age_norm",
    "credit_score",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered score columns to df. Returns a new DataFrame."""
    df = df.copy()

    # GST Compliance Score (0-100)
    ontime_ratio = df["gst_ontime_filings"] / df["gst_total_filings"].replace(0, 1)
    delay_norm = np.clip(df["gst_avg_delay_days"] / 90.0, 0, 1)
    df["gst_compliance_score"] = np.clip(
        (0.60 * ontime_ratio + 0.40 * (1 - delay_norm)) * 100, 0, 100
    )

    # Payment Behaviour Score (0-100)
    late_norm = np.clip(df["late_payment_count_per_month"] / 20.0, 0, 1)
    pay_delay_norm = np.clip(df["avg_payment_delay_days"] / 90.0, 0, 1)
    df["payment_behaviour_score"] = np.clip(
        (
            0.30 * df["payment_punctuality_ratio"]
            + 0.30 * (1 - late_norm)
            + 0.20 * (1 - pay_delay_norm)
            + 0.20 * (1 - df["outstanding_liability_ratio"])
        )
        * 100,
        0,
        100,
    )

    # Marketplace Performance Score (0-100)
    rating_norm = np.clip(df["seller_rating"] / 5.0, 0, 1)
    df["marketplace_performance_score"] = np.clip(
        (
            0.30 * rating_norm
            + 0.25 * df["order_fulfillment_rate"]
            + 0.25 * (1 - df["return_rate"])
            + 0.20 * (1 - df["cancellation_rate"])
        )
        * 100,
        0,
        100,
    )

    # Seller age normalised (0-1)
    df["seller_age_norm"] = np.clip(df["seller_account_age_months"] / 120.0, 0, 1)

    # Composite Credit Score (0-100)
    df["credit_score"] = (
        0.35 * df["gst_compliance_score"]
        + 0.35 * df["payment_behaviour_score"]
        + 0.30 * df["marketplace_performance_score"]
    )

    return df


def get_feature_columns() -> list[str]:
    """Returns the ordered list of feature columns used by the ML model."""
    return _RAW_FEATURES + _ENGINEERED_FEATURES


def score_to_tier(score: float) -> str:
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Medium Risk"
    return "High Risk"
