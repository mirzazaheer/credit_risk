"""
Load the trained model and run inference on a single MSME input dict.
"""

import os
import sys
import json
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.engineer import engineer_features, score_to_tier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_columns.json")

_model = None
_feature_cols = None


def _load():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feature_cols = json.load(f)


def predict(input_dict: dict) -> dict:
    """
    Args:
        input_dict: raw MSME feature values (matches columns in msme_data.csv,
                    excluding msme_name, gstin, risk_label)
    Returns:
        {
          "credit_score": float,        # 0-100
          "risk_tier": str,             # Low / Medium / High Risk
          "probabilities": {label: float},
          "sub_scores": {"gst": float, "payment": float, "marketplace": float},
          "engineered_df": pd.DataFrame  # for SHAP
        }
    """
    _load()

    df = pd.DataFrame([input_dict])
    df = engineer_features(df)

    X = df[_feature_cols]
    proba = _model.predict_proba(X)[0]
    classes = _model.classes_

    credit_score = float(df["credit_score"].iloc[0])
    risk_tier = score_to_tier(credit_score)

    return {
        "credit_score": round(credit_score, 1),
        "risk_tier": risk_tier,
        "probabilities": {cls: round(float(p), 3) for cls, p in zip(classes, proba)},
        "sub_scores": {
            "gst": round(float(df["gst_compliance_score"].iloc[0]), 1),
            "payment": round(float(df["payment_behaviour_score"].iloc[0]), 1),
            "marketplace": round(float(df["marketplace_performance_score"].iloc[0]), 1),
        },
        "engineered_df": df,
    }
