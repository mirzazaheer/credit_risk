"""
Generates synthetic MSME dataset for training the credit risk model.
Run: python data/generate_synthetic.py
Output: data/msme_data.csv
"""

import numpy as np
import pandas as pd
from faker import Faker
import os

fake = Faker("en_IN")
np.random.seed(42)

N = 1200  # number of MSME records


def _generate_cohort(n, profile):
    """Generate n records following a given risk profile."""
    p = profile
    records = []
    for _ in range(n):
        gst_total = np.random.choice([4, 12], p=[0.4, 0.6])
        ontime_frac = np.clip(np.random.beta(p["gst_alpha"], p["gst_beta"]), 0, 1)
        gst_ontime = int(round(ontime_frac * gst_total))

        record = {
            "msme_name": fake.company(),
            "gstin": fake.bothify(text="??#?????????", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            "gst_total_filings": gst_total,
            "gst_ontime_filings": gst_ontime,
            "gst_avg_delay_days": int(np.clip(np.random.exponential(p["delay_mean"]), 0, 90)),
            "gst_annual_turnover_lakhs": round(np.random.uniform(*p["turnover_range"]), 2),
            "seller_rating": round(np.clip(np.random.normal(p["rating_mean"], 0.5), 1.0, 5.0), 1),
            "order_fulfillment_rate": round(np.clip(np.random.normal(p["fulfil_mean"], 0.08), 0.50, 1.00), 3),
            "return_rate": round(np.clip(np.random.normal(p["return_mean"], 0.05), 0.00, 0.40), 3),
            "cancellation_rate": round(np.clip(np.random.normal(p["cancel_mean"], 0.04), 0.00, 0.30), 3),
            "seller_account_age_months": int(np.clip(np.random.normal(p["age_mean"], 15), 1, 120)),
            "payment_punctuality_ratio": round(np.clip(np.random.normal(p["pay_punct_mean"], 0.12), 0.0, 1.0), 3),
            "late_payment_count_per_month": int(np.clip(np.random.exponential(p["late_count_mean"]), 0, 20)),
            "avg_payment_delay_days": int(np.clip(np.random.exponential(p["pay_delay_mean"]), 0, 90)),
            "outstanding_liability_ratio": round(np.clip(np.random.normal(p["liab_mean"], 0.10), 0.0, 1.0), 3),
        }
        records.append(record)
    return records


# Three cohort profiles matching risk tiers
profiles = {
    "Low Risk": {
        "gst_alpha": 9, "gst_beta": 1.5,      # high on-time filing ratio
        "delay_mean": 3,                        # short delays
        "turnover_range": (80, 500),
        "rating_mean": 4.4,
        "fulfil_mean": 0.95,
        "return_mean": 0.06,
        "cancel_mean": 0.04,
        "age_mean": 60,
        "pay_punct_mean": 0.92,
        "late_count_mean": 1.0,
        "pay_delay_mean": 4,
        "liab_mean": 0.18,
    },
    "Medium Risk": {
        "gst_alpha": 5, "gst_beta": 3,
        "delay_mean": 12,
        "turnover_range": (30, 200),
        "rating_mean": 3.6,
        "fulfil_mean": 0.82,
        "return_mean": 0.16,
        "cancel_mean": 0.12,
        "age_mean": 36,
        "pay_punct_mean": 0.70,
        "late_count_mean": 4.0,
        "pay_delay_mean": 18,
        "liab_mean": 0.42,
    },
    "High Risk": {
        "gst_alpha": 2, "gst_beta": 6,
        "delay_mean": 35,
        "turnover_range": (5, 80),
        "rating_mean": 2.5,
        "fulfil_mean": 0.65,
        "return_mean": 0.28,
        "cancel_mean": 0.22,
        "age_mean": 18,
        "pay_punct_mean": 0.40,
        "late_count_mean": 9.0,
        "pay_delay_mean": 40,
        "liab_mean": 0.68,
    },
}

cohort_sizes = {"Low Risk": 400, "Medium Risk": 400, "High Risk": 400}

rows = []
for label, profile in profiles.items():
    cohort = _generate_cohort(cohort_sizes[label], profile)
    for r in cohort:
        r["risk_label"] = label
    rows.extend(cohort)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), "msme_data.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} records → {out_path}")
print(df["risk_label"].value_counts().to_string())
