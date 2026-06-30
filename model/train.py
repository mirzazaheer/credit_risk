"""
Train the RandomForest credit risk classifier and save artifacts.
Run from project root: python model/train.py
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.engineer import engineer_features, get_feature_columns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "msme_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_columns.json")

LABEL_ORDER = ["Low Risk", "Medium Risk", "High Risk"]


def train():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=LABEL_ORDER))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred, labels=LABEL_ORDER))

    joblib.dump(clf, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Feature list saved → {FEATURES_PATH}")


if __name__ == "__main__":
    train()
