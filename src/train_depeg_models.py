# src/train_depeg_models.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CLEAN, OUTPUTS


def walk_forward_split(df: pd.DataFrame, train_frac: float = 0.7):
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def main():
    df = pd.read_csv(DATA_CLEAN / "usdt_depeg_dataset.csv")
    df = df.sort_values("date")

    y = df["depeg"]

    feature_cols = [c for c in df.columns if c not in ["date", "depeg", "is_offpeg"]]

    X = df[feature_cols].copy()

    train_df, test_df = walk_forward_split(df)
    X_train = train_df[feature_cols]
    y_train = train_df["depeg"]
    X_test = test_df[feature_cols]
    y_test = test_df["depeg"]

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Logistic Regression ---
    logit = LogisticRegression(max_iter=2000)
    logit.fit(X_train_scaled, y_train)
    logit_probs = logit.predict_proba(X_test_scaled)[:, 1]

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    results = []

    for name, probs in [("Logistic", logit_probs), ("RandomForest", rf_probs)]:
        results.append({
            "model": name,
            "AUROC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs),
            "Brier": brier_score_loss(y_test, probs)
        })

    out = pd.DataFrame(results)
    print("\nDepeg Prediction Performance (Walk-Forward):")
    print(out.to_string(index=False))

    # Save predictions for later calibration plot
    pred_df = test_df.copy()
    pred_df["logit_prob"] = logit_probs
    pred_df["rf_prob"] = rf_probs

    pred_path = DATA_CLEAN / "depeg_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print("\nSaved predictions:", pred_path)


if __name__ == "__main__":
    main()
