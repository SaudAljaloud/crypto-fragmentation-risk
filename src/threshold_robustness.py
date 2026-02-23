# src/threshold_robustness.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CLEAN, OUTPUTS


THRESHOLDS = [0.0010, 0.0015, 0.0020]  # 10, 15, 20 bps
TRAIN_FRAC = 0.70


def build_dataset(usdt: pd.DataFrame, thresh: float) -> pd.DataFrame:
    df = usdt.sort_values("date").copy()

    # label (today off-peg)
    df["abs_dev"] = (df["close_px"] - 1.0).abs()
    df["is_offpeg"] = (df["abs_dev"] > thresh).astype(int)

    # predict NEXT DAY depeg (no leakage)
    df["depeg"] = df["is_offpeg"].shift(-1)

    # features (lag-only)
    df["ret"] = np.log(df["close_px"]).diff()
    df["ret_lag1"] = df["ret"].shift(1)
    df["ret_lag2"] = df["ret"].shift(2)
    df["ret_lag3"] = df["ret"].shift(3)

    for w in [7, 14, 30]:
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()

    df["volume_lag1"] = df["volume_base"].shift(1)
    df["volume_lag3"] = df["volume_base"].shift(3)

    for w in [7, 14]:
        df[f"volume_vol_{w}"] = df["volume_base"].rolling(w).std()

    # clean
    df = df.dropna().reset_index(drop=True)

    # keep only modeling columns
    feature_cols = [
        "ret_lag1", "ret_lag2", "ret_lag3",
        "ret_vol_7", "ret_vol_14", "ret_vol_30",
        "volume_lag1", "volume_lag3",
        "volume_vol_7", "volume_vol_14",
    ]
    keep = ["date", "depeg"] + feature_cols
    return df[keep]


def walk_forward_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def safe_auc(y_true, probs) -> float:
    # AUROC undefined if only one class in y_true
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, probs)


def safe_auprc(y_true, probs) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, probs)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Load USDT Kraken series
    usdt = pd.read_csv(DATA_CLEAN / "usdt_kraken.csv")
    usdt = usdt.rename(columns={"date_utc": "date"})
    usdt["date"] = pd.to_datetime(usdt["date"])
    usdt = usdt[["date", "close_px", "volume_base"]].copy()

    rows = []

    for thresh in THRESHOLDS:
        df = build_dataset(usdt, thresh)

        train_df, test_df = walk_forward_split(df)
        feature_cols = [c for c in df.columns if c not in ["date", "depeg"]]

        X_train = train_df[feature_cols]
        y_train = train_df["depeg"].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df["depeg"].astype(int)

        # Logistic (scaled)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        logit = LogisticRegression(max_iter=2000)
        logit.fit(X_train_s, y_train)
        logit_probs = logit.predict_proba(X_test_s)[:, 1]

        # RandomForest (no scaling)
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            random_state=42
        )
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        base_rate_all = df["depeg"].mean()
        base_rate_test = y_test.mean()

        for model_name, probs in [("Logistic", logit_probs), ("RandomForest", rf_probs)]:
            rows.append({
                "threshold": thresh,
                "threshold_bps": int(round(thresh * 10000)),
                "model": model_name,

                "n_total": len(df),
                "n_train": len(train_df),
                "n_test": len(test_df),

                "events_total": int(df["depeg"].sum()),
                "events_test": int(y_test.sum()),
                "base_rate_all": float(base_rate_all),
                "base_rate_test": float(base_rate_test),

                "AUROC": float(safe_auc(y_test, probs)),
                "AUPRC": float(safe_auprc(y_test, probs)),
                "Brier": float(brier_score_loss(y_test, probs)),
            })

        print(f"Done threshold={thresh} (bps={int(round(thresh*10000))}) "
              f"events_total={int(df['depeg'].sum())} base_rate={base_rate_all:.4f}")

    out = pd.DataFrame(rows).sort_values(["threshold", "model"]).reset_index(drop=True)

    out_path = OUTPUTS / "threshold_robustness.csv"
    out.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("\nRobustness table (key columns):")
    print(out[["threshold_bps", "model", "events_total", "base_rate_all", "AUROC", "AUPRC", "Brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
