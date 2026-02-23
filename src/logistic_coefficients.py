# src/logistic_coefficients.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CLEAN


THRESH = 0.001  # 10 bps


def build_dataset(usdt: pd.DataFrame) -> pd.DataFrame:
    df = usdt.sort_values("date").copy()

    df["abs_dev"] = (df["close_px"] - 1.0).abs()
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
    df["depeg"] = df["is_offpeg"].shift(-1)

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

    df = df.dropna().reset_index(drop=True)

    return df


def main():
    usdt = pd.read_csv(DATA_CLEAN / "usdt_kraken.csv")
    usdt = usdt.rename(columns={"date_utc": "date"})
    usdt["date"] = pd.to_datetime(usdt["date"])
    usdt = usdt[["date", "close_px", "volume_base"]]

    df = build_dataset(usdt)

    split = int(len(df) * 0.7)
    train = df.iloc[:split].copy()

    feature_cols = [
        "ret_lag1", "ret_lag2", "ret_lag3",
        "ret_vol_7", "ret_vol_14", "ret_vol_30",
        "volume_lag1", "volume_lag3",
        "volume_vol_7", "volume_vol_14",
    ]

    X = train[feature_cols]
    y = train["depeg"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coef": model.coef_[0]
    })

    coefs["abs_coef"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs_coef", ascending=False)

    print("\nLogistic Coefficients (standardized features, 10 bps threshold):")
    print(coefs[["feature", "coef"]].to_string(index=False))


if __name__ == "__main__":
    main()
