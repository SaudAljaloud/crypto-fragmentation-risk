# src/build_depeg_dataset_02.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_CLEAN

# ---------------------------------
# Label definition
# ---------------------------------
THRESH = 0.0015   # 15 basis points
MIN_DAYS = 1


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    df["peg"] = 1.0
    df["dev"] = df["close_px"] - df["peg"]
    df["abs_dev"] = df["dev"].abs()

    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)

    # Predict NEXT DAY depeg (no leakage)
    df["depeg"] = df["is_offpeg"].shift(-1)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # --- Returns ---
    df["ret"] = np.log(df["close_px"]).diff()

    # Lagged returns (past info only)
    df["ret_lag1"] = df["ret"].shift(1)
    df["ret_lag2"] = df["ret"].shift(2)
    df["ret_lag3"] = df["ret"].shift(3)

    # Rolling volatility (past window only)
    for w in [7, 14, 30]:
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()

    # Volume features (lagged)
    df["volume_lag1"] = df["volume_base"].shift(1)
    df["volume_lag3"] = df["volume_base"].shift(3)

    # Rolling volume volatility
    for w in [7, 14]:
        df[f"volume_vol_{w}"] = df["volume_base"].rolling(w).std()

    return df


def main() -> None:
    path = DATA_CLEAN / "usdt_kraken.csv"
    df = pd.read_csv(path)

    df = df.rename(columns={"date_utc": "date"})
    df["date"] = pd.to_datetime(df["date"])

    df = df[["date", "close_px", "volume_base"]].copy()

    df = add_labels(df)
    df = add_features(df)

    # Remove rows with NaN (from lags/rolling/shift)
    df = df.dropna().reset_index(drop=True)

    out_path = DATA_CLEAN / "usdt_depeg_dataset.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path, "rows=", len(df))
    print("Depeg rate:", df["depeg"].mean())
    print("Depeg count:", int(df["depeg"].sum()))


if __name__ == "__main__":
    main()
