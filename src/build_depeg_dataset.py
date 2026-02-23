# src/build_depeg_dataset.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_CLEAN

# --- Label definition (adjust later for robustness) ---
THRESH = 0.0015   # 15 basis points
MIN_DAYS = 1        # event if condition holds for >= MIN_DAYS (daily)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["peg"] = 1.0
    df["dev"] = df["close_px"] - df["peg"]
    df["abs_dev"] = df["dev"].abs()

    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)

    # Consecutive-day filter (MIN_DAYS)
    if MIN_DAYS > 1:
        run = 0
        kept = []
        for x in df["is_offpeg"].values:
            run = run + 1 if x == 1 else 0
            kept.append(1 if run >= MIN_DAYS else 0)
        df["depeg"] = kept
    else:
        df["depeg"] = df["is_offpeg"]

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # Returns of USDT/USD
    df["ret"] = np.log(df["close_px"]).diff()

    # Rolling stats (use simple windows for speed)
    for w in [7, 14, 30]:
        df[f"abs_dev_ma_{w}"] = df["abs_dev"].rolling(w).mean()
        df[f"abs_dev_max_{w}"] = df["abs_dev"].rolling(w).max()
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()

    # Momentum of deviation (change in deviation)
    df["abs_dev_change_1"] = df["abs_dev"].diff()
    df["abs_dev_change_3"] = df["abs_dev"].diff(3)

    # Level features
    df["close_minus_1"] = df["close_px"] - 1.0

    return df


def main() -> None:
    path = DATA_CLEAN / "usdt_kraken.csv"
    df = pd.read_csv(path)

    # Standardize column names from earlier pipeline
    # our saved usdt_kraken.csv columns include: asset,date_utc,timestamp_utc,close_px,volume_base,source
    df = df.rename(columns={"date_utc": "date", "close_px": "close_px"})
    df["date"] = pd.to_datetime(df["date"])

    # Keep just what we need
    keep_cols = [c for c in df.columns if c in ["date", "close_px", "volume_base"]]
    df = df[keep_cols].copy()

    df = add_labels(df)
    df = add_features(df)

    # Drop initial NaNs from rolling windows
    df = df.dropna().reset_index(drop=True)

    out_path = DATA_CLEAN / "usdt_depeg_dataset.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path, "rows=", len(df))
    print("Depeg rate:", df["depeg"].mean())
    print("Depeg count:", int(df["depeg"].sum()))


if __name__ == "__main__":
    main()

