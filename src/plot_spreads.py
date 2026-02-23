# src/plot_spreads.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DATA_CLEAN, OUTPUTS

ROLL_WINDOW = 30      # days
Z_THRESH = 3.0        # anomaly threshold


def ensure_outputs_dir() -> Path:
    out_dir = OUTPUTS
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    path = DATA_CLEAN / "spreads_overlap_btc_eth.csv"
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date_utc"])
    df = df.sort_values(["asset", "date"])

    # Rolling z-score per asset
    anomalies = []
    out_dir = ensure_outputs_dir()

    for asset, g in df.groupby("asset"):
        g = g.copy().reset_index(drop=True)

        roll_mean = g["spread_rel"].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
        roll_std = g["spread_rel"].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std()

        g["z_spread_rel"] = (g["spread_rel"] - roll_mean) / roll_std
        g["is_anomaly"] = g["z_spread_rel"].abs() >= Z_THRESH

        # Save anomalies
        a = g[g["is_anomaly"]].copy()
        if not a.empty:
            anomalies.append(a[["asset", "date_utc", "spread_rel", "spread_abs", "z_spread_rel"]])

        # Plot 1: spread_rel time series
        plt.figure()
        plt.plot(g["date"], g["spread_rel"])
        plt.title(f"{asset}: Relative Spread (Binance/Kraken - 1)")
        plt.xlabel("Date")
        plt.ylabel("spread_rel")
        plt.tight_layout()
        p1 = out_dir / f"spread_rel_timeseries_{asset}.png"
        plt.savefig(p1, dpi=200)
        plt.close()

        # Plot 2: z-score with threshold lines
        plt.figure()
        plt.plot(g["date"], g["z_spread_rel"])
        plt.axhline(Z_THRESH, linestyle="--")
        plt.axhline(-Z_THRESH, linestyle="--")
        plt.title(f"{asset}: Rolling Z-score of Relative Spread (window={ROLL_WINDOW})")
        plt.xlabel("Date")
        plt.ylabel("z_spread_rel")
        plt.tight_layout()
        p2 = out_dir / f"spread_rel_zscore_{asset}.png"
        plt.savefig(p2, dpi=200)
        plt.close()

        print(f"Saved plots for {asset}: {p1.name}, {p2.name}")

    # Write anomaly table
    if anomalies:
        out_a = pd.concat(anomalies, ignore_index=True).sort_values(["asset", "date_utc"])
    else:
        out_a = pd.DataFrame(columns=["asset", "date_utc", "spread_rel", "spread_abs", "z_spread_rel"])

    anomalies_path = DATA_CLEAN / "spread_anomalies.csv"
    out_a.to_csv(anomalies_path, index=False)
    print("Saved anomaly table:", anomalies_path, "rows=", len(out_a))


if __name__ == "__main__":
    main()
