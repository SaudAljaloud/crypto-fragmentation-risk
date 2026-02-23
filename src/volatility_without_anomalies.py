# src/volatility_without_anomalies.py
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import DATA_CLEAN


def compute_ewma_vol(r: pd.Series, lam: float = 0.94) -> float:
    var = 0.0
    for x in r:
        var = lam * var + (1 - lam) * x**2
    return np.sqrt(var)


def main() -> None:
    panel = pd.read_csv(DATA_CLEAN / "panel_overlap_btc_eth.csv")
    anomalies = pd.read_csv(DATA_CLEAN / "spread_anomalies.csv")

    panel = panel.sort_values(["asset", "date_utc"])

    # remove anomaly dates
    merged = panel.merge(
        anomalies[["asset", "date_utc"]],
        on=["asset", "date_utc"],
        how="left",
        indicator=True
    )

    clean = merged[merged["_merge"] == "left_only"].copy()

    results = []

    for asset, g in clean.groupby("asset"):
        g = g.copy().reset_index(drop=True)

        g["ret_binance"] = np.log(g["close_binance"]).diff()
        g["ret_kraken"] = np.log(g["close_kraken"]).diff()
        g["ret_mid"] = np.log((g["close_binance"] + g["close_kraken"]) / 2).diff()

        for label in ["ret_binance", "ret_kraken", "ret_mid"]:
            r = g[label].dropna()

            results.append({
                "asset": asset,
                "series": label,
                "sample_vol_clean": r.std(),
                "ewma_vol_clean": compute_ewma_vol(r)
            })

    out = pd.DataFrame(results)
    print("\nVolatility WITHOUT anomaly days:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
