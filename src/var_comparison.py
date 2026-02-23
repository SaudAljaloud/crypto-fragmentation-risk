# src/var_comparison.py
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import DATA_CLEAN


def historical_var(r: pd.Series, alpha: float) -> float:
    return np.quantile(r, alpha)


def main() -> None:
    panel = pd.read_csv(DATA_CLEAN / "panel_overlap_btc_eth.csv")
    panel = panel.sort_values(["asset", "date_utc"])

    results = []

    for asset, g in panel.groupby("asset"):
        g = g.copy()

        g["ret_binance"] = np.log(g["close_binance"]).diff()
        g["ret_kraken"] = np.log(g["close_kraken"]).diff()
        g["ret_mid"] = np.log((g["close_binance"] + g["close_kraken"]) / 2).diff()

        for label in ["ret_binance", "ret_kraken", "ret_mid"]:
            r = g[label].dropna()

            results.append({
                "asset": asset,
                "series": label,
                "VaR_1pct": historical_var(r, 0.01),
                "VaR_5pct": historical_var(r, 0.05),
            })

    out = pd.DataFrame(results)

    print("\nHistorical VaR Comparison:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
