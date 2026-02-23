# src/volatility_comparison.py
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import DATA_CLEAN


LAMBDA = 0.94  # EWMA decay


def compute_ewma_vol(r: pd.Series, lam: float = LAMBDA) -> float:
    var = 0.0
    for x in r:
        var = lam * var + (1 - lam) * x**2
    return np.sqrt(var)


def main() -> None:
    path = DATA_CLEAN / "panel_overlap_btc_eth.csv"
    df = pd.read_csv(path)

    df = df.sort_values(["asset", "date_utc"])

    results = []

    for asset, g in df.groupby("asset"):
        g = g.copy().reset_index(drop=True)

        # log returns
        g["ret_binance"] = np.log(g["close_binance"]).diff()
        g["ret_kraken"] = np.log(g["close_kraken"]).diff()
        g["ret_mid"] = np.log((g["close_binance"] + g["close_kraken"]) / 2).diff()

        for label in ["ret_binance", "ret_kraken", "ret_mid"]:
            r = g[label].dropna()

            sample_vol = r.std()
            ewma_vol = compute_ewma_vol(r)

            results.append({
                "asset": asset,
                "series": label,
                "sample_vol": sample_vol,
                "ewma_vol": ewma_vol
            })

    out = pd.DataFrame(results)
    print("\nVolatility Comparison:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
