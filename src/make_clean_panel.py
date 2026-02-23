# src/make_clean_panel.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import DATA_RAW, DATA_CLEAN


def latest_csv(folder: Path) -> Path:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    return files[-1]


def load_binance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map Binance symbols to assets
    sym_map = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
    df["asset"] = df["symbol"].map(sym_map)
    df = df[df["asset"].notna()].copy()

    # Standard columns
    df["source"] = "binance"
    df = df.rename(columns={
        "close": "close_px",
        "volume": "volume_base",
        "time_utc": "timestamp_utc"
    })

    keep = ["asset", "date_utc", "timestamp_utc", "close_px", "volume_base", "source"]
    return df[keep].sort_values(["asset", "date_utc"]).reset_index(drop=True)


def load_kraken(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    pair_map = {"XBTUSD": "BTC", "ETHUSD": "ETH", "USDTUSD": "USDT"}
    df["asset"] = df["pair"].map(pair_map)
    df = df[df["asset"].notna()].copy()

    df["source"] = "kraken"
    df = df.rename(columns={
        "close": "close_px",
        "volume": "volume_base",
        "time_utc": "timestamp_utc"
    })

    keep = ["asset", "date_utc", "timestamp_utc", "close_px", "volume_base", "source"]
    return df[keep].sort_values(["asset", "date_utc"]).reset_index(drop=True)


def main() -> None:
    # Load latest raw files
    bn_path = latest_csv(DATA_RAW / "binance")
    kr_path = latest_csv(DATA_RAW / "kraken")

    print("Using Binance:", bn_path)
    print("Using Kraken :", kr_path)

    bn = load_binance(bn_path)
    kr = load_kraken(kr_path)

    # Save USDT/USD from Kraken for depeg modeling
    usdt = kr[kr["asset"] == "USDT"].copy()
    usdt_out = DATA_CLEAN / "usdt_kraken.csv"
    usdt.to_csv(usdt_out, index=False)
    print("Saved:", usdt_out, "rows=", len(usdt))

    # Focus overlap comparison for BTC/ETH only
    bn_be = bn[bn["asset"].isin(["BTC", "ETH"])].copy()
    kr_be = kr[kr["asset"].isin(["BTC", "ETH"])].copy()

    # Determine overlap window per asset
    overlaps = []
    for a in ["BTC", "ETH"]:
        b = bn_be[bn_be["asset"] == a]
        k = kr_be[kr_be["asset"] == a]
        if b.empty or k.empty:
            continue
        start = max(b["date_utc"].min(), k["date_utc"].min())
        end = min(b["date_utc"].max(), k["date_utc"].max())
        overlaps.append((a, start, end))

    print("Overlap windows:", overlaps)

    # Filter to overlap window
    def in_window(df: pd.DataFrame, asset: str, start: str, end: str) -> pd.DataFrame:
        return df[(df["asset"] == asset) & (df["date_utc"] >= start) & (df["date_utc"] <= end)].copy()

    bn_f = []
    kr_f = []
    for a, start, end in overlaps:
        bn_f.append(in_window(bn_be, a, start, end))
        kr_f.append(in_window(kr_be, a, start, end))

    bn_f = pd.concat(bn_f, ignore_index=True)
    kr_f = pd.concat(kr_f, ignore_index=True)

    # Wide format panel: close prices by source
    bn_w = bn_f[["asset", "date_utc", "close_px"]].rename(columns={"close_px": "close_binance"})
    kr_w = kr_f[["asset", "date_utc", "close_px"]].rename(columns={"close_px": "close_kraken"})

    panel = bn_w.merge(kr_w, on=["asset", "date_utc"], how="inner").sort_values(["asset", "date_utc"])
    panel_out = DATA_CLEAN / "panel_overlap_btc_eth.csv"
    panel.to_csv(panel_out, index=False)
    print("Saved:", panel_out, "rows=", len(panel))

    # Spread diagnostics: relative difference
    spreads = panel.copy()
    spreads["spread_abs"] = (spreads["close_binance"] - spreads["close_kraken"]).abs()
    spreads["spread_rel"] = (spreads["close_binance"] / spreads["close_kraken"]) - 1.0
    spreads_out = DATA_CLEAN / "spreads_overlap_btc_eth.csv"
    spreads.to_csv(spreads_out, index=False)
    print("Saved:", spreads_out)

    # Print quick summary stats
    print("\nSpread summary (relative):")
    print(spreads.groupby("asset")["spread_rel"].agg(["mean", "std", "min", "max"]))


if __name__ == "__main__":
    main()
