# src/make_clean_panel_3venues.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

from src.config import DATA_RAW, DATA_CLEAN, OUTPUTS


def latest_csv(folder: Path) -> Path:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    # choose most recently modified
    return max(files, key=lambda p: p.stat().st_mtime)


def _std_asset_from_binance_symbol(sym: str) -> str:
    if sym.startswith("BTC"):
        return "BTC"
    if sym.startswith("ETH"):
        return "ETH"
    return sym


def load_binance_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected: symbol,date_utc,timestamp_utc,close_px,volume_base,source (your pipeline)
    if "symbol" not in df.columns:
        raise ValueError("Binance file missing 'symbol' column")
    df = df.copy()
    df["asset"] = df["symbol"].map(_std_asset_from_binance_symbol)
    df["venue"] = "binance"
    df["date_utc"] = df["date_utc"].astype(str)
    # standard name for close
    if "close_px" in df.columns:
        df["close"] = df["close_px"]
    elif "close" in df.columns:
        df["close"] = df["close"]
    else:
        raise ValueError("Binance file missing close column (close_px or close)")
    keep = ["date_utc", "asset", "venue", "close"]
    return df[keep]


def _std_asset_from_kraken_pair(pair: str) -> str:
    # XBTUSD -> BTC, ETHUSD -> ETH
    if pair.upper().startswith("XBT"):
        return "BTC"
    if pair.upper().startswith("ETH"):
        return "ETH"
    return pair


def load_kraken_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected: pair,date_utc,timestamp_utc,close_px,volume_base,source (your pipeline)
    if "pair" not in df.columns:
        raise ValueError("Kraken file missing 'pair' column")
    df = df.copy()
    df["asset"] = df["pair"].map(_std_asset_from_kraken_pair)
    df["venue"] = "kraken"
    df["date_utc"] = df["date_utc"].astype(str)
    if "close_px" in df.columns:
        df["close"] = df["close_px"]
    elif "close" in df.columns:
        df["close"] = df["close"]
    else:
        raise ValueError("Kraken file missing close column (close_px or close)")
    keep = ["date_utc", "asset", "venue", "close"]
    return df[keep]


def _std_asset_from_coinbase_symbol(sym: str) -> str:
    # BTC/USD -> BTC, ETH/USD -> ETH
    s = sym.upper()
    if s.startswith("BTC/"):
        return "BTC"
    if s.startswith("ETH/"):
        return "ETH"
    return sym


def load_coinbase_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected from your script: asset,symbol,timestamp_utc,date_utc,open,high,low,close,volume
    df = df.copy()
    if "asset" not in df.columns:
        if "symbol" in df.columns:
            df["asset"] = df["symbol"].map(_std_asset_from_coinbase_symbol)
        else:
            raise ValueError("Coinbase file missing both 'asset' and 'symbol'")
    df["venue"] = "coinbase"
    df["date_utc"] = df["date_utc"].astype(str)
    if "close" not in df.columns:
        raise ValueError("Coinbase file missing 'close'")
    keep = ["date_utc", "asset", "venue", "close"]
    return df[keep]


def build_panel_3venues(bin_df: pd.DataFrame, kra_df: pd.DataFrame, cb_df: pd.DataFrame) -> pd.DataFrame:
    # pivot each venue to columns close_<venue>
    def pivot_close(df: pd.DataFrame, venue: str) -> pd.DataFrame:
        out = df[df["venue"] == venue].pivot(index=["date_utc", "asset"], columns="venue", values="close").reset_index()
        out = out.rename(columns={venue: f"close_{venue}"})
        return out

    b = pivot_close(bin_df, "binance")
    k = pivot_close(kra_df, "kraken")
    c = pivot_close(cb_df, "coinbase")

    panel = b.merge(k, on=["date_utc", "asset"], how="inner").merge(c, on=["date_utc", "asset"], how="inner")
    panel = panel.sort_values(["asset", "date_utc"]).reset_index(drop=True)
    return panel


def compute_spreads(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    # relative spreads
    out["spread_rel_bk"] = out["close_binance"] / out["close_kraken"] - 1.0
    out["spread_rel_bc"] = out["close_binance"] / out["close_coinbase"] - 1.0
    out["spread_rel_kc"] = out["close_kraken"] / out["close_coinbase"] - 1.0
    return out


def overlap_windows(panel: pd.DataFrame) -> list[tuple[str, str, str]]:
    wins = []
    for a, g in panel.groupby("asset"):
        wins.append((a, g["date_utc"].min(), g["date_utc"].max()))
    return wins


def main() -> None:
    # locate latest raw files
    bin_path = latest_csv(DATA_RAW / "binance")
    kra_path = latest_csv(DATA_RAW / "kraken")
    cb_path = latest_csv(DATA_RAW / "coinbase")

    print("Using Binance :", bin_path)
    print("Using Kraken  :", kra_path)
    print("Using Coinbase:", cb_path)

    bin_df = load_binance_daily(bin_path)
    kra_df = load_kraken_daily(kra_path)
    cb_df = load_coinbase_daily(cb_path)

    # keep only BTC/ETH
    bin_df = bin_df[bin_df["asset"].isin(["BTC", "ETH"])].copy()
    kra_df = kra_df[kra_df["asset"].isin(["BTC", "ETH"])].copy()
    cb_df = cb_df[cb_df["asset"].isin(["BTC", "ETH"])].copy()

    panel = build_panel_3venues(bin_df, kra_df, cb_df)

    out_panel_path = DATA_CLEAN / "panel_overlap_btc_eth_3venues.csv"
    panel.to_csv(out_panel_path, index=False)

    spreads = compute_spreads(panel)
    out_spreads_path = DATA_CLEAN / "spreads_overlap_btc_eth_3venues.csv"
    spreads.to_csv(out_spreads_path, index=False)

    wins = overlap_windows(panel)
    print("Overlap windows:", wins)
    print(f"Saved: {out_panel_path} rows={len(panel)}")
    print(f"Saved: {out_spreads_path} rows={len(spreads)}")

    # summary table (mean/std/min/max) by asset and spread type
    long = spreads.melt(
        id_vars=["date_utc", "asset"],
        value_vars=["spread_rel_bk", "spread_rel_bc", "spread_rel_kc"],
        var_name="pair",
        value_name="spread_rel",
    )
    summ = long.groupby(["asset", "pair"])["spread_rel"].agg(["mean", "std", "min", "max"]).reset_index()
    out_summary_path = OUTPUTS / "spread_summary_3venues.csv"
    summ.to_csv(out_summary_path, index=False)

    print("\nSpread summary (relative):")
    print(summ)

    print(f"\nSaved: {out_summary_path}")


if __name__ == "__main__":
    main()
