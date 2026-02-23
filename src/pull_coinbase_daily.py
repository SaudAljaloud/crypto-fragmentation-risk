# src/pull_coinbase_daily.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import ccxt

from src.config import DATA_RAW, MARKETS_COINBASE, START_DATE, END_DATE


def to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD (UTC) to milliseconds since epoch."""
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fetch_daily_ohlcv(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> pd.DataFrame:
    """
    Fetch daily OHLCV in chunks using ccxt.

    Returns dataframe with columns:
    symbol, timestamp_utc, date_utc, open, high, low, close, volume
    """
    all_rows = []
    timeframe = "1d"
    limit = 300  # chunk size; Coinbase supports paging

    ms = since_ms
    end_ms = to_ms(END_DATE) if END_DATE is not None else None

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=ms, limit=limit)
        if not batch:
            break

        all_rows.extend(batch)

        last_ts = batch[-1][0]
        # Advance by 1 day in ms to avoid duplicates
        ms = last_ts + 24 * 60 * 60 * 1000

        # Stop if END_DATE set and we passed it
        if end_ms is not None and ms > end_ms:
            break

        # If batch smaller than limit, likely done
        if len(batch) < limit:
            break

        # Polite sleep to avoid rate limits
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["timestamp_utc", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
    df["date_utc"] = df["timestamp_utc"].dt.date.astype(str)
    df.insert(0, "symbol", symbol)
    return df


def main() -> None:
    out_dir: Path = DATA_RAW / "coinbase"
    out_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.coinbase({"enableRateLimit": True})

    # Load markets once (avoids BadSymbol and speeds up)
    exchange.load_markets()

    since_ms = to_ms(START_DATE)

    frames = []
    # MARKETS_COINBASE is a dict: {"BTC":"BTC/USD","ETH":"ETH/USD"}
    for asset, symbol in MARKETS_COINBASE.items():
        print(f"[Coinbase] fetching {asset} ({symbol}) ...")

        # Defensive: ensure market exists
        if symbol not in exchange.markets:
            print(f"[Coinbase] WARNING: market not found on Coinbase: {symbol}")
            continue

        df = fetch_daily_ohlcv(exchange, symbol, since_ms)
        if df.empty:
            print(f"[Coinbase] WARNING: no data returned for {symbol}")
            continue

        df.insert(0, "asset", asset)   # helpful for merging later
        frames.append(df)

    if not frames:
        raise RuntimeError("No Coinbase data fetched. Check symbols / connectivity.")

    out = pd.concat(frames, ignore_index=True)

    tag = now_utc_tag()
    out_path = out_dir / f"daily_{tag}.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}  rows={len(out)}")


if __name__ == "__main__":
    main()
