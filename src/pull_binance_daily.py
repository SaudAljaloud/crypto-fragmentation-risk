# src/pull_binance_daily.py
from __future__ import annotations

import time
import requests
import pandas as pd
from datetime import datetime, timezone

from src.config import DATA_RAW, SYMBOLS_BINANCE, START_DATE, END_DATE

BINANCE_BASE = "https://api.binance.com"
INTERVAL = "1d"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)


def fetch_klines(symbol: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    start_dt = _parse_date(start_date)
    end_dt = datetime.now(timezone.utc) if end_date is None else _parse_date(end_date)

    url = f"{BINANCE_BASE}/api/v3/klines"

    all_rows = []
    start_ms = _ms(start_dt)
    end_ms = _ms(end_dt)

    while True:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_start = last_open_time + 24 * 60 * 60 * 1000
        if next_start > end_ms or next_start <= start_ms:
            break

        start_ms = next_start
        time.sleep(0.35)

    cols = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["date_utc"] = df["time_utc"].dt.date.astype(str)
    df.insert(0, "symbol", symbol)

    out = df[[
        "symbol", "date_utc", "time_utc",
        "open", "high", "low", "close",
        "volume", "num_trades"
    ]].sort_values(["symbol", "date_utc"]).reset_index(drop=True)

    # Optional end_date trim
    if end_date is not None:
        out = out[out["date_utc"] <= end_date].reset_index(drop=True)

    return out


def main() -> None:
    stamp = utc_stamp()
    out_dir = DATA_RAW / "binance"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sym in SYMBOLS_BINANCE:
        print(f"[Binance] fetching {sym} ...")
        frames.append(fetch_klines(sym, START_DATE, END_DATE))

    out = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"daily_{stamp}.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}  rows={len(out)}")


if __name__ == "__main__":
    main()
