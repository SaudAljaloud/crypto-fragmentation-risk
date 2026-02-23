# src/pull_kraken_daily.py
from __future__ import annotations

import time
import requests
import pandas as pd
from datetime import datetime, timezone

from src.config import DATA_RAW, START_DATE, END_DATE

KRAKEN_BASE = "https://api.kraken.com/0/public/OHLC"
INTERVAL_MIN = 1440  # daily candles

KRAKEN_PAIRS = ["XBTUSD", "ETHUSD", "USDTUSD"]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_unix(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_ohlc(pair: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    since = to_unix(start_date)

    params = {
        "pair": pair,
        "interval": INTERVAL_MIN,
        "since": since
    }

    r = requests.get(KRAKEN_BASE, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    if js.get("error"):
        raise RuntimeError(f"Kraken API error for {pair}: {js['error']}")

    result = js["result"]
    data_key = [k for k in result.keys() if k != "last"][0]
    rows = result[data_key]

    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])

    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["date_utc"] = df["time_utc"].dt.date.astype(str)
    df.insert(0, "pair", pair)

    out = df[[
        "pair", "date_utc", "time_utc",
        "open", "high", "low", "close",
        "volume", "count"
    ]].sort_values(["pair", "date_utc"]).reset_index(drop=True)

    if END_DATE is not None:
        out = out[out["date_utc"] <= END_DATE].reset_index(drop=True)

    return out


def main() -> None:
    stamp = utc_stamp()
    out_dir = DATA_RAW / "kraken"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for pair in KRAKEN_PAIRS:
        print(f"[Kraken] fetching {pair} ...")
        df = fetch_ohlc(pair, START_DATE, END_DATE)
        frames.append(df)
        time.sleep(0.35)

    out = pd.concat(frames, ignore_index=True)

    out_path = out_dir / f"daily_{stamp}.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}  rows={len(out)}")


if __name__ == "__main__":
    main()
