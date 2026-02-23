# src/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUTS = PROJECT_ROOT / "outputs"

for p in [DATA_RAW, DATA_CLEAN, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)

# CoinGecko IDs
ASSETS_COINGECKO = ["bitcoin", "ethereum", "tether"]
VS_CURRENCY = "usd"

# Binance symbols
SYMBOLS_BINANCE = ["BTCUSDT", "ETHUSDT"]

#Kraken 
KRAKEN_PAIRS = ["XBTUSD", "ETHUSD", "USDTUSD"]

#Coinbase
MARKETS_COINBASE = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD"
}
START_DATE = "2019-01-01"
END_DATE = None
