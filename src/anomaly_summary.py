# src/anomaly_summary.py
import pandas as pd
from src.config import DATA_CLEAN

def main():
    path = DATA_CLEAN / "spread_anomalies.csv"
    df = pd.read_csv(path)

    if df.empty:
        print("No anomalies found.")
        return

    print("\nAnomaly count by asset:")
    print(df.groupby("asset").size())

    print("\nTop 5 largest absolute spreads:")
    df["abs_spread"] = df["spread_rel"].abs()
    print(df.sort_values("abs_spread", ascending=False)
            .head(5)[["asset", "date_utc", "spread_rel", "z_spread_rel"]])

if __name__ == "__main__":
    main()
