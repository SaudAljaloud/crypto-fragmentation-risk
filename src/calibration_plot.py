# src/calibration_plot.py
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import DATA_CLEAN, OUTPUTS


def main() -> None:
    # Load predictions
    df = pd.read_csv(DATA_CLEAN / "depeg_predictions.csv")
    df = df.sort_values("date")

    probs = df["logit_prob"]
    y = df["depeg"]

    # --- Use 5 bins for stability ---
    bins = np.linspace(0, 1, 6)  # 5 equal-width bins
    df["bin"] = pd.cut(probs, bins, include_lowest=True)

    calib = (
        df.groupby("bin", observed=False)
          .agg(
              mean_prob=("logit_prob", "mean"),
              actual_rate=("depeg", "mean"),
              count=("depeg", "size")
          )
          .dropna()
          .reset_index()
    )

    # --- Plot ---
    plt.figure()
    plt.plot(calib["mean_prob"], calib["actual_rate"])
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Frequency")
    plt.title("Calibration Plot (Logistic, 5 Bins)")
    plt.tight_layout()

    out_path = OUTPUTS / "calibration_logistic.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)
    print("\nCalibration table:")
    print(calib.to_string(index=False))


if __name__ == "__main__":
    main()
