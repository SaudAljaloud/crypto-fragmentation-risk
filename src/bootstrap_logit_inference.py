# src/bootstrap_logit_inference.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CLEAN, OUTPUTS

# Match the setting you used for the coefficient table
THRESH = 0.001  # 10 bps
TRAIN_FRAC = 0.70

# Bootstrap settings
B = 500              # increase to 1000 later if you want
BLOCK_LEN = 14       # ~2 trading weeks (reasonable for daily dependence)
RANDOM_SEED = 42


FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]


def build_dataset(usdt: pd.DataFrame) -> pd.DataFrame:
    df = usdt.sort_values("date").copy()

    # label built from abs deviation but predicted NEXT DAY (no leakage in features)
    df["abs_dev"] = (df["close_px"] - 1.0).abs()
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
    df["depeg"] = df["is_offpeg"].shift(-1)

    # lag-only features
    df["ret"] = np.log(df["close_px"]).diff()
    df["ret_lag1"] = df["ret"].shift(1)
    df["ret_lag2"] = df["ret"].shift(2)
    df["ret_lag3"] = df["ret"].shift(3)

    for w in [7, 14, 30]:
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()

    df["volume_lag1"] = df["volume_base"].shift(1)
    df["volume_lag3"] = df["volume_base"].shift(3)

    for w in [7, 14]:
        df[f"volume_vol_{w}"] = df["volume_base"].rolling(w).std()

    df = df.dropna().reset_index(drop=True)
    return df


def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Return bootstrap indices of length n using moving blocks."""
    if block_len <= 1:
        return rng.integers(0, n, size=n)

    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    idx = []
    for s in starts:
        idx.extend(range(s, s + block_len))
    return np.array(idx[:n], dtype=int)


def fit_logit_coefs(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=4000)
    model.fit(Xs, y)
    return model.coef_[0].copy()


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    usdt = pd.read_csv(DATA_CLEAN / "usdt_kraken.csv").rename(columns={"date_utc": "date"})
    usdt["date"] = pd.to_datetime(usdt["date"])
    usdt = usdt[["date", "close_px", "volume_base"]].copy()

    df = build_dataset(usdt)

    split = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split].copy()

    X_train = train[FEATURE_COLS].copy()
    y_train = train["depeg"].astype(int).copy()

    # Point estimate on training set
    point = fit_logit_coefs(X_train, y_train)

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(train)

    boot = np.zeros((B, len(FEATURE_COLS)), dtype=float)
    failed = 0

    for b in range(B):
        idx = moving_block_bootstrap_indices(n, BLOCK_LEN, rng)
        Xb = X_train.iloc[idx]
        yb = y_train.iloc[idx]

        # In rare-event settings, some resamples can become single-class; skip those
        if yb.nunique() < 2:
            failed += 1
            continue

        try:
            boot[b] = fit_logit_coefs(Xb, yb)
        except Exception:
            failed += 1

    # Drop failed rows (all-zeros)
    valid = ~(boot == 0).all(axis=1)
    boot = boot[valid]
    B_eff = boot.shape[0]

    # Percentile CI
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)

    # Two-sided bootstrap p-value (sign test style)
    # p = 2 * min(P(beta<=0), P(beta>=0))
    pvals = []
    for j in range(len(FEATURE_COLS)):
        p_le = np.mean(boot[:, j] <= 0)
        p_ge = np.mean(boot[:, j] >= 0)
        pvals.append(2 * min(p_le, p_ge))
    pvals = np.array(pvals)

    out = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coef_point": point,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_boot": pvals,
    }).sort_values("coef_point", key=lambda s: s.abs(), ascending=False)

    out_path = OUTPUTS / "logit_coef_bootstrap_10bps.csv"
    out.to_csv(out_path, index=False)

    print(f"Bootstrap done. Effective draws: {B_eff}/{B} (failed={failed})")
    print("Saved:", out_path)
    print("\nTop coefficients with 95% CI:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
