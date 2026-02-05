#!/usr/bin/env python3
"""Compare TA-Lib vectorized outputs vs stateful incremental outputs.

This script uses a curated list of TA-Lib supported indicators and compares
vectorized results (talib=True) to stateful incremental results.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd
import pandas_ta_stateful as ta
from pandas_ta_stateful import Study
from pandas_ta_stateful.maps import Imports


DEFAULT_EXCLUDE = ["long_run", "short_run", "tsignals", "xsignals"]

DEFAULT_SPECS = [
    {"kind": "ema", "length": 10, "talib": True},
    {"kind": "rsi", "length": 14, "talib": True},
    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9, "talib": True},
    {"kind": "atr", "length": 14, "talib": True},
    {"kind": "adx", "length": 14, "talib": True},
    {"kind": "bbands", "length": 20, "lower_std": 2.0, "upper_std": 2.0, "talib": True},
    {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3, "talib": True},
]


def make_ohlcv(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=rows, freq="1min")
    base = 100 + rng.standard_normal(rows).cumsum()
    close = base + rng.normal(0, 0.2, rows)
    open_ = base + rng.normal(0, 0.2, rows)
    high = np.maximum(open_, close) + rng.random(rows) * 0.5
    low = np.minimum(open_, close) - rng.random(rows) * 0.5
    volume = rng.integers(100, 1000, rows)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    # Some indicators expect a numeric 'timestamp' column. Provide seconds.
    df["timestamp"] = df.index.view("int64") // 1_000_000_000
    return df


def compare_frames(ref: pd.DataFrame, test: pd.DataFrame, eps: float) -> pd.DataFrame:
    diff = (test - ref).abs()
    rel = diff / (ref.abs() + eps)
    return pd.DataFrame(
        {
            "nan_ref": ref.isna().sum(),
            "nan_test": test.isna().sum(),
            "max_abs": diff.max(),
            "mean_abs": diff.mean(),
            "mean_rel": rel.mean(),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--split", type=int, default=1500)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    if not Imports.get("talib", False):
        raise SystemExit("[X] TA-Lib not available. Install ta-lib to run this script.")

    df_full = make_ohlcv(args.rows, args.seed)
    base_cols = list(df_full.columns)

    if args.split >= args.rows:
        raise SystemExit("--split must be < --rows")

    study = Study(name="talib_compare", ta=DEFAULT_SPECS)

    # Vectorized (TA-Lib) reference
    df_vec = df_full.copy()
    df_vec.ta.study(
        study,
        append=True,
        verbose=False,
        timed=False,
        ordered=True,
        cores=0,
        exclude=DEFAULT_EXCLUDE,
    )

    # Seed + incremental (stateful)
    df_seed = df_full.iloc[: args.split + 1].copy()
    res_seed, state = df_seed.ta.study_stateful(
        study,
        append=True,
        returns=True,
        returns_state=True,
        verbose=False,
        timed=False,
        ordered=True,
        cores=0,
        exclude=DEFAULT_EXCLUDE,
        state=None,
    )

    df_inc = df_full.copy()
    split_ts = df_full.index[args.split]
    res_inc, _ = df_inc.ta.study_stateful(
        study,
        append=True,
        returns=True,
        returns_state=True,
        verbose=False,
        timed=False,
        ordered=True,
        cores=0,
        exclude=DEFAULT_EXCLUDE,
        state=state,
        state_timestamp=split_ts,
    )

    indicator_cols = [c for c in df_vec.columns if c not in base_cols]
    seed_cols = [c for c in indicator_cols if c in res_seed.columns]
    inc_cols = [c for c in indicator_cols if c in res_inc.columns]
    common_cols = sorted(set(seed_cols) & set(inc_cols))
    missing_seed = sorted(set(indicator_cols) - set(seed_cols))
    missing_inc = sorted(set(indicator_cols) - set(inc_cols))

    if missing_seed:
        print(f"[i] missing in seed (ignored): {len(missing_seed)}")
    if missing_inc:
        print(f"[i] missing in incremental (ignored): {len(missing_inc)}")

    combined = res_inc.reindex(df_full.index)
    combined.loc[:split_ts, common_cols] = res_seed.loc[:split_ts, common_cols]

    compare_idx = df_full.index[-args.tail :]
    compare_cols = [c for c in indicator_cols if c in combined.columns]
    ref = df_vec.loc[:, compare_cols]
    test = combined.loc[:, compare_cols]
    compare_idx = compare_idx.intersection(ref.index).intersection(test.index)
    ref = ref.loc[compare_idx]
    test = test.loc[compare_idx]

    summary = compare_frames(ref, test, args.eps)

    print("[i] rows:", args.rows)
    print("[i] split index:", args.split)
    print("[i] compare rows:", len(compare_idx))
    print("[i] indicator columns:", len(indicator_cols))
    print("\nTop 15 by max_abs:")
    print(summary.sort_values("max_abs", ascending=False).head(15))
    print("\nTop 15 by mean_abs:")
    print(summary.sort_values("mean_abs", ascending=False).head(15))


if __name__ == "__main__":
    main()
