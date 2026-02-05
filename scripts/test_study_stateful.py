#!/usr/bin/env python3
"""Study vs study_stateful incremental comparison.

This script compares full vectorized `study()` results to a two-phase
`study_stateful()` workflow:
1) seed on t=0..split
2) update on t=split+1..end using state + state_timestamp

By default it excludes long_run/short_run/tsignals/xsignals to match
study_stateful's built-in exclusions.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import pandas_ta_stateful as ta


DEFAULT_EXCLUDE = ["long_run", "short_run", "tsignals", "xsignals"]


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


def parse_exclude(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


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
    ap.add_argument("--rows", type=int, default=1011)
    ap.add_argument("--split", type=int, default=1005, help="seed end index")
    ap.add_argument("--tail", type=int, default=10, help="compare last N rows (combined)")
    ap.add_argument("--seed-tail", type=int, default=5, help="compare last N rows of seed segment")
    ap.add_argument("--inc-tail", type=int, default=5, help="compare first N rows of incremental segment")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", type=str, default="", help="comma-separated kinds to exclude")
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    df_full = make_ohlcv(args.rows, args.seed)
    base_cols = list(df_full.columns)

    if args.split >= args.rows:
        raise SystemExit("--split must be < --rows")

    exclude = DEFAULT_EXCLUDE + parse_exclude(args.exclude)

    # Vectorized study reference
    df_study = df_full.copy()
    df_study.ta.study(
        append=True,
        verbose=False,
        timed=False,
        ordered=True,
        cores=0,
        exclude=exclude,
    )

    # Stateful seed (t=0..split)
    df_seed = df_full.iloc[: args.split + 1].copy()
    res_seed, state = df_seed.ta.study_stateful(
        append=True,
        returns=True,
        returns_state=True,
        verbose=False,
        timed=False,
        ordered=True,
        cores=0,
        exclude=exclude,
        state=None,
    )

    # Incremental update (t=split+1..end)
    df_inc = df_full.copy()
    split_ts = df_full.index[args.split]
    res_inc, _ = df_inc.ta.study_stateful(
        append=True,
        returns=True,
        returns_state=True,
        verbose=False,
        timed=False,
        ordered=True,
        cores=0,
        exclude=exclude,
        state=state,
        state_timestamp=split_ts,
    )

    # Combine seed + incremental
    indicator_cols = [c for c in df_study.columns if c not in base_cols]
    seed_cols = [c for c in indicator_cols if c in res_seed.columns]
    inc_cols = [c for c in indicator_cols if c in res_inc.columns]
    common_cols = sorted(set(seed_cols) & set(inc_cols))
    missing_seed = sorted(set(indicator_cols) - set(seed_cols))
    missing_inc = sorted(set(indicator_cols) - set(inc_cols))

    if missing_seed:
        print(f"[i] missing in seed (ignored): {len(missing_seed)}")
        print("[i] missing in seed:", ", ".join(missing_seed))
    if missing_inc:
        print(f"[i] missing in incremental (ignored): {len(missing_inc)}")
        print("[i] missing in incremental:", ", ".join(missing_inc))

    combined = res_inc.reindex(df_full.index)
    combined.loc[:split_ts, common_cols] = res_seed.loc[:split_ts, common_cols]

    # Compare combined last N rows
    compare_cols = [c for c in indicator_cols if c in combined.columns]
    ref_all = df_study.loc[:, compare_cols]
    test_all = combined.loc[:, compare_cols]
    compare_idx = df_full.index[-args.tail :]
    compare_idx = compare_idx.intersection(ref_all.index).intersection(test_all.index)
    ref = ref_all.loc[compare_idx]
    test = test_all.loc[compare_idx]
    summary = compare_frames(ref, test, args.eps)

    # Compare seed segment (t=split-seed_tail+1 .. split)
    seed_start = max(0, args.split - args.seed_tail + 1)
    seed_idx = df_full.index[seed_start: args.split + 1]
    seed_cols = [c for c in indicator_cols if c in res_seed.columns]
    seed_ref = df_study.loc[seed_idx, seed_cols]
    seed_test = res_seed.loc[seed_idx, seed_cols]
    seed_summary = compare_frames(seed_ref, seed_test, args.eps)

    # Compare incremental segment (t=split+1 .. split+inc_tail)
    inc_start = args.split + 1
    inc_end = min(args.rows, inc_start + args.inc_tail)
    inc_idx = df_full.index[inc_start:inc_end]
    inc_cols = [c for c in indicator_cols if c in res_inc.columns]
    inc_ref = df_study.loc[inc_idx, inc_cols]
    inc_test = res_inc.loc[inc_idx, inc_cols]
    inc_summary = compare_frames(inc_ref, inc_test, args.eps)

    # Output
    print("[i] rows:", args.rows)
    print("[i] split index:", args.split)
    print("[i] compare rows:", len(compare_idx))
    print("[i] indicator columns:", len(indicator_cols))
    print("\nTop 15 by max_abs (combined):")
    print(summary.sort_values("max_abs", ascending=False).head(15))
    print("\nTop 15 by mean_abs (combined):")
    print(summary.sort_values("mean_abs", ascending=False).head(15))

    print("\nTop 15 by max_abs (seed segment):")
    print(seed_summary.sort_values("max_abs", ascending=False).head(15))
    print("\nTop 15 by mean_abs (seed segment):")
    print(seed_summary.sort_values("mean_abs", ascending=False).head(15))

    print("\nTop 15 by max_abs (incremental segment):")
    print(inc_summary.sort_values("max_abs", ascending=False).head(15))
    print("\nTop 15 by mean_abs (incremental segment):")
    print(inc_summary.sort_values("mean_abs", ascending=False).head(15))


if __name__ == "__main__":
    main()
