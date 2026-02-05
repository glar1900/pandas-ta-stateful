#!/usr/bin/env python3
"""Benchmark seed time for study_stateful.

Runs study_stateful in seed mode (state=None) on synthetic OHLCV data and
reports elapsed time. Use this to estimate full-seed runtime by scaling
with row count.
"""
from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter
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
    df["timestamp"] = df.index.view("int64") // 1_000_000_000
    return df


def parse_exclude(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", type=str, default="", help="comma-separated kinds to exclude")
    ap.add_argument("--stateful-only", action="store_true", help="force stateful-only (no vectorized)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs (not timed)")
    ap.add_argument("--runs", type=int, default=3, help="timed runs")
    args = ap.parse_args()

    exclude = DEFAULT_EXCLUDE + parse_exclude(args.exclude)
    df = make_ohlcv(args.rows, args.seed)

    # Warmup
    for _ in range(max(args.warmup, 0)):
        df_copy = df.copy()
        df_copy.ta.study_stateful(
            append=False,
            returns=True,
            returns_state=False,
            verbose=True,
            timed=True,
            ordered=True,
            cores=0,
            exclude=exclude,
            state=None,
            stateful_only=args.stateful_only,
            progress=True,
        )

    # Timed runs
    times = []
    for _ in range(max(args.runs, 1)):
        df_copy = df.copy()
        start = perf_counter()
        df_copy.ta.study_stateful(
            append=False,
            returns=True,
            returns_state=False,
            verbose=False,
            timed=False,
            ordered=True,
            cores=0,
            exclude=exclude,
            state=None,
            stateful_only=args.stateful_only,
        )
        elapsed = perf_counter() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"[i] rows: {args.rows}")
    print(f"[i] runs: {len(times)} (warmup: {args.warmup})")
    print(f"[i] stateful_only: {args.stateful_only}")
    print(f"[i] avg seconds: {avg:.3f}")
    if args.rows > 0:
        print(f"[i] seconds per 100k rows: {avg / args.rows * 100_000:.3f}")


if __name__ == "__main__":
    main()
