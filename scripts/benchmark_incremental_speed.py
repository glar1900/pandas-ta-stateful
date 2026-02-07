#!/usr/bin/env python3
"""Benchmark incremental update speed for study_stateful.

Measures how long incremental updates take as total history grows.
Two modes:
  - full: pass full history + tail rows (input prep cost grows with history)
  - tail: pass only tail rows (true streaming cost; should be ~O(tail))
"""
from __future__ import annotations

import argparse
import copy
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


def parse_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_exclude(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def time_call(fn, runs: int) -> float:
    times = []
    for _ in range(max(runs, 1)):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    return sum(times) / len(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        type=str,
        default="10000,50000,100000,200000,500000",
        help="comma-separated total row counts",
    )
    ap.add_argument("--tail", type=int, default=1, help="new rows per update")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", type=str, default="", help="comma-separated kinds to exclude")
    ap.add_argument("--stateful-only", action="store_true", help="force stateful-only (no vectorized)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs (not timed)")
    ap.add_argument("--runs", type=int, default=3, help="timed runs")
    ap.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("full", "tail", "both"),
        help="benchmark input mode",
    )
    args = ap.parse_args()

    sizes = parse_list(args.sizes)
    exclude = DEFAULT_EXCLUDE + parse_exclude(args.exclude)

    print(f"[i] sizes: {sizes}")
    print(f"[i] tail: {args.tail}")
    print(f"[i] runs: {args.runs} (warmup: {args.warmup})")
    print(f"[i] stateful_only: {args.stateful_only}")
    print(f"[i] mode: {args.mode}")

    for rows in sizes:
        if rows <= args.tail + 1:
            print(f"[i] skip rows={rows} (need > tail+1)")
            continue

        df = make_ohlcv(rows, args.seed)
        split = rows - args.tail
        df_hist = df.iloc[:split]
        df_full = df
        df_tail = df.iloc[split:]

        # Seed state from history (not timed)
        _, base_state = df_hist.ta.study_stateful(
            append=False,
            returns=True,
            returns_state=True,
            verbose=False,
            timed=False,
            ordered=True,
            cores=0,
            exclude=exclude,
            state=None,
            stateful_only=args.stateful_only,
        )
        state_ts = df_hist.index[-1]

        def run_full():
            state_copy = copy.deepcopy(base_state)
            df_full.ta.study_stateful(
                append=False,
                returns=False,
                returns_state=False,
                verbose=False,
                timed=False,
                ordered=True,
                cores=0,
                exclude=exclude,
                state=state_copy,
                state_timestamp=state_ts,
                stateful_only=args.stateful_only,
            )

        def run_tail():
            state_copy = copy.deepcopy(base_state)
            df_tail.ta.study_stateful(
                append=False,
                returns=False,
                returns_state=False,
                verbose=False,
                timed=False,
                ordered=True,
                cores=0,
                exclude=exclude,
                state=state_copy,
                state_timestamp=state_ts,
                stateful_only=args.stateful_only,
            )

        # Warmup
        for _ in range(max(args.warmup, 0)):
            if args.mode in ("full", "both"):
                run_full()
            if args.mode in ("tail", "both"):
                run_tail()

        # Timed runs
        if args.mode in ("full", "both"):
            avg_full = time_call(run_full, args.runs)
            print(
                f"[full] rows={rows} tail={args.tail} avg_s={avg_full:.6f} "
                f"s_per_tail={avg_full / max(args.tail, 1):.6f}"
            )
        if args.mode in ("tail", "both"):
            avg_tail = time_call(run_tail, args.runs)
            print(
                f"[tail] rows={rows} tail={args.tail} avg_s={avg_tail:.6f} "
                f"s_per_tail={avg_tail / max(args.tail, 1):.6f}"
            )


if __name__ == "__main__":
    main()
