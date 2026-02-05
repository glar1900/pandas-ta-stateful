# -*- coding: utf-8 -*-
"""pandas-ta stateful -- lookahead indicators (conditional / approximate).

This module implements indicators that typically use future data (lookahead)
but are made streaming-compatible through restrictions or approximations:

  1. dpo        -- CONDITIONAL: centered=False (no lookahead shift)
  2. ichimoku   -- CONDITIONAL: lookahead=False (exclude chikou span)
  3. tos_stdevall -- APPROXIMATE: expanding window with runtime warning
  4. vp         -- APPROXIMATE: expanding window Volume Profile with warning

Each section follows the pattern:
  1. State dataclass
  2. init / update / output_names helpers
  3. LOOKAHEAD_REGISTRY["<kind>"] = StatefulIndicator(...)
  4. SEED_REGISTRY["<kind>"] = seed_fn (where applicable)
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._base import (
    NAN,
    _is_nan,
    _param,
    _as_int,
    _as_float,
    StatefulIndicator,
    STATEFUL_REGISTRY,
    SEED_REGISTRY,
    LOOKAHEAD_REGISTRY,
    replay_seed,
)


# ===========================================================================
# DPO  (CONDITIONAL lookahead: centered=False ONLY)
# ===========================================================================
# Detrended Price Oscillator: close - SMA(close, length).shift(t)
# where t = int(0.5 * length) + 1.
# The vectorized version allows centered=True (uses future data via shift(-t)).
# Stateful version FIXES centered=False (no lookahead).

@dataclass
class DPOState:
    """DPO state with fixed centered=False.

    RESTRICTION: centered=False is FIXED in stateful mode to avoid lookahead.
    The centered=True variant uses close.shift(t) - ma which looks into the future.
    Only the centered=False variant (close - ma.shift(t)) is streaming-compatible.
    """
    length: int
    t: int                           # shift amount = int(0.5*length)+1
    sma_sum: float = 0.0
    sma_count: int = 0
    sma_buffer: deque = field(default_factory=deque)  # rolling window for SMA
    ma_buffer: deque = field(default_factory=deque)   # shifted MA values


def _dpo_init(params: Dict[str, Any]) -> DPOState:
    """Initialize DPO state. ALWAYS uses centered=False."""
    length = _as_int(_param(params, "length", 20), 20)
    t = int(0.5 * length) + 1
    return DPOState(
        length=length,
        t=t,
        sma_buffer=deque(maxlen=length),
        ma_buffer=deque(maxlen=t),
    )


def _dpo_update(
    state: DPOState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], DPOState]:
    """Update DPO: close - SMA(close, length).shift(t).

    Note: centered=False is FIXED in stateful mode to avoid lookahead.
    """
    close = bar["close"]

    # Update SMA rolling window
    # Track oldest value before appending (deque with maxlen auto-removes)
    oldest: Optional[float] = None
    if len(state.sma_buffer) >= state.length:
        oldest = state.sma_buffer[0]

    state.sma_buffer.append(close)
    state.sma_sum += close
    state.sma_count += 1

    # Remove oldest value from sum if buffer was full
    if oldest is not None:
        state.sma_sum -= oldest
        state.sma_count = state.length

    # Calculate current SMA
    if state.sma_count >= state.length:
        current_ma = state.sma_sum / state.length
    else:
        current_ma = None

    # Add MA to shift buffer
    if current_ma is not None:
        state.ma_buffer.append(current_ma)

    # DPO = close - shifted_ma
    # We need MA from t bars ago
    dpo_val: Optional[float] = None
    if len(state.ma_buffer) >= state.t:
        shifted_ma = state.ma_buffer[0]  # oldest value in the t-length buffer
        dpo_val = close - shifted_ma

    return [dpo_val], state


def _dpo_output_names(params: Dict[str, Any]) -> List[str]:
    length = _as_int(_param(params, "length", 20), 20)
    return [f"DPO_{length}"]


def _dpo_seed(series: Dict[str, Any], params: Dict[str, Any]) -> DPOState:
    """Seed DPO from historical data.

    For precise state, we need the SMA buffer and the shifted MA buffer.
    Since this is complex, we use replay_seed approach.
    """
    # DPO is effectively replay_only for accurate seeding
    # because we need both the SMA window and the shift buffer
    return replay_seed("dpo", series, params)


LOOKAHEAD_REGISTRY["dpo"] = StatefulIndicator(
    kind="dpo",
    inputs=("close",),
    init=_dpo_init,
    update=_dpo_update,
    output_names=_dpo_output_names,
)
SEED_REGISTRY["dpo"] = _dpo_seed


# ===========================================================================
# ICHIMOKU  (CONDITIONAL lookahead: lookahead=False, exclude chikou)
# ===========================================================================
# Ichimoku Kinko Hyo with forward-looking spans.
# The chikou span uses close.shift(-kijun+1) which is lookahead.
# Stateful version FIXES lookahead=False (exclude chikou span).
# Outputs: tenkan_sen, kijun_sen, span_a, span_b (4 outputs)

@dataclass
class IchimokuState:
    """Ichimoku state without chikou span (lookahead=False).

    RESTRICTION: lookahead=False is FIXED in stateful mode to avoid lookahead.
    The chikou span uses close.shift(-kijun+1) which looks into the future.
    Only the 4 components (tenkan, kijun, span_a, span_b) are streaming-compatible.
    The chikou span is EXCLUDED from stateful output.
    """
    tenkan: int
    kijun: int
    senkou: int
    high_tenkan: deque = field(default_factory=deque)
    low_tenkan: deque = field(default_factory=deque)
    high_kijun: deque = field(default_factory=deque)
    low_kijun: deque = field(default_factory=deque)
    high_senkou: deque = field(default_factory=deque)
    low_senkou: deque = field(default_factory=deque)
    tenkan_buffer: deque = field(default_factory=deque)  # for span_a shift
    kijun_buffer: deque = field(default_factory=deque)   # for span_a shift
    span_b_buffer: deque = field(default_factory=deque)  # for span_b shift


def _ichimoku_init(params: Dict[str, Any]) -> IchimokuState:
    """Initialize Ichimoku state. ALWAYS uses lookahead=False (no chikou)."""
    tenkan = _as_int(_param(params, "tenkan", 9), 9)
    kijun = _as_int(_param(params, "kijun", 26), 26)
    senkou = _as_int(_param(params, "senkou", 52), 52)

    return IchimokuState(
        tenkan=tenkan,
        kijun=kijun,
        senkou=senkou,
        high_tenkan=deque(maxlen=tenkan),
        low_tenkan=deque(maxlen=tenkan),
        high_kijun=deque(maxlen=kijun),
        low_kijun=deque(maxlen=kijun),
        high_senkou=deque(maxlen=senkou),
        low_senkou=deque(maxlen=senkou),
        tenkan_buffer=deque(maxlen=kijun - 1),
        kijun_buffer=deque(maxlen=kijun - 1),
        span_b_buffer=deque(maxlen=kijun - 1),
    )


def _ichimoku_update(
    state: IchimokuState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], IchimokuState]:
    """Update Ichimoku without chikou span (lookahead=False).

    Returns: [tenkan_sen, kijun_sen, span_a, span_b]

    Note: lookahead=False is FIXED in stateful mode to avoid future data.
          Chikou span is excluded.
    """
    high = bar["high"]
    low = bar["low"]

    # Update rolling windows
    state.high_tenkan.append(high)
    state.low_tenkan.append(low)
    state.high_kijun.append(high)
    state.low_kijun.append(low)
    state.high_senkou.append(high)
    state.low_senkou.append(low)

    # Calculate midprice for each period
    tenkan_sen: Optional[float] = None
    if len(state.high_tenkan) >= state.tenkan:
        tenkan_sen = 0.5 * (max(state.high_tenkan) + min(state.low_tenkan))

    kijun_sen: Optional[float] = None
    if len(state.high_kijun) >= state.kijun:
        kijun_sen = 0.5 * (max(state.high_kijun) + min(state.low_kijun))

    span_b_current: Optional[float] = None
    if len(state.high_senkou) >= state.senkou:
        span_b_current = 0.5 * (max(state.high_senkou) + min(state.low_senkou))

    # Span A = average of tenkan and kijun (unshifted)
    span_a_current: Optional[float] = None
    if tenkan_sen is not None and kijun_sen is not None:
        span_a_current = 0.5 * (tenkan_sen + kijun_sen)

    # Store current values for shifting
    if tenkan_sen is not None:
        state.tenkan_buffer.append(tenkan_sen)
    if kijun_sen is not None:
        state.kijun_buffer.append(kijun_sen)
    if span_b_current is not None:
        state.span_b_buffer.append(span_b_current)

    # Output shifted spans (shift forward by kijun-1)
    # This means we output values calculated kijun-1 bars ago
    span_a: Optional[float] = None
    span_b: Optional[float] = None

    if len(state.tenkan_buffer) >= state.kijun - 1 and len(state.kijun_buffer) >= state.kijun - 1:
        # Get values from kijun-1 bars ago
        old_tenkan = state.tenkan_buffer[0]
        old_kijun = state.kijun_buffer[0]
        span_a = 0.5 * (old_tenkan + old_kijun)

    if len(state.span_b_buffer) >= state.kijun - 1:
        span_b = state.span_b_buffer[0]

    return [tenkan_sen, kijun_sen, span_a, span_b], state


def _ichimoku_output_names(params: Dict[str, Any]) -> List[str]:
    tenkan = _as_int(_param(params, "tenkan", 9), 9)
    kijun = _as_int(_param(params, "kijun", 26), 26)
    return [
        f"ITS_{tenkan}",      # Tenkan Sen
        f"IKS_{kijun}",       # Kijun Sen
        f"ISA_{tenkan}",      # Span A (shifted)
        f"ISB_{kijun}",       # Span B (shifted)
    ]


def _ichimoku_seed(series: Dict[str, Any], params: Dict[str, Any]) -> IchimokuState:
    """Seed Ichimoku from historical data.

    Requires replay to build all rolling windows and shift buffers.
    """
    return replay_seed("ichimoku", series, params)


LOOKAHEAD_REGISTRY["ichimoku"] = StatefulIndicator(
    kind="ichimoku",
    inputs=("high", "low", "close"),
    init=_ichimoku_init,
    update=_ichimoku_update,
    output_names=_ichimoku_output_names,
)
SEED_REGISTRY["ichimoku"] = _ichimoku_seed


# ===========================================================================
# TOS_STDEVALL  (APPROXIMATE: expanding window with warning)
# ===========================================================================
# TD Ameritrade's Think or Swim Standard Deviation All
# Original uses all bars or last N bars for linear regression + std bands.
# Stateful version uses EXPANDING window (growing) and emits warning.

@dataclass
class TosStdevallState:
    """TOS_STDEVALL state using expanding window (approximate).

    APPROXIMATION: The vectorized version computes linear regression and std
    over ALL historical bars or a fixed window. In stateful mode, we use an
    EXPANDING window that grows with each new bar. This means results will
    differ from training data as the window keeps expanding. A runtime warning
    is emitted on first update.
    """
    stds: Tuple[int, ...]           # standard deviation bands
    ddof: int                        # degrees of freedom for std
    prices: List[float] = field(default_factory=list)
    warned: bool = False             # track if warning was emitted


def _tos_stdevall_init(params: Dict[str, Any]) -> TosStdevallState:
    """Initialize TOS_STDEVALL with expanding window."""
    stds_param = params.get("stds", [1, 2, 3])
    if not isinstance(stds_param, (list, tuple)):
        stds_param = [1, 2, 3]
    stds = tuple(sorted(stds_param))
    ddof = _as_int(_param(params, "ddof", 1), 1)

    return TosStdevallState(stds=stds, ddof=ddof)


def _tos_stdevall_update(
    state: TosStdevallState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], TosStdevallState]:
    """Update TOS_STDEVALL using expanding window.

    Returns: [lr, L_1, U_1, L_2, U_2, ...] for each std band.

    Warning: This is an APPROXIMATION. The expanding window grows with each
    bar, unlike the vectorized version which uses a fixed historical window.
    Results will differ from training data.
    """
    close = bar["close"]

    # Emit warning once per state instance
    if not state.warned:
        warnings.warn(
            "TOS_STDEVALL in stateful mode uses EXPANDING window (approximate). "
            "Results will differ from vectorized training data which uses a fixed window.",
            UserWarning,
            stacklevel=2
        )
        state.warned = True

    # Add price to expanding window
    state.prices.append(close)
    n = len(state.prices)

    # Need at least 2 points for linear regression
    if n < 2:
        # Return None for all outputs
        num_outputs = 1 + 2 * len(state.stds)  # LR + (L_i, U_i) for each std
        return [None] * num_outputs, state

    # Linear regression: y = mx + b
    # Using simple least squares
    x_vals = list(range(n))
    sum_x = sum(x_vals)
    sum_y = sum(state.prices)
    sum_xx = sum(x * x for x in x_vals)
    sum_xy = sum(x * y for x, y in zip(x_vals, state.prices))

    # Slope and intercept
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    b = (sum_y - m * sum_x) / n

    # Current LR value (at position n-1)
    lr = m * (n - 1) + b

    # Calculate standard deviation
    mean_price = sum_y / n
    variance = sum((p - mean_price) ** 2 for p in state.prices) / (n - state.ddof)
    stdev = variance ** 0.5

    # Build output list: [lr, L_1, U_1, L_2, U_2, ...]
    outputs: List[Optional[float]] = [lr]
    for std_mult in state.stds:
        outputs.append(lr - std_mult * stdev)  # Lower band
        outputs.append(lr + std_mult * stdev)  # Upper band

    return outputs, state


def _tos_stdevall_output_names(params: Dict[str, Any]) -> List[str]:
    stds_param = params.get("stds", [1, 2, 3])
    if not isinstance(stds_param, (list, tuple)):
        stds_param = [1, 2, 3]
    stds = sorted(stds_param)

    names = ["TOS_STDEVALL_LR"]
    for std_mult in stds:
        names.append(f"TOS_STDEVALL_L_{std_mult}")
        names.append(f"TOS_STDEVALL_U_{std_mult}")
    return names


def _tos_stdevall_seed(series: Dict[str, Any], params: Dict[str, Any]) -> TosStdevallState:
    """Seed TOS_STDEVALL from historical data.

    We need to replay all historical prices to build the expanding window.
    """
    return replay_seed("tos_stdevall", series, params)


LOOKAHEAD_REGISTRY["tos_stdevall"] = StatefulIndicator(
    kind="tos_stdevall",
    inputs=("close",),
    init=_tos_stdevall_init,
    update=_tos_stdevall_update,
    output_names=_tos_stdevall_output_names,
)
SEED_REGISTRY["tos_stdevall"] = _tos_stdevall_seed


# ===========================================================================
# VP  (APPROXIMATE: expanding window Volume Profile with warning)
# ===========================================================================
# Volume Profile bins price/volume into ranges.
# Original is a distribution snapshot, not a time series.
# Stateful version uses EXPANDING window VP and emits warning.

@dataclass
class VPState:
    """VP state using expanding window (approximate).

    APPROXIMATION: Volume Profile is NOT a time series indicator but a
    distribution snapshot. The vectorized version bins price/volume over a
    fixed historical period. In stateful mode, we use an EXPANDING window
    that grows with each bar, making this a very rough approximation. Results
    will differ significantly from training data. A runtime warning is emitted
    on first update.
    """
    width: int                       # number of price bins
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    warned: bool = False             # track if warning was emitted


def _vp_init(params: Dict[str, Any]) -> VPState:
    """Initialize VP with expanding window."""
    width = _as_int(_param(params, "width", 10), 10)
    return VPState(width=width)


def _vp_update(
    state: VPState, bar: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[List[Optional[float]], VPState]:
    """Update VP using expanding window.

    Returns: [low_price, mean_price, high_price, pos_volume, neg_volume,
              neut_volume, total_volume] for the most active bin.

    Warning: This is an APPROXIMATION. The expanding window grows with each
    bar. Results will differ significantly from vectorized training data.
    """
    close = bar["close"]
    volume = bar["volume"]

    # Emit warning once per state instance
    if not state.warned:
        warnings.warn(
            "VP in stateful mode uses EXPANDING window (approximate). "
            "Volume Profile is not a time series indicator. "
            "Results will differ significantly from vectorized training data.",
            UserWarning,
            stacklevel=2
        )
        state.warned = True

    # Add to expanding window
    state.prices.append(close)
    state.volumes.append(volume)
    n = len(state.prices)

    # Need at least width bars to create meaningful bins
    if n < state.width:
        return [None] * 7, state

    # Calculate price direction
    prev_close = state.prices[-2] if n > 1 else close
    price_change = close - prev_close

    if price_change > 0:
        pos_vol = volume
        neg_vol = 0.0
    elif price_change < 0:
        pos_vol = 0.0
        neg_vol = volume
    else:
        pos_vol = 0.0
        neg_vol = 0.0

    # Bin the data
    min_price = min(state.prices)
    max_price = max(state.prices)

    # Avoid division by zero
    if max_price == min_price:
        price_range = 1.0
    else:
        price_range = max_price - min_price

    bin_width = price_range / state.width

    # Initialize bins
    bins = [
        {
            "low": min_price + i * bin_width,
            "high": min_price + (i + 1) * bin_width,
            "pos_vol": 0.0,
            "neg_vol": 0.0,
            "neut_vol": 0.0,
            "count": 0,
            "price_sum": 0.0,
        }
        for i in range(state.width)
    ]

    # Assign each price/volume to a bin
    for i, (p, v) in enumerate(zip(state.prices, state.volumes)):
        # Determine bin index
        if p == max_price:
            bin_idx = state.width - 1
        else:
            bin_idx = min(int((p - min_price) / bin_width), state.width - 1)

        bins[bin_idx]["count"] += 1
        bins[bin_idx]["price_sum"] += p

        # Determine volume direction
        if i > 0:
            prev_p = state.prices[i - 1]
            if p > prev_p:
                bins[bin_idx]["pos_vol"] += v
            elif p < prev_p:
                bins[bin_idx]["neg_vol"] += v
            else:
                bins[bin_idx]["neut_vol"] += v
        else:
            bins[bin_idx]["neut_vol"] += v

    # Find bin with highest total volume
    max_vol = 0.0
    max_bin = bins[0]
    for b in bins:
        total_vol = b["pos_vol"] + b["neg_vol"] + b["neut_vol"]
        if total_vol > max_vol:
            max_vol = total_vol
            max_bin = b

    # Calculate mean price for the bin
    if max_bin["count"] > 0:
        mean_price = max_bin["price_sum"] / max_bin["count"]
    else:
        mean_price = 0.5 * (max_bin["low"] + max_bin["high"])

    total_volume = max_bin["pos_vol"] + max_bin["neg_vol"]

    return [
        max_bin["low"],
        mean_price,
        max_bin["high"],
        max_bin["pos_vol"],
        max_bin["neg_vol"],
        max_bin["neut_vol"],
        total_volume,
    ], state


def _vp_output_names(params: Dict[str, Any]) -> List[str]:
    width = _as_int(_param(params, "width", 10), 10)
    # Using generic close/volume names since input names vary
    return [
        f"VP_{width}_low_price",
        f"VP_{width}_mean_price",
        f"VP_{width}_high_price",
        f"VP_{width}_pos_volume",
        f"VP_{width}_neg_volume",
        f"VP_{width}_neut_volume",
        f"VP_{width}_total_volume",
    ]


def _vp_seed(series: Dict[str, Any], params: Dict[str, Any]) -> VPState:
    """Seed VP from historical data.

    We need to replay all historical prices/volumes to build the expanding window.
    """
    return replay_seed("vp", series, params)


LOOKAHEAD_REGISTRY["vp"] = StatefulIndicator(
    kind="vp",
    inputs=("close", "volume"),
    init=_vp_init,
    update=_vp_update,
    output_names=_vp_output_names,
)
SEED_REGISTRY["vp"] = _vp_seed
