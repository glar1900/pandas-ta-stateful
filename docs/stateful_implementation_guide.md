# Stateful Implementation Guide (for External AI)

This guide explains how to implement `stateful.py` + `seed_state` support
for all indicators listed in `docs/stateful_indicator_classification.md`.

## Scope
- Implement streaming updates for **stateful** indicators.
- Implement seed extraction for **output_only** and **internal_series**.
- **Lookahead** indicators are excluded in streaming (except noted approximations).
- **Replay_only** indicators must support incremental update, but seed is
  obtained by a one‑time replay.

## Files and Conventions
### Primary code
- `pandas_ta/stateful.py`
  - Add dataclasses for state
  - Add `init(params)` and `update(state, inputs, params)` for each indicator
  - Add to `STATEFUL_REGISTRY`

### Expected interface
```
init(params: dict) -> state
update(state, inputs: dict, params: dict) -> (list[float|None], state)
```

Where:
- `inputs` contains the named inputs (close/high/low/open/volume etc)
- return list order must match `output_names(params)`

### Naming & output order
Follow existing indicator output names from vectorized versions.
Use `output_names(params)` helpers in `stateful.py`.

### TA‑Lib compatibility
When `talib=True` is used in the original indicator, the **stateful update must
match TA‑Lib output** (initialization and smoothing). Match:
- EMA: SMA seed, `alpha=2/(n+1)`
- RMA/Wilder: SMA seed, `alpha=1/n`
- ATR/RSI/ADX etc: use Wilder smoothing

## Classification‑Driven Rules
Use `docs/stateful_indicator_classification.md` as the source of truth.

### memory_type = window
- Do **not** implement in `stateful.py`.
- These are computed via vectorized rolling window from full df.

### memory_type = lookahead
- By default, exclude from streaming
- **Conditional streaming is allowed** when the indicator exposes a no‑lookahead option:
  - `dpo`: use `centered=False`
  - `ichimoku`: use `lookahead=False` (exclude chikou)
- Two **approximate** exceptions are allowed:
  - `tos_stdevall`: use **expanding window** (growing) and warn
  - `vp`: use **expanding window VP** and warn

### stateful + seed_method
- `output_only`: seed state from last output value(s)
- `internal_series`: seed state from intermediate series (avg_gain, atr, etc)
- `replay_only`: seed by one‑time replay (stateful update from start)

## Examples

### Output‑only (EMA)
```
state = { "ema": last_ema_value }
update: ema = alpha * x + (1-alpha) * prev_ema
```

### Internal‑series (RSI, RMA)
```
state = { "avg_gain": g, "avg_loss": l, "prev_close": c }
update:
  gain = max(delta, 0); loss = max(-delta, 0)
  avg_gain = (avg_gain*(n-1)+gain)/n
  avg_loss = (avg_loss*(n-1)+loss)/n
  rsi = 100 * avg_gain / (avg_gain + avg_loss)
```

### Replay‑only (PSAR)
PSAR’s state includes: SAR value, AF, EP, direction. These are not derivable
from final output alone. You must:
1) implement `update` for streaming, and
2) seed by replaying from the beginning up to `state_timestamp`

### What each seed_method means (Implementation Notes)
- **output_only**
  - State fields are just the last output(s) needed to continue.
  - Example: EMA, OBV, PVT, KAMA.
  - Seed: take the last output value(s) from vectorized calc.
  - Update: simple recurrence using last output.
- **internal_series**
  - Output alone is insufficient. Need intermediate series.
  - Example: RSI (avg_gain/avg_loss), ATR (prev_atr), ADX (DM/ATR).
  - Seed: compute intermediate series vectorized and take last values.
  - Update: recurrence uses those internal values.
- **replay_only**
  - Internal state is complex and not derivable from vectorized outputs.
  - Example: PSAR, ZigZag, RSX, STC.
  - Seed: run stateful update from the beginning once (replay) to build state.
  - Update: use the standard stateful update loop.

## Guidance by Category

### MAs / Overlap
- EMA, RMA, SMMA, KAMA, T3, TEMA, ZLMA: output_only or internal_series per table
- DEMA, TRIX: **internal_series** (needs multiple EMA states)

### Momentum
- RSI/CMO/CRSI: internal_series (avg_gain/avg_loss)
- MACD/PPO: output_only but needs 2–3 EMA states
- QQE, STC, RSX: replay_only (multi‑stage loops)

### Volatility
- ATR/NATR: internal_series (ATR state)
- ATRTS: internal_series (needs ATR/MA states + prev ATRTS)

### Trend
- ADX/RWI/CKSP: internal_series (ATR + DM/MA states)
- PSAR, ZigZag: replay_only

### Volume
- AD/OBV/PVT/NVI/PVI: output_only (cumulative sums)
- VWAP: internal_series (cum_pv, cum_vol, session key)

## Seed State Extraction (seed_state)
Implement a `seed_state()` utility per indicator:
- For output_only: take last output value(s)
- For internal_series: compute intermediate series vectorized and take last
  values (avg_gain/avg_loss etc)
- For replay_only: do not implement seed; require replay

Add a `seed_state_registry` mapping:
```
SEED_REGISTRY = {
  "ema": ema_seed_state,
  "rsi": rsi_seed_state,
  ...
}
```

## Lookahead Warnings
- If `lookahead` indicator is requested in streaming mode, skip it and
  emit a warning.
- For `tos_stdevall` and `vp`, allow **expanding window** approximation with
  a clear warning message.

## Testing Expectations
- Provide a minimal comparison test for each implemented indicator:
  - vectorized TA‑Lib output vs stateful update
  - seed_state + incremental update must match vectorized output

## Deliverable Checklist
- State dataclasses added
- init/update/output_names implemented
- Registry entry added
- Seed function (if output_only or internal_series)
- Warnings for lookahead or approximations
