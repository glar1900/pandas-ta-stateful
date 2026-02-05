# Classification Handoff (for External AI)

This document is an instruction sheet for the AI (or reviewer) who will fill
`docs/stateful_indicator_classification.md`.

## Goal
Classify each indicator into memory type and (if stateful) seed method, with a
short reason. The output should enable streaming behavior with:
- window indicators using only lookback data
- stateful indicators using persisted state + incremental updates
- lookahead indicators excluded in streaming

## File to Fill
- `docs/stateful_indicator_classification.md`

## Columns and Allowed Values
- `memory_type`: `window` | `stateful` | `lookahead` | `unknown`
- `seed_method` (stateful only): `output_only` | `internal_series` | `replay_only` | `not_applicable`
- `lookback`: integer or short formula if window‑based
- `state_fields`: comma‑separated internal fields needed for state
- `reason`: short justification (1–2 sentences)
- `notes`: optional (edge cases, options like talib/presma/mamode)

## Definitions
- **window**: output depends only on a fixed recent window (no long‑memory)
- **stateful**: recursive/long‑memory; must persist state across updates
- **lookahead**: uses future data or writes to future timestamps (exclude)
- **output_only**: last output(s) alone are enough to continue
- **internal_series**: needs intermediate series (e.g., avg_gain/avg_loss)
- **replay_only**: cannot seed from vectorized outputs; must replay once

## Classification Guidance (How to Decide)
1. **Lookahead?**
   - If the indicator writes future spans or uses future bars → `lookahead`
2. **Fixed window only?**
   - If output depends only on last *N* bars (rolling stats, candle patterns),
     then `window` and fill `lookback`
3. **Otherwise it’s stateful**
   - Recursive MA / smoothing / trend state → `stateful`
   - Decide seed method:
     - `output_only`: next value depends on last output only (EMA-like)
     - `internal_series`: needs internal series for exact seed (RSI, ATR, ADX)
     - `replay_only`: if internal state not derivable from vectorized outputs

## Examples (Reference)
- EMA: `stateful`, `output_only`, state_fields: `ema`
- RSI (RMA): `stateful`, `internal_series`, state_fields: `avg_gain, avg_loss, prev_close`
- SMA: `window`, lookback: `length`
- Candle patterns: `window`, lookback: usually 2–10 bars
- Ichimoku (future spans): `lookahead`

## Expectations
- Fill **every row** in the table.
- Use **short, consistent reasons**.
- If unsure, mark `unknown` and explain in `notes`.
