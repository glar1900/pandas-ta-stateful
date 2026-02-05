# Stateful Indicator Classification

Fill the table with classification, seed method, and reasons.

Allowed values:
- memory_type: window | stateful | lookahead | unknown
- seed_method (stateful only): output_only | internal_series | replay_only | not_applicable

Definitions:
- window: fixed lookback only; no long-term state required
- stateful: recursive/long-memory; requires state to continue
- lookahead: uses future data or writes future spans (exclude in streaming)
- output_only: last output value(s) fully define next state
- internal_series: requires intermediate series (e.g., avg_gain/avg_loss)
- replay_only: seed not derivable from vectorized outputs; must replay to extract state

| kind | category | memory_type | seed_method | lookback | state_fields | reason | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aberration | volatility | stateful | internal_series | - | atr_state, sma_state | Depends on ATR (stateful RMA by default) and SMA of HLC3. ATR requires internal state for RMA smoothing. | mamode parameter affects ATR statefulness |
| accbands | volatility | stateful | output_only | - | ma_state (for _lower, close, _upper) | Uses MA (default SMA) on transformed high-low ratios. MA state depends on mamode parameter. | If mamode=sma, becomes window; if ema/rma becomes stateful with output_only |
| ad | volume | stateful | output_only | 0 | cumsum_value | Uses cumsum to accumulate volume-weighted price position | Can seed from last AD value alone |
| adosc | volume | stateful | internal_series | 0 | ad_cumsum, fast_ema_state, slow_ema_state | Computes EMA(fast) - EMA(slow) of AD; AD uses cumsum | Requires AD cumsum + both EMA states (prev value + alpha) |
| adx | trend | stateful | internal_series | - | atr_state, pos_ma_state, neg_ma_state, dx_ma_state | ADX uses RMA (or other MA) on ATR, directional movements, and DX. Requires persisting MA states for ATR, DMP, DMN, and ADX smoothing. | Depends on mamode parameter; talib vs tvmode have different initialization |
| alligator | overlap | stateful | output_only | - | jaw_smma, teeth_smma, lips_smma | Three SMMA lines with recursive formula: SMMA[i] = ((n-1)*SMMA[i-1] + close[i])/n | Each SMMA line needs its last value to continue |
| alma | overlap | window | not_applicable | length | - | Gaussian-weighted moving average using fixed sliding window with pre-computed weights | No recursion, pure window operation |
| alphatrend | trend | stateful | internal_series | - | atr_state, rsi_state (or mfi_state), prev_at | Uses ATR and RSI/MFI (both stateful), plus nb_alpha loop references previous result. Needs state for all underlying indicators and previous AT value. | State depends on volume presence (RSI vs MFI) |
| amat | trend | stateful | internal_series | 2 | fast_ma_state, slow_ma_state | Computes fast/slow MAs (stateful) then applies increasing/decreasing with lookback=2 (window). Overall stateful due to MA components. | MA states dominate; lookback is just for trend detection |
| ao | momentum | window | not_applicable | 34 | - | Uses SMA of median price over fixed windows | AO = SMA(median, 5) - SMA(median, 34) |
| aobv | volume | stateful | internal_series | 0 | obv_cumsum, ma_fast_state, ma_slow_state | Uses OBV (cumsum) with MAs and rolling min/max | Needs OBV cumsum + MA states; rolling is window-based |
| apo | momentum | stateful | output_only | - | fast_ma, slow_ma | Uses two MAs (default SMA but supports EMA) | If EMA mode: stateful; if SMA: window |
| aroon | trend | window | not_applicable | length+1 | - | Uses rolling window to find periods since highest high and lowest low within fixed window. Pure window-based indicator. | No recursive state needed |
| atr | volatility | stateful | internal_series | - | tr_series, rma_state | Uses RMA (default) of true range. With presma=True, initializes with SMA then applies RMA recursively. | mamode=rma (default) is stateful; presma affects initialization |
| atrts | volatility | stateful | internal_series | - | prev_atrts, atr_state, ma_state | Recursive trailing stop uses previous result plus ATR/MA state. Can seed from those internal values without full replay. | Uses nb_atrts; needs prev result and current up/dn from price vs MA |
| bbands | volatility | window | not_applicable | length | - | Rolling standard deviation and MA (default SMA). Purely window-based calculation. | If mamode changed to EMA/RMA, becomes stateful |
| bias | momentum | stateful | output_only | - | ma_value | Depends on mamode; default SMA=window, EMA=stateful | (close / MA) - 1 |
| bop | momentum | window | not_applicable | 0 | - | Direct OHLC calculation, no lookback | BOP = (close - open) / (high - low) |
| brar | momentum | window | not_applicable | 26 | - | Uses rolling sum over fixed window | AR and BR from rolling sums of ranges |
| cci | momentum | window | not_applicable | 14 | - | SMA and MAD over fixed window | CCI = (TP - SMA(TP)) / (c * MAD(TP)) |
| cdl_doji | candle | window | not_applicable | 10 | - | Compares candle body to SMA of high-low range over fixed window; body < 0.1% of avg range identifies doji. | Uses SMA(hl_range, 10) for comparison; no recursion |
| cdl_inside | candle | window | not_applicable | 1 | - | Identifies inside bars where high/low are within previous bar's range using roll(x, 1). | Compares current bar with previous only; fixed 1-bar lookback |
| cdl_pattern | candle | window | not_applicable | 1-3 | - | Pattern recognition compares current candle with 1-3 previous bars using fixed lookback; no recursive state. | Most TA-Lib patterns use 1-3 bar lookback. Wrapper includes doji (10-bar SMA) and inside (1-bar comparison). |
| cdl_z | candle | window | not_applicable | length | - | Applies rolling Z-score (SMA and stdev) to OHLC over fixed window; pure rolling statistics. | Default length=30. When full=True, uses expanding window but still window-based. |
| cfo | momentum | window | not_applicable | 9 | - | Linear regression over fixed window | Forecast oscillator using linreg |
| cg | momentum | window | not_applicable | 10 | - | Weighted rolling window calculation | Center of Gravity indicator |
| chandelier_exit | volatility | stateful | internal_series | - | atr_state, direction_ffill | Uses ATR (stateful) and direction with forward-fill. Direction state needs previous values. | ATR is stateful with RMA; direction uses replace(0, nan).ffill() |
| chop | trend | window | not_applicable | length | - | Uses rolling max/min and rolling sum of ATR over fixed window. Although ATR is stateful, CHOP only uses rolling sum which is window-based. | ATR is computed but used in rolling sum (window operation) |
| cksp | trend | stateful | internal_series | - | atr_state | Chande Kroll Stop uses ATR (stateful RMA/SMA), then applies rolling max/min (window). Overall stateful due to ATR dependency. | Two-stage: stateful ATR then window operations |
| cmf | volume | window | not_applicable | 20 | - | Rolling sum of money flow ratio over fixed window | Pure window indicator using rolling sums |
| cmo | momentum | stateful | internal_series | - | pos_avg, neg_avg | Uses RMA (EMA-based) for smoothing gains/losses | Needs separate avg_gain and avg_loss |
| coppock | momentum | window | not_applicable | 35 | - | ROC and WMA both window-based | WMA(ROC(11) + ROC(14), 10) |
| crsi | momentum | stateful | internal_series | - | rsi_state, streak_rsi_state | RSI requires RMA state for gains/losses | Composite of RSI, streak RSI, percent rank |
| cti | momentum | window | not_applicable | 12 | - | Linear regression correlation over window | Wrapper for linreg(r=True) |
| decay | trend | stateful | output_only | - | prev_decay | Recursive decay: result[i] = max(0, x[i], result[i-1] * rate) or result[i-1] - rate. Previous output alone defines next state. | Linear or exponential mode affects formula only |
| decreasing | trend | window | not_applicable | length | - | Checks if close.diff(length) < 0 or strict mode checks each step. Pure comparison over fixed window, no long-term state. | Strict mode uses loop but still window-based |
| dema | overlap | stateful | internal_series | - | ema1, ema2 | DEMA = 2*EMA(close) - EMA(EMA(close)); requires two cascaded EMA states | Cannot reconstruct from final output alone; needs both intermediate EMAs |
| dm | momentum | stateful | output_only | - | dmp_ma_state, dmn_ma_state | Directional Movement smoothed with MA (default RMA). Each direction needs MA state. | Default mamode='rma' is stateful; uses ma(pos_dm) and ma(neg_dm) |
| drawdown | performance | stateful | output_only | 0 | max_close | Tracks cumulative maximum close to compute drawdown; max_close = close.cummax(). | Needs last cummax value to continue; DD = max_close - close |
| donchian | volatility | window | not_applicable | max(lower_length, upper_length) | - | Rolling min/max over fixed windows. No recursive state. | Pure window indicator |
| dpo | trend | lookahead | not_applicable | - | - | Detrended Price Oscillator shifts data by t=(0.5*length)+1. When centered=True (default), uses .shift(-t) which is lookahead. | Streaming allowed if centered=False (no lookahead) |
| ebsw | cycle | stateful | internal_series | - | lastHP, lastClose, filtHist[3] | Recursive highpass and SuperSmoother filters require previous HP values and 3-element filter history. Cannot seed from output alone. | Both initial and default versions are stateful; filtHist stores intermediate filtered values needed for next iteration. |
| efi | volume | stateful | output_only | 13 | ma_state | Price-volume momentum smoothed with MA (default EMA) | MA state only (EMA is output_only) |
| ema | overlap | stateful | output_only | - | ema | Recursive: EMA[t] = α*close[t] + (1-α)*EMA[t-1] via pandas ewm | Classic exponential smoothing; last output sufficient |
| entropy | statistics | window | not_applicable | length | - | Rolling entropy over fixed window using sum operations | Standard rolling window calculation |
| eom | volume | window | not_applicable | 14 | - | Distance/box-ratio smoothed with SMA | SMA is window-based with fixed lookback |
| er | momentum | window | not_applicable | 10 | - | Rolling sum of absolute volatility | Efficiency ratio = net change / sum(abs changes) |
| eri | momentum | stateful | output_only | - | ema_value | Uses EMA which is recursive | Bull/Bear power from high/low vs EMA |
| exhc | momentum | stateful | output_only | - | prev_up, prev_dn, prev_sign | Exhaustion counts are consecutive run lengths; next values can be derived from last counts and last diff sign. | Cap/show_all/nozeros affect output formatting only |
| fisher | momentum | stateful | replay_only | - | v, prev_fisher | Iterative with dependencies on previous values | Loop: v=0.66*pos+0.67*v_prev, fisher uses prev |
| fwma | overlap | window | not_applicable | length | - | Fibonacci-weighted moving average using rolling window with Fibonacci weights | Fixed window with static weight pattern |
| ha | candle | stateful | output_only | 0 | HA_open, HA_close | Recursive calculation: ha_open[i] = 0.5 * (ha_open[i-1] + ha_close[i-1]); each bar depends on previous HA values. | Can seed from last HA_open and HA_close outputs alone; no internal hidden state required. |
| hilo | overlap | stateful | output_only | - | hilo | Tracks trend using MAs of high/low; hilo[i] = hilo[i-1] when no trend change | Single state value; MAs can be recomputed in window mode but hilo itself is recursive |
| hl2 | overlap | window | not_applicable | 1 | - | Simple arithmetic: (high + low) / 2 | Current bar only, no memory |
| hlc3 | overlap | window | not_applicable | 1 | - | Simple arithmetic: (high + low + close) / 3 | Current bar only, no memory |
| hma | overlap | window | not_applicable | length + sqrt(length) | - | Hull MA uses WMA(2*WMA(close,n/2) - WMA(close,n), sqrt(n)); default mamode=wma is window-based | If mamode=ema, becomes stateful; default WMA is pure window |
| ht_trendline | trend | stateful | internal_series | - | wma4, dt, q1, i1, ji, jq, i2, q2, re, im, period, smp, i_trend | Hilbert Transform maintains complex recursive state across multiple internal series. Cannot seed from output alone; requires full internal state. | Complex Ehlers algorithm with many interdependent series |
| hwc | volatility | stateful | replay_only | - | last_a, last_v, last_f, last_var, last_price, last_result | Holt-Winters with explicit loop maintaining 6 state variables. Must replay to extract state. | Triple exponential smoothing state |
| hwma | overlap | stateful | replay_only | - | last_a, last_v, last_f | Holt-Winter MA with 3 interdependent state variables updated recursively in loop. | F, V, A each depend on previous values; similar to hwc triple exponential smoothing |
| ichimoku | overlap | lookahead | not_applicable | max(9,26,52) | - | Chikou span uses negative shift: close.shift(-kijun+1); spans shifted forward. | Streaming allowed if lookahead=False (exclude chikou); spans still forward-shifted but based only on past data |
| increasing | trend | window | not_applicable | length | - | Checks if close.diff(length) > 0 or strict mode checks each step. Pure comparison over fixed window, no long-term state. | Mirror of decreasing indicator |
| inertia | momentum | stateful | internal_series | - | rvi_state | RVI is EMA-based (stateful) | Linreg smoothing of stateful RVI |
| jma | overlap | stateful | internal_series | 7 | ma1, ma2, det0, det1, volty, v_sum, uBand, lBand, kv | Uses recursive smoothing with multiple internal state variables that cannot be derived from output alone | Jurik MA has complex state including volatility bands, detrender values, and multiple adaptive filter components |
| kama | overlap | stateful | output_only | 10 | prev_kama | Kaufman's Adaptive MA uses recursive formula: `kama[i] = sc[i] * close[i] + (1 - sc[i]) * kama[i-1]` | Seeds from initial SMA, then previous output is sufficient for continuation |
| kc | volatility | stateful | internal_series | - | ma_state (basis, band) | Uses MA (default EMA) on close and true range/high-low range. EMA requires state. | If mamode=sma, becomes window; EMA/RMA is stateful |
| kdj | momentum | stateful | output_only | - | k_ema_state, d_ema_state | pd_rma is EMA with alpha=1/n | K=RMA(fastk), D=RMA(K), J=3K-2D |
| kst | momentum | window | not_applicable | 45 | - | Multiple ROCs smoothed by SMA | Weighted sum of 4 smoothed ROCs |
| kurtosis | statistics | window | not_applicable | length | - | Rolling kurtosis using pandas rolling.kurt() | Fixed window statistical measure |
| kvo | volume | stateful | internal_series | 0 | signed_vol_cumsum, fast_ema_state, slow_ema_state, signal_ema_state | Signed volume with fast/slow EMAs and signal line | Needs cumsum + 3 EMA states |
| linreg | overlap | window | not_applicable | 14 | - | Linear regression over fixed rolling window using sliding window view | Pure window-based calculation, no recursion |
| log_return | performance | window | not_applicable | length | - | Non-cumulative mode (default) computes log(close[i]/close[i-length]), which only requires a fixed lookback window. | Cumulative mode requires state (initial close value), but default behavior is window-based with lookback=length. |
| long_run | trend | window | not_applicable | length | - | Applies increasing/decreasing (both window-based) to fast/slow series with fixed lookback. No recursive state. | Composite of window indicators |
| macd | momentum | stateful | output_only | - | ema_fast, ema_slow, ema_signal | Three recursive EMAs | MACD=EMA(12)-EMA(26), Signal=EMA(MACD,9) |
| mad | statistics | window | not_applicable | length | - | Rolling mean absolute deviation using fixed window | Applies custom MAD function over rolling window |
| mama | overlap | stateful | internal_series | 3 | wma4, dt, smp, i1, i2, ji, jq, q1, q2, re, im, alpha, period, phase, mama, fama | Complex Hilbert Transform with many interdependent recursive series | Ehlers MESA requires extensive internal state that cannot be reconstructed from outputs |
| massi | volatility | stateful | internal_series | - | ema_state (hl_ema1, hl_ema2) | Double EMA of high-low range, then rolling sum. EMA components are stateful. | Two levels of EMA require internal state |
| mcgd | overlap | stateful | output_only | 10 | prev_mcgd | McGinley Dynamic uses formula: `mcgd[i] = mcgd[i-1] + (close[i] - mcgd[i-1]) / divisor` | Rolling window with function that references previous output value |
| median | statistics | window | not_applicable | length | - | Rolling median using pandas rolling.median() | Fixed window statistical measure |
| mfi | volume | window | not_applicable | 14 | - | Money flow ratio using rolling sums via convolution | Fixed window using convolve for rolling sums |
| midpoint | overlap | window | not_applicable | 2 | - | Average of rolling min and max over fixed window | Simple rolling statistics, no recursion |
| midprice | overlap | window | not_applicable | 2 | - | Average of rolling high/low over fixed window | Simple rolling statistics on high/low, no recursion |
| mom | momentum | window | not_applicable | 10 | - | Simple n-period difference | close[i] - close[i-n] |
| natr | volatility | stateful | internal_series | - | atr_state | Normalized ATR (scalar/close * ATR). Depends on ATR which is stateful. | Inherits statefulness from ATR |
| nvi | volume | stateful | output_only | 0 | cumsum_value | Cumulative ROC on negative volume days | Can seed from last NVI value |
| obv | volume | stateful | output_only | 0 | cumsum_value | Cumulative signed volume | Can seed from last OBV value alone |
| ohlc4 | overlap | window | not_applicable | 1 | - | Simple average of OHLC for current bar only | Point-wise calculation, no window or state |
| pdist | volatility | window | not_applicable | 1 | - | Point calculation using current and previous bar (drift=1). No long memory. | Simple window of size drift+1 |
| percent_return | performance | window | not_applicable | length | - | Non-cumulative mode (default) computes (close[i]/close[i-length])-1, which only requires a fixed lookback window. | Cumulative mode requires state (initial close value), but default behavior is window-based with lookback=length. |
| pgo | momentum | stateful | internal_series | - | ema_state (for ATR EMA), sma_state, atr_prev_close | Combines SMA and EMA of ATR | Uses SMA(close) and EMA(ATR()), both require state |
| pivots | overlap | stateful | internal_series | anchor period | prev_period_ohlc, current_period_ohlc | Pivot levels use prior period OHLC and are forward-filled during the next period; no future data if computed at period close. | Requires anchor/session rollover handling |
| ppo | momentum | stateful | output_only | - | fast_ma_state, slow_ma_state, signal_ema_state | Percentage difference of two MAs with signal | Uses two MAs (default SMA) and EMA signal line |
| psar | trend | stateful | replay_only | - | sar, af, ep, falling | Parabolic SAR maintains state for SAR value, acceleration factor, extreme point, and direction. State interdependencies make seeding from output impractical. | Complex state evolution with reversals |
| psl | momentum | window | not_applicable | 12 | - | Rolling sum of sign changes | Uses rolling window sum of binary indicators |
| pvi | volume | stateful | output_only | 0 | cumsum_value | Recursive: pvi[i] = pvi[i-1] * (close[i]/close[i-1]) | Last PVI value sufficient to continue |
| pvo | volume | stateful | internal_series | 0 | fast_ema_state, slow_ema_state, signal_ema_state | EMAs of volume (fast-slow)/slow, plus signal | Needs 3 EMA states |
| pvol | volume | window | not_applicable | 0 | - | Instantaneous price * volume product | No memory required, point calculation |
| pvr | volume | window | not_applicable | 1 | - | Categorizes price/volume direction using diff | Uses diff with drift=1, no long-term state |
| pvt | volume | stateful | output_only | 0 | cumsum_value | Cumulative ROC(close) * volume | Can seed from last PVT value |
| pwma | overlap | window | not_applicable | 10 | - | Pascal's Triangle weighted rolling average | Fixed window with static weights, no recursion |
| qqe | momentum | stateful | replay_only | - | rsi_state, rsi_ma_state, long/short_state, trend_state | Complex iterative logic with RSI smoothing | Loop-based calculation with conditional state updates (lines 104-137) |
| qstick | trend | stateful | internal_series | - | ma_state | Q Stick is MA of (close - open). Stateful due to MA component (depends on mamode). | Seed method depends on mamode (e.g., EMA vs SMA) |
| quantile | statistics | window | not_applicable | length | - | Rolling quantile using pandas rolling.quantile() | Fixed window statistical measure |
| reflex | cycle | stateful | internal_series | - | _f (SuperSmoother series, last 2 values), _ms (last EMA value) | Recursive 2nd-order SuperSmoother filter and EMA-smoothed mean square require intermediate series state. | The SuperSmoother needs `_f[i-1]` and `_f[i-2]`, and the normalization needs the smoothed MS from `_ms[i-1]`. |
| rma | overlap | stateful | output_only | 10 | prev_rma | Wilder's MA is EMA with alpha=1/length, uses `ewm(alpha=alpha, adjust=False)` | EMA variant, previous output sufficient to continue |
| roc | momentum | window | not_applicable | 10 | - | Rate of change over fixed period | Simply (close - close[n]) / close[n] * 100 |
| rsi | momentum | stateful | internal_series | - | avg_gain, avg_loss, prev_close | Needs internal RMA/EMA series | Uses RMA smoothing of gains/losses (default mamode='rma') |
| rsx | momentum | stateful | replay_only | - | f0-f90, v4-v20 state variables | Jurik's complex iterative algorithm | Many internal state variables updated in loop (lines 52-112) |
| rvgi | momentum | window | not_applicable | 14 + 4 - 1 = 17 | - | SWMA smoothing with rolling sum | Uses SWMA (weighted window) and rolling sums |
| rvi | volatility | stateful | internal_series | - | stdev_state, ma_state (pos_avg, neg_avg) | Uses stdev (window) and MA (default EMA) of pos/neg std products. MA is stateful. | stdev is window, but EMA smoothing makes it stateful |
| rwi | trend | stateful | internal_series | - | atr_state | Random Walk Index uses ATR (stateful) then applies window operations. Overall stateful due to ATR. | ATR with RMA/SMA makes it stateful |
| short_run | trend | window | not_applicable | length | - | Applies increasing/decreasing (both window-based) to fast/slow series with fixed lookback. No recursive state. | Mirror of long_run |
| sinwma | overlap | window | not_applicable | 14 | - | Sine-weighted rolling average with static weights | Fixed window with precomputed sine weights, no recursion |
| skew | statistics | window | not_applicable | length | - | Rolling skewness using pandas rolling.skew() | Fixed window statistical measure |
| slope | momentum | window | not_applicable | 1 | - | Simple difference over period | (close - close[n]) / n, optionally as angle |
| sma | overlap | window | not_applicable | 10 | - | Simple rolling average using convolution or rolling sum | Classic window-based moving average |
| smc | momentum | window | not_applicable | max(14,50,20)+1 | - | Pattern detection with rolling stats | Uses rolling max/min and shifting, no recursion |
| smi | momentum | stateful | internal_series | - | tsi_states | Wrapper around TSI with signal | Delegates to TSI which uses double EMA smoothing |
| smma | overlap | stateful | output_only | 7 | prev_smma | Smoothed MA uses: `smma[i] = ((length-1) * smma[i-1] + close[i]) / length` | Seeds from initial SMA, then previous output sufficient |
| squeeze | momentum | window | not_applicable | max(bb,kc,mom)+1 | - | Compares BB and KC bands | Uses BBands, KC, and momentum with MAs, all window-based |
| squeeze_pro | momentum | window | not_applicable | max(bb,kc,mom)+1 | - | Enhanced squeeze with multiple KC levels | Similar to squeeze with additional band comparisons |
| ssf | overlap | stateful | internal_series | 20 | result[i-1], result[i-2] | Ehlers Super Smoother Filter uses recursive IIR formula requiring 2 previous output values | Needs last 2 outputs for 2-pole filter, but these are stored in result array |
| ssf3 | overlap | stateful | internal_series | 20 | result[i-1], result[i-2], result[i-3] | Ehlers 3-pole filter uses recursive IIR formula requiring 3 previous output values | Needs last 3 outputs for 3-pole filter |
| stc | momentum | stateful | replay_only | - | pf[], pff[], stoch1[], stoch2[] arrays | Schaff Trend Cycle iterative calculation | Loop-based with lookback within loop for min/max (lines 24-51) |
| stdev | statistics | window | not_applicable | length | - | Rolling standard deviation via variance sqrt | Uses rolling variance, no recursion |
| stoch | momentum | window | not_applicable | k + d + smooth_k | - | Stochastic oscillator | Rolling min/max with MA smoothing, fixed window |
| stochf | momentum | window | not_applicable | k + d - 1 | - | Fast stochastic | Rolling min/max with MA, simpler than stoch |
| stochrsi | momentum | stateful | internal_series | - | rsi_state, rsi_ma_states | Stochastic of RSI | Needs RSI state, then applies rolling window to RSI output |
| supertrend | overlap | stateful | internal_series | 7 | dir_[i-1], lb[i-1], ub[i-1] | Trend direction and bands depend on previous state, not just previous output | Requires tracking direction and adjusted band levels across bars |
| swma | overlap | window | not_applicable | 10 | - | Symmetric triangle weighted rolling average | Fixed window with static symmetric weights, no recursion |
| t3 | overlap | stateful | output_only | 10 | prev_ema_outputs | Six cascaded EMAs (e1-e6), each seedable from output, combined linearly | Composition of EMAs which are individually seedable from output |
| tema | overlap | stateful | output_only | 10 | prev_ema_outputs | Triple EMA: `3*(ema1 - ema2) + ema3`, each EMA seedable from output | Composition of three EMAs which are individually seedable |
| thermo | volatility | stateful | output_only | - | ma_state | Computes max of thermoL and thermoH, then applies MA (default EMA). EMA is stateful. | If mamode=sma, becomes window |
| tmo | momentum | stateful | output_only | - | ema_states (3 levels) | Triple EMA smoothing | Uses 3 sequential EMAs on signed delta sum |
| tos_stdevall | statistics | lookahead | not_applicable | varies | - | Linear regression with std bands over all/specified bars | For streaming, approximate with expanding (growing) window and warn about drift vs training |
| trendflex | trend | stateful | internal_series | - | _f, _ms | Ehlers Trendflex maintains internal filter state (_f) and mean square state (_ms) recursively. Requires both series for continuation. | Recursive super smoother plus trend calculation |
| trima | overlap | window | not_applicable | 10 | - | Double SMA: `sma(sma(close, half_length), half_length)` | Nested rolling windows, no long-term recursion |
| trix | momentum | stateful | internal_series | - | ema1, ema2, ema3 | Triple cascaded EMA: ema3 = EMA(EMA(EMA(close))); requires all 3 intermediate EMA states. | TRIX = pct_change(ema3); signal = rolling.mean(TRIX) |
| true_range | volatility | window | not_applicable | 1 | - | Max of hl_range, high-pc, pc-low where pc is close.shift(drift). Window of drift+1 bars. | No recursive state |
| tsi | momentum | stateful | output_only | - | slow_ema, fast_slow_ema, abs_slow_ema, abs_fast_slow_ema, signal_ma | Double EMA smoothing | Two parallel double-EMA chains plus signal line |
| ttm_trend | trend | window | not_applicable | 6 | - | Averages hl2 over length periods using shifts in loop; trend = close > avg(hl2, length). | Loop sums shifted hl2 values but no recursion; fixed window calculation |
| tsv | volume | window | not_applicable | 18 | - | Rolling sum of close-diff-weighted signed volume, smoothed with SMA | Default mamode is SMA (window-based) |
| ui | volatility | window | not_applicable | 2*length-1 | - | Rolling max for highest_close and rolling sum of squared downside. Pure window operations. | No recursive state, even with everget option |
| uo | momentum | window | not_applicable | max(7,14,28)+1 | - | Ultimate Oscillator | Rolling sums of BP and TR over 3 periods |
| variance | statistics | window | not_applicable | length | - | Rolling variance using pandas rolling.var() | Fixed window statistical measure |
| vhf | trend | window | not_applicable | length | - | Vertical Horizontal Filter uses rolling max, min, and sum over fixed window. No recursive state needed. | Pure window-based statistics |
| vhm | volume | window | not_applicable | 610 | - | Volume deviation from MA, normalized by stdev | Uses SMA and pstdev over fixed window |
| vidya | overlap | stateful | internal_series | 14 | prev_vidya, cmo_components | Variable Index Dynamic Average with adaptive alpha based on CMO rolling calculations | Requires previous output and CMO calculation involves rolling sums |
| vortex | trend | window | not_applicable | length | - | Vortex uses true_range (stateless) and rolling sums over fixed window. No long-term state required. | True range is instant calculation, rolling sum is window |
| vp | volume | lookahead | not_applicable | width | - | Volume Profile bins price/volume data into ranges; not a time series but a distribution snapshot. | For streaming, approximate with expanding window VP and warn; not strictly time-series |
| vwap | volume | stateful | internal_series | anchor period | cum_pv, cum_vol, session_key | VWAP is cumulative sums of price*volume and volume within an anchor period. Seed from last sums and current session key. | Reset on anchor/session boundary |
| vwma | volume | window | not_applicable | 10 | - | Rolling weighted average: SMA(price*vol) / SMA(vol) | Two SMAs over fixed window |
| wcp | overlap | window | not_applicable | 1 | - | Weighted close price: `(high + low + 2*close) / 4` | Point-wise calculation, no window or state |
| willr | momentum | window | not_applicable | 14 | - | Williams %R | Rolling min/max calculation |
| wma | overlap | window | not_applicable | 10 | - | Linearly weighted rolling average | Fixed window with linear weights, no recursion |
| zigzag | trend | stateful | replay_only | - | pivot_history, zz_state | ZigZag identifies swing highs/lows based on deviation threshold using complex numba loops. | nb_rolling_hl finds pivots, nb_find_zz filters by deviation; backtest mode differs from live |
| zlma | overlap | stateful | output_only | 10 | prev_ma | Zero-lag applies lag-adjusted input to chosen MA (default EMA) | Inherits statefulness from underlying MA; for EMA, output-only seedable |
| zscore | statistics | window | not_applicable | length | - | Z-score from rolling mean and stdev | Combines two window-based indicators (SMA + STDEV) |
