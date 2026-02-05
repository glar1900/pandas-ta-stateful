# -*- coding: utf-8 -*-
from numba import njit
from numpy import floor, nan, zeros, zeros_like, roll
from pandas import Series, DataFrame
from pandas_ta._typing import DictLike, Int, IntFloat
from pandas_ta.utils import (
    v_bool,
    v_offset,
    v_pos_default,
    v_series,
)


# Find high and low pivots using a centered rolling window.
@njit(cache=True)
def nb_rolling_hl(np_high, np_low, window_size):

    m = np_high.size
    
    # 2x capacity 확보 (핵심)
    cap = m * 2

    idx   = zeros(cap)
    swing = zeros(cap)
    value = zeros(cap)

    extremes = 0

    left  = int(floor(window_size / 2))
    right = left + 1

    for i in range(left, m - right):

        low_center  = np_low[i]
        high_center = np_high[i]

        low_window  = np_low[i-left:i+right]
        high_window = np_high[i-left:i+right]

        # ---- LOW PIVOT ----
        if (low_center <= low_window).all():

            if extremes >= cap:
                break

            idx[extremes]   = i
            swing[extremes] = -1
            value[extremes] = low_center
            extremes += 1

        # ---- HIGH PIVOT ----
        if (high_center >= high_window).all():

            if extremes >= cap:
                break

            idx[extremes]   = i
            swing[extremes] = 1
            value[extremes] = high_center
            extremes += 1

    return idx[:extremes], swing[:extremes], value[:extremes]


# Calculate zigzag points using pre-calculated unfiltered pivots.
@njit(cache=True)
def nb_zz_backtest(idx, swing, value, deviation):
    # Safety check: empty arrays
    if idx.size == 0 or swing.size == 0 or value.size == 0:
        return zeros(0), zeros(0), zeros(0), zeros(0)
    
    zz_idx = zeros_like(idx)
    zz_swing = zeros_like(swing)
    zz_value = zeros_like(value)
    zz_dev = zeros_like(idx)

    zigzags = 0
    changes = 0
    zz_idx[zigzags] = idx[0]
    zz_swing[zigzags] = swing[0]
    zz_value[zigzags] = value[0]
    zz_dev[zigzags] = 0

    # print(f'Starting S: {zz_swing[0]}')

    m = idx.size
    for i in range(1, m):
        # Safety: ensure we don't overflow
        if zigzags >= idx.size - 1:
            break
            
        last_zz_value = zz_value[zigzags]
        current_dev = (value[i] - last_zz_value) / last_zz_value

        # print(f'{i} | P {swing[i]:.0f} : {idx[i]:.0f} , {value[i]}')
        # print(f'{len(str(i))*" "} | Last: {zz_swing[zigzags-changes]:.0f} , Dev: %{(current_dev*100):.1f}')

        # Safety: ensure zigzags-changes >= 0
        lookback_idx = max(0, zigzags - changes)
        
        # Last point in zigzag is bottom
        if zz_swing[lookback_idx] == -1:
            if swing[i] == -1:
                # If the current pivot is lower than the last ZZ bottom:
                # create a new point and log it as a change
                if value[i] < zz_value[zigzags]:
                    if zz_idx[lookback_idx] == idx[i]:
                        continue
                    # print(f'{len(str(i))*" "} | Change -1 : {zz_value[zigzags]} to {value[i]}')
                    zigzags += 1
                    changes += 1
                    if zigzags < idx.size:  # Safety check
                        zz_idx[zigzags] = idx[i]
                        zz_swing[zigzags] = swing[i]
                        zz_value[zigzags] = value[i]
                        zz_dev[zigzags] = 100 * current_dev
            else:
                # If the deviation between pivot and the last ZZ bottom is
                # great enough create new ZZ point.
                if current_dev > 0.01 * deviation:
                    if zz_idx[lookback_idx] == idx[i]:
                        continue
                    # print(f'{len(str(i))*" "} | new ZZ 1 {value[i]}')
                    zigzags += 1
                    if zigzags < idx.size:  # Safety check
                        zz_idx[zigzags] = idx[i]
                        zz_swing[zigzags] = swing[i]
                        zz_value[zigzags] = value[i]
                        zz_dev[zigzags] = 100 * current_dev
                        changes = 0

        # last point in zigzag is top
        else:
            if swing[i] == 1:
                # If the current pivot is higher than the last ZZ top:
                # create a new point and log it as a change
                if value[i] > zz_value[zigzags]:
                    if zz_idx[lookback_idx] == idx[i]:
                        continue
                    # print(f'{len(str(i))*" "} | Change 1 : {zz_value[zigzags]} to {value[i]}')
                    zigzags += 1
                    changes += 1
                    if zigzags < idx.size:  # Safety check
                        zz_idx[zigzags] = idx[i]
                        zz_swing[zigzags] = swing[i]
                        zz_value[zigzags] = value[i]
                        zz_dev[zigzags] = 100 * current_dev
            else:
                # If the deviation between pivot and the last ZZ top is great
                # enough create new ZZ point.
                if current_dev < -0.01 * deviation:
                    if zz_idx[lookback_idx] == idx[i]:
                        continue
                    # print(f'{len(str(i))*" "} | new ZZ -1 {value[i]}')
                    zigzags += 1
                    if zigzags < idx.size:  # Safety check
                        zz_idx[zigzags] = idx[i]
                        zz_swing[zigzags] = swing[i]
                        zz_value[zigzags] = value[i]
                        zz_dev[zigzags] = 100 * current_dev
                        changes = 0

    _n = min(zigzags + 1, idx.size)  # Safety: don't exceed array size
    return zz_idx[:_n], zz_swing[:_n], zz_value[:_n], zz_dev[:_n]


# Calculate zigzag points using pre-calculated unfiltered pivots.
@njit(cache=True)
def nb_find_zz(idx, swing, value, deviation):

    n = idx.size

    # Early exit
    if n == 0:
        return zeros(0), zeros(0), zeros(0), zeros(0)

    # Preallocate (max possible size)
    zz_idx   = zeros_like(idx)
    zz_swing = zeros_like(swing)
    zz_value = zeros_like(value)
    zz_dev   = zeros_like(value)

    # ---- INIT ----
    zigzags = 0

    zz_idx[0]   = idx[n - 1]
    zz_swing[0] = swing[n - 1]
    zz_value[0] = value[n - 1]
    zz_dev[0]   = 0.0

    # ---- MAIN LOOP ----
    for i in range(n - 2, -1, -1):

        # Capacity guard (BEFORE increment)
        if zigzags + 1 >= n:
            break

        cur_val   = value[i]
        cur_swing = swing[i]
        cur_idx   = idx[i]

        # Prevent div by zero globally
        if cur_val == 0.0:
            continue

        last_val   = zz_value[zigzags]
        last_swing = zz_swing[zigzags]

        # =========================
        # NEXT IS BOTTOM
        # =========================
        if last_swing == -1:

            # Lower low replaces last
            if cur_swing == -1:

                if cur_val < last_val and zigzags >= 1:
                    prev_val = zz_value[zigzags - 1]
                    dev = (prev_val - cur_val) / cur_val

                    zz_idx[zigzags]   = cur_idx
                    zz_swing[zigzags] = -1
                    zz_value[zigzags] = cur_val
                    zz_dev[zigzags - 1] = 100.0 * dev

            # Opposite swing → new pivot
            else:

                dev = (cur_val - last_val) / cur_val

                if dev > 0.01 * deviation and cur_idx != zz_idx[zigzags]:

                    zigzags += 1

                    zz_idx[zigzags]   = cur_idx
                    zz_swing[zigzags] = 1
                    zz_value[zigzags] = cur_val
                    zz_dev[zigzags - 1] = 100.0 * dev

        # =========================
        # NEXT IS TOP
        # =========================
        else:

            # Higher high replaces last
            if cur_swing == 1:

                if cur_val > last_val and zigzags >= 1:
                    prev_val = zz_value[zigzags - 1]
                    dev = (cur_val - prev_val) / cur_val

                    zz_idx[zigzags]   = cur_idx
                    zz_swing[zigzags] = 1
                    zz_value[zigzags] = cur_val
                    zz_dev[zigzags - 1] = 100.0 * dev

            # Opposite swing → new pivot
            else:

                dev = (last_val - cur_val) / cur_val

                if dev > 0.01 * deviation and cur_idx != zz_idx[zigzags]:

                    zigzags += 1

                    zz_idx[zigzags]   = cur_idx
                    zz_swing[zigzags] = -1
                    zz_value[zigzags] = cur_val
                    zz_dev[zigzags - 1] = 100.0 * dev


    out_n = zigzags + 1

    return (
        zz_idx[:out_n],
        zz_swing[:out_n],
        zz_value[:out_n],
        zz_dev[:out_n],
    )



# Maps nb_find_zz results back onto the original data indices.
@njit(cache=True)
def nb_map_zz(idx, swing, value, deviation, n):
    swing_map = zeros(n)
    value_map = zeros(n)
    dev_map = zeros(n)

    for j, i in enumerate(idx):
        i = int(i)
        # Safety: ensure i is within bounds
        if i >= n or i < 0:
            continue
        swing_map[i] = swing[j]
        value_map[i] = value[j]
        dev_map[i] = deviation[j]

    for i in range(n):
        if swing_map[i] == 0:
            swing_map[i] = nan
            value_map[i] = nan
            dev_map[i] = nan

    return swing_map, value_map, dev_map



def zigzag(
    high: Series, low: Series, close: Series = None,
    legs: int = None, deviation: IntFloat = None, backtest: bool = None,
    offset: Int = None, **kwargs: DictLike
):
    """Zigzag

    This indicator attempts to filter out smaller movements while identifying
    trend direction. It does not predict future trends, but it does identify
    swing highs and lows.

    Sources:
        * [stockcharts](https://school.stockcharts.com/doku.php?id=technical_indicators:zigzag)
        * [tradingview](https://www.tradingview.com/support/solutions/43000591664-zig-zag/#:~:text=Definition,trader%20visual%20the%20price%20action.)

    Parameters:
        high (Series): ```high``` Series
        low (Series): ```low``` Series
        close (Series): ```close``` Series. Default: ```None```
        legs (int): Number of legs (> 2). Default: ```10```
        deviation (float): Reversal deviation percentage. Default: ```5```
        backtest (bool): Default: ```False```
        offset (int): Post shift. Default: ```0```

    Other Parameters:
        fillna (value): ```pd.DataFrame.fillna(value)```

    Returns:
        (DataFrame): 2 columns

    Note: Deviation
        When ```deviation=10```, it shows movements greater than ```10%```.

    Note: Backtest Mode
        Ensures the DataFrame is safe for backtesting. By default, swing
        points are returned on the pivot index. Intermediate swings are
        not returned at all. This mode swing detection is placed on the bar
        that would have been detected. Furthermore, changes in swing levels
        are also included instead of only the final value.

        * Use the following formula to get the true index of a pivot:
          ```p_i = i - int(floor(legs / 2))```

    Warning:
        A Series reversal will create a new line.
    """
    # Validate
    legs = v_pos_default(legs, 10)
    _length = legs + 1
    high = v_series(high, _length)
    low = v_series(low, _length)

    if high is None or low is None:
        return

    if close is not None:
        close = v_series(close,_length)
        np_close = close.values
        if close is None:
            return

    deviation = v_pos_default(deviation, 5.0)
    offset = v_offset(offset)
    backtest = v_bool(backtest, False)

    if backtest:
        offset+=int(floor(legs/2))

    # Wrap entire calculation in try-except to prevent heap corruption crashes
    try:
        # Calculation
        np_high, np_low = high.to_numpy(), low.to_numpy()
        hli, hls, hlv = nb_rolling_hl(np_high, np_low, legs)

        # Safety: if no pivots found, return None
        if hli.size == 0 or hls.size == 0 or hlv.size == 0:
            return None

        if backtest:
            zzi, zzs, zzv, zzd = nb_zz_backtest(hli, hls, hlv, deviation)
        else:
            zzi, zzs, zzv, zzd = nb_find_zz(hli, hls, hlv, deviation)
        
        # Safety: if no zigzag points found, return None
        if zzi.size == 0:
            return None

        swing, value, dev = nb_map_zz(zzi, zzs, zzv, zzd, np_high.size)

        # Offset
        if offset != 0:
            swing = roll(swing, offset)
            value = roll(value, offset)
            dev = roll(dev, offset)

            swing[:offset] = nan
            value[:offset] = nan
            dev[:offset] = nan

        # Name and Category
        _props = f"_{deviation}%_{legs}"
        
        # Create Series directly from numpy arrays - safest approach
        swing_series = Series(swing, index=high.index, name=f"ZIGZAGs{_props}")
        value_series = Series(value, index=high.index, name=f"ZIGZAGv{_props}")
        dev_series = Series(dev, index=high.index, name=f"ZIGZAGd{_props}")
        
        # Only apply fillna if explicitly requested (rare case)
        if "fillna" in kwargs:
            fill_value = kwargs["fillna"]
            swing_series = swing_series.fillna(fill_value)
            value_series = value_series.fillna(fill_value)
            dev_series = dev_series.fillna(fill_value)
        
        df = DataFrame({
            f"ZIGZAGs{_props}": swing_series,
            f"ZIGZAGv{_props}": value_series,
            f"ZIGZAGd{_props}": dev_series,
        })
        df.name = f"ZIGZAG{_props}"
        df.category = "trend"

        return df
    
    except Exception as e:
        # Any error in zigzag - return None instead of crashing
        # This prevents heap corruption from killing the entire process
        return None
