"""Indicator primitives used by the Contraction Scan strategy.

All functions take/return pandas Series indexed by date, ascending order
(oldest first), one per ticker.
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Wilder-smoothed Average True Range. length=1 degenerates to the raw
    true range of the current bar, which is how the scan's `atr(1)` reads."""
    tr = true_range(high, low, close)
    if length <= 1:
        return tr
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def natr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    return atr(high, low, close, length) / close * 100.0


def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result[avg_loss == 0] = 100.0
    return result


def pgo(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Pretty Good Oscillator: (close - SMA(length)) / ATR(length)."""
    return (close - sma(close, length)) / atr(high, low, close, length)


def arange_(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """Average daily range over `length` bars. length=0 means "today's bar
    only" (the range of the current bar, no averaging)."""
    bar_range = high - low
    if length <= 0:
        return bar_range
    return bar_range.rolling(length).mean()


def advol(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Average dollar volume over `length` bars, in millions of dollars."""
    dollar_vol = close * volume
    return dollar_vol.rolling(length).mean() / 1_000_000.0


def trend_up(series: pd.Series, length: int) -> pd.Series:
    """True where the series is higher than it was `length` bars ago."""
    return series > series.shift(length)


def trend_dn(series: pd.Series, length: int) -> pd.Series:
    """True where the series is lower than it was `length` bars ago."""
    return series < series.shift(length)


def no_cross_under_in_window(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """True at bar t if `a < b` was NOT true at any offset 0..window bars
    ago (inclusive) -- i.e. no bearish state within that lookback."""
    cross_under = (a < b)
    # rolling window of size (window+1) covering offsets 0..window
    any_cross_under = cross_under.rolling(window + 1).max().astype(bool)
    return ~any_cross_under
