"""Indicator primitives for the biotech condition-discovery research.

All functions take/return pandas Series indexed by date, ascending order
(oldest first), one per ticker. Ported from contraction_scan_backtest's
indicators.py with a few additions (Bollinger width, 52-week range
position, volume ratio, gap) needed for the broader feature sweep here.
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
    return (close - sma(close, length)) / atr(high, low, close, length)


def bollinger_width(close: pd.Series, length: int, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width as a fraction of the mid-band price -- a
    standard "squeeze" measure: low values mean the bands have pinched in,
    often preceding a volatility expansion either direction."""
    mid = sma(close, length)
    std = close.rolling(length).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (upper - lower) / mid


def pct_from_52w_high(close: pd.Series) -> pd.Series:
    rolling_high = close.rolling(252, min_periods=60).max()
    return (close - rolling_high) / rolling_high * 100.0


def pct_from_52w_low(close: pd.Series) -> pd.Series:
    rolling_low = close.rolling(252, min_periods=60).min()
    return (close - rolling_low) / rolling_low * 100.0


def volume_ratio(volume: pd.Series, length: int) -> pd.Series:
    """Today's volume relative to its trailing average -- a spike > 1
    means above-average participation."""
    avg = volume.rolling(length).mean().shift(1)  # prior-day average, no lookahead
    return volume / avg


def gap_pct(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap: today's open vs yesterday's close, as a percent."""
    prev_close = close.shift(1)
    return (open_ - prev_close) / prev_close * 100.0


def trend_up(series: pd.Series, length: int) -> pd.Series:
    return series > series.shift(length)


def trend_dn(series: pd.Series, length: int) -> pd.Series:
    return series < series.shift(length)
