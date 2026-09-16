"""Pluggable OHLCV data sources.

The screener only needs a `get_bars(symbol, timeframe) -> DataFrame` call
with columns [open, high, low, close, volume] indexed by an ascending
datetime index. Swap in whatever provider you have access to.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd

from .bars import REQUIRED_COLUMNS

RESAMPLE_RULE = {"week": "W-FRI", "month": "ME"}
RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


class PriceDataSource(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """timeframe is one of 'day', 'week', 'month'."""
        raise NotImplementedError


class InMemoryDataSource(PriceDataSource):
    """Serves pre-built DataFrames, keyed by (symbol, timeframe).

    Used for tests and for feeding data pulled from a provider (e.g. a
    brokerage API) that isn't wired up as a first-class data source here.
    """

    def __init__(self, data: Dict[Tuple[str, str], pd.DataFrame]):
        self._data = data

    def get_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._data[(symbol, timeframe)]


class YFinanceDataSource(PriceDataSource):
    """Fetches daily bars from Yahoo Finance and resamples up for week/month.

    Resampling from daily (rather than fetching each interval separately)
    guarantees the week/month bars are built from the same underlying
    prices as the daily series and only costs one network call per symbol.
    """

    def __init__(self, period: str = "15y"):
        self.period = period
        self._daily_cache: Dict[str, pd.DataFrame] = {}

    def _get_daily(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._daily_cache:
            import yfinance as yf

            df = yf.download(
                symbol, period=self.period, interval="1d",
                progress=False, auto_adjust=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            df.index.name = "date"
            if df.empty:
                raise ValueError(f"no data returned for {symbol!r}")
            self._daily_cache[symbol] = df[REQUIRED_COLUMNS].dropna()
        return self._daily_cache[symbol]

    def get_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        daily = self._get_daily(symbol)
        if timeframe == "day":
            return daily
        rule = RESAMPLE_RULE[timeframe]
        return daily.resample(rule).agg(RESAMPLE_AGG).dropna()


def robinhood_bars_to_df(bars: list) -> pd.DataFrame:
    """Convert Robinhood MCP `get_equity_historicals` bar dicts to a DataFrame.

    Each bar dict looks like:
      {"begins_at": "...", "open_price": "...", "high_price": "...",
       "low_price": "...", "close_price": "...", "volume": ...}

    Bars flagged "interpolated" are synthetic gap-fill with no real trade
    data (flat OHLC, zero volume) and are dropped — keeping them would
    inject a spurious inside-bar into the type sequence and corrupt
    pattern detection right at the most recent, most-actionable bar.
    """
    bars = [b for b in bars if not b.get("interpolated")]
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["begins_at"])
    df = df.rename(columns={
        "open_price": "open", "high_price": "high",
        "low_price": "low", "close_price": "close",
    })
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.set_index("date").sort_index()
    return df[REQUIRED_COLUMNS]
