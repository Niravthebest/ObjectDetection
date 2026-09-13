"""TheStrat bar (candle) classification.

Every bar on any timeframe is classified relative to the immediately
preceding bar on the *same* timeframe:

  1  (inside)    high <= prev_high and low  >= prev_low
  2u (directional up)    high  > prev_high and low  >= prev_low
  2d (directional down)  low   < prev_low  and high <= prev_high
  3  (outside)   high  > prev_high and low  <  prev_low

A bar's "direction" (used for Full Time Frame Continuity checks) is simply
whether it closed green or red: close >= open is up, otherwise down.
"""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def classify_types(df: pd.DataFrame) -> pd.Series:
    """Return a Series of '1'/'2u'/'2d'/'3' aligned to df.index.

    The first row has no prior bar to compare against and is set to None.
    """
    high, low = df["high"], df["low"]
    prev_high, prev_low = high.shift(1), low.shift(1)
    broke_up = high > prev_high
    broke_down = low < prev_low

    types = pd.Series(index=df.index, dtype=object)
    types[broke_up & broke_down] = "3"
    types[broke_up & ~broke_down] = "2u"
    types[~broke_up & broke_down] = "2d"
    types[~broke_up & ~broke_down] = "1"
    if len(types):
        types.iloc[0] = None
    return types


def bar_direction(df: pd.DataFrame) -> pd.Series:
    """Return a Series of 'up'/'down' per bar based on close vs open."""
    return pd.Series(
        np.where(df["close"] >= df["open"], "up", "down"), index=df.index
    )
