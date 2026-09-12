"""Translates the Contraction Scan condition string into a per-bar boolean
signal for one ticker's OHLCV history. See README.md for the mapping from
each clause of the original scan syntax to the code below, and for the
assumptions made where the syntax was ambiguous.
"""
import pandas as pd

import indicators as ind


def compute_signals(df: pd.DataFrame, rsi_lag: int = 1) -> pd.DataFrame:
    """df must have columns: date, open, high, low, close, volume, sorted
    ascending by date. Returns df with an added boolean `entry_signal`
    column plus the underlying indicator columns (for inspection)."""
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    df["sma20"] = ind.sma(c, 20)
    df["sma50"] = ind.sma(c, 50)
    df["sma100"] = ind.sma(c, 100)
    df["sma200"] = ind.sma(c, 200)

    df["atr1"] = ind.atr(h, l, c, 1)
    df["atr5"] = ind.atr(h, l, c, 5)
    df["atr20"] = ind.atr(h, l, c, 20)
    df["atr50"] = ind.atr(h, l, c, 50)
    df["natr50"] = ind.natr(h, l, c, 50)

    df["pgo20"] = ind.pgo(h, l, c, 20)
    df["pgo50"] = ind.pgo(h, l, c, 50)

    df["rsi7"] = ind.rsi(c, 7)
    df["advol30"] = ind.advol(c, v, 30)

    df["arange0"] = ind.arange_(h, l, 0)

    # exch/type/advol(30) > 3 -- liquidity gate
    cond_liquidity = df["advol30"] > 3

    # !(sma(20) < sma(50))@{0..20} -- no bearish 20/50 cross-under in the
    # last 21 bars (today plus the prior 20)
    cond_no_recent_crossunder = ind.no_cross_under_in_window(df["sma20"], df["sma50"], 20)

    # !(price < sma(50) and sma(50) trend_dn 20)
    cond_not_breakdown = ~(
        (c < df["sma50"]) & ind.trend_dn(df["sma50"], 20)
    )

    # price > sma(100) or price > sma(200)
    cond_above_long_ma = (c > df["sma100"]) | (c > df["sma200"])

    # sma(200) trend_up 60
    cond_long_uptrend = ind.trend_up(df["sma200"], 60)

    # natr(50) > 1.5
    cond_volatility_floor = df["natr50"] > 1.5

    # price > sma(50) - arange(0)
    cond_near_50sma = c > (df["sma50"] - df["arange0"])

    # contraction: today's true range much smaller than recent ATR, either
    # outright or (as an inside day) with a looser threshold
    prev_high, prev_low = h.shift(1), l.shift(1)
    is_inside_day = (c < prev_high) & (c > prev_low)
    tight_strict = (
        (df["atr1"] < df["atr5"] * 0.5)
        | (df["atr1"] < df["atr20"] * 0.5)
        | (df["atr1"] < df["atr50"] * 0.5)
    )
    tight_loose = (
        (df["atr1"] < df["atr5"] * 0.75)
        | (df["atr1"] < df["atr20"] * 0.75)
        | (df["atr1"] < df["atr50"] * 0.75)
    )
    cond_contraction = tight_strict | (is_inside_day & tight_loose)

    # pgo(50) < 2.5 or pgo(20) < 2.5
    cond_not_extended = (df["pgo50"] < 2.5) | (df["pgo20"] < 2.5)

    # (rsi(7) < 60)@1 -- evaluated on the PRIOR bar's RSI, per the @1 offset
    cond_rsi = df["rsi7"].shift(rsi_lag) < 60

    df["entry_signal"] = (
        cond_liquidity
        & cond_no_recent_crossunder
        & cond_not_breakdown
        & cond_above_long_ma
        & cond_long_uptrend
        & cond_volatility_floor
        & cond_near_50sma
        & cond_contraction
        & cond_not_extended
        & cond_rsi
    )
    return df
