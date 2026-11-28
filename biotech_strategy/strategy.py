"""Biotech Capitulation Bounce -- derived from the condition-discovery
sweep in analyze_conditions.py.

That sweep found the strongest, most reliable (largest sample, most
consistent across horizons) EDGE in this biotech universe was not a
smooth technical pattern but a volatility-clustering + partial-reversal
effect: biotech names that are already oversold AND trading near their
52-week low AND seeing a pickup in volume (capitulation selling, often
after a bad catalyst) have a meaningfully elevated chance of a sharp
bounce over the following 10 trading days, versus the universe baseline.

Entry signal (True on the day the condition is observed; traded at
next bar's open, matching the contraction_scan_backtest convention):
  - RSI(14) < 35          -- oversold
  - price within 20% of its trailing 52-week low  -- capitulation zone
  - volume >= 1.5x its trailing 20-day average     -- participation/confirmation
  - NOT already up more than 10% in the last 5 days -- avoid buying into
    a bounce that's already underway (reduces late entries)
"""
import pandas as pd

from indicators import rsi, pct_from_52w_low, volume_ratio


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, vol = out["close"], out["volume"]

    out["rsi14"] = rsi(close, 14)
    out["pct_from_52w_low"] = pct_from_52w_low(close)
    out["vol_ratio20"] = volume_ratio(vol, 20)
    out["ret_5d"] = close.pct_change(5)

    out["cond_oversold"] = out["rsi14"] < 35
    out["cond_near_low"] = out["pct_from_52w_low"] < 20
    out["cond_volume"] = out["vol_ratio20"] >= 1.5
    out["cond_not_already_bounced"] = out["ret_5d"] < 0.10

    out["entry_signal"] = (
        out["cond_oversold"]
        & out["cond_near_low"]
        & out["cond_volume"]
        & out["cond_not_already_bounced"]
    )
    return out
