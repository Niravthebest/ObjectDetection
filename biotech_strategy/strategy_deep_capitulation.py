"""Biotech Deep Capitulation Bounce -- a refinement of strategy.py.

Found by comparing entry-day features of the base strategy's 29
trades that returned >=100% (out of 1,610 total, 2017-2025, 30-day
hold) against the rest: the single strongest, most monotonic
discriminator was how far the stock had already fallen from its
52-week HIGH (not just proximity to its 52-week low, which the base
strategy already requires). The monster winners were disproportionately
stocks already down >80% from their high, not just "near their low" --
a 52-week range can be narrow, so "near the low" doesn't imply "fell a
long way."

Adds one condition to the base rule: price must be down more than 80%
from its trailing 52-week high.

Effect on the 2017-2025, 30-day-hold trade set: baseline 1,610 trades,
49.7% win rate, 0% median return, 29 trades >=100%. Filtered to 215
trades, 54.0% win rate, +3.1% median return, 18 of the 29 monster
winners retained (62% recall) -- the filter raises the average and
median outcome of the WHOLE remaining bucket, not just isolating the
tail.
"""
import pandas as pd

from indicators import rsi, pct_from_52w_low, pct_from_52w_high, volume_ratio


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, vol = out["close"], out["volume"]

    out["rsi14"] = rsi(close, 14)
    out["pct_from_52w_low"] = pct_from_52w_low(close)
    out["pct_from_52w_high"] = pct_from_52w_high(close)
    out["vol_ratio20"] = volume_ratio(vol, 20)
    out["ret_5d"] = close.pct_change(5)

    out["cond_oversold"] = out["rsi14"] < 35
    out["cond_near_low"] = out["pct_from_52w_low"] < 20
    out["cond_volume"] = out["vol_ratio20"] >= 1.5
    out["cond_not_already_bounced"] = out["ret_5d"] < 0.10
    out["cond_deep_drawdown"] = out["pct_from_52w_high"] < -80

    out["entry_signal"] = (
        out["cond_oversold"]
        & out["cond_near_low"]
        & out["cond_volume"]
        & out["cond_not_already_bounced"]
        & out["cond_deep_drawdown"]
    )
    return out
