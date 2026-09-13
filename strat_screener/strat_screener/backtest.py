"""Historical backtest of TheStrat combo patterns, bucketed by timeframe,
pattern, direction, and whether the *other* two timeframes were in Full
Time Frame Continuity (FTFC) with the trade direction at the time.

Lookahead safety: when checking what the "other" timeframes were doing as
of a given daily/weekly/monthly bar, we only ever look at the last bar of
the other timeframe that had *fully closed* before the current bar opened
(strictly before its `begins_at`). This deliberately ignores the
still-forming higher-timeframe candle (which is what a live screener would
show) in favor of the last confirmed one, so the backtest never peeks at
information that wasn't yet final at that point in time.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bars import classify_types, bar_direction
from .patterns import detect_pattern

TIMEFRAMES = ("day", "week", "month")


def prepare_symbol_frames(data_source, symbol: str, timeframes: Sequence[str] = TIMEFRAMES) -> Dict[str, pd.DataFrame]:
    frames = {}
    for tf in timeframes:
        df = data_source.get_bars(symbol, tf).copy()
        df["type"] = classify_types(df)
        df["direction"] = bar_direction(df)
        frames[tf] = df
    return frames


def other_direction_asof(other_index: pd.DatetimeIndex, other_direction: pd.Series, current_begins_at) -> Optional[str]:
    """Direction of the last `other` bar that closed strictly before `current_begins_at`."""
    pos = other_index.searchsorted(current_begins_at, side="left")
    if pos == 0:
        return None
    return other_direction.iloc[pos - 1]


def backtest_symbol(frames: Dict[str, pd.DataFrame], horizon: int = 6) -> List[dict]:
    """Scan every timeframe's history for pattern occurrences and record the
    forward outcome `horizon` bars later (in that same timeframe's units).
    """
    events: List[dict] = []
    for tf in TIMEFRAMES:
        df = frames[tf]
        types = df["type"].values
        n = len(df)
        for i in range(2, n - horizon):
            match = detect_pattern(types, i)
            if match is None:
                continue

            window = df.iloc[i - match.n_bars + 1: i + 1]
            range_high = window["high"].max()
            range_low = window["low"].min()
            entry = df["close"].iloc[i]
            stop = range_low if match.direction == "up" else range_high
            risk = abs(entry - stop)
            if risk <= 0:
                continue

            exit_price = df["close"].iloc[i + horizon]
            raw_pnl = (exit_price - entry) if match.direction == "up" else (entry - exit_price)
            r_multiple = raw_pnl / risk

            others = [t for t in TIMEFRAMES if t != tf]
            other_dirs = []
            for ot in others:
                od = other_direction_asof(frames[ot].index, frames[ot]["direction"], df.index[i])
                if od is not None:
                    other_dirs.append(od)
            ftfc_aligned = len(other_dirs) == len(others) and all(d == match.direction for d in other_dirs)

            events.append({
                "timeframe": tf,
                "pattern": match.name,
                "direction": match.direction,
                "ftfc_aligned": ftfc_aligned,
                "r_multiple": r_multiple,
                "return_pct": raw_pnl / entry,
                "win": bool(r_multiple > 0),
                "date": df.index[i],
            })
    return events


def _bucket_row(g: pd.DataFrame) -> dict:
    return {
        "n": int(len(g)),
        "win_rate": float(g["win"].mean()),
        "avg_r": float(g["r_multiple"].mean()),
        "median_r": float(g["r_multiple"].median()),
        "avg_return_pct": float(g["return_pct"].mean()),
    }


def aggregate_stats(all_events: List[dict]) -> dict:
    """Build a lookup of bucket key -> {n, win_rate, avg_r, median_r, avg_return_pct}.

    Two key granularities are stored so scoring can fall back gracefully:
      "<tf>|<pattern>|<direction>|ftfc=<bool>"   (specific)
      "<tf>|<pattern>|<direction>"               (pattern-only, ignores FTFC)
    """
    if not all_events:
        return {}
    df = pd.DataFrame(all_events)
    stats: Dict[str, dict] = {}

    df["bucket"] = df["timeframe"] + "|" + df["pattern"] + "|" + df["direction"] + "|ftfc=" + df["ftfc_aligned"].astype(str)
    for bucket, g in df.groupby("bucket"):
        stats[bucket] = _bucket_row(g)

    df["bucket_generic"] = df["timeframe"] + "|" + df["pattern"] + "|" + df["direction"]
    for bucket, g in df.groupby("bucket_generic"):
        stats[bucket] = _bucket_row(g)

    return stats


def run_backtest(data_source, symbols: Sequence[str], horizon: int = 6, timeframes: Sequence[str] = TIMEFRAMES):
    """Returns (stats, per_symbol_frames, all_events)."""
    all_events: List[dict] = []
    per_symbol_frames: Dict[str, dict] = {}
    for sym in symbols:
        try:
            frames = prepare_symbol_frames(data_source, sym, timeframes)
        except Exception as exc:  # noqa: BLE001 - keep scanning the rest of the universe
            print(f"skip {sym}: {exc}")
            continue
        per_symbol_frames[sym] = frames
        all_events.extend(backtest_symbol(frames, horizon=horizon))
    stats = aggregate_stats(all_events)
    return stats, per_symbol_frames, all_events
