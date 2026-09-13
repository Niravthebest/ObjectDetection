"""Rank the currently-actionable TheStrat setups across a symbol universe."""
from typing import Dict, List, Sequence

import pandas as pd

from .backtest import TIMEFRAMES, other_direction_asof, prepare_symbol_frames
from .patterns import detect_pattern
from .scoring import composite_score, find_bucket

VOLUME_LOOKBACK = 20


def current_setups_for_symbol(frames: Dict[str, pd.DataFrame], stats: dict) -> List[dict]:
    results = []
    for tf in TIMEFRAMES:
        df = frames[tf]
        n = len(df)
        if n < 3:
            continue
        i = n - 1  # last fully completed bar
        match = detect_pattern(df["type"].values, i)
        if match is None:
            continue

        others = [t for t in TIMEFRAMES if t != tf]
        aligned, total = 0, 0
        for ot in others:
            odf = frames[ot]
            od = other_direction_asof(odf.index, odf["direction"], df.index[i])
            if od is not None:
                total += 1
                if od == match.direction:
                    aligned += 1

        volume = df["volume"]
        volume_ratio = None
        if len(volume) > VOLUME_LOOKBACK:
            avg_volume = volume.iloc[i - VOLUME_LOOKBACK:i].mean()
            if avg_volume:
                volume_ratio = volume.iloc[i] / avg_volume

        score, backtest_n = composite_score(stats, tf, match.name, match.direction, aligned, total, volume_ratio)
        fully_aligned = total > 0 and aligned == total
        bucket = find_bucket(stats, tf, match.name, match.direction, fully_aligned)

        window = df.iloc[max(0, i - match.n_bars + 1): i + 1]
        results.append({
            "timeframe": tf,
            "pattern": match.name,
            "direction": match.direction,
            "score": round(score, 1),
            "backtest_n": backtest_n,
            "win_rate": round(bucket["win_rate"], 3) if bucket else None,
            "avg_return_pct": round(bucket["avg_return_pct"], 4) if bucket else None,
            "avg_r": round(bucket["avg_r"], 2) if bucket else None,
            "ftfc": f"{aligned}/{total}",
            "bar_date": str(df.index[i].date()),
            "close": float(df["close"].iloc[i]),
            "trigger_high": float(window["high"].max()),
            "trigger_low": float(window["low"].min()),
        })
    return results


def run_screen(data_source, symbols: Sequence[str], stats: dict, timeframes: Sequence[str] = TIMEFRAMES) -> List[dict]:
    rows = []
    for sym in symbols:
        try:
            frames = prepare_symbol_frames(data_source, sym, timeframes)
        except Exception as exc:  # noqa: BLE001 - keep scanning the rest of the universe
            print(f"skip {sym}: {exc}")
            continue
        for row in current_setups_for_symbol(frames, stats):
            row["symbol"] = sym
            rows.append(row)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows
