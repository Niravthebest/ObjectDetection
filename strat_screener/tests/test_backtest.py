import pandas as pd

from strat_screener.backtest import backtest_symbol, other_direction_asof


def _df(dates, highs, lows, closes, types, directions, opens=None, volumes=None):
    idx = pd.to_datetime(dates)
    n = len(dates)
    return pd.DataFrame({
        "open": opens or closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes or [1000] * n,
        "type": types,
        "direction": directions,
    }, index=idx)


def test_other_direction_asof_ignores_same_or_later_bars():
    other_index = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
    other_direction = pd.Series(["down", "up", "down"], index=other_index)

    # Before any bar closed -> None.
    assert other_direction_asof(other_index, other_direction, pd.Timestamp("2023-12-31")) is None
    # Exactly at a bar's begins_at -> that bar hasn't closed yet, use the prior one.
    assert other_direction_asof(other_index, other_direction, pd.Timestamp("2024-01-08")) == "down"
    # Well after the second bar started -> still the last one strictly before it, i.e. the first.
    assert other_direction_asof(other_index, other_direction, pd.Timestamp("2024-01-10")) == "up"


def test_backtest_symbol_detects_pattern_and_computes_r_multiple():
    # idx1 = 1 (inside rel. idx0), idx2 = 2u (rel. idx1), idx3 = 1 (rel. idx2),
    # idx4 = 2u (rel. idx3) -> "2-1-2 continuation" up detected at i=4.
    # window = bars[2:5] -> high max = 20, low min = 12.
    # entry = close[4] = 16, stop = 12, risk = 4.
    # horizon=1 -> exit = close[5] = 30 -> raw_pnl = 14 -> r_multiple = 3.5, win=True.
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    day = _df(
        dates,
        highs=[20, 18, 19, 17, 20, 30],
        lows=[10, 12, 12, 13, 13, 16],
        closes=[15, 13, 16, 15, 16, 30],
        types=[None, "1", "2u", "1", "2u", "2u"],
        directions=["up", "down", "up", "down", "up", "up"],
    )
    # Weekly/monthly: single earlier bar, both "up", closed well before any daily bar,
    # so FTFC should read as fully aligned for an "up" daily setup.
    week = _df(["2023-12-25"], [20], [5], [18], [None], ["up"])
    month = _df(["2023-12-01"], [20], [5], [18], [None], ["up"])

    frames = {"day": day, "week": week, "month": month}
    events = backtest_symbol(frames, horizon=1)

    day_events = [e for e in events if e["timeframe"] == "day"]
    assert len(day_events) == 1
    event = day_events[0]
    assert event["pattern"] == "2-1-2 continuation"
    assert event["direction"] == "up"
    assert event["ftfc_aligned"] is True
    assert abs(event["r_multiple"] - 3.5) < 1e-9
    assert event["win"] is True


def test_backtest_symbol_skips_zero_risk_setups():
    # Same "2-1-2 continuation" type sequence as above, but the window bars
    # (idx 2-4) are all flat (high == low == 10) so risk == 0. The event
    # must be skipped rather than dividing by zero.
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    day = _df(
        dates,
        highs=[20, 18, 10, 10, 10, 12],
        lows=[10, 12, 10, 10, 10, 9],
        closes=[15, 13, 10, 10, 10, 11],
        types=[None, "1", "2u", "1", "2u", "2u"],
        directions=["up", "down", "up", "up", "up", "up"],
    )
    week = _df(["2023-12-25"], [10], [10], [10], [None], ["up"])
    month = _df(["2023-12-01"], [10], [10], [10], [None], ["up"])
    events = backtest_symbol({"day": day, "week": week, "month": month}, horizon=1)
    assert events == []
