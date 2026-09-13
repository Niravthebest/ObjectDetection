import pandas as pd

from strat_screener.bars import classify_types, bar_direction


def make_df(rows):
    """rows: list of (open, high, low, close, volume)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    return df


def test_classify_types_all_cases():
    # bar0: baseline
    # bar1: inside  (10-15 contained within 8-20)
    # bar2: 2u      (breaks high 20->22, low stays >= 8)
    # bar3: 2d      (breaks low 8->5, high stays <= 22)
    # bar4: outside (breaks both sides of bar3's 5-22 range... use 3-25 range)
    rows = [
        (10, 20, 8, 15, 100),
        (12, 15, 10, 13, 100),
        (14, 22, 11, 20, 100),
        (18, 21, 5, 6, 100),
        (5, 25, 3, 24, 100),
    ]
    df = make_df(rows)
    types = classify_types(df)
    assert types.iloc[0] is None
    assert types.iloc[1] == "1"
    assert types.iloc[2] == "2u"
    assert types.iloc[3] == "2d"
    assert types.iloc[4] == "3"


def test_bar_direction():
    rows = [
        (10, 12, 9, 11, 100),  # close > open -> up
        (10, 12, 9, 9, 100),   # close < open -> down
        (10, 12, 9, 10, 100),  # close == open -> up (tie goes up)
    ]
    df = make_df(rows)
    directions = bar_direction(df)
    assert list(directions) == ["up", "down", "up"]
