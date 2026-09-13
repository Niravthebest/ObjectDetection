from strat_screener.scoring import backtest_edge, composite_score


def make_stats(n, win_rate, avg_r, ftfc_aligned=None, timeframe="day", pattern="2-1-2 continuation", direction="up"):
    suffix = f"|ftfc={ftfc_aligned}" if ftfc_aligned is not None else ""
    key = f"{timeframe}|{pattern}|{direction}{suffix}"
    return {key: {"n": n, "win_rate": win_rate, "avg_r": avg_r, "median_r": avg_r, "avg_return_pct": 0.01}}


def test_missing_bucket_returns_neutral():
    score, n = backtest_edge({}, "day", "2-1-2 continuation", "up", True)
    assert score == 50.0
    assert n == 0


def test_small_sample_shrinks_toward_neutral():
    # A perfect 2-for-2 record shouldn't score near 100; shrinkage pulls it
    # most of the way back to 50 because n is far below the shrink constant.
    stats = make_stats(n=2, win_rate=1.0, avg_r=2.0)
    score, n = backtest_edge(stats, "day", "2-1-2 continuation", "up", True)
    assert n == 2
    assert 50.0 < score < 60.0


def test_large_sample_reaches_full_strength():
    stats = make_stats(n=200, win_rate=0.65, avg_r=1.0)
    score, n = backtest_edge(stats, "day", "2-1-2 continuation", "up", True)
    expected_base = 100 * (0.7 * 0.65 + 0.3 * ((0.5 + 1) / 2))
    assert abs(score - expected_base) < 0.5


def test_falls_back_to_generic_bucket_when_ftfc_specific_missing():
    stats = make_stats(n=50, win_rate=0.6, avg_r=0.5)  # no ftfc suffix -> generic key
    score, n = backtest_edge(stats, "day", "2-1-2 continuation", "up", True)
    assert n == 50


def test_composite_score_adds_ftfc_and_volume_bonus():
    stats = make_stats(n=200, win_rate=0.5, avg_r=0.0, ftfc_aligned=True)
    base_score, _ = backtest_edge(stats, "day", "2-1-2 continuation", "up", True)

    no_bonus, _ = composite_score(stats, "day", "2-1-2 continuation", "up", 0, 2, volume_ratio=1.0)
    with_ftfc, _ = composite_score(stats, "day", "2-1-2 continuation", "up", 2, 2, volume_ratio=1.0)
    with_both, _ = composite_score(stats, "day", "2-1-2 continuation", "up", 2, 2, volume_ratio=2.0)

    assert no_bonus == round(base_score, 10) or abs(no_bonus - base_score) < 1e-6
    assert with_ftfc > no_bonus
    assert with_both > with_ftfc


def test_composite_score_clipped_to_100():
    stats = make_stats(n=1000, win_rate=1.0, avg_r=5.0, ftfc_aligned=True)
    score, _ = composite_score(stats, "day", "2-1-2 continuation", "up", 2, 2, volume_ratio=3.0)
    assert score == 100.0
