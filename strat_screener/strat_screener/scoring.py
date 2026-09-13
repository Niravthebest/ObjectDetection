"""Turn backtest stats for a setup into a single 0-100 score.

score = shrunk_backtest_edge + ftfc_bonus + volume_bonus

- shrunk_backtest_edge: derived from the historical win rate and average
  R-multiple for this (timeframe, pattern, direction[, ftfc]) bucket,
  pulled toward a neutral 50 when the sample size is small so a pattern
  seen twice can't claim a 100 just because both trades won.
- ftfc_bonus: up to 15 points, proportional to how many of the *other* two
  timeframes currently agree with the trade direction.
- volume_bonus: 5 points if the triggering bar's volume is at least 1.5x
  its trailing 20-bar average (participation confirming the move).
"""
from typing import Optional, Tuple

SHRINK_K = 30  # samples at which the backtest edge is taken at ~full weight
FTFC_MAX_BONUS = 15.0
VOLUME_BONUS = 5.0
VOLUME_RATIO_THRESHOLD = 1.5


def _shrink_to_neutral(base: float, n: int, k: int = SHRINK_K) -> float:
    weight = min(1.0, n / k) if k else 1.0
    return 50.0 + (base - 50.0) * weight


def find_bucket(stats: dict, timeframe: str, pattern: str, direction: str, ftfc_aligned: bool) -> Optional[dict]:
    """Look up the raw backtest bucket {n, win_rate, avg_r, median_r, avg_return_pct}
    for a setup, preferring the FTFC-specific bucket and falling back to the
    FTFC-agnostic one. Returns None if the pattern has no history at all.
    """
    specific_key = f"{timeframe}|{pattern}|{direction}|ftfc={ftfc_aligned}"
    generic_key = f"{timeframe}|{pattern}|{direction}"
    return stats.get(specific_key) or stats.get(generic_key)


def backtest_edge(stats: dict, timeframe: str, pattern: str, direction: str, ftfc_aligned: bool) -> Tuple[float, int]:
    """Returns (shrunk_score_0_100, sample_size)."""
    entry = find_bucket(stats, timeframe, pattern, direction, ftfc_aligned)
    if not entry:
        return 50.0, 0

    win_rate = entry["win_rate"]
    avg_r = entry["avg_r"]
    norm_r = max(-1.0, min(1.0, avg_r / 2.0))  # +2R -> 1.0, -2R -> -1.0
    base = 100.0 * (0.7 * win_rate + 0.3 * ((norm_r + 1.0) / 2.0))
    return _shrink_to_neutral(base, entry["n"]), entry["n"]


def composite_score(
    stats: dict,
    timeframe: str,
    pattern: str,
    direction: str,
    ftfc_aligned_count: int,
    ftfc_total: int,
    volume_ratio: Optional[float],
) -> Tuple[float, int]:
    """Returns (score_0_100, backtest_sample_size)."""
    fully_aligned = ftfc_total > 0 and ftfc_aligned_count == ftfc_total
    base, n = backtest_edge(stats, timeframe, pattern, direction, fully_aligned)

    ftfc_bonus = FTFC_MAX_BONUS * (ftfc_aligned_count / ftfc_total) if ftfc_total else 0.0
    volume_bonus = VOLUME_BONUS if (volume_ratio is not None and volume_ratio >= VOLUME_RATIO_THRESHOLD) else 0.0

    score = base + ftfc_bonus + volume_bonus
    return max(0.0, min(100.0, score)), n
