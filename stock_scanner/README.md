# SMA-14 Reclaim Scanner (S&P 500 + Nasdaq)

**Signal (daily closes):** yesterday's close was below the 14-day SMA, today's close is above
the 14-day SMA, and today's close is also above the 30-day SMA.

**Backtest:** buy at the close on the signal day, sell at the close 5 trading days later.
A trade counts as a win if that 5-day return is above 0. Prices are split- and dividend-adjusted.

**Score:** the Wilson 95% lower bound of each stock's win rate. A high win rate built on only
a few signals gets marked down, so 3 wins from 3 signals doesn't outrank 40 wins from 55.
**Edge:** the signal's win rate minus the stock's win rate when bought on any day and held 5 days.

## Daily use
```bash
pip install -r requirements.txt
python sma_cross_scanner.py backtest              # refresh scores (weekly or monthly is enough)
python sma_cross_scanner.py scan --min-signals 20 # run every day after 4:15pm ET
```
Useful flags: `--universe both|sp500|nasdaq|nasdaq100|sp500+nasdaq100`, `--list-source auto|slickcharts|github`, `--nasdaq-min-cap 1e9`, `--tickers AAPL,MSFT`,
`--min-score 0.5`, `--top 30`, `--years 5`, `--asof 2026-11-20` (replay a past day).

Output is written to `results/`:
- `backtest_scores.csv`: one row per stock with signals, wins, win rate, score, average and median 5-day return, profit factor and edge.
- `backtest_trades.csv`: every historical signal (this file isn't committed).
- `scan_YYYY-MM-DD.csv`: today's hits, joined with each stock's backtest score.

To run it automatically, schedule it with cron, for example `20 16 * * 1-5 cd /path/stock_scanner && python sma_cross_scanner.py scan`.

## Results (5 years to 2026-11-23, 1,315 stocks, 62,747 signals)
| | Win rate (5 days) | Average 5-day return |
|---|---|---|
| All signals | 50.3% | +0.26% |
| Any day (baseline) | 51.3% | n/a |

**Caveat:** across all stocks combined, this signal performs about the same as a coin flip
and slightly worse than buying on a random day. Per-stock scores barely hold up over time.
I ranked stocks by score using signals before Nov 2024, then checked their signals from Nov 2024 onward:

| Score group (pre-Nov 2024 signals) | Win rate, Nov 2024 onward |
|---|---|
| Bottom 20% | 50.6% |
| Top 20% | 51.3% |

Treat a high score as a weak tiebreaker, not as evidence of an edge. Also note that the stock
universe is today's index members, so the backtest has survivorship bias (stocks that dropped
out of the indexes aren't tested).

## Data sources
- Prices come from the Yahoo Finance chart API.
- Index membership comes from [slickcharts.com](https://www.slickcharts.com) (`/sp500`, `/nasdaq100`) by default.
  Slickcharts sits behind a Cloudflare bot check. From a home connection it may load; from cloud
  servers it is usually blocked. When it can't be read, the script says so and falls back to GitHub:
  - S&P 500: datasets/s-and-p-500-companies (the exact index list).
  - Nasdaq-100: approximated as the 100 largest non-financial Nasdaq-listed companies (from
    rreichel3/US-Stock-Symbols), which is close to the real index but not identical.
- `--list-source slickcharts` uses slickcharts only and fails if it's blocked, instead of falling back.
- `--universe nasdaq` / `both` (all Nasdaq-listed stocks above `--nasdaq-min-cap`) always use the
  rreichel3 list, since slickcharts only publishes the Nasdaq-100.
