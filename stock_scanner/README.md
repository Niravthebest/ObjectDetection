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

---

# Scanner 2: near the 52-week high (port of a Chartink scan)

`near_52wh_scanner.py` applies the "Stock near 5% of 52 week high" Chartink scan to US stocks.
A stock passes when all of these are true:

- The close is within 5% of the 52-week high.
- The close is above where it was 65 days ago.
- The close is above $100.
- The 20-day EMA is above the 50-day EMA, and the close is above the 21-day EMA.
- Volume is above 50,000 shares.
- Return on capital employed (ROCE) is above 15% and return on equity (ROE) is at least 15%, using yearly figures.

Fundamentals are point-in-time: each fiscal year's numbers are used only from 90 days after the
year ends, roughly when the annual report is public. Yahoo provides about 4 years of annual data,
so the full scan, including ROCE and ROE, can only be tested from 2022-03-31 onward.

```bash
python near_52wh_scanner.py scan                  # today's list (add --fresh-only for new entries only)
python near_52wh_scanner.py backtest --min-price 100
```

## Results (S&P 500 + Nasdaq above $1B, 1,315 stocks)
A fresh entry is the first day a stock passes after not passing the day before.

| Test (2022-03-31 to 2026-11-23, $100 floor) | Signals | Win rate, 5 days | Median, 5 days | Win rate, 20 days | Median, 20 days |
|---|---|---|---|---|---|
| Baseline (every stock, every day) | 1,407,907 | 51.7% | +0.19% | 53.0% | +0.70% |
| Price and trend conditions only, fresh entry | 14,364 | 51.1% | +0.11% | 51.7% | +0.35% |
| Full scan, every day it passes | 30,406 | 52.5% | +0.20% | 53.6% | +0.69% |
| Full scan, fresh entry | 4,014 | 52.6% | +0.23% | 52.6% | +0.58% |

The full scan beats the baseline by only about 1 percentage point, and its average returns are
lower than the baseline's. The price and trend conditions alone lost to buying at random in
2021 and 2022, and roughly matched it from 2023 to 2026. With a $10 price floor the results are
nearly the same. Year-by-year results and per-stock scores are saved in `results/near52wh_p*_*.csv`.

---

# Portfolio simulation: trade the stock with the best expectancy each day

`portfolio_sim.py --signals near52|sma14` simulates trading the signals. Each day it buys the
triggered stock with the highest **expectancy** (that stock's average 5-day return on its past
signals, or win rate × average win − loss rate × average loss), holds it 5 days, and charges a
0.10% round-trip cost.

Expectancy is **walk-forward**: on each day it uses only trades that had already exited, so the
test never uses future results. For comparison, the script also runs the same strategy with
expectancy from the full backtest (which does see the future), a random pick among that day's
triggered stocks, and buying and holding SPY.

**Results:** with honest, walk-forward expectancy, picking the best-expectancy stock did no better
than picking a triggered stock at random, and often did worse. The version that sees the future
looks spectacular, which shows how much in-sample rankings overstate results. On the 14-day
average signal, the results depend mostly on a few biotech stocks that jumped 40–300% in 5 days.
See `results/sim_*` for full tables and trade lists.
