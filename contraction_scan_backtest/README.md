# Contraction Scan Backtest

Backtests this scan:

```
exch(nyse,nasdaq) and type(stock) and advol(30) > 3
and ! (sma(20) < sma(50))@{0..20}
and ! (price < sma(50) and sma(50) trend_dn 20)
and (price > sma(100) or price > sma(200))
and (sma(200) trend_up 60)
and natr(50) > 1.5
and price > sma(50) - arange(0)
and (
      (atr(1) < atr(5) * 0.5 or atr(1) < atr(20) * 0.5 or atr(1) < atr(50) * 0.5)
      or
      (price < high[1] and price > low[1]
       and (atr(1) < atr(5) * 0.75 or atr(1) < atr(20) * 0.75 or atr(1) < atr(50) * 0.75))
    )
and (pgo(50) < 2.5 or pgo(20) < 2.5)
and (rsi(7) < 60)@1
```

## Universe

The scan syntax targets all NYSE/NASDAQ common stocks. Robinhood's own
scanner tools are live-only (they evaluate filters against current market
data, they can't replay history), and there's no bulk historical-data
endpoint for the full ~8,000-ticker NYSE/NASDAQ universe — `get_equity_historicals`
takes up to 10 symbols per call, so a literal "all tickers" backtest would
need many thousands of calls.

As a practical stand-in, the universe here is the **~1,000 most liquid
NYSE/NASDAQ stocks** (avg volume(30) > 500k shares, price > $5, market cap
down to roughly $6.8B), built via Robinhood's own scanner in market-cap
slices. This approximates the scan's own `advol(30) > 3` liquidity gate —
most micro/small caps below that liquidity bar wouldn't clear the scan's
technical filters anyway — but it does mean small- and micro-cap names are
excluded here even where they'd pass the literal scan. See `universe.txt`.

## Data

Daily OHLCV, split-adjusted, from 2022-06-01 through today, pulled via
`get_equity_historicals`. The pre-2023 data is warm-up only (200-day SMA +
`trend_up 60` need ~260 bars of history before a signal is valid); actual
trade entries are restricted to the last 3 years.

## Interpretation of the syntax

This is a proprietary screener DSL (StockFetcher/TrendSpider-style); a few
constructs needed an explicit choice since there's no single universal
spec. All are implemented in `strategy.py` / `indicators.py`:

- **`@{0..20}`** — "true at some point in the last 21 bars (today + 20 back)".
  `! (sma(20) < sma(50))@{0..20}` is read as: sma(20) has NOT been below
  sma(50) at any offset 0..20 — i.e. no bearish 20/50 cross-under in the
  last 21 bars.
- **`@1`** — "evaluated 1 bar ago" (yesterday), not today. `(rsi(7) < 60)@1`
  checks *yesterday's* RSI(7), not today's.
- **`trend_up N` / `trend_dn N`** — value today vs. value N bars ago
  (`sma(200) trend_up 60` = sma(200) now > sma(200) 60 bars ago).
- **`arange(0)`** — "average range" with length 0 taken as just today's
  bar range (high − low), since there's nothing to average over 0 bars.
- **`atr(1)`** — a 1-period ATR is just today's true range (no smoothing).
- **`pgo(n)`** — Pretty Good Oscillator, `(close − SMA(n)) / ATR(n)`.
- **`advol(30) > 3`** — assumed to be 30-day average *dollar* volume in
  millions (i.e. > $3M/day). This is a very loose floor by design — it's a
  base liquidity gate, not the strategy's real filter.
- **`price < high[1] and price > low[1]`** — an inside day relative to
  yesterday's range.

If any of these differ from the platform you originally wrote this scan
for, the fix is localized: each clause maps to one named `cond_*` variable
in `strategy.py`.

## Exit rule (not specified by the scan)

The scan only defines entries. As agreed, this backtest uses a **fixed
holding period**: enter at the next bar's open after a signal, exit at the
close `N` trading days later (default `N=10`, configurable via
`--hold-days`). No stop-loss or profit target. Only one open position per
symbol at a time — a new signal while already in a trade is ignored.

## Running it

```
python3 backtest.py --data-dir data --out-dir results --hold-days 10
```

Outputs `results/trades.csv` (every simulated trade) and
`results/summary.json` (win rate, average return, profit factor, a
sequential equal-weight equity curve's CAGR/max-drawdown, etc).

## Known limitations

- Survivorship bias: the universe is today's liquid-stock list, not the
  historical constituents as of each date, and delisted names never
  Robinhood-tradable are absent.
- No slippage, commissions, or borrow costs.
- No portfolio-level position sizing or capital constraints — the equity
  curve treats every trade as sequential and equal-weight, not concurrent.
- ATR/PGO use Wilder smoothing, a common but not the only convention.
