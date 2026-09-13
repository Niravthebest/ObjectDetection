# TheStrat Multi-Timeframe Screener

A screener for **TheStrat**, the daily/weekly/monthly candlestick
classification method popularized by Rob Smith. It classifies every bar,
detects TheStrat's combo patterns, checks multi-timeframe continuity, and
scores each currently-actionable setup using win rate / R-multiple stats
measured from that pattern's own history — not a fixed, made-up weighting.

## The strategy logic

**Bar types** — every bar is classified relative to the *immediately
preceding bar on the same timeframe*:

| Type | Rule | Meaning |
|---|---|---|
| `1` inside | high ≤ prev high and low ≥ prev low | consolidation, not tradeable on its own |
| `2u` directional up | high > prev high, low ≥ prev low | buyers took control |
| `2d` directional down | low < prev low, high ≤ prev high | sellers took control |
| `3` outside | high > prev high and low < prev low | both sides taken out, price discovery |

**Combo patterns** detected on the last 2-3 bars (`patterns.py`):

- `2-1-2 continuation` / `2-1-2 reversal` — directional bar, inside bar,
  then a directional bar either continuing or reversing the first.
- `3-1-2` — outside bar, inside bar, directional break.
- `1-2-2 continuation` — inside bar followed by two directional bars the
  same way.
- `2-2 reversal` ("rev strat") — two directional bars in opposite
  directions back to back.

**Full Time Frame Continuity (FTFC)** — for any setup on one timeframe
(say, daily), we check whether the *other two* timeframes (weekly,
monthly) are currently pointed the same direction (green/red close vs.
open). More agreement = higher conviction, per TheStrat's core premise
that price direction is best read across timeframes rather than lagging
indicators.

Reference: [rickyzcarroll/the-strat](https://github.com/rickyzcarroll/the-strat)
(readme summarizing Rob Smith's TheStrat, linked from the doc this was
built from). This screener implements the well-documented public rules of
that method; it does not encode anyone's private/paid indicator logic.

## Scoring model

Each currently-actionable setup gets a 0-100 score:

```
score = shrunk_backtest_edge + ftfc_bonus + volume_bonus
```

- **shrunk_backtest_edge** — the historical win rate and average
  R-multiple for this exact `(timeframe, pattern, direction, ftfc_aligned)`
  combination, computed by `backtest.py` from as much history as you feed
  it. Falls back to the FTFC-agnostic bucket, then to a neutral 50 if the
  pattern has never fired before for that symbol/universe. Small samples
  are *shrunk* toward 50 (see `SHRINK_K` in `scoring.py`) so a pattern that
  won its only two occurrences doesn't score near 100.
- **ftfc_bonus** — up to 15 points, proportional to how many of the other
  two timeframes agree with the trade direction right now.
- **volume_bonus** — 5 points if the triggering bar's volume is ≥1.5x its
  trailing 20-bar average.

Backtest outcome for each historical pattern occurrence: enter at the
close of the bar that completed the pattern, stop at the far side of the
pattern's range, and measure the close `horizon` bars later (same
timeframe units — e.g. 6 trading days for a daily setup, 6 weeks for a
weekly one). Win = positive R-multiple.

**Lookahead safety**: when the backtest checks what the *other* timeframes
were doing at the time of a historical setup, it only looks at the last
bar of that other timeframe that had **fully closed strictly before** the
setup's bar opened — never the bar that was still forming. See
`other_direction_asof` in `backtest.py`.

**Note on live screening**: the *screener* (as opposed to the backtest)
naturally uses the most recent bar in your data, which for weekly/monthly
timeframes is often the still-forming current period — that's intentional
and matches how TheStrat is actually traded (you watch the forming
candle's setup intra-week/intra-month), but it means a weekly/monthly
signal can still change before that period closes.

## Layout

```
strat_screener/
  bars.py          bar type + direction classification
  patterns.py      combo pattern detection
  data_sources.py  pluggable OHLCV providers (yfinance, in-memory)
  backtest.py       historical pattern scan -> win rate / R-multiple stats
  scoring.py        composite 0-100 score from backtest stats + FTFC + volume
  screener.py       ranks current actionable setups across a symbol list
  cli.py            `backtest` and `screen` subcommands
tests/               pytest unit tests for the above
symbols.txt          default symbol universe: S&P 500 + Nasdaq-100, deduped (~516 tickers)
```

`symbols.txt` uses the yfinance/dash convention for dual-class tickers
(`BRK-B`, `BF-B`). If you're sourcing data from Robinhood instead, translate
those two to dot notation (`BRK.B`, `BF.B`) — that's the format its API
expects.

## Usage

```bash
pip install -r requirements.txt

# 1. Backtest a symbol universe to build pattern edge stats (run this
#    periodically, e.g. weekly, to keep the stats current).
python -m strat_screener backtest --symbols-file symbols.txt --period 15y --out strat_stats.json

# 2. Screen for today's actionable setups, ranked by score.
python -m strat_screener screen --symbols-file symbols.txt --stats strat_stats.json --top 25
```

Default data source is Yahoo Finance via `yfinance` (daily bars,
resampled locally to weekly/monthly so all three timeframes come from the
same prices). If you have another OHLCV provider, implement
`PriceDataSource.get_bars(symbol, timeframe)` in `data_sources.py` and
swap it in.

### Using Robinhood data instead of yfinance

If you're running this from a Claude session with the Robinhood MCP tools
connected (`get_equity_historicals`), you don't need yfinance at all:
fetch daily bars for your symbols, convert each symbol's bar list with
`data_sources.robinhood_bars_to_df`, resample to week/month with the same
`RESAMPLE_RULE`/`RESAMPLE_AGG` used internally, and feed the result into
an `InMemoryDataSource`. Pass that data source to `run_backtest` /
`run_screen` exactly as you would `YFinanceDataSource`. This was how the
whole pipeline was validated end-to-end while building it (10 years of
AAPL/MSFT/SPY daily bars -> 3,560 historical pattern occurrences and a
ranked list of live setups).

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

## Limitations / disclaimer

- FTFC and pattern outcomes are computed from price action only; the
  hammer/shooter/broadening-formation/PMG signals described in TheStrat
  community material are not implemented here, only the core bar-type and
  combo-pattern logic plus FTFC.
- The backtest's forward-return simulation is a simplification (fixed
  bar-count exit, no partial fills or slippage) — it's meant to rank
  patterns relative to each other, not to be a production P&L simulator.
- This is an educational screener, not investment advice.
