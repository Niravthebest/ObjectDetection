# Biotech Capitulation Bounce

A new strategy built by scanning daily price/volume data for 210 biotech
stocks (Morningstar industry code 20610010, market cap > $500M) over
2023-09-01 through 2026-11-25, to find which technical conditions
actually predict a subsequent large price move in this sector.

## Method

1. **Universe**: pulled via Robinhood's scanner (`Sector = Healthcare` +
   `morningstarIndustryCode = 20610010`, excluding pharma/devices/health
   plans), filtered to market cap > $500M for liquidity, giving 210
   names (96 above $2B, 114 between $500M-$2B).
2. **Condition sweep** (`analyze_conditions.py`): computed ~25 candidate
   technical conditions per symbol-day (volatility contraction, RSI
   extremes, proximity to 52-week high/low, volume spikes, gaps,
   momentum extremes, moving-average trend state, Bollinger squeeze),
   then measured, for each condition and each forward horizon
   (3/5/10/20 trading days), the lift in probability of a "big move"
   (|forward return| >= 10/15/20/30%) versus the unconditional baseline
   for this universe.
3. **Finding**: the single strongest and most consistent predictor of
   an upcoming big move, across every horizon, was **prior 20-day
   momentum extremes** (a stock already up or down >30% in the last
   month is much likelier to keep moving sharply) -- a volatility-
   clustering effect, not a directional signal by itself. Volume spikes
   (2-5x average) and large overnight gaps showed the same pattern.
   On pure directional tilt, the clearest (if more modest) edge was
   capitulation-style: oversold + near the 52-week low + a volume
   pickup showed an elevated chance of a bounce (up move) over the next
   10 days versus baseline.
4. **Strategy** (`strategy.py`): built from that capitulation-bounce
   finding --
   - RSI(14) < 35 (oversold)
   - price within 20% of its trailing 52-week low
   - volume >= 1.5x its trailing 20-day average (confirms participation)
   - not already up >10% in the last 5 days (avoid chasing a bounce
     that's already underway)
5. **Backtest** (`backtest.py`): same mechanics as the generic
   contraction-scan project -- enter at next bar's open, hold 10
   trading days, exit at close, portfolio equity built from daily
   mark-to-market across all open positions (never naive sequential
   compounding).

## Results (2023-11-25 to 2026-11-25, 10-day hold)

| Metric | Value |
|---|---|
| Total trades | 687 (179 unique symbols) |
| Win rate | 55.6% |
| Avg return/trade | +3.5% |
| Median return/trade | +1.6% |
| Avg win / avg loss | +14.2% / -9.8% |
| Profit factor | 1.81 |
| Avg concurrent positions | 9.4 |
| Portfolio CAGR | 114.8% |
| Max drawdown | **-54.8%** |
| Sharpe (annualized) | 1.80 |

## Important caveats -- read before trading this

- **This is a high-volatility, concentrated strategy.** With only ~9
  average concurrent positions (vs ~125 in the generic 1000-stock
  contraction scan), a single name's move swings the whole portfolio
  much more. The -54.8% max drawdown is real and would be very hard to
  sit through.
- **Signal clustering on macro events inflates the trade count.**
  86 of the 687 trades (12.5%) entered within four days in early April
  2025 (04-01, 04-02, 04-07, 04-08) -- almost certainly the broad
  market selloff that week, not 80+ independent biotech-specific
  catalysts. Treat the realized edge as resting on meaningfully fewer
  truly independent bets than "687 trades" suggests.
- **The correlation-sweep z-scores in `results/condition_scan.csv`
  overstate statistical confidence.** They're computed on overlapping
  daily windows (adjacent days share most of their forward-return
  window), which is not an independent sample, so treat those z-scores
  as a ranking heuristic, not a real significance test.
- **Returns are driven by fat right tails typical of biotech catalysts**
  (trial readouts, FDA decisions can double or halve a stock in days).
  The top 5 winning trades account for ~24% of the total return sum --
  not fully tail-dependent, but concentrated enough to flag.
- **No slippage/spread modeling.** Many of these are lower-liquidity
  small/mid-caps; buying right after a volume-spike capitulation day
  (exactly this strategy's entry trigger) is when spreads and slippage
  are often worst.
- **No catalyst-calendar awareness.** This is pure price/volume
  technicals -- it doesn't know an earnings date, FDA PDUFA date, or
  trial readout is imminent, which in biotech specifically is often the
  actual driver of the next move.

## Files

- `universe.txt` -- the 210 biotech tickers
- `indicators.py` -- technical indicator primitives
- `analyze_conditions.py` -- the condition-discovery sweep
- `strategy.py` -- the derived entry rule
- `backtest.py` -- the backtest harness
- `results/condition_scan.csv` -- full sweep output (every condition x
  horizon x threshold combination)
- `results/trades.csv`, `results/summary.json`, `results/equity_curve.csv`
  -- backtest output
