# Scan Universe Expansion — 2026-09-03

Added 7 tickers to the saved 3-criteria screen (scan id `c592f940-0171-4859-8eb8-49d666586e44`), permanently — future `run_scan` calls include them automatically. Universe is now 515 tickers (508 S&P500+Nasdaq100 + these 7).

## Added tickers
RBRK (Rubrik), ASTS (AST SpaceMobile), SAIL (SailPoint), IREN (Iris Energy), GCT (GigaCloud Technology), MNTN (MNTN, Inc.), RKLB (Rocket Lab)

## Historical backtest (2023-09-05 to 2026-09-02, 10-day/+5% win definition)

| Ticker | Triggers (evaluated) | Wins | Win Rate | Avg Fwd Return |
|---|---|---|---|---|
| SAIL | 4 | 4 | 100% | +6.95% (too few to trust) |
| IREN | 37 | 28 | 75.7% | +2.97% |
| RKLB | 43 | 31 | 72.1% | +3.77% |
| RBRK | 16 | 8 | 50.0% | +0.15% |
| GCT | 42 | 18 | 42.9% | -0.63% |
| ASTS | 42 | 17 | 40.5% | -5.19% |
| MNTN | 0 evaluable | — | N/A | Only 1 trigger ever, too recent |

**Aggregate: 184 triggers, 106 wins → 57.6% pooled win rate.**

## Read
- **RKLB and IREN** — best of the batch, large samples (43/37 triggers) with strong win rates and positive average returns. Worth watching for future live triggers.
- **ASTS** — fires often but loses on average (-5.19%); high trigger frequency here is not a good sign.
- **GCT** — triggered live on 2026-09-03 alongside MNTN, but has a large sample with a net-negative average return historically. Treat today's GCT signal with skepticism.
- **RBRK** — coin-flip, near-zero average return.
- **SAIL, MNTN** — too little history to trust yet.

## Data notes
- RBRK, SAIL, MNTN are recent IPOs/relistings; leading placeholder (pre-listing) bars were stripped before computation. All 7 tickers now have 322+ real trading days, enough for the full 252-day rolling-low window.
