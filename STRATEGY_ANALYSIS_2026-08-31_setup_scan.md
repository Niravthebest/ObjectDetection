# 3-Criteria Momentum Scan — 2026-08-31

Scan: **70%+ above 52-week low, ADR proxy (ATR14/Close) ≥ 4.5%, price above EMA8 and EMA21**, run against the S&P 500 + Nasdaq 100 universe (508 tickers, saved scan id `c592f940-0171-4859-8eb8-49d666586e44`).

## Live triggers (as of scan run, market open 2026-08-31)

| Symbol | Price | % Change (day) | % Above 52wk Low | ADR proxy | Historical win rate* |
|---|---|---|---|---|---|
| MU | $956.09 | +2.5% | +737% | 6.3% | 64.7% (17 triggers, +3.82% avg) |
| CRWD | $229.55 | +5.1% | +168% | 5.3% | 61.5% (13 triggers, +2.13% avg) |
| PANW | $382.12 | +2.8% | +174% | 4.8% | 66.7% (6 triggers, +3.55% avg) |
| MDB | $455.14 | +1.9% | +111% | 4.9% | 50.0% (10 triggers, +2.49% avg) |
| TEAM | $194.75 | +2.3% | +248% | 4.7% | 100% (2 triggers — too thin to trust) |
| WDAY | $197.65 | -3.5% | +79% | 5.3% | 0% (1 trigger, lost) |
| HPQ | $30.03 | -1.6% | +71% | 4.8% | No prior history — first-ever trigger |

*Historical win rate = full 2-year backtest (2024-09-03 to 2026-08-27) across the entire 508-ticker universe: win = closing price reaches entry+5% within 10 trading days. See prior session backtest for full universe results (632 total triggers, 341 wins, 53.96% aggregate win rate).

## Change vs. prior scan run (earlier same day)
**ANET dropped off the list; MU newly appeared.** Everything else (MDB, PANW, CRWD, WDAY, TEAM, HPQ) persisted.

## Read for tomorrow
- **Strongest setups:** MU (new trigger, strongest historical sample + win rate), CRWD (still up, robust sample size), PANW (strong win rate, smaller sample).
- **Weak/no track record despite triggering:** WDAY (red today, single historical trigger was a loser), HPQ (never triggered before this backtest window).
- **Too little data to trust:** TEAM (2 historical triggers only, despite the eye-catching 100%).

## Caveats
- ADR proxy = ATR(14)/Close, not a literal Average Daily Range calculation — no native ADR% filter exists in the scanner.
- Win rate only measures whether +5% was hit within 10 trading days; it does not account for drawdown before that point or model a stop-loss.
- This is analysis only — no orders placed.
