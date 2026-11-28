# Crushed S&P 500 reversal — daily tracker

Alert-only. Nothing is ever bought or sold automatically.

## Strategy
- Universe: S&P 500.
- Signal (all on the day's close): RSI(14) < 35; close within 20% of 52-week low;
  volume >= 1.5x 20-day average; 5-day return < +10%; close more than 60% below 52-week high. No stop-loss.
- Robinhood saved scan: "Crushed S&P 500 reversal (>60% down)",
  scan_id `9435407b-05d5-41c7-a52a-b8a849791a6a`.
- Trade: buy at the next trading day's open; hold 30 trading days; sell at the open of
  the 31st trading day. One position per symbol; repeat signals while held are ignored.

## State: `strategy/positions.json`
```
{"positions": [ {"symbol", "signal_date", "status": "pending_entry"|"open"|"exit_due"|"closed",
                 "entry_date", "entry_price", "exit_date", "exit_price", "return"} ],
 "last_run": "YYYY-MM-DD"}
```

## Daily procedure (weekdays after the close)
1. Get today's daily bar for SPY with `get_equity_historicals` (interval `day`, start a few days
   back). If there is no bar dated today (market holiday), stop without alerting.
2. For each `pending_entry`: set `entry_date` = today, `entry_price` = today's open
   (daily bar `open_price`), status `open`. Event: "Position opened".
3. For each `open`: count daily bars from `entry_date` through today inclusive. If the count
   is >= 30, set status `exit_due`. Event: "Sell SYMBOL at tomorrow's open".
4. For each `exit_due` whose exit was flagged on an earlier run: set `exit_date` = today,
   `exit_price` = today's open, `return` = exit_price / entry_price - 1, status `closed`.
   Event: "Position closed" with the return.
   (Do step 4 before step 3 so a position isn't flagged and closed on the same run.)
5. Run the saved scan. For each result whose symbol has no `pending_entry`, `open` or
   `exit_due` position, add `pending_entry` with `signal_date` = today.
   Event: "New signal: buy SYMBOL at tomorrow's open".
6. Set `last_run`, commit and push `positions.json`.
7. If there were events, end with a short summary of them; otherwise end with "No change".

Reminder for every alert: backtest 2017-2025 had ~63% wins, median +7% per trade, worst -49%;
most profit came from rare large winners and crash clusters (March 2020). Judge it over many
trades, not a handful.
