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
- Instrument: **call option, ~2 months to expiry, ~60 delta, $5,000 of premium per signal.**
  - Expiry: the listed expiration closest to 60 calendar days after the signal date
    (acceptable range 50–75 days).
  - Strike: the call on that expiry whose delta (from the option quote's greeks) is closest
    to 0.60.
  - Size: contracts = floor(5000 / (mid price x 100)); if that is 0, report "one contract
    exceeds $5,000" and give the cost of 1 contract.
  - If the stock has no listed options or no expiry in range, fall back to $5,000 of stock
    and say so in the alert.
- Positions seeded on 2026-11-28 (COIN, RDDT, INTU, APTV, ALB) are stock positions
  (`instrument: "stock"`); no option fields.

## State: `strategy/positions.json`
```
{"positions": [ {"symbol", "signal_date", "status": "pending_entry"|"open"|"exit_due"|"closed",
                 "instrument": "call"|"stock",
                 "entry_date", "entry_price", "expected_exit_date",
                 "option": {"expiration", "strike", "delta_at_signal", "contracts",
                            "signal_mid", "entry_premium", "exit_premium"},
                 "exit_date", "exit_price", "return"} ],
 "last_run": "YYYY-MM-DD"}
```
`entry_price` / `exit_price` are always the stock's opens; for calls, `return` is
exit_premium / entry_premium - 1 (the stock return is also reported).

## Tools (Robinhood)
- Stock prices: `get_equity_historicals` (interval `day`).
- Signals: `run_scan` with the scan_id above.
- Options: `get_option_chains` (expirations), `get_option_instruments` (calls for one expiry),
  `get_option_quotes` (bid/ask/mid and greeks), `get_option_historicals` (daily bars for the
  entry and exit opens). If an option's daily open is unavailable, use that day's mid from
  `get_option_quotes` and note it.

## Daily procedure (weekdays after the close)
1. Get today's daily bar for SPY with `get_equity_historicals` (interval `day`, start a few days
   back). If there is no bar dated today (market holiday), stop without alerting.
2. For each `pending_entry`: set `entry_date` = today, `entry_price` = the stock's open today,
   and for calls `entry_premium` = the contract's open today. Status `open`.
   Event: "Position opened" (with premium and contracts for calls).
3. For each `exit_due` whose exit was flagged on an earlier run: set `exit_date` = today,
   `exit_price` = stock open today, for calls `exit_premium` = contract open today; compute
   `return`; status `closed`. Event: "Position closed" with the return and $ P&L.
4. For each `open`: count daily bars from `entry_date` through today inclusive. If the count
   is >= 30, set status `exit_due`. Event: "Sell SYMBOL (and the call) at tomorrow's open".
   (Step 3 runs before step 4 so a position isn't flagged and closed on the same run.)
5. Run the saved scan. For each result whose symbol has no `pending_entry`, `open` or
   `exit_due` position: pick the contract as described above, add `pending_entry` with
   `signal_date` = today, `instrument` = "call" and the option fields filled from today's quote.
   Event: "New signal: buy N x SYMBOL EXPIRY STRIKE call (~0.60 delta, ~$mid) at tomorrow's open".
6. Set `last_run`, commit and push `positions.json`.
7. If there were events, end with a short summary of them; otherwise end with "No change".

Reminder for every alert: modelled 2026 results for 60-delta, 2-month calls: ~52% wins,
median +7%, several trades lost ~100% of premium; profit depended on a few big winners.
Judge it over many trades, not a handful.
