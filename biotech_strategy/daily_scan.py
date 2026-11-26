"""Daily live-scan updater for the Biotech Deep Capitulation Bounce strategy.

Run once per trading day, after fresh OHLCV bars for the universe have
been fetched and saved to a CSV (symbol,date,open,high,low,close,volume).
This script:

1. Cleans the new rows (drops any flat zero-volume bar -- a known
   Robinhood data defect -- as a safety net; interpolated rows should
   already be filtered out by the caller before this runs).
2. Merges them into data/prices_long.csv.
3. Recomputes strategy_deep_capitulation signals on the full history.
4. Advances the live-tracking state file (results/live_state.json),
   mirroring the backtest's own mechanics exactly -- one position per
   symbol at a time, entry at the next trading day's open after a
   signal, exit after 30 trading days held (counted in actual bars,
   never a calendar/business-day guess, since a freshly-opened
   position has no future bars yet to compute a target date from):
     - a pending entry is promoted to an open position once the next
       bar after its signal day is available (entry = that bar's open)
     - an open position closes once its entry index + 29 trading-day
       bars exist in the data (exit = that bar's close), logged to
       results/live_closed_trades.csv
     - any symbol not already open/pending whose entry_signal fires on
       the latest date becomes a new pending entry
5. Prints a plain-text summary and a NOTABLE=yes/no line so the caller
   knows whether to send a push notification.

Usage:
    python3 daily_scan.py --new-data /tmp/new_bars.csv
"""
import argparse
import json
import os

import pandas as pd

from backtest import load_universe, MIN_HISTORY_BARS
from strategy_deep_capitulation import compute_signals

HOLD_DAYS = 30
DATA_PATH = "data/prices_long.csv"
STATE_PATH = "results/live_state.json"
CLOSED_LOG_PATH = "results/live_closed_trades.csv"


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_scan_date": None, "open_positions": {}, "pending_entries": {}}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-data", required=True,
                     help="CSV of freshly fetched bars: symbol,date,open,high,low,close,volume")
    args = ap.parse_args()

    new_df = pd.read_csv(args.new_data)
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df = new_df[~((new_df["volume"] == 0) & (new_df["open"] == new_df["high"])
                       & (new_df["high"] == new_df["low"]) & (new_df["low"] == new_df["close"]))]

    old_df = pd.read_csv(DATA_PATH)
    old_df["date"] = pd.to_datetime(old_df["date"])
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(["symbol", "date"])
    combined.to_csv(DATA_PATH, index=False)

    universe = load_universe("data")
    latest_date = max(g["date"].max() for g in universe.values())

    state = load_state()
    if state["last_scan_date"] == str(latest_date.date()):
        print(f"No new trading day since last scan ({latest_date.date()}). Nothing to do.")
        print("NOTABLE=no")
        return

    sig_frames = {}
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            continue
        sig_frames[symbol] = compute_signals(df).reset_index(drop=True)

    events = {"opened": [], "closed": [], "new_signal": []}

    # 1. promote pending entries whose next-day bar is now available
    still_pending = {}
    for symbol, pend in state["pending_entries"].items():
        sig = sig_frames.get(symbol)
        if sig is None:
            continue
        signal_date = pd.Timestamp(pend["signal_date"])
        after = sig[sig["date"] > signal_date]
        if len(after) == 0:
            still_pending[symbol] = pend  # next bar not out yet
            continue
        entry_row = after.iloc[0]
        entry_price = entry_row["open"]
        if pd.isna(entry_price) or entry_price <= 0:
            still_pending[symbol] = pend
            continue
        state["open_positions"][symbol] = {
            "signal_date": pend["signal_date"],
            "entry_date": str(entry_row["date"].date()),
            "entry_price": float(entry_price),
        }
        events["opened"].append((symbol, str(entry_row["date"].date()), float(entry_price)))
    state["pending_entries"] = still_pending

    # 2. close any open position whose 30th held bar now exists
    still_open = {}
    for symbol, pos in state["open_positions"].items():
        sig = sig_frames.get(symbol)
        if sig is None:
            still_open[symbol] = pos
            continue
        entry_date = pd.Timestamp(pos["entry_date"])
        entry_rows = sig.index[sig["date"] == entry_date]
        if len(entry_rows) == 0:
            still_open[symbol] = pos
            continue
        entry_idx = entry_rows[0]
        exit_idx = entry_idx + HOLD_DAYS - 1
        if exit_idx < len(sig):
            exit_row = sig.iloc[exit_idx]
            exit_price = float(exit_row["close"])
            ret = exit_price / pos["entry_price"] - 1.0
            closed_row = {**pos, "symbol": symbol, "exit_date": str(exit_row["date"].date()),
                          "exit_price": exit_price, "return": ret}
            events["closed"].append(closed_row)
            header_needed = not os.path.exists(CLOSED_LOG_PATH)
            pd.DataFrame([closed_row]).to_csv(CLOSED_LOG_PATH, mode="a", header=header_needed, index=False)
        else:
            still_open[symbol] = pos
    state["open_positions"] = still_open

    # 3. register new signals on the latest date (skip symbols already open/pending)
    for symbol, sig in sig_frames.items():
        if symbol in state["open_positions"] or symbol in state["pending_entries"]:
            continue
        last_row = sig[sig["date"] == latest_date]
        if len(last_row) == 0:
            continue
        if bool(last_row.iloc[0]["entry_signal"]):
            state["pending_entries"][symbol] = {"signal_date": str(latest_date.date())}
            events["new_signal"].append(symbol)

    state["last_scan_date"] = str(latest_date.date())
    save_state(state)

    print(f"=== Daily scan: {latest_date.date()} ===")
    print(f"Opened: {events['opened']}")
    closed_fmt = [(c["symbol"], f"{c['return']:+.1%}") for c in events["closed"]]
    print(f"Closed: {closed_fmt}")
    print(f"New signal (pending entry at next open): {events['new_signal']}")
    print(f"Currently open: {list(state['open_positions'].keys())}")
    print(f"Currently pending: {list(state['pending_entries'].keys())}")

    notable = bool(events["opened"] or events["closed"] or events["new_signal"])
    print(f"\nNOTABLE={'yes' if notable else 'no'}")


if __name__ == "__main__":
    main()
