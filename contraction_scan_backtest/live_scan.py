"""Run the Contraction Scan against the most recent bar of data to find
current picks -- stocks whose entry condition is true as of the latest
complete trading day. These would be entered at the NEXT trading day's
open, per the same entry-timing convention used in the backtest.

Usage:
    python3 live_scan.py --data-dir data --out results/live_picks.csv
"""
import argparse
import os

import pandas as pd

from backtest import load_universe, MIN_HISTORY_BARS
from strategy import compute_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/live_picks.csv")
    ap.add_argument("--as-of", default=None,
                     help="YYYY-MM-DD; treat this date as the signal day, ignoring any later bars "
                          "(e.g. to skip a known-bad or incomplete final day). Defaults to the latest "
                          "date present in the data.")
    args = ap.parse_args()

    universe = load_universe(args.data_dir)
    as_of_date = pd.Timestamp(args.as_of) if args.as_of else None
    if as_of_date is not None:
        universe = {sym: df[df["date"] <= as_of_date].reset_index(drop=True)
                    for sym, df in universe.items()}
        universe = {sym: df for sym, df in universe.items() if len(df)}
    latest_date = max(g["date"].max() for g in universe.values())
    print(f"Scanning as of: {latest_date.date()}" + (" (later bars ignored)" if as_of_date is not None else ""))

    picks = []
    skipped = 0
    stale = []
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        if df["date"].iloc[-1] != latest_date:
            stale.append(symbol)  # this symbol's data doesn't reach the latest date
            continue
        sig_df = compute_signals(df)
        last = sig_df.iloc[-1]
        if bool(last["entry_signal"]):
            picks.append({
                "symbol": symbol,
                "signal_date": last["date"],
                "close": last["close"],
                "sma20": last["sma20"],
                "sma50": last["sma50"],
                "sma200": last["sma200"],
                "natr50": last["natr50"],
                "pgo20": last["pgo20"],
                "pgo50": last["pgo50"],
                "rsi7_prior_day": sig_df["rsi7"].iloc[-2],
                "advol30_millions": last["advol30"],
                "atr1": last["atr1"],
                "atr20": last["atr20"],
            })

    picks_df = pd.DataFrame(picks).sort_values("natr50", ascending=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    picks_df.to_csv(args.out, index=False)

    print(f"Universe: {len(universe)} symbols ({skipped} skipped for insufficient history, "
          f"{len(stale)} stale/not reaching latest date)")
    print(f"Picks as of {latest_date.date()}: {len(picks_df)}")
    if len(picks_df):
        print(picks_df[["symbol", "close", "natr50", "pgo20", "pgo50", "rsi7_prior_day"]].to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
