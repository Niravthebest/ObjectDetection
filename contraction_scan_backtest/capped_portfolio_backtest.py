"""Position-capped portfolio backtest.

The weekend-scan simulations (picking a single "best trailing
expectancy" name per week) showed that concentrating into very few
positions, chosen by a ranking signal, badly underperforms the full
diversified basket -- the ranking signal itself isn't predictive
enough to justify concentration, and a handful of bad trades can wipe
out months of gains when each one is a big fraction of the book.

This script asks a different question: if you cap the portfolio at N
concurrently open positions (so you're never running the full universe
at once), but admit candidates WITHOUT trying to rank or pick
"the best" one -- just first-available, uninformed -- how much
performance do you actually give up versus the full ~124-position
baseline? This isolates the cost of reduced diversification itself
from the cost of a bad selection signal.

Mechanics: walk chronologically through trading days. Each day, close
any position that has reached its hold_days. Then, among symbols whose
entry_signal fired the prior trading day (so today is their entry
day) and that aren't already held, admit candidates in a fixed,
information-free order (alphabetical by symbol) until the position
cap is filled; any signal beyond the cap that day is simply skipped
(a missed trade, not queued for later). Daily portfolio return is the
equal-weighted mean return of whatever is currently open (same
mark-to-market convention as backtest.py).

Usage:
    python3 capped_portfolio_backtest.py --data-dir data --out-dir results \
        --hold-days 10 --position-cap 5
"""
import argparse
import os

import numpy as np
import pandas as pd

from backtest import load_universe, max_drawdown, MIN_HISTORY_BARS
from strategy import compute_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--hold-days", type=int, default=10)
    ap.add_argument("--position-cap", type=int, default=5)
    ap.add_argument("--test-start", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    universe = load_universe(args.data_dir)
    latest_date = max(g["date"].max() for g in universe.values())
    test_start = (
        pd.Timestamp(args.test_start)
        if args.test_start
        else latest_date - pd.DateOffset(years=3)
    )

    # Precompute signals and a per-symbol date->row-index map so we can
    # walk a shared trading calendar and look up each symbol's bar for
    # a given date in O(1).
    sig_frames = {}
    date_index = {}
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        sig = compute_signals(df)
        sig_frames[symbol] = sig
        date_index[symbol] = pd.Series(sig.index.values, index=sig["date"].values)

    all_dates = sorted(set().union(*[set(sig["date"]) for sig in sig_frames.values()]))
    all_dates = [d for d in all_dates if d >= test_start]

    # candidates[date] = list of symbols whose entry_signal fired on `date`
    candidates_by_date = {}
    for symbol, sig in sig_frames.items():
        fired = sig[sig["entry_signal"] & (sig["date"] >= test_start)]
        for d in fired["date"]:
            candidates_by_date.setdefault(d, []).append(symbol)
    for d in candidates_by_date:
        candidates_by_date[d].sort()  # information-free, deterministic order

    open_positions = {}  # symbol -> dict(entry_idx, entry_price, exit_idx_date)
    trades = []
    daily_returns = []
    concurrent_counts = []
    skipped_for_cap = 0

    for i, today in enumerate(all_dates):
        # 1. close anything whose hold is up
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            if today == pos["exit_date"]:
                sig = sig_frames[symbol]
                exit_idx = date_index[symbol].get(np.datetime64(today))
                exit_price = sig["close"].iloc[exit_idx]
                ret = exit_price / pos["entry_price"] - 1.0
                trades.append({
                    "symbol": symbol,
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": today,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "return": ret,
                })
                del open_positions[symbol]

        # 2. daily mark-to-market across whatever's open (using yesterday's->today's close move)
        day_rets = []
        for symbol, pos in open_positions.items():
            sig = sig_frames[symbol]
            idx = date_index[symbol].get(np.datetime64(today))
            if idx is None or idx == 0:
                continue
            prev_close = sig["close"].iloc[idx - 1]
            today_close = sig["close"].iloc[idx]
            if pd.isna(prev_close) or pd.isna(today_close) or prev_close <= 0:
                continue
            day_rets.append(today_close / prev_close - 1.0)
        if day_rets:
            daily_returns.append((today, np.mean(day_rets)))
        concurrent_counts.append(len(open_positions))

        # 3. admit new candidates whose signal was yesterday (entry = today), up to the cap
        if i == 0:
            continue
        yesterday = all_dates[i - 1]
        for symbol in candidates_by_date.get(yesterday, []):
            if len(open_positions) >= args.position_cap:
                skipped_for_cap += 1
                continue
            if symbol in open_positions:
                continue
            sig = sig_frames[symbol]
            idx = date_index[symbol].get(np.datetime64(today))
            if idx is None:
                continue
            entry_price = sig["open"].iloc[idx]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            exit_idx = min(idx + args.hold_days - 1, len(sig) - 1)
            exit_date = sig["date"].iloc[exit_idx]
            open_positions[symbol] = {
                "entry_date": today,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "signal_date": yesterday,
            }

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] <= 0]
    gross_win = wins["return"].sum()
    gross_loss = -losses["return"].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    daily_df = pd.DataFrame(daily_returns, columns=["date", "ret"]).set_index("date")
    equity = (1 + daily_df["ret"]).cumprod()
    years_span = max(len(equity) / 252.0, 1e-6)
    cagr = equity.iloc[-1] ** (1 / years_span) - 1.0 if len(equity) else np.nan
    daily_ret_series = equity.pct_change().dropna()
    sharpe = (
        daily_ret_series.mean() / daily_ret_series.std() * np.sqrt(252)
        if daily_ret_series.std() and daily_ret_series.std() > 0
        else np.nan
    )

    summary = {
        "position_cap": args.position_cap,
        "hold_days": args.hold_days,
        "total_trades": n,
        "unique_symbols": trades_df["symbol"].nunique() if n else 0,
        "win_rate": len(wins) / n if n else np.nan,
        "avg_return_per_trade": trades_df["return"].mean() if n else np.nan,
        "median_return_per_trade": trades_df["return"].median() if n else np.nan,
        "profit_factor": profit_factor,
        "avg_concurrent_positions": float(np.mean(concurrent_counts)),
        "candidates_skipped_for_cap": skipped_for_cap,
        "portfolio_total_return": equity.iloc[-1] - 1.0 if len(equity) else np.nan,
        "portfolio_cagr": cagr,
        "portfolio_max_drawdown": max_drawdown(equity) if len(equity) else np.nan,
        "portfolio_sharpe_annualized": sharpe,
        "test_start": str(test_start.date()),
        "test_end": str(latest_date.date()),
    }

    print(f"Universe: {len(sig_frames)} symbols ({skipped} skipped for insufficient history)")
    print(f"Position cap: {args.position_cap}  |  hold: {args.hold_days} trading days")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    import json
    suffix = f"cap{args.position_cap}_hold{args.hold_days}d"
    trades_path = os.path.join(args.out_dir, f"trades_{suffix}.csv")
    trades_df.to_csv(trades_path, index=False)
    with open(os.path.join(args.out_dir, f"summary_{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    equity.rename("equity").to_csv(os.path.join(args.out_dir, f"equity_{suffix}.csv"), index_label="date")
    print(f"\nWrote {trades_path}")


if __name__ == "__main__":
    main()
