"""Backtest runner for the Contraction Scan strategy.

Loads per-symbol daily OHLCV data (CSV files with columns
symbol,date,open,high,low,close,volume), computes the scan's entry
condition on each bar, and simulates a fixed-holding-period trade whenever
the signal fires. See README.md for interpretation notes and how to run.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from strategy import compute_signals

MIN_HISTORY_BARS = 260  # 200sma + trend_up(60) warm-up


def load_universe(data_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.csv")))
    if not files:
        raise SystemExit(f"No chunk_*.csv files found under {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.drop_duplicates(subset=["symbol", "date"])
    all_df = all_df.sort_values(["symbol", "date"])
    return {sym: g.reset_index(drop=True) for sym, g in all_df.groupby("symbol")}


def simulate_trades(df: pd.DataFrame, symbol: str, test_start: pd.Timestamp,
                     hold_days: int) -> list:
    trades = []
    in_position_until = -1
    n = len(df)
    for t in range(len(df)):
        if t <= in_position_until:
            continue
        if not df["entry_signal"].iloc[t]:
            continue
        if df["date"].iloc[t] < test_start:
            continue
        entry_idx = t + 1
        exit_idx = entry_idx + hold_days - 1
        if exit_idx >= n:
            continue  # not enough future bars to complete the trade
        entry_price = df["open"].iloc[entry_idx]
        exit_price = df["close"].iloc[exit_idx]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue
        ret = exit_price / entry_price - 1.0
        trades.append({
            "symbol": symbol,
            "signal_date": df["date"].iloc[t],
            "entry_date": df["date"].iloc[entry_idx],
            "exit_date": df["date"].iloc[exit_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
        })
        in_position_until = exit_idx
    return trades


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def summarize(trades_df: pd.DataFrame, hold_days: int) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"total_trades": 0}
    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] <= 0]
    gross_win = wins["return"].sum()
    gross_loss = -losses["return"].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    ordered = trades_df.sort_values("entry_date")
    equity = (1.0 + ordered["return"]).cumprod()

    years_span = max(
        (ordered["exit_date"].max() - ordered["entry_date"].min()).days / 365.25,
        1e-6,
    )
    cagr = equity.iloc[-1] ** (1 / years_span) - 1.0

    daily_like_return_std = ordered["return"].std()
    sharpe_like = (
        ordered["return"].mean() / daily_like_return_std * np.sqrt(252 / hold_days)
        if daily_like_return_std and daily_like_return_std > 0
        else np.nan
    )

    return {
        "total_trades": n,
        "unique_symbols": trades_df["symbol"].nunique(),
        "win_rate": len(wins) / n,
        "avg_return": trades_df["return"].mean(),
        "median_return": trades_df["return"].median(),
        "avg_win": wins["return"].mean() if len(wins) else np.nan,
        "avg_loss": losses["return"].mean() if len(losses) else np.nan,
        "profit_factor": profit_factor,
        "expectancy": trades_df["return"].mean(),
        "compounded_total_return": equity.iloc[-1] - 1.0,
        "cagr_equal_weight_sequential": cagr,
        "max_drawdown_equal_weight_sequential": max_drawdown(equity),
        "sharpe_like_annualized": sharpe_like,
        "date_range_start": trades_df["entry_date"].min(),
        "date_range_end": trades_df["exit_date"].max(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--hold-days", type=int, default=10)
    ap.add_argument("--test-start", default=None,
                     help="YYYY-MM-DD; defaults to 3 years before the latest date in the data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    universe = load_universe(args.data_dir)

    latest_date = max(g["date"].max() for g in universe.values())
    test_start = (
        pd.Timestamp(args.test_start)
        if args.test_start
        else latest_date - pd.DateOffset(years=3)
    )

    all_trades = []
    skipped_short_history = []
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped_short_history.append(symbol)
            continue
        sig_df = compute_signals(df)
        trades = simulate_trades(sig_df, symbol, test_start, args.hold_days)
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    trades_path = os.path.join(args.out_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)

    summary = summarize(trades_df, args.hold_days) if len(trades_df) else {"total_trades": 0}
    summary["universe_size"] = len(universe)
    summary["skipped_short_history"] = len(skipped_short_history)
    summary["test_start"] = str(test_start.date())
    summary["hold_days"] = args.hold_days

    summary_path = os.path.join(args.out_dir, "summary.json")
    pd.Series(summary).to_json(summary_path, indent=2, date_format="iso")

    print(f"Universe: {len(universe)} symbols ({len(skipped_short_history)} skipped for insufficient history)")
    print(f"Test window: {test_start.date()} .. {latest_date.date()}  |  hold: {args.hold_days} trading days")
    print(f"Trades: {len(trades_df)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {trades_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
