"""Backtest for the Biotech Capitulation Bounce strategy.

Same mechanics as contraction_scan_backtest/backtest.py: enter at the
next bar's open after a signal day, hold a fixed number of trading
days, exit at that bar's close. Daily returns across all open
positions are used to build a proper mark-to-market portfolio equity
curve (never naive sequential compounding -- many positions overlap).

Usage:
    python3 backtest.py --data-dir data --out-dir results --hold-days 10
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from strategy import compute_signals

MIN_HISTORY_BARS = 260


def load_universe(data_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.drop_duplicates(subset=["symbol", "date"])
    all_df = all_df.sort_values(["symbol", "date"])
    return {sym: g.reset_index(drop=True) for sym, g in all_df.groupby("symbol")}


def simulate_trades(df: pd.DataFrame, symbol: str, test_start: pd.Timestamp,
                     hold_days: int) -> tuple:
    trades = []
    daily_returns = []
    in_position_until = -1
    n = len(df)
    for t in range(n):
        if t <= in_position_until:
            continue
        if not df["entry_signal"].iloc[t]:
            continue
        if df["date"].iloc[t] < test_start:
            continue
        entry_idx = t + 1
        if entry_idx >= n:
            continue
        entry_price = df["open"].iloc[entry_idx]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        exit_idx = min(entry_idx + hold_days - 1, n - 1)
        exit_price = df["close"].iloc[exit_idx]
        if pd.isna(exit_price):
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
        prev_price = entry_price
        for i in range(entry_idx, exit_idx + 1):
            price = df["close"].iloc[i]
            if pd.isna(price) or prev_price <= 0:
                break
            daily_returns.append((df["date"].iloc[i], price / prev_price - 1.0))
            prev_price = price
        in_position_until = exit_idx
    return trades, daily_returns


def build_portfolio_equity_curve(daily_returns: list) -> pd.Series:
    if not daily_returns:
        return pd.Series(dtype=float)
    df = pd.DataFrame(daily_returns, columns=["date", "ret"])
    by_day = df.groupby("date")["ret"].mean()
    by_day = by_day.sort_index()
    equity = (1 + by_day).cumprod()
    return equity


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd.min()


def summarize(trades_df: pd.DataFrame, daily_returns: list) -> tuple:
    n = len(trades_df)
    if n == 0:
        return {"total_trades": 0}, pd.Series(dtype=float)
    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] <= 0]
    gross_win = wins["return"].sum()
    gross_loss = -losses["return"].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    equity = build_portfolio_equity_curve(daily_returns)
    years_span = max(len(equity) / 252.0, 1e-6)
    cagr = equity.iloc[-1] ** (1 / years_span) - 1.0 if len(equity) else np.nan

    daily_ret_series = equity.pct_change().dropna()
    sharpe = (
        daily_ret_series.mean() / daily_ret_series.std() * np.sqrt(252)
        if daily_ret_series.std() and daily_ret_series.std() > 0
        else np.nan
    )

    return {
        "total_trades": n,
        "unique_symbols": trades_df["symbol"].nunique(),
        "win_rate": len(wins) / n,
        "avg_return_per_trade": trades_df["return"].mean(),
        "median_return_per_trade": trades_df["return"].median(),
        "avg_win": wins["return"].mean() if len(wins) else np.nan,
        "avg_loss": losses["return"].mean() if len(losses) else np.nan,
        "profit_factor": profit_factor,
        "avg_concurrent_positions": len(daily_returns) / max(len(equity), 1),
        "portfolio_total_return": equity.iloc[-1] - 1.0 if len(equity) else np.nan,
        "portfolio_cagr": cagr,
        "portfolio_max_drawdown": max_drawdown(equity),
        "portfolio_sharpe_annualized": sharpe,
    }, equity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--hold-days", type=int, default=10)
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

    all_trades = []
    all_daily_returns = []
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        sig_df = compute_signals(df)
        trades, daily_returns = simulate_trades(sig_df, symbol, test_start, args.hold_days)
        all_trades.extend(trades)
        all_daily_returns.extend(daily_returns)

    trades_df = pd.DataFrame(all_trades)
    trades_path = os.path.join(args.out_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)

    summary, equity = summarize(trades_df, all_daily_returns) if len(trades_df) else ({"total_trades": 0}, pd.Series(dtype=float))
    summary["universe_size"] = len(universe)
    summary["skipped_short_history"] = skipped
    summary["test_start"] = str(test_start.date())
    summary["hold_days"] = args.hold_days

    print(f"Universe: {len(universe)} symbols ({skipped} skipped for insufficient history)")
    print(f"Test window: {test_start.date()} .. {latest_date.date()}  |  hold: {args.hold_days} trading days")
    print(f"Trades: {len(trades_df)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    import json
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if len(equity):
        equity.rename("equity").to_csv(os.path.join(args.out_dir, "equity_curve.csv"), index_label="date")

    print(f"\nWrote {trades_path}")
    print(f"Wrote {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
