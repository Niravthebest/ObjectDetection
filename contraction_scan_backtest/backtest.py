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
                     hold_days: int) -> tuple:
    """Returns (trades, daily_returns) where daily_returns is a list of
    (date, day_return) covering every day each trade was held -- used to
    build a proper mark-to-market portfolio equity curve, since with
    thousands of trades many positions overlap in time."""
    trades = []
    daily_returns = []
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
        # day-by-day mark-to-market return while this trade is held
        prev_price = entry_price
        for i in range(entry_idx, exit_idx + 1):
            price = df["close"].iloc[i]
            if pd.isna(price) or prev_price <= 0:
                break
            daily_returns.append((df["date"].iloc[i], price / prev_price - 1.0))
            prev_price = price
        in_position_until = exit_idx
    return trades, daily_returns


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def build_portfolio_equity_curve(daily_returns: list) -> pd.Series:
    """Equal-weight, mark-to-market across every position open on a given
    day (positions overlap constantly with 1000 tickers, so this -- not
    sequential per-trade compounding -- is the correct way to build an
    equity curve). Days with no open positions contribute a 0% return."""
    if not daily_returns:
        return pd.Series(dtype=float)
    dr = pd.DataFrame(daily_returns, columns=["date", "ret"])
    daily_portfolio_ret = dr.groupby("date")["ret"].mean()
    full_range = pd.date_range(daily_portfolio_ret.index.min(),
                                daily_portfolio_ret.index.max(), freq="B")
    daily_portfolio_ret = daily_portfolio_ret.reindex(full_range, fill_value=0.0)
    return (1.0 + daily_portfolio_ret).cumprod()


def summarize(trades_df: pd.DataFrame, daily_returns: list) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"total_trades": 0}
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
    avg_concurrent_positions = (
        pd.DataFrame(daily_returns, columns=["date", "ret"]).groupby("date").size().mean()
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
        "expectancy_per_trade": trades_df["return"].mean(),
        "avg_concurrent_positions": avg_concurrent_positions,
        "portfolio_total_return": equity.iloc[-1] - 1.0 if len(equity) else np.nan,
        "portfolio_cagr": cagr,
        "portfolio_max_drawdown": max_drawdown(equity) if len(equity) else np.nan,
        "portfolio_sharpe_annualized": sharpe,
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
    all_daily_returns = []
    skipped_short_history = []
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped_short_history.append(symbol)
            continue
        sig_df = compute_signals(df)
        trades, daily_returns = simulate_trades(sig_df, symbol, test_start, args.hold_days)
        all_trades.extend(trades)
        all_daily_returns.extend(daily_returns)

    trades_df = pd.DataFrame(all_trades)
    trades_path = os.path.join(args.out_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)

    summary = summarize(trades_df, all_daily_returns) if len(trades_df) else {"total_trades": 0}
    summary["universe_size"] = len(universe)
    summary["skipped_short_history"] = len(skipped_short_history)
    summary["test_start"] = str(test_start.date())
    summary["hold_days"] = args.hold_days

    summary_path = os.path.join(args.out_dir, "summary.json")
    pd.Series(summary).to_json(summary_path, indent=2, date_format="iso")

    equity = build_portfolio_equity_curve(all_daily_returns)
    if len(equity):
        equity.rename("equity").to_csv(os.path.join(args.out_dir, "equity_curve.csv"),
                                        index_label="date")

    print(f"Universe: {len(universe)} symbols ({len(skipped_short_history)} skipped for insufficient history)")
    print(f"Test window: {test_start.date()} .. {latest_date.date()}  |  hold: {args.hold_days} trading days")
    print(f"Trades: {len(trades_df)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {trades_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
