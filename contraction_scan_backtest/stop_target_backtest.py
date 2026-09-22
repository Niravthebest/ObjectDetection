"""Grid search over stop-loss / take-profit combinations, fixed 5-day max
hold. Reuses the same entry-signal logic as backtest.py; only the exit
rule changes: exit early on a stop-loss or profit-cap hit (checked via
each day's high/low), otherwise exit at the close of day 5.

Usage:
    python3 stop_target_backtest.py --data-dir data --out-dir results/stop_target \
        --max-hold-days 5 --stops 0.05,0.10,0.15,0.20 --targets 0.05,0.10,0.15
"""
import argparse
import os

import numpy as np
import pandas as pd

from backtest import (
    load_universe, max_drawdown, build_portfolio_equity_curve,
    MIN_HISTORY_BARS,
)
from strategy import compute_signals


def simulate_trades_stop_target(df: pd.DataFrame, symbol: str, test_start: pd.Timestamp,
                                 max_hold_days: int, stop_pct: float, target_pct: float) -> tuple:
    """Same entry timing as the fixed-hold backtest (enter at next bar's
    open after a signal), but exits early if a day's low/high crosses the
    stop or target price, else exits at the close of day `max_hold_days`.
    If a single day's range crosses BOTH levels, the stop is assumed to
    have been hit first (conservative), unless the day's open already
    gapped through one level, in which case that's the fill price."""
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
        stop_price = entry_price * (1 - stop_pct)
        target_price = entry_price * (1 + target_pct)
        last_possible = min(entry_idx + max_hold_days - 1, n - 1)
        if last_possible < entry_idx:
            continue

        exit_idx, exit_price, exit_reason = None, None, None
        for i in range(entry_idx, last_possible + 1):
            o, h, l = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i]
            if pd.isna(h) or pd.isna(l) or pd.isna(o):
                break
            hit_stop = l <= stop_price
            hit_target = h >= target_price
            if hit_stop and hit_target:
                if o <= stop_price:
                    exit_idx, exit_price, exit_reason = i, o, "stop_gap"
                elif o >= target_price:
                    exit_idx, exit_price, exit_reason = i, o, "target_gap"
                else:
                    exit_idx, exit_price, exit_reason = i, stop_price, "stop"
                break
            elif hit_stop:
                exit_price = o if o < stop_price else stop_price
                exit_idx, exit_reason = i, "stop"
                break
            elif hit_target:
                exit_price = o if o > target_price else target_price
                exit_idx, exit_reason = i, "target"
                break
        if exit_idx is None:
            exit_idx = last_possible
            exit_price = df["close"].iloc[exit_idx]
            exit_reason = "time"
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
            "exit_reason": exit_reason,
        })
        prev_price = entry_price
        for i in range(entry_idx, exit_idx + 1):
            price = df["close"].iloc[i] if i < exit_idx else exit_price
            if pd.isna(price) or prev_price <= 0:
                break
            daily_returns.append((df["date"].iloc[i], price / prev_price - 1.0))
            prev_price = price
        in_position_until = exit_idx
    return trades, daily_returns


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
    reason_counts = trades_df["exit_reason"].value_counts(normalize=True).to_dict()

    return {
        "total_trades": n,
        "win_rate": len(wins) / n,
        "avg_return_per_trade": trades_df["return"].mean(),
        "median_return_per_trade": trades_df["return"].median(),
        "profit_factor": profit_factor,
        "portfolio_total_return": equity.iloc[-1] - 1.0 if len(equity) else np.nan,
        "portfolio_cagr": cagr,
        "portfolio_max_drawdown": max_drawdown(equity) if len(equity) else np.nan,
        "portfolio_sharpe_annualized": sharpe,
        "pct_stopped_out": reason_counts.get("stop", 0) + reason_counts.get("stop_gap", 0),
        "pct_target_hit": reason_counts.get("target", 0) + reason_counts.get("target_gap", 0),
        "pct_time_exit": reason_counts.get("time", 0),
    }, equity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results/stop_target")
    ap.add_argument("--max-hold-days", type=int, default=5)
    ap.add_argument("--stops", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--targets", default="0.05,0.10,0.15")
    ap.add_argument("--test-start", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stops = [float(x) for x in args.stops.split(",")]
    targets = [float(x) for x in args.targets.split(",")]

    universe = load_universe(args.data_dir)
    latest_date = max(g["date"].max() for g in universe.values())
    test_start = (
        pd.Timestamp(args.test_start)
        if args.test_start
        else latest_date - pd.DateOffset(years=3)
    )

    # Compute signals once per symbol (indicators don't depend on the exit
    # rule) and reuse across every stop/target combo -- this is the
    # expensive part, no need to redo it 12 times.
    signal_dfs = {}
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        signal_dfs[symbol] = compute_signals(df)

    results = []
    equity_curves = {}
    for stop_pct in stops:
        for target_pct in targets:
            all_trades, all_daily_returns = [], []
            for symbol, sig_df in signal_dfs.items():
                trades, daily_returns = simulate_trades_stop_target(
                    sig_df, symbol, test_start, args.max_hold_days, stop_pct, target_pct
                )
                all_trades.extend(trades)
                all_daily_returns.extend(daily_returns)
            trades_df = pd.DataFrame(all_trades)
            summary, equity = summarize(trades_df, all_daily_returns)
            summary["stop_pct"] = stop_pct
            summary["target_pct"] = target_pct
            results.append(summary)
            equity_curves[(stop_pct, target_pct)] = equity
            label = f"stop={stop_pct:.0%} target={target_pct:.0%}"
            print(f"{label:28s} trades={summary.get('total_trades',0):5d}  "
                  f"win_rate={summary.get('win_rate',float('nan')):.1%}  "
                  f"avg_ret={summary.get('avg_return_per_trade',float('nan')):+.2%}  "
                  f"PF={summary.get('profit_factor',float('nan')):.2f}  "
                  f"CAGR={summary.get('portfolio_cagr',float('nan')):+.1%}  "
                  f"MaxDD={summary.get('portfolio_max_drawdown',float('nan')):.1%}  "
                  f"Sharpe={summary.get('portfolio_sharpe_annualized',float('nan')):.2f}")

    grid_df = pd.DataFrame(results)
    grid_path = os.path.join(args.out_dir, "grid_results.csv")
    grid_df.to_csv(grid_path, index=False)

    best_by_sharpe = grid_df.sort_values("portfolio_sharpe_annualized", ascending=False).iloc[0]
    best_by_cagr = grid_df.sort_values("portfolio_cagr", ascending=False).iloc[0]
    best_by_dd = grid_df.sort_values("portfolio_max_drawdown", ascending=False).iloc[0]

    print(f"\nWrote {grid_path}")
    print(f"\nBest Sharpe:  stop={best_by_sharpe.stop_pct:.0%} target={best_by_sharpe.target_pct:.0%}  "
          f"Sharpe={best_by_sharpe.portfolio_sharpe_annualized:.2f}")
    print(f"Best CAGR:    stop={best_by_cagr.stop_pct:.0%} target={best_by_cagr.target_pct:.0%}  "
          f"CAGR={best_by_cagr.portfolio_cagr:+.1%}")
    print(f"Best MaxDD:   stop={best_by_dd.stop_pct:.0%} target={best_by_dd.target_pct:.0%}  "
          f"MaxDD={best_by_dd.portfolio_max_drawdown:.1%}")

    # save equity curves for the best-Sharpe combo and the unconstrained baseline for comparison
    for key in [(best_by_sharpe.stop_pct, best_by_sharpe.target_pct)]:
        eq = equity_curves[key]
        if len(eq):
            fname = f"equity_stop{int(key[0]*100)}_target{int(key[1]*100)}.csv"
            eq.rename("equity").to_csv(os.path.join(args.out_dir, fname), index_label="date")

    # save ALL equity curves compactly (weekly resample) for charting
    all_eq = {}
    for (s, tp), eq in equity_curves.items():
        if len(eq):
            weekly = eq.resample("W").last().dropna()
            all_eq[f"stop{int(s*100)}_target{int(tp*100)}"] = weekly
    if all_eq:
        combined = pd.DataFrame(all_eq)
        combined.to_csv(os.path.join(args.out_dir, "all_equity_curves_weekly.csv"), index_label="date")


if __name__ == "__main__":
    main()
