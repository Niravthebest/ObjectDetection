#!/usr/bin/env python3
"""
Portfolio simulation: each day, buy the triggered stock with the best EXPECTANCY,
hold it 5 trading days, repeat.

    Expectancy (per stock) = win% * avg win - loss% * avg loss = mean 5-day return per past trade

WALK-FORWARD (no look-ahead): on day t a stock's expectancy is computed only from its own
earlier signals whose 5-day exit had already happened by day t. A stock needs at least
--min-trades completed trades and a positive expectancy to be eligible.

Modes
    single : 100% of capital in one stock; next trade only after the current one exits.
    5-slot : capital split into 5 slots of 20%; every day the free slot buys that day's top
             pick, so a new position starts each day and 5 overlap.

Comparisons
    look-ahead : same rules but expectancy from the FULL backtest (includes future trades) -
                 shows how much an in-sample ranking flatters results.
    random     : same mode, but pick a random triggered stock (mean of N runs).
    SPY        : buy & hold over the same dates.

Entry = signal-day close, exit = close 5 trading days later (adjusted prices),
--cost per round trip (default 0.10%).

USAGE
    python portfolio_sim.py --signals near52 [--start 2022-03-31]
    python portfolio_sim.py --signals sma14
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import sma_cross_scanner as base

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = 5


def load_signals(kind: str, results: str) -> pd.DataFrame:
    if kind == "near52":
        t = pd.read_csv(os.path.join(results, "near52wh_p100_trades.csv"), parse_dates=["date"])
        t = t[t.pass_all_fresh]
    else:
        t = pd.read_csv(os.path.join(results, "backtest_trades.csv"), parse_dates=["date"])
    return t[["date", "symbol", "fwd5"]].dropna().reset_index(drop=True)


def simulate(sig: pd.DataFrame, cal: pd.DatetimeIndex, start, slots: int, pick: str,
             min_trades: int, cost: float, rng=None, insample: dict | None = None):
    """pick: 'walk' | 'insample' | 'random'. Returns (trades df, final multiple)."""
    pos = {d: i for i, d in enumerate(cal)}
    sig = sig[sig.date.isin(pos)].copy()
    sig["i"] = sig.date.map(pos)
    by_day = {i: g for i, g in sig.groupby("i")}
    exits = defaultdict(list)                      # exit idx -> [(sym, ret)]
    for r in sig.itertuples():
        exits[r.i + HOLD].append((r.symbol, r.fwd5))
    hist_sum, hist_n = defaultdict(float), defaultdict(int)

    capital = [1.0 / slots] * slots
    busy = [-1] * slots
    held = [None] * slots
    trades = []
    start_i = cal.searchsorted(pd.Timestamp(start))
    last_i = len(cal) - 1 - HOLD
    for t in range(len(cal)):
        for sym, ret in exits.get(t, ()):          # trades that have closed by today's close
            hist_sum[sym] += ret
            hist_n[sym] += 1
        if t < start_i or t > last_i or t not in by_day:
            continue
        free = [k for k in range(slots) if busy[k] <= t]
        if not free:
            continue
        today = by_day[t]
        cur = {held[k] for k in range(slots) if busy[k] > t}
        today = today[~today.symbol.isin(cur)]
        if pick == "random":
            order = today.sample(frac=1, random_state=rng.integers(1 << 31))
            order = order.assign(exp=np.nan, n=0)
        else:
            if pick == "walk":
                n = today.symbol.map(hist_n).fillna(0)
                e = today.symbol.map(lambda s: hist_sum[s] / hist_n[s] if hist_n[s] else np.nan)
            else:
                n = today.symbol.map(lambda s: insample.get(s, (np.nan, 0))[1])
                e = today.symbol.map(lambda s: insample.get(s, (np.nan, 0))[0])
            order = today.assign(exp=e.values, n=n.values)
            order = order[(order.n >= min_trades) & (order.exp > 0)]
            order = order.sort_values(["exp", "n"], ascending=False)
        for k, r in zip(free, order.itertuples()):
            net = r.fwd5 - cost
            capital[k] *= 1 + net
            busy[k], held[k] = t + HOLD, r.symbol
            trades.append({"date": cal[t], "exit": cal[t + HOLD], "symbol": r.symbol,
                           "expectancy": r.exp, "prior_trades": r.n, "ret": r.fwd5, "net": net,
                           "slot": k})
    return pd.DataFrame(trades), sum(capital)


def equity_curve(trades: pd.DataFrame, slots: int) -> pd.Series:
    """Realized equity after each exit."""
    cap = [1.0 / slots] * slots
    pts = []
    for r in trades.sort_values("exit").itertuples():
        cap[r.slot] *= 1 + r.net
        pts.append((r.exit, sum(cap)))
    return pd.Series(dict(pts)) if pts else pd.Series(dtype=float)


def metrics(trades, final, slots, start, end) -> dict:
    yrs = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    eq = equity_curve(trades, slots)
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    return {"trades": len(trades),
            "win_rate": round(float((trades.net > 0).mean()) * 100, 1) if len(trades) else None,
            "avg_trade": round(float(trades.net.mean()) * 100, 2) if len(trades) else None,
            "total_ret": round((final - 1) * 100, 1),
            "cagr": round((final ** (1 / yrs) - 1) * 100, 1),
            "max_dd": round(dd * 100, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signals", choices=["near52", "sma14"], default="near52")
    p.add_argument("--start", default="", help="first trade date (default: earliest signal + warm-up)")
    p.add_argument("--min-trades", type=int, default=5)
    p.add_argument("--cost", type=float, default=0.001, help="round-trip cost as a fraction")
    p.add_argument("--random-runs", type=int, default=200)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    args = p.parse_args()

    sig = load_signals(args.signals, args.results)
    spy = base.fetch_history("SPY", "6y")
    cal = spy.index
    start = pd.Timestamp(args.start) if args.start else sig.date.min() + pd.DateOffset(months=6)
    end = cal[-1 - HOLD]
    spy_ret = float(spy.adj.loc[start:end].iloc[-1] / spy.adj.loc[start:end].iloc[0])

    ins = sig.groupby("symbol").fwd5.agg(["mean", "size"])
    insample = {s: (m, n) for s, (m, n) in ins.iterrows()}

    rows, saved = [], {}
    for slots, mode in ((1, "single"), (5, "5-slot")):
        for pick in ("walk", "insample"):
            tr, fin = simulate(sig, cal, start, slots, pick, args.min_trades, args.cost, insample=insample)
            rows.append({"mode": mode, "pick": {"walk": "top expectancy (walk-forward)",
                                                 "insample": "top expectancy (LOOK-AHEAD)"}[pick],
                         **metrics(tr, fin, slots, start, end)})
            saved[(mode, pick)] = tr
        rng = np.random.default_rng(7)
        rnd = [metrics(*simulate(sig, cal, start, slots, "random", 0, args.cost, rng=rng), slots, start, end)
               for _ in range(args.random_runs)]
        rd = pd.DataFrame(rnd).median(numeric_only=True).round(1).to_dict()
        rows.append({"mode": mode, "pick": f"random triggered stock (median of {args.random_runs})", **rd})
    yrs = (end - start).days / 365.25
    rows.append({"mode": "-", "pick": "SPY buy & hold", "trades": 1, "total_ret": round((spy_ret - 1) * 100, 1),
                 "cagr": round((spy_ret ** (1 / yrs) - 1) * 100, 1),
                 "max_dd": round(float((spy.adj.loc[start:end] / spy.adj.loc[start:end].cummax() - 1).min()) * 100, 1)})
    res = pd.DataFrame(rows)

    # yearly: walk-forward single + 5-slot vs SPY
    yearly = []
    for y in range(start.year, end.year + 1):
        row = {"year": y}
        for (mode, pick), tr in saved.items():
            if pick != "walk":
                continue
            slots = 1 if mode == "single" else 5
            eq = equity_curve(tr, slots)
            eq = pd.concat([pd.Series({start - pd.Timedelta(days=1): 1.0}), eq])
            a = eq[eq.index < pd.Timestamp(f"{y}-01-01")]
            b = eq[eq.index <= pd.Timestamp(f"{y}-12-31")]
            row[f"{mode} %"] = round((b.iloc[-1] / (a.iloc[-1] if len(a) else 1.0) - 1) * 100, 1)
        s = spy.adj.loc[start:end]
        sa = s[s.index < pd.Timestamp(f"{y}-01-01")]
        sb = s[s.index <= pd.Timestamp(f"{y}-12-31")]
        row["SPY %"] = round((sb.iloc[-1] / (sa.iloc[-1] if len(sa) else s.iloc[0]) - 1) * 100, 1)
        yearly.append(row)
    yearly = pd.DataFrame(yearly)

    tag = args.signals
    res.to_csv(os.path.join(args.results, f"sim_{tag}_summary.csv"), index=False)
    yearly.to_csv(os.path.join(args.results, f"sim_{tag}_yearly.csv"), index=False)
    saved[("single", "walk")].to_csv(os.path.join(args.results, f"sim_{tag}_trades_single.csv"), index=False)
    saved[("5-slot", "walk")].to_csv(os.path.join(args.results, f"sim_{tag}_trades_5slot.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"Signals: {tag}  |  {start:%Y-%m-%d} .. {end:%Y-%m-%d}  |  min prior trades {args.min_trades}, "
          f"cost {args.cost*100:.2f}% round trip\n")
    print(res.to_string(index=False))
    print("\nCalendar-year returns (walk-forward):")
    print(yearly.to_string(index=False))
    tr = saved[("single", "walk")]
    print("\nMost-traded stocks (single, walk-forward):")
    print(tr.groupby("symbol").net.agg(trades="size", win_rate=lambda r: round((r > 0).mean() * 100),
                                       avg_pct=lambda r: round(r.mean() * 100, 2))
            .sort_values("trades", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
