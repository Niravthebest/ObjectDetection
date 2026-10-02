"""
Run the Stine superstock backtest.

  In-sample  (IS):  2017-01-01 .. 2025-12-31  -- parameters are chosen here only
  Out-of-sample (OOS): 2026-01-01 .. latest   -- run once with the IS-chosen params

Data: weekly OHLCV CSVs in data/<SYMBOL>.csv (columns: date,open,high,low,close,volume)
or, with --download, pulled from Yahoo Finance's chart API (daily -> weekly).

usage: python run.py [--download] [--universe universe.txt]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from stine import Params, backtest, stats

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"
WARMUP_START = "2015-06-01"   # 30/40-week averages + 26-week base need history
IS_START, IS_END = "2017-01-01", "2025-12-31"
OOS_START, OOS_END = "2026-01-01", "2099-12-31"


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    w = daily.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return w.dropna(subset=["close"])


def fetch_daily(sym: str) -> pd.DataFrame:
    """Daily bars from Yahoo's chart API, adjusted for splits and dividends."""
    import time
    import requests
    p1 = int(pd.Timestamp(WARMUP_START).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={p1}&period2={int(time.time())}&interval=1d&events=split,div")
    for attempt in range(4):
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(2 ** (attempt + 1))
    res = (r.json().get("chart") or {}).get("result")
    if r.status_code != 200 or not res or "timestamp" not in res[0]:
        return pd.DataFrame()
    res = res[0]
    q = res["indicators"]["quote"][0]
    d = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]},
                     index=pd.to_datetime(res["timestamp"], unit="s").normalize())
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is not None:
        f = pd.Series(adj, index=d.index) / d["close"]
        for c in ["open", "high", "low", "close"]:
            d[c] = d[c] * f
    return d.dropna(subset=["close"])


def download(symbols: list[str]) -> None:
    DATA.mkdir(exist_ok=True)
    for s in symbols:
        d = fetch_daily(s)
        if d.empty:
            print("no data:", s)
            continue
        to_weekly(d).rename_axis("date").to_csv(DATA / f"{s}.csv")


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{sym}.csv", parse_dates=["date"]).set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


GRID = dict(vol_mult=[1.5, 2.0, 3.0], parabolic_pct=[0.6, 0.8, 1.2], base_weeks=[26, 52])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--universe", default=str(HERE / "universe.txt"))
    a = ap.parse_args()

    symbols = list(dict.fromkeys(
        t.upper() for l in open(a.universe) for t in l.split("#")[0].split()))
    if a.download:
        download(symbols + ["SPY"])

    spy = load("SPY")
    data = {s: load(s) for s in symbols if (DATA / f"{s}.csv").exists()}
    print(f"universe: {len(data)} symbols with data")

    # ---- in-sample parameter search (objective: Sharpe, require >= 20 trades)
    rows = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        p = Params(**dict(zip(keys, combo)))
        eq, tr = backtest(data, spy, p, IS_START, IS_END)
        st = stats(eq, tr)
        rows.append({**dict(zip(keys, combo)), **st})
    grid = pd.DataFrame(rows)
    ok = grid[grid.trades >= 20]
    best = (ok if len(ok) else grid).sort_values("sharpe", ascending=False).iloc[0]
    bp = Params(**{k: (int(best[k]) if k == "base_weeks" else float(best[k])) for k in keys})

    OUT.mkdir(exist_ok=True)
    grid.to_csv(OUT / "is_grid.csv", index=False)

    res = {"params": dc_asdict(bp)}
    curves = {}
    for name, (s, e) in {"in_sample": (IS_START, IS_END), "out_of_sample": (OOS_START, OOS_END)}.items():
        eq, tr = backtest(data, spy, bp, s, e)
        res[name] = stats(eq, tr)
        spy_seg = spy.loc[(spy.index >= s) & (spy.index <= e), "close"]
        res[name]["spy_buy_hold"] = spy_seg.iloc[-1] / spy_seg.iloc[0] - 1
        res[name]["period"] = f"{eq.index[0].date()} .. {eq.index[-1].date()}"
        tr.to_csv(OUT / f"trades_{name}.csv", index=False)
        eq.to_csv(OUT / f"equity_{name}.csv")
        curves[name] = (eq["equity"], spy_seg)

    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    plot(curves)


def dc_asdict(p):
    import dataclasses
    return dataclasses.asdict(p)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    for ax, (name, (eq, spy)) in zip(axes, curves.items()):
        ax.plot(eq.index, eq / eq.iloc[0], label="Stine strategy", color="#1f6feb")
        ax.plot(spy.index, spy / spy.iloc[0], label="SPY buy & hold", color="#8b949e")
        ax.set_title(name.replace("_", "-") + " (growth of $1)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}x"))
        if name == "out_of_sample":
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=120)


if __name__ == "__main__":
    main()
