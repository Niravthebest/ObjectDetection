"""Crushed-stock reversal backtest.

Entry (all on signal-day close):
  1. RSI(14) < 35
  2. close <= 1.20 * 52-week low
  3. volume / avg volume of prior 20 days >= 1.5
  4. 5-day return < +10%
  5. close <= (1 - DD) * 52-week high   (DD = 0.80 by default)
Trade: buy next day's open, hold 30 trading days, sell at that day's open.
One open position per symbol; repeat signals while holding are ignored.

Input: a directory of CSVs, one per symbol, columns date,open,high,low,close,volume
(split-adjusted). Usage: python backtest.py DATA_DIR [--dd 0.8] [--start 2017-01-01] [--end 2025-12-31]
"""
import argparse, glob, os
import numpy as np
import pandas as pd

HOLD = 30


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def signals(df, dd):
    hi52 = df.high.rolling(252, min_periods=252).max()
    lo52 = df.low.rolling(252, min_periods=252).min()
    volr = df.volume / df.volume.shift(1).rolling(20).mean()
    return (
        (rsi(df.close) < 35)
        & (df.close <= 1.2 * lo52)
        & (volr >= 1.5)
        & (df.close / df.close.shift(5) - 1 < 0.10)
        & (df.close <= (1 - dd) * hi52)
    )


def trades_for(sym, df, dd, start, end):
    sig = signals(df, dd).to_numpy()
    dates, opens = df.index, df.open.to_numpy()
    out, i, n = [], 0, len(df)
    while i < n - 1:
        if sig[i] and start <= dates[i] <= end:
            e, x = i + 1, i + 1 + HOLD
            closed = x < n
            xp = opens[x] if closed else df.close.iloc[-1]
            out.append(dict(symbol=sym, signal=dates[i], entry=dates[e], entry_px=opens[e],
                            exit=dates[x] if closed else dates[-1], exit_px=xp,
                            ret=xp / opens[e] - 1, closed=closed))
            i = x  # flat again only after exit
            continue
        i += 1
    return out


def load(path):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return df[(df.open > 0) & (df.close > 0)].dropna()


def summarize(t):
    t = t[t.closed]
    if t.empty:
        return "0 closed trades"
    r = t.ret
    big = t.loc[r.idxmax()]
    return (f"trades {len(t)} | win {100*(r>0).mean():.1f}% | mean {100*r.mean():+.1f}% | "
            f"median {100*r.median():+.1f}% | best {big.symbol} {100*big.ret:+.0f}% | "
            f"worst {100*r.min():+.0f}% | mean ex-top-5 {100*r.sort_values().iloc[:-5].mean():+.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dd", type=float, nargs="+", default=[0.8])
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="trades.csv")
    a = ap.parse_args()
    data = {os.path.basename(p)[:-4]: load(p) for p in glob.glob(os.path.join(a.data_dir, "*.csv"))}
    s, e = pd.Timestamp(a.start), pd.Timestamp(a.end)
    for dd in a.dd:
        t = pd.DataFrame([x for sym, df in data.items() for x in trades_for(sym, df, dd, s, e)])
        print(f"drawdown > {dd:.0%}: " + (summarize(t) if len(t) else "0 trades"))
        if len(t):
            t.to_csv(a.out.replace(".csv", f"_dd{int(dd*100)}.csv"), index=False)
