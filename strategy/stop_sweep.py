"""Stop-loss sweep for the >60% crushed-stock strategy (stock and 60-delta ~2-month calls).

Signals use point-in-time S&P 500 membership. Per trade: enter at next open; exit after
30 trading days at the open, or earlier if the stop triggers. After a stop, the symbol can
be re-entered on a later signal.

Stop types
  stock X%   : stock trades at entry*(1-X) intraday (gap below fills at the open).
               The call (if held) is sold at the model value at that stock price.
  premium X% : call's model value at a close <= entry premium*(1-X); sold at that value.

Option prices are Black-Scholes with IV = 30-day realized vol (or x1.2), 5% round-trip cost.
Usage: python stop_sweep.py DATA_DIR MEMBERSHIP_CSV START END
"""
import glob, os, sys
from math import erf, exp, log, sqrt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import backtest as b
import two_week as tw

HOLD, DTE, DELTA, COST, R = 30, 60, 0.6, 0.05, 0.03
N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))


def call(S, K, T, s):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (R + s * s / 2) * T) / (s * sqrt(T))
    return S * N(d1) - K * exp(-R * T) * N(d1 - s * sqrt(T))


def strike(S, T, s):
    lo, hi = S * 0.2, S * 6
    for _ in range(60):
        k = (lo + hi) / 2
        d = N((log(S / k) + (R + s * s / 2) * T) / (s * sqrt(T)))
        lo, hi = (k, hi) if d > DELTA else (lo, k)
    return k


def load(data_dir, mem_csv):
    m = pd.read_csv(mem_csv)
    m["end_date"] = m.end_date.fillna("2099-12-31")
    win = {}
    for t, a, e in m[["ticker", "start_date", "end_date"]].itertuples(index=False):
        win.setdefault(tw.RENAMES.get(t, t.replace(".", "-")), []).append((pd.Timestamp(a), pd.Timestamp(e)))
    out = {}
    for p in glob.glob(os.path.join(data_dir, "*.csv")):
        k = os.path.basename(p)[:-4]
        if k in ("KDP", "SPY", "PARA") or k not in win:
            continue
        df = b.load(p)[["open", "high", "low", "close", "volume"]]
        if len(df) < 260:
            continue
        df["hv"] = np.log(df.close).diff().rolling(30).std() * np.sqrt(252)
        df["sig"] = b.signals(df, 0.6)
        mem = np.zeros(len(df), bool)
        for a, e in win[k]:
            mem |= (df.index >= a) & (df.index < e)
        df["sig"] &= mem
        out[k] = df
    return out


def sim(data, start, end, kind=None, level=None, ivm=1.0):
    """kind: None | 'stock' | 'premium'. Returns DataFrame of trades (stock ret, call ret)."""
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for k, df in data.items():
        o, lo, c, hv, sig = (df[x].to_numpy() for x in ("open", "low", "close", "hv", "sig"))
        d, n, i = df.index, len(df), 0
        while i < n - 1:
            if not (sig[i] and s0 <= d[i] <= s1):
                i += 1
                continue
            e, x = i + 1, i + 1 + HOLD
            if x >= n:
                break
            iv0 = min(max(hv[i], .15), 3) * ivm
            K = strike(o[e], DTE / 365, iv0)
            p0 = call(o[e], K, DTE / 365, iv0)
            ex, spx, opx, stopped = x, o[x], None, False
            for j in range(e, x):
                ivj = min(max(hv[j], .15), 3) * ivm
                T = max(DTE - (d[j] - d[e]).days, 0) / 365
                if kind == "stock":
                    sp = o[e] * (1 - level)
                    if lo[j] <= sp:
                        spx = min(sp, o[j]) if j > e else sp
                        opx, ex, stopped = call(spx, K, T, ivj), j, True
                        break
                elif kind == "premium":
                    v = call(c[j], K, T, ivj)
                    if v <= p0 * (1 - level):
                        spx, opx, ex, stopped = c[j], v, j, True
                        break
            if opx is None:
                ivx = min(max(hv[x], .15), 3) * ivm
                opx = call(o[x], K, max(DTE - (d[x] - d[e]).days, 0) / 365, ivx)
            rows.append(dict(sym=k, signal=d[i], stock=spx / o[e] - 1,
                             opt=max(opx / p0 - 1 - COST, -1), stopped=stopped))
            i = ex  # flat again after exit
    return pd.DataFrame(rows)


def summary(t, col, stake):
    r = t[col]
    return dict(trades=len(t), stopped=int(t.stopped.sum()), win=f"{100*(r>0).mean():.0f}%",
                median=f"{100*r.median():+.1f}%", worst=f"{100*r.min():+.0f}%",
                pnl=round((r * stake).sum()))


if __name__ == "__main__":
    data_dir, mem, start, end = sys.argv[1:5]
    data = load(data_dir, mem)
    halves = [(start, "2020-12-31"), ("2021-01-01", end)]
    rules = [(None, None)] + [("stock", x) for x in (.10, .15, .20, .25, .30, .40)] + \
            [("premium", x) for x in (.30, .50, .70)]
    for col, stake, label in (("stock", 10000, "STOCK, $10k/trade"), ("opt", 5000, "60-DELTA CALLS, $5k/trade")):
        print(f"\n=== {label}: {start[:4]}-{end[:4]} ===")
        for kind, lvl in rules:
            if col == "stock" and kind == "premium":
                continue
            name = "no stop" if kind is None else f"{kind} -{int(lvl*100)}%"
            full = summary(sim(data, start, end, kind, lvl), col, stake)
            h = [round((sim(data, a, z, kind, lvl)[col] * stake).sum()) for a, z in halves]
            extra = ""
            if col == "opt":
                extra = f" | IVx1.2 ${round((sim(data, start, end, kind, lvl, 1.2)[col]*stake).sum()):>9,}"
            print(f"{name:14s} trades {full['trades']:3d} stopped {full['stopped']:3d} win {full['win']:>4s} "
                  f"median {full['median']:>7s} worst {full['worst']:>5s} | P&L ${full['pnl']:>9,} "
                  f"| 1st half ${h[0]:>8,} 2nd half ${h[1]:>8,}{extra}")
