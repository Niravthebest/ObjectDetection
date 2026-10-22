"""Search for an S&P 500 exposure rule that beats buy-and-hold.

"Beat" is fixed up front: higher CAGR AND a max drawdown no worse than
buy-and-hold, in data not used to choose the rule.

Returns: SPY total return (dividends reinvested), 1993-2026. Exposure e is set at
the close and earns the next day: r = cash + e * (spy - cash), i.e. ES futures
with T-bill collateral (e > 1 is futures leverage). Cash = 13-week T-bill (^IRX).
Cost: 2 bps per unit of exposure traded; exposure only changes when the target
moves by more than 0.1 (no daily dithering).

Data split:
  EXPLORE   1993-2005  (choose the rule and its parameters here only)
  HOLDOUT   2006-2026  (untouched until the rule is fixed)
  OLD       1960-1992  S&P 500 price index (no dividends for either side)
  plus a walk-forward that re-picks parameters every year from the prior 10 years.

Rule families:
  bh            e = 1
  trend n       e = 1 above the n-day SMA, else 0
  dip k         e = 1, plus k while an RSI(2) dip trade is open (RSI(2) < 10 above
                the 200-day SMA, until the close > 5-day SMA)
  vol t/c       e = t / (20-day realised vol), capped at c
  vol+trend     vol rule, but 0 below the 200-day SMA
  vol+trend+dip vol+trend, plus k during an RSI(2) dip trade (still capped at c+k)

Usage: python beat_bh.py DATA_PKL OUT_DIR {explore|test}
"""
import itertools, json, pickle, sys
import numpy as np, pandas as pd
from backtest import rsi

COST, BAND = 2 / 1e4, 0.10
EXPLORE, HOLDOUT, OLD = ("1993-01-01", "2005-12-31"), ("2006-01-01", "2026-12-31"), ("1960-01-01", "1992-12-31")

def frame(px, irx):
    d = pd.DataFrame({"px": px}).dropna()
    d["ret"] = d.px.pct_change()
    d["cash"] = (irx.reindex(d.index).ffill() / 100 / 252).fillna(0)
    c = d.px
    d["sma5"], d["sma100"], d["sma150"], d["sma200"] = (c.rolling(n).mean() for n in (5, 100, 150, 200))
    d["rsi2"] = rsi(c, 2)
    d["vol20"] = d.ret.rolling(20).std() * np.sqrt(252)
    # RSI(2) dip-trade state at each close
    dip, on = np.zeros(len(d)), False
    for i, (cl, s5, s2, r) in enumerate(zip(c.values, d.sma5.values, d.sma200.values, d.rsi2.values)):
        if on and cl > s5:
            on = False
        elif not on and r < 10 and cl > s2:
            on = True
        dip[i] = on
    d["dip"] = dip
    return d.dropna()

def target(d, kind, p):
    above = lambda n: (d.px > d[f"sma{n}"]).astype(float)
    if kind == "bh":
        return pd.Series(1.0, index=d.index)
    if kind == "trend":
        return above(p["n"])
    if kind == "dip":
        return 1 + p["k"] * d.dip
    v = (p["t"] / d.vol20).clip(upper=p["c"])
    if kind == "vol":
        return v
    v = v * above(200)
    if kind == "vol+trend":
        return v
    return v + p["k"] * d.dip  # vol+trend+dip

def run(d, e_target):
    """Apply banding, then daily returns with next-day exposure and costs."""
    tgt, e, cur = e_target.values, np.zeros(len(d)), 0.0
    for i, x in enumerate(tgt):
        if abs(x - cur) > BAND or (x == 0) != (cur == 0):
            cur = x
        e[i] = cur
    e_prev = np.r_[0.0, e[:-1]]  # exposure set at yesterday's close
    r = d.cash.values + e_prev * (d.ret.values - d.cash.values) - COST * np.abs(np.diff(np.r_[0.0, e]))
    return pd.Series(r, index=d.index), pd.Series(e, index=d.index)

def stats(r, e=None):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    out = dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, max_dd=dd, sharpe=r.mean() / r.std() * np.sqrt(252),
               vol=r.std() * np.sqrt(252), worst_year=r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1).min())
    if e is not None:
        out.update(avg_exp=e.mean(), max_exp=e.max(), in_mkt=(e > 0).mean())
    return out

GRID = ([("bh", {})] + [("trend", {"n": n}) for n in (100, 150, 200)] + [("dip", {"k": k}) for k in (0.5, 1.0)]
        + [("vol", {"t": t, "c": c}) for t in (0.12, 0.15, 0.18, 0.21) for c in (1.5, 2.0)]
        + [("vol+trend", {"t": t, "c": c}) for t in (0.12, 0.15, 0.18, 0.21) for c in (1.5, 2.0)]
        + [("vol+trend+dip", {"t": t, "c": c, "k": k}) for t in (0.12, 0.15, 0.18, 0.21) for c in (1.5, 2.0) for k in (0.5, 1.0)])

def name(kind, p):
    return kind + ("" if not p else " " + " ".join(f"{k}={v:g}" for k, v in p.items()))

def table(d, span):
    rows = {}
    for kind, p in GRID:
        r, e = run(d, target(d, kind, p))
        m = (r.index >= span[0]) & (r.index <= span[1])
        rows[name(kind, p)] = stats(r[m], e[m])
    return pd.DataFrame(rows).T

def show(T):
    s = T.copy()
    for k in ("cagr", "max_dd", "vol", "worst_year", "in_mkt"):
        s[k] = (s[k] * 100).round(1)
    return s.round(2).to_string()

if __name__ == "__main__":
    data = pickle.load(open(sys.argv[1], "rb"))
    out_dir, mode = sys.argv[2], sys.argv[3]
    irx = data["^IRX"].close
    spy = frame(data["SPY"].adj, irx)
    pd.set_option("display.width", 250)
    if mode == "explore":
        T = table(spy, EXPLORE)
        b = T.loc["bh"]
        T["beats"] = (T.cagr > b.cagr) & (T.max_dd >= b.max_dd)
        print(f"== EXPLORE {EXPLORE[0][:4]}-{EXPLORE[1][:4]} (SPY total return) ==")
        print(show(T.sort_values("cagr", ascending=False)))
        T.to_csv(f"{out_dir}/explore.csv")

CHOSEN = [("vol+trend+dip", {"t": 0.15, "c": 1.5, "k": 1.0}),   # fixed after EXPLORE, before HOLDOUT
          ("dip", {"k": 1.0})]

def walk_forward(d, start_year, train=10):
    """Each year pick the grid rule with the best CAGR whose max drawdown over the prior
    `train` years is no worse than buy-and-hold's; trade it for the year."""
    runs = {name(k, p): run(d, target(d, k, p))[0] for k, p in GRID}
    yrs = d.index.year
    out, picks = [], []
    for y in range(start_year, yrs.max() + 1):
        tr = (yrs >= y - train) & (yrs < y)
        if tr.sum() < 200 * train:
            continue
        st = {n: stats(r[tr]) for n, r in runs.items()}
        ok = {n: s for n, s in st.items() if s["max_dd"] >= st["bh"]["max_dd"]}
        pick = max(ok, key=lambda n: ok[n]["cagr"])
        picks.append((y, pick))
        out.append(runs[pick][yrs == y])
    return pd.concat(out), picks, runs["bh"]

def boot(r_s, r_b, n=2000, seed=11):
    rng = np.random.default_rng(seed)
    key = [r_s.index.year, r_s.index.month]
    ms, mb = [g.values for _, g in r_s.groupby(key)], [g.values for _, g in r_b.groupby(key)]
    res = []
    for _ in range(n):
        idx = rng.integers(0, len(ms), len(ms))
        a, b = np.concatenate([ms[i] for i in idx]), np.concatenate([mb[i] for i in idx])
        ea, eb = np.cumprod(1 + a), np.cumprod(1 + b)
        res.append((ea[-1] > eb[-1], (ea / np.maximum.accumulate(ea) - 1).min() >= (eb / np.maximum.accumulate(eb) - 1).min()))
    res = np.array(res)
    return res[:, 0].mean(), res[:, 1].mean(), (res[:, 0] & res[:, 1]).mean()

def test(data, out_dir):
    irx = data["^IRX"].close
    spy = frame(data["SPY"].adj, irx)
    gspc = frame(data["^GSPC"].close, irx)
    es = frame(data["ES=F"].close, irx * 0)  # futures price already excludes the cash yield
    res = {}
    for label, d, span in [("HOLDOUT 2006-2026 (SPY total return)", spy, HOLDOUT),
                           ("OLD 1960-1992 (S&P 500 price index, no dividends either side)", gspc, OLD),
                           ("ES futures 2001-2026 (futures excess return)", es, ("2001-07-01", "2026-12-31"))]:
        T = table(d, span)
        b = T.loc["bh"]
        T["beats"] = (T.cagr > b.cagr) & (T.max_dd >= b.max_dd)
        print(f"\n== {label} ==")
        keep = ["bh", "trend n=200"] + [name(k, p) for k, p in CHOSEN]
        print(show(T.loc[keep]))
        print(f"grid rules that beat B&H here: {int(T.beats.sum())} of {len(T) - 1}")
        res[label] = T
    # Yearly table and bootstrap on the holdout
    m = (spy.index >= HOLDOUT[0])
    rb = run(spy, target(spy, "bh", {}))[0][m]
    for k, p in CHOSEN:
        rs, e = run(spy, target(spy, k, p))
        rs = rs[m]
        yr = pd.DataFrame({"strategy": rs.groupby(rs.index.year).apply(lambda x: (1 + x).prod() - 1),
                           "bh": rb.groupby(rb.index.year).apply(lambda x: (1 + x).prod() - 1)})
        print(f"\n{name(k, p)}: beats B&H in {(yr.strategy > yr.bh).mean():.0%} of holdout years")
        print((yr * 100).round(1).T.to_string())
        pc, pd_, pboth = boot(rs, rb)
        print(f"bootstrap (monthly blocks): P(higher end value) {pc:.0%}, P(drawdown no worse) {pd_:.0%}, P(both) {pboth:.0%}")
        for cost in (5, 10):
            global COST
            old, COST = COST, cost / 1e4
            print(f"  cost {cost} bps/unit: holdout CAGR {stats(run(spy, target(spy, k, p))[0][m])['cagr']:.1%}")
            COST = old
    # Walk-forward on the SPY years after the first 10 (training never sees the future)
    wf, picks, rbh = walk_forward(spy, 2003)
    mm = wf.index >= HOLDOUT[0]
    print("\n== Walk-forward (rule re-picked yearly from prior 10 years), 2006-2026 ==")
    print("WF  :", {k: round(v, 3) for k, v in stats(wf[mm]).items()})
    print("B&H :", {k: round(v, 3) for k, v in stats(rbh[rbh.index >= HOLDOUT[0]]).items()})
    print("picks:", ", ".join(f"{y}:{n}" for y, n in picks if y >= 2006))
    pickle.dump(dict(res=res, picks=picks), open(f"{out_dir}/test.pkl", "wb"))

if __name__ == "__main__" and sys.argv[3] == "test":
    test(pickle.load(open(sys.argv[1], "rb")), sys.argv[2])
