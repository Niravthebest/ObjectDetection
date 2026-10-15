"""Stop-loss study for the RSI(2) stock strategy.

Base rules (unchanged): buy at the close when RSI(2) < 10 and close > 200-day SMA;
sell at the close when close > 5-day SMA. Cost 5 bps per side. Every signal is
taken with the same $1 size, so per-trade statistics and the combined daily P&L
of all open trades describe the strategy as the scanner reports it.

Exit add-ons tested (all checked on the close, like the scanner at 3:45pm):
  pct  : sell if the close is X% or more below the entry price
  vol  : sell if the close is k x (20-day stdev of daily returns) x sqrt(days held + 1)
         below entry -- a stop that widens for volatile stocks and longer holds
  time : sell after N trading days if the 5-day SMA exit has not fired
After a stop or time exit the stock is blocked from re-entry until it closes
above its 5-day SMA (the trade the base rules would still have been in).

Data split (by entry date):
  EXPLORE  2012-2018 -- used for every chart, bucket and choice of rule
  HOLDOUT  1991-2011 and 2019-2026 -- untouched until the rule is fixed

Usage: python stop_study.py DATA_DIR {explore|test}
DATA_DIR holds sp500.pkl (fetch_stocks.py) and index.pkl; JSON for the charts is
written there.
"""
import json, pickle, sys
import numpy as np, pandas as pd
from backtest import rsi

COST = 5 / 1e4
EXPLORE = ("2012-01-01", "2018-12-31")
PCT = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
VOLK = [1.5, 2.0, 2.5, 3.0, 4.0]
TIME = [5, 7, 10, 15]
VARIANTS = ([("none", None)] + [("pct", p) for p in PCT] + [("vol", k) for k in VOLK]
            + [("time", n) for n in TIME])

def vname(kind, p):
    if kind == "none":
        return "base"
    return {"pct": f"stop {p:.0%}", "vol": f"vol {p:g}x", "time": f"time {p}d"}[kind]

def build(prices):
    """Aligned arrays (days x symbols) on a common calendar."""
    close = pd.DataFrame({s: df["close"].astype(float) for s, df in prices.items()}).sort_index()
    close = close[close.index >= "1990-01-01"]
    r = close.pct_change(fill_method=None)
    return dict(
        dates=close.index, syms=list(close.columns), c=close.values,
        sma5=close.rolling(5).mean().values, sma200=close.rolling(200).mean().values,
        rsi2=close.apply(lambda s: rsi(s.dropna(), 2)).reindex(close.index).values,
        vol20=r.rolling(20).std().values, ret1=r.values,
        ret5=(close / close.shift(5) - 1).values)

def simulate(A, kind, p, cols=None, size_vol=None):
    """Every signal, $1 each (or size_vol / 20-day vol dollars, clipped 0.25-3, if size_vol is set). Returns trades DataFrame and daily P&L / open-count arrays."""
    c, s5, s200, r2, v20 = A["c"], A["sma5"], A["sma200"], A["rsi2"], A["vol20"]
    nd = len(A["dates"])
    pnl, nopen = np.zeros(nd), np.zeros(nd, dtype=int)
    rows = []
    for j in (range(c.shape[1]) if cols is None else cols):
        cj, s5j, s2j, rj, vj = c[:, j], s5[:, j], s200[:, j], r2[:, j], v20[:, j]
        entry_ok = (rj < 10) & (cj > s2j)
        i, blocked = 0, False
        cand = np.flatnonzero(entry_ok)
        k = 0
        while k < len(cand):
            e = cand[k]
            if e < i:
                k += 1
                continue
            if blocked:  # wait for a close above the 5-day SMA before allowing re-entry
                rel = np.flatnonzero(cj[i:e + 1] > s5j[i:e + 1])
                if len(rel) == 0:
                    k += 1
                    continue
                blocked = False
                i = i + rel[0] + 1
                continue
            ep, mae, mfe, reason = cj[e], 0.0, 0.0, "sma5"
            w = 1.0 if size_vol is None else float(np.clip(size_vol / vj[e], 0.25, 3.0))
            pnl[e] -= COST * w
            d = e + 1
            while d < nd:
                if np.isnan(cj[d]):
                    d += 1
                    continue
                pnl[d] += w * (cj[d] - cj[d - 1]) / ep if not np.isnan(cj[d - 1]) else 0.0
                nopen[d] += 1
                rr = cj[d] / ep - 1
                mae, mfe = min(mae, rr), max(mfe, rr)
                held = d - e
                if cj[d] > s5j[d]:
                    break
                if kind == "pct" and rr <= -p:
                    reason = "stop"; break
                if kind == "vol" and rr <= -p * vj[e] * np.sqrt(held + 1):
                    reason = "stop"; break
                if kind == "time" and held >= p:
                    reason = "time"; break
                d += 1
            if d >= nd:  # still open at the end of the data
                break
            pnl[d] -= COST * w
            rows.append((A["syms"][j], A["dates"][e], A["dates"][d], cj[d] / ep * (1 - COST) ** 2 - 1,
                         d - e, reason, mae, mfe, rj[e], cj[e] / s2j[e] - 1, vj[e],
                         A["ret1"][e, j], A["ret5"][e, j]))
            blocked = reason != "sma5"
            i = d + 1
            k += 1
    t = pd.DataFrame(rows, columns=["symbol", "entry", "exit", "ret", "days", "reason", "mae", "mfe",
                                    "rsi2", "dist200", "vol20", "ret1", "ret5"])
    return t, pnl, nopen

def summarize(t):
    w, l = t.ret[t.ret > 0], t.ret[t.ret <= 0]
    return dict(trades=len(t), win=len(w) / len(t), exp=t.ret.mean(), median=t.ret.median(),
                avg_win=w.mean(), avg_loss=l.mean(), pf=w.sum() / -l.sum(), p5=t.ret.quantile(.05),
                p1=t.ret.quantile(.01), worst=t.ret.min(), lt10=(t.ret < -.10).mean(),
                days=t.days.mean(), stopped=(t.reason != "sma5").mean())

def curve(pnl, nopen, dates, start, end):
    """Combined P&L of every open trade ($1 each): total, max drawdown, worst month, peak open."""
    m = (dates >= start) & (dates <= end)
    eq = np.cumsum(pnl[m])
    dd = eq - np.maximum.accumulate(eq)
    s = pd.Series(pnl[m], index=dates[m])
    mon = s.groupby([s.index.year, s.index.month]).sum()
    return dict(total=eq[-1], max_dd=dd.min(), worst_month=mon.min(), peak_open=int(nopen[m].max()),
                mar=eq[-1] / -dd.min() if dd.min() < 0 else np.nan)

def box(x):
    q = np.nanpercentile(x, [5, 25, 50, 75, 95])
    return dict(lo=q[0], q1=q[1], med=q[2], q3=q[3], hi=q[4], mean=float(np.nanmean(x)), n=int(np.isfinite(x).sum()))

def span(t, start, end):
    return t[(t.entry >= start) & (t.entry <= end)]

def explore(A, base_dir):
    s, e = EXPLORE
    out, rows, sims = {}, [], {}
    for kind, p in VARIANTS:
        t, pnl, nopen = simulate(A, kind, p)
        sims[vname(kind, p)] = (t, pnl, nopen)
        te = span(t, s, e)
        rows.append(dict(variant=vname(kind, p), **summarize(te), **curve(pnl, nopen, A["dates"], s, e)))
    R = pd.DataFrame(rows).set_index("variant")
    out["variants"] = R.reset_index().to_dict("records")
    out["variant_box"] = {k: box(span(v[0], s, e).ret.values) for k, v in sims.items()}

    base = span(sims["base"][0], s, e).copy()
    # Weak-spot buckets: deciles of entry features (and MAE quintile) vs base trade return
    feats = {"vol20": "20-day volatility", "dist200": "Distance above 200-day SMA",
             "rsi2": "Entry RSI(2)", "ret1": "Entry-day return", "ret5": "5-day return into entry"}
    out["deciles"] = {}
    for f, label in feats.items():
        q = pd.qcut(base[f], 10, labels=False, duplicates="drop")
        out["deciles"][f] = dict(label=label, buckets=[
            dict(d=int(b), from_=float(g[f].min()), to=float(g[f].max()), **box(g.ret.values),
                 win=float((g.ret > 0).mean()), tail=float((g.ret < -.10).mean()))
            for b, g in base.groupby(q)])
    # Recovery: once a trade has closed X% under entry, what does holding on to the base exit earn
    # from that point (vs selling there)?
    t, _, _ = sims["base"]
    rec = []
    c = A["c"]; si = {s_: j for j, s_ in enumerate(A["syms"])}; di = {d: i for i, d in enumerate(A["dates"])}
    for x in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        fw = []
        for r in base[base.mae <= -x].itertuples():
            j, i0, i1 = si[r.symbol], di[r.entry], di[r.exit]
            path = c[i0:i1 + 1, j] / c[i0, j] - 1
            hit = np.flatnonzero(path <= -x)[0]
            fw.append((1 + r.ret) / (1 + path[hit]) - 1)
        fw = np.array(fw)
        rec.append(dict(x=x, share=len(fw) / len(base), **box(fw), win=float((fw > 0).mean())))
    out["recovery"] = rec
    out["mae_box"] = box(base.mae.values)
    # Days held vs return
    db = base.assign(b=base.days.clip(upper=10))
    out["days"] = [dict(d=int(b), **box(g.ret.values)) for b, g in db.groupby("b")]
    # Combined P&L curves (every signal, $1 each) for base and a few variants
    m = (A["dates"] >= s) & (A["dates"] <= e)
    out["curves"] = {k: dict(dates=[d.strftime("%Y-%m-%d") for d in A["dates"][m][::5]],
                             eq=np.cumsum(v[1][m])[::5].round(3).tolist())
                     for k, v in sims.items() if k in ("base", "stop 10%", "stop 15%", "vol 3x", "time 10d")}
    json.dump(out, open(f"{base_dir}/explore.json", "w"), default=float)

    pd.set_option("display.width", 250)
    show = R.copy()
    for k in ("win", "exp", "median", "avg_win", "avg_loss", "p5", "p1", "worst", "lt10", "stopped"):
        show[k] = (show[k] * 100).round(2)
    print(f"== EXPLORE {s[:4]}-{e[:4]}: every signal, all S&P 500 stocks ==")
    print(show.round(2).to_string())
    print("\nRecovery after being down X% (forward return to the base exit):")
    print(pd.DataFrame(rec)[["x", "n", "share", "mean", "med", "win", "lo", "hi"]].round(4).to_string(index=False))
    for f in feats:
        d = pd.DataFrame(out["deciles"][f]["buckets"])[["d", "from_", "to", "n", "mean", "med", "win", "tail", "lo"]]
        print(f"\nDeciles of {f}:\n" + d.round(4).to_string(index=False))
    return sims

HOLDOUT = {"1991-2011": ("1991-01-01", "2011-12-31"), "2019-2026": ("2019-01-01", "2026-12-31")}
CANDIDATES = ["stop 15%", "vol 3x", "time 10d"]  # fixed after EXPLORE, before looking at HOLDOUT

def maxdd(x):
    eq = np.cumsum(x)
    return (eq - np.maximum.accumulate(eq)).min()

def test(A, base_dir, scanner_syms, IDX):
    sims = {vname(k, p): simulate(A, k, p) for k, p in VARIANTS}
    dates, out, rows = A["dates"], {}, []
    for pname, (s, e) in HOLDOUT.items():
        for v, (t, pnl, no) in sims.items():
            rows.append(dict(period=pname, variant=v, **summarize(span(t, s, e)), **curve(pnl, no, dates, s, e)))
    R = pd.DataFrame(rows)
    out["holdout"] = R.to_dict("records")
    out["holdout_box"] = {p_: {v: box(span(t, s, e).ret.values) for v, (t, _, _) in sims.items()}
                          for p_, (s, e) in HOLDOUT.items()}

    # Walk-forward: each year, pick the variant with the best P&L / max drawdown over the prior 5 years
    years = pd.Index(dates.year)
    wf = []
    for y in range(1996, int(years.max()) + 1):
        tr = (years >= y - 5) & (years <= y - 1)
        te = years == y
        score = {v: (pnl[tr].sum() / -maxdd(pnl[tr]) if maxdd(pnl[tr]) < 0 else np.inf) for v, (_, pnl, _) in sims.items()}
        pick = max(score, key=score.get)
        wf.append(dict(year=y, pick=pick, wf=sims[pick][1][te].sum(), base=sims["base"][1][te].sum(),
                       holdout=not (2012 <= y <= 2018)))
    W = pd.DataFrame(wf)
    out["walkforward"] = W.to_dict("records")
    # chained OOS curves (holdout years only)
    m = ~((years >= 2012) & (years <= 2018)) & (years >= 1996)
    pick_by_year = dict(zip(W.year, W.pick))
    wf_daily = np.array([sims[pick_by_year[y]][1][i] if m[i] else 0.0 for i, y in enumerate(years)])
    step = 5
    out["wf_curve"] = dict(dates=[d.strftime("%Y-%m-%d") for d in dates[m][::step]],
                           wf=np.cumsum(wf_daily[m])[::step].round(2).tolist(),
                           base=np.cumsum(sims["base"][1][m])[::step].round(2).tolist(),
                           **{c: np.cumsum(sims[c][1][m])[::step].round(2).tolist() for c in CANDIDATES})

    # Monthly block bootstrap of candidate minus base on the holdout
    rng = np.random.default_rng(7)
    mon = pd.Series(np.arange(len(dates)), index=dates)[m].groupby([dates[m].year, dates[m].month]).apply(list).tolist()
    boot = {}
    for c_ in CANDIDATES:
        b, s_ = sims["base"][1], sims[c_][1]
        res = []
        for _ in range(2000):
            idx = np.concatenate([mon[i] for i in rng.integers(0, len(mon), len(mon))])
            res.append((s_[idx].sum() - b[idx].sum(), maxdd(s_[idx]) - maxdd(b[idx])))
        res = np.array(res)
        boot[c_] = dict(total=dict(p05=np.percentile(res[:, 0], 5), med=np.median(res[:, 0]), p95=np.percentile(res[:, 0], 95),
                                   share_better=float((res[:, 0] > 0).mean()),
                                   hist=np.histogram(res[:, 0], 30)[0].tolist(), edges=np.histogram(res[:, 0], 30)[1].round(2).tolist()),
                        dd=dict(p05=np.percentile(res[:, 1], 5), med=np.median(res[:, 1]), p95=np.percentile(res[:, 1], 95),
                                share_better=float((res[:, 1] > 0).mean())))
    out["bootstrap"] = boot

    # Year by year and stock by stock: how often does each candidate beat base?
    yr = {}
    for c_ in CANDIDATES:
        d = pd.Series(sims[c_][1] - sims["base"][1], index=dates)[m]
        g = d.groupby(d.index.year).sum()
        yr[c_] = dict(years=g.index.tolist(), diff=g.round(2).tolist(), share_better=float((g > 0).mean()))
        tb, tc = sims["base"][0], sims[c_][0]
        hb = tb[~tb.entry.dt.year.between(2012, 2018)]; hc = tc[~tc.entry.dt.year.between(2012, 2018)]
        per = (hc.groupby("symbol").ret.mean() - hb.groupby("symbol").ret.mean()).dropna()
        yr[c_]["stock_share_better"] = float((per > 0).mean())
    out["by_year"] = yr

    # The 98-stock scanner list, holdout
    cols = [i for i, s_ in enumerate(A["syms"]) if s_ in scanner_syms]
    sub = []
    for v in ["base"] + CANDIDATES:
        k, p = next((k, p) for k, p in VARIANTS if vname(k, p) == v)
        t, pnl, no = simulate(A, k, p, cols=cols)
        for pname, (s, e) in HOLDOUT.items():
            sub.append(dict(period=pname, variant=v, **summarize(span(t, s, e)), **curve(pnl, no, dates, s, e)))
    out["scanner_list"] = sub

    # Index trades (SPY, Nasdaq-100) with tighter stops, all years outside EXPLORE
    idx_rows = []
    for k, p in [("none", None), ("pct", 0.03), ("pct", 0.05), ("pct", 0.08), ("pct", 0.10), ("time", 10)]:
        t, _, _ = simulate(IDX, k, p)
        h = t[~t.entry.dt.year.between(2012, 2018)]
        for sym, g in h.groupby("symbol"):
            idx_rows.append(dict(symbol=sym, variant=vname(k, p), **summarize(g)))
    out["index"] = idx_rows
    json.dump(out, open(f"{base_dir}/test.json", "w"), default=float)

    pd.set_option("display.width", 250)
    fmt = lambda df: df.assign(**{k: (df[k] * 100).round(2) for k in ("win", "exp", "avg_loss", "p5", "worst", "lt10", "stopped") if k in df})
    cols_ = ["variant", "trades", "win", "exp", "avg_loss", "p5", "worst", "lt10", "stopped", "total", "max_dd", "worst_month", "mar"]
    for pname in HOLDOUT:
        print(f"\n== HOLDOUT {pname}: all S&P 500 stocks ==")
        print(fmt(R[R.period == pname])[cols_].round(2).to_string(index=False))
    print("\n== Walk-forward picks (holdout years) ==")
    Wh = W[W.holdout]
    print(Wh.pick.value_counts().to_string())
    print(f"WF total {Wh.wf.sum():.1f} vs base {Wh.base.sum():.1f}; years WF beat base {(Wh.wf > Wh.base).mean():.0%}")
    print("\n== Bootstrap (holdout, candidate minus base) ==")
    for c_, b in boot.items():
        print(c_, "total P&L diff 5/50/95%:", [round(b["total"][q], 1) for q in ("p05", "med", "p95")],
              f"P(better) {b['total']['share_better']:.0%}", "| maxDD diff:", [round(b["dd"][q], 1) for q in ("p05", "med", "p95")],
              f"P(shallower) {b['dd']['share_better']:.0%}")
    for c_, y_ in yr.items():
        print(f"{c_}: beats base in {y_['share_better']:.0%} of holdout years, {y_['stock_share_better']:.0%} of stocks")
    print("\n== Scanner list (98 stocks), holdout ==")
    print(fmt(pd.DataFrame(sub))[["period"] + cols_].round(2).to_string(index=False))
    print("\n== Index trades outside EXPLORE ==")
    print(fmt(pd.DataFrame(idx_rows))[["symbol", "variant", "trades", "win", "exp", "avg_loss", "worst", "stopped"]].round(2).to_string(index=False))

if __name__ == "__main__":
    base_dir, mode = sys.argv[1], sys.argv[2]
    data = pickle.load(open(f"{base_dir}/sp500.pkl", "rb"))
    A = build(data["prices"])
    if mode == "explore":
        explore(A, base_dir)
    else:
        st = pd.read_csv(__file__.replace("stop_study.py", "stock_rsi2_per_symbol.csv"), index_col=0)
        scanner_syms = set(st[(st.trades >= 30) & (st.expectancy >= 0.007)].index)
        ix = pickle.load(open(f"{base_dir}/index.pkl", "rb"))
        IDX = build({"SPY": ix["SPY"], "NDX": ix["^NDX"]})
        test(A, base_dir, scanner_syms, IDX)
