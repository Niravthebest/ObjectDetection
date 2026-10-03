"""Out-of-sample test of the scanner's stock filter (expectancy >= 0.7%/trade,
>= 30 trades), with no look-ahead:

1. Single split: pick stocks on trades that exited 1991-2015, trade 2016-2026.
2. Walk-forward: each year, pick stocks on all trades that exited before that
   year began, trade that year only.

Reports per-trade stats and a 10-slot portfolio vs. trading all stocks and
vs. SPY buy & hold. Survivorship bias (current members only) still applies.
"""
import pickle, sys
import numpy as np, pandas as pd
from backtest import load
from stocks_rsi2 import trades_for, summarize, portfolio

MIN_EXP, MIN_TRADES = 0.007, 30

def pick(T, before):
    t = T[T.exit < before]
    g = t.groupby("symbol").ret.agg(["mean", "size"])
    return set(g[(g["mean"] >= MIN_EXP) & (g["size"] >= MIN_TRADES)].index)

def masked_panel(panel, allowed_by_year):
    """Hide entry signals for stocks not selected in a given year."""
    p = dict(panel)
    rsi = panel["rsi2"].copy()
    for yr, allowed in allowed_by_year.items():
        rows = rsi.index.year == yr
        rsi.loc[rows, [c for c in rsi.columns if c not in allowed]] = np.nan
    p["rsi2"] = rsi
    return p

def fmt(s):
    return (f"trades {int(s.trades):>6}  win {s.win_rate:6.1%}  expectancy {s.expectancy:+.2%}  "
            f"avg win {s.avg_win:+.2%}  avg loss {s.avg_loss:+.2%}  PF {s.profit_factor:.2f}  worst {s.worst:+.1%}")

def port_line(name, panel, start, end):
    e, st = portfolio(panel, start, end)
    return f"{name:<38} CAGR {st['cagr']:6.1%}  maxDD {st['max_dd']:6.1%}  Sharpe {st['sharpe']:.2f}", e

if __name__ == "__main__":
    base = sys.argv[1]
    data = pickle.load(open(f"{base}/sp500.pkl", "rb"))
    allt, panel = [], {k: {} for k in ("close", "rsi2", "sma5", "sma200")}
    for sym, df in data["prices"].items():
        df, t = trades_for(df)
        for k in panel:
            panel[k][sym] = df[k]
        if len(t):
            t["symbol"] = sym
            allt.append(t)
    T = pd.concat(allt, ignore_index=True)
    panel = {k: pd.DataFrame(v).sort_index() for k, v in panel.items()}
    spy = load(f"{base}/SPY.json")["close"]

    def spy_line(start, end):
        s = spy.loc[start:end]; yrs = (s.index[-1] - s.index[0]).days / 365.25
        return f"{'SPY buy & hold':<38} CAGR {(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1:6.1%}  maxDD {(s / s.cummax() - 1).min():6.1%}"

    # 1. Single split
    sel = pick(T, "2016-01-01")
    oos = T[T.exit >= "2016-01-01"]
    print(f"== 1. Selected on 1991-2015 ({len(sel)} stocks), traded 2016-2026 ==")
    print("Selected stocks :", fmt(summarize(oos[oos.symbol.isin(sel)])))
    print("All stocks      :", fmt(summarize(oos)))
    print("Rejected stocks :", fmt(summarize(oos[~oos.symbol.isin(sel)])))
    yrs = range(2016, 2027)
    print(port_line("Portfolio, selected stocks (10 slots)", masked_panel(panel, {y: sel for y in yrs}), "2016", "2026")[0])
    print(port_line("Portfolio, all stocks (10 slots)", panel, "2016", "2026")[0])
    print(spy_line("2016", "2026"))

    # 2. Walk-forward, re-selected every January
    print("\n== 2. Walk-forward: re-select each January on all prior trades ==")
    rows, sels = [], {}
    for y in range(2001, 2027):
        sels[y] = pick(T, f"{y}-01-01")
        t = T[(T.exit.dt.year == y) & (T.entry >= f"{y}-01-01")]
        a, s = summarize(t), summarize(t[t.symbol.isin(sels[y])]) if t.symbol.isin(sels[y]).any() else None
        rows.append(dict(year=y, n_selected=len(sels[y]), sel_trades=int(s.trades) if s is not None else 0,
                         sel_win=s.win_rate if s is not None else np.nan, sel_exp=s.expectancy if s is not None else np.nan,
                         all_win=a.win_rate, all_exp=a.expectancy))
    r = pd.DataFrame(rows).set_index("year")
    wf = T[T.apply(lambda x: x.entry.year >= 2001 and x.entry.year == x.exit.year and x.symbol in sels[x.exit.year], axis=1)]
    for k in ("sel_win", "sel_exp", "all_win", "all_exp"):
        r[k] = (r[k] * 100).round(2)
    print(r.to_string())
    print("\nWalk-forward selected, 2001-2026:", fmt(summarize(wf)))
    print("Walk-forward selected, 2016-2026:", fmt(summarize(wf[wf.exit >= "2016"])))
    print("Walk-forward selected, 2021-2026:", fmt(summarize(wf[wf.exit >= "2021"])))
    wp = masked_panel(panel, sels)
    for s, e in [("2001", "2026"), ("2016", "2026"), ("2021", "2026")]:
        print(f"\n{s}-{e}:")
        print(port_line("Portfolio, walk-forward selection", wp, s, e)[0])
        print(port_line("Portfolio, all stocks", panel, s, e)[0])
        print(spy_line(s, e))
