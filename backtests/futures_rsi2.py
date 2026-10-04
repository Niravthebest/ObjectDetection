"""Dip-200 / RSI(2) rules on Yahoo continuous front-month futures.

Buy at close when RSI(2) < 10 and close > 200-day SMA; sell at close when
close > 5-day SMA. Returns are % of contract price (unlevered), 2 bps/side.
Develop window 2012-2018, test window 2022-2026, plus full history.
Caveat: Yahoo's continuous contracts are NOT back-adjusted, so contract rolls
show up as price jumps (worst for energy, grains and livestock).
"""
import sys
import numpy as np, pandas as pd
import backtest
from backtest import load, rsi, run
from stocks_rsi2 import summarize

backtest.COST_BPS = 2
SECTORS = {
    "Equity index": ["ES", "NQ", "YM", "RTY"], "Energy": ["CL", "BZ", "NG", "RB", "HO"],
    "Metals": ["GC", "SI", "HG", "PL", "PA"], "Rates": ["ZT", "ZF", "ZN", "ZB"],
    "Grains": ["ZC", "ZS", "ZW", "ZL", "ZM"], "Softs": ["KC", "SB", "CC", "CT"],
    "Livestock": ["LE", "HE", "GF"], "Currencies": ["6E", "6J", "6B", "6A", "6C", "6S"]}
WINDOWS = {"full": (None, None), "dev 2012-18": ("2012", "2018"), "test 2022-26": ("2022", "2026")}

def trades(df):
    c = df["close"]
    df = df.assign(sma5=c.rolling(5).mean(), sma200=c.rolling(200).mean(), rsi2=rsi(c, 2)).dropna()
    cv, s5 = df.close.values, df.sma5.values
    d, t = run(df, (df.rsi2 < 10) & (df.close > df.sma200), lambda i, e: cv[i] > s5[i])
    return d, t

if __name__ == "__main__":
    base = sys.argv[1]
    rows, allt = [], []
    for sector, syms in SECTORS.items():
        for s in syms:
            try:
                df = load(f"{base}/fut/{s}.json")
            except Exception:
                continue
            daily, t = trades(df)
            t["symbol"], t["sector"] = s, sector
            allt.append(t)
            row = dict(sector=sector, symbol=s, since=df.index[0].year)
            for w, (a, b) in WINDOWS.items():
                tt = t if a is None else t[(t.exit >= a) & (t.exit <= f"{b}-12-31")]
                dd = daily if a is None else daily.loc[a:b]
                if len(tt) < 5:
                    continue
                st = summarize(tt)
                yrs = (dd.index[-1] - dd.index[0]).days / 365.25
                row |= {f"{w}|trades": int(st.trades), f"{w}|win": st.win_rate, f"{w}|exp": st.expectancy,
                        f"{w}|avg_win": st.avg_win, f"{w}|avg_loss": st.avg_loss, f"{w}|pf": st.profit_factor,
                        f"{w}|worst": st.worst, f"{w}|cagr": (1 + dd).prod() ** (1 / yrs) - 1}
            rows.append(row)
    R = pd.DataFrame(rows).set_index(["sector", "symbol"])
    R.to_csv(f"{base}/futures_rsi2.csv")
    T = pd.concat(allt)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for w in WINDOWS:
        cols = [c for c in R.columns if c.startswith(w + "|")]
        x = R[cols].copy(); x.columns = [c.split("|")[1] for c in cols]
        for k in ("win", "exp", "avg_win", "avg_loss", "worst", "cagr"):
            x[k] = (x[k] * 100).round(2)
        print(f"\n== {w} ==\n" + x.round(2).to_string())
    print("\n== By sector (pooled trades) ==")
    out = {}
    for sec, g in T.groupby("sector"):
        for w, (a, b) in WINDOWS.items():
            gg = g if a is None else g[(g.exit >= a) & (g.exit <= f"{b}-12-31")]
            if len(gg) >= 5:
                st = summarize(gg)
                out[(sec, w)] = dict(trades=int(st.trades), win=round(st.win_rate * 100, 1),
                                     exp=round(st.expectancy * 100, 2), pf=round(st.profit_factor, 2))
    print(pd.DataFrame(out).T.unstack(1).to_string())
