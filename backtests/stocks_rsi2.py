"""RSI(2) strategy on every current S&P 500 stock: per-trade statistics,
per-stock / per-sector / per-year breakdowns, and a 10-slot portfolio.

Rules: buy at close when RSI(2) < 10 and close > 200-day SMA; sell at close
when close > 5-day SMA. Cost 5 bps per side.
Caveat: current constituents only (survivorship bias flatters results).
"""
import pickle, sys
import numpy as np, pandas as pd
import backtest
from backtest import rsi, run

backtest.COST_BPS = 5
SLOTS = 10

def trades_for(df):
    df = df.copy()
    c = df["close"].astype(float)
    df["close"], df["open"] = c, df["open"].astype(float)
    df["sma5"], df["sma200"], df["rsi2"] = c.rolling(5).mean(), c.rolling(200).mean(), rsi(c, 2)
    df = df.dropna()
    if len(df) < 50:
        return df, pd.DataFrame()
    s5, cv = df.sma5.values, df.close.values
    _, t = run(df, (df.rsi2 < 10) & (df.close > df.sma200), lambda i, e: cv[i] > s5[i])
    return df, t

def summarize(t):
    w, l = t.ret[t.ret > 0], t.ret[t.ret <= 0]
    return pd.Series(dict(
        trades=len(t), win_rate=len(w) / len(t), expectancy=t.ret.mean(), median_trade=t.ret.median(),
        avg_win=w.mean(), avg_loss=l.mean(), payoff=w.mean() / -l.mean(),
        profit_factor=w.sum() / -l.sum(), avg_days=t.days.mean(),
        p5=t.ret.quantile(.05), worst=t.ret.min(), pct_below_minus10=(t.ret < -.10).mean()))

def portfolio(panel, start, end):
    """Up to SLOTS positions, each 1/SLOTS of equity at entry; lowest RSI(2) first."""
    close, rsi2, s5, s200 = (panel[k].loc[start:end] for k in ("close", "rsi2", "sma5", "sma200"))
    cash, pos, eq = 1.0, {}, []
    cost = backtest.COST_BPS / 1e4
    for dt in close.index:
        px = close.loc[dt]
        for s in list(pos):
            if not np.isnan(px[s]) and px[s] > s5.loc[dt, s]:
                cash += pos.pop(s) * px[s] * (1 - cost)
        held = sum(sh * (px[s] if not np.isnan(px[s]) else 0) for s, sh in pos.items())
        equity = cash + held
        free = SLOTS - len(pos)
        if free > 0:
            sig = rsi2.loc[dt][(rsi2.loc[dt] < 10) & (px > s200.loc[dt])].drop(list(pos), errors="ignore")
            for s in sig.sort_values().index[:free]:
                alloc = min(equity / SLOTS, cash)
                pos[s] = alloc * (1 - cost) / px[s]
                cash -= alloc
        eq.append(cash + sum(sh * px[s] for s, sh in pos.items() if not np.isnan(px[s])))
    e = pd.Series(eq, index=close.index)
    d = e.pct_change().fillna(0)
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return e, dict(cagr=e.iloc[-1] ** (1 / yrs) - 1, max_dd=(e / e.cummax() - 1).min(),
                   sharpe=d.mean() / d.std() * np.sqrt(252))

def pct(df, cols):
    df = df.copy()
    for k in cols:
        if k in df:
            df[k] = (df[k] * 100).round(2)
    return df.round(2)

PCT = ["win_rate", "expectancy", "median_trade", "avg_win", "avg_loss", "avg_days_", "p5", "worst",
       "pct_below_minus10", "cagr", "max_dd"]

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
    T["year"] = T.exit.dt.year
    T.to_csv(f"{base}/stock_trades.csv", index=False)
    panel = {k: pd.DataFrame(v).sort_index() for k, v in panel.items()}
    pd.set_option("display.width", 250)

    print(f"== All trades, {T.symbol.nunique()} stocks, {T.entry.min():%Y}-{T.exit.max():%Y} ==")
    periods = {"All (1991-2026)": T, "1991-2009": T[T.year <= 2009], "2010-2019": T[(T.year >= 2010) & (T.year <= 2019)],
               "2020-2026": T[T.year >= 2020], "2021-2026": T[T.year >= 2021]}
    print(pct(pd.DataFrame({k: summarize(v) for k, v in periods.items()}).T, PCT).to_string())

    per = T.groupby("symbol").apply(summarize)
    per = per[per.trades >= 30]
    print(f"\n== Per stock ({len(per)} stocks with >=30 trades) ==")
    print(f"positive expectancy: {(per.expectancy > 0).mean():.0%}   win rate >60%: {(per.win_rate > .6).mean():.0%}   "
          f"profit factor >1.5: {(per.profit_factor > 1.5).mean():.0%}")
    dist = per[["win_rate", "expectancy", "profit_factor"]].quantile([.1, .25, .5, .75, .9]).T
    dist.loc[["win_rate", "expectancy"]] *= 100
    print("distribution across stocks (win rate and expectancy in %):\n" + dist.round(2).to_string())
    per.to_csv(f"{base}/stock_rsi2_per_symbol.csv")
    cols = ["trades", "win_rate", "expectancy", "avg_win", "avg_loss", "profit_factor", "worst"]
    print("\nTop 15 by expectancy:\n" + pct(per.sort_values("expectancy", ascending=False).head(15)[cols], PCT).to_string())
    print("\nBottom 15 by expectancy:\n" + pct(per.sort_values("expectancy").head(15)[cols], PCT).to_string())

    yr = T.groupby("year").apply(summarize)[["trades", "win_rate", "expectancy", "profit_factor"]]
    print("\n== By exit year ==\n" + pct(yr, PCT).to_string())

    print(f"\n== Portfolio: up to {SLOTS} positions, 10% each, lowest RSI(2) first ==")
    for s, e in [("1995", "2026"), ("2010", "2026"), ("2021", "2026")]:
        eqc, st = portfolio(panel, s, e)
        ew = panel["close"].loc[s:e].pct_change().mean(axis=1).fillna(0)
        ewe = (1 + ew).cumprod(); yrs = (ewe.index[-1] - ewe.index[0]).days / 365.25
        print(f"{s}-{e}: RSI(2) portfolio CAGR {st['cagr']:.1%}, maxDD {st['max_dd']:.1%}, Sharpe {st['sharpe']:.2f} | "
              f"equal-weight buy&hold CAGR {ewe.iloc[-1] ** (1 / yrs) - 1:.1%}, maxDD {(ewe / ewe.cummax() - 1).min():.1%}")
