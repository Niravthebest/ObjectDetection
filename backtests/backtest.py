"""Backtest RSI(2) Connors mean reversion and MA-pullback strategies on index data.

Data: Yahoo Finance chart API JSON files (daily). Signals on the close, filled at
the same close (classic Connors convention) and, as a robustness check, at the
next day's open. Costs: COST_BPS per side.
"""
import json, sys, numpy as np, pandas as pd

COST_BPS = 2  # per side, applied to returns

def load(path):
    r = json.load(open(path))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s").normalize())
    if "adjclose" in r["indicators"]:  # total-return adjust OHLC for ETFs
        f = np.array(r["indicators"]["adjclose"][0]["adjclose"], dtype=float) / df["close"].values
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].mul(f, axis=0)
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df[(df["open"] > 0) & (df["low"] > 0)]
    return df

def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def run(df, entry_sig, exit_fn, fill="close", max_hold=None):
    """entry_sig: bool Series evaluated at close. exit_fn(i, entry_i) -> bool at close i."""
    o, c = df["open"].values, df["close"].values
    trades, pos, ei, ep = [], False, None, None
    daily = np.zeros(len(df))  # strategy daily returns
    sig = entry_sig.values
    for i in range(1, len(df)):
        if pos:
            prev = c[i - 1] if i - 1 >= ei else ep
            daily[i] = c[i] / prev - 1
            if exit_fn(i, ei) or (max_hold and i - ei >= max_hold):
                daily[i] -= COST_BPS / 1e4
                trades.append((df.index[ei], df.index[i], c[i] / ep - 1 - 2 * COST_BPS / 1e4, i - ei))
                pos = False
            continue
        if fill == "close" and sig[i]:
            pos, ei, ep = True, i, c[i]
            daily[i] -= COST_BPS / 1e4
        elif fill == "open" and sig[i - 1]:
            pos, ei, ep = True, i, o[i]
            daily[i] = c[i] / o[i] - 1 - COST_BPS / 1e4
            if exit_fn(i, ei):
                trades.append((df.index[ei], df.index[i], c[i] / ep - 1 - 2 * COST_BPS / 1e4, 0))
                pos = False
    return pd.Series(daily, index=df.index), pd.DataFrame(trades, columns=["entry", "exit", "ret", "days"])

def stats(name, daily, trades, df):
    eq = (1 + daily).cumprod()
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    out = dict(strategy=name, cagr=cagr, max_dd=dd, sharpe=sharpe, trades=len(trades))
    if len(trades):
        w, l = trades.ret[trades.ret > 0], trades.ret[trades.ret <= 0]
        out.update(win_rate=len(w) / len(trades), avg_trade=trades.ret.mean(),
                   avg_win=w.mean() if len(w) else 0, avg_loss=l.mean() if len(l) else 0,
                   worst_trade=trades.ret.min(),
                   profit_factor=w.sum() / -l.sum() if len(l) and l.sum() else np.inf,
                   avg_days=trades.days.mean(),
                   exposure=(trades.days + 1).sum() / len(df), trades_per_yr=len(trades) / yrs)
    return out

def backtest(df, label, start=None):
    df = df.copy()
    c = df["close"]
    df["sma5"], df["sma20"], df["sma50"], df["sma200"] = (c.rolling(n).mean() for n in (5, 20, 50, 200))
    df["rsi2"] = rsi(c, 2)
    if start:
        df = df[df.index >= start]
    df = df.dropna()
    c, lo = df["close"].values, df["low"].values
    s5, s20, s50, s200 = (df[k].values for k in ("sma5", "sma20", "sma50", "sma200"))
    hi10 = df["close"].shift(1).rolling(10).max().values
    uptrend = (df.close > df.sma200) & (df.sma50 > df.sma200)
    rows = []

    bh = df["close"].pct_change().fillna(0)
    rows.append(stats("Buy & hold", bh, pd.DataFrame(columns=["ret", "days"]), df) | {"exposure": 1.0})

    # 1. Connors RSI(2): RSI2<10 & close>SMA200, exit close>SMA5
    sig = (df.rsi2 < 10) & (df.close > df.sma200)
    ex = lambda i, e: c[i] > s5[i]
    for fill in ("close", "open"):
        d, t = run(df, sig, ex, fill)
        rows.append(stats(f"RSI(2)<10 >200MA, exit >5MA [{fill} fill]", d, t, df))
    # Sensitivity: RSI(2)<5 and no trend filter
    d, t = run(df, (df.rsi2 < 5) & (df.close > df.sma200), ex)
    rows.append(stats("RSI(2)<5 >200MA (variant)", d, t, df))
    d, t = run(df, df.rsi2 < 10, ex)
    rows.append(stats("RSI(2)<10, NO 200MA filter", d, t, df))

    # 2a. Pullback to 20MA in uptrend: low touches 20MA, close holds 50MA,
    #     was >2% above 20MA within prior 10 days. Exit new 10-day closing high,
    #     stop on close < 50MA, 30-day time stop.
    stretched = (df.close > df.sma20 * 1.02).shift(1).rolling(10).max().astype(bool)
    sig = uptrend & (df.low <= df.sma20) & (df.close > df.sma50) & stretched
    ex = lambda i, e: i > e and (c[i] > hi10[i] or c[i] < s50[i])
    d, t = run(df, sig, ex, max_hold=30)
    rows.append(stats("Dip to 20MA in uptrend", d, t, df))

    # 2b. Pullback to 50MA in uptrend: low touches 50MA, close > 200MA,
    #     was >3% above 50MA within prior 20 days. Exit new 10-day closing high,
    #     stop on close 3% below 50MA, 40-day time stop.
    stretched = (df.close > df.sma50 * 1.03).shift(1).rolling(20).max().astype(bool)
    sig = (df.close > df.sma200) & (df.low <= df.sma50) & (df.close > df.sma200) & stretched
    ex = lambda i, e: i > e and (c[i] > hi10[i] or c[i] < s50[i] * 0.97)
    d, t = run(df, sig, ex, max_hold=40)
    rows.append(stats("Dip to 50MA in uptrend", d, t, df))

    res = pd.DataFrame(rows)
    res.insert(0, "market", f"{label} ({df.index[0]:%Y}-{df.index[-1]:%Y})")
    return res

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    sets = [("SPY", "SPY.json", None), ("S&P 500 index", "%5EGSPC.json", "1960-01-01"),
            ("NYSE U.S. 100 index", "%5ENY.json", None), ("Nasdaq-100 index", "%5ENDX.json", None)]
    allres = pd.concat([backtest(load(f"{base}/{f}"), lbl, s) for lbl, f, s in sets])
    allres.to_csv(f"{base}/results.csv", index=False)
    pct = ["cagr", "max_dd", "win_rate", "avg_trade", "avg_win", "avg_loss", "worst_trade", "exposure"]
    show = allres.copy()
    for k in pct:
        show[k] = (show[k] * 100).round(2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(show.round(2).to_string(index=False))
