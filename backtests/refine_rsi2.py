"""Refine the RSI(2) strategy with an in-sample / out-of-sample split.

Grid-searches entry threshold, entry type, exit rule and time stop on SPY
1993-2012 (in-sample), picks by Sharpe, then reports untouched
out-of-sample results on SPY 2013-2026, the S&P 500 index 1970-1992 and
the NYSE U.S. 100 index 2004-2026. Also tests a VIX filter and earning
T-bill interest while flat.
"""
import itertools, sys
import numpy as np, pandas as pd
from backtest import load, rsi, run, stats

def prep(df):
    df = df.copy()
    c = df["close"]
    for n in (5, 10, 200):
        df[f"sma{n}"] = c.rolling(n).mean()
    df["rsi2"] = rsi(c, 2)
    df["cum_rsi2"] = df["rsi2"] + df["rsi2"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    return df.dropna()

ENTRIES = {f"RSI2<{x}": ("rsi2", x) for x in (5, 10, 15, 20)} | \
          {f"2-day cum RSI2<{x}": ("cum_rsi2", x) for x in (20, 35, 50)}
EXITS = ["close>SMA5", "close>SMA10", "RSI2>50", "RSI2>70", "close>prior high", "first up close"]
HOLDS = [None, 5, 10]

def exit_rule(df, name):
    c = df["close"].values
    if name == "close>SMA5":       a = df["sma5"].values;      return lambda i, e: c[i] > a[i]
    if name == "close>SMA10":      a = df["sma10"].values;     return lambda i, e: c[i] > a[i]
    if name == "RSI2>50":          a = df["rsi2"].values;      return lambda i, e: a[i] > 50
    if name == "RSI2>70":          a = df["rsi2"].values;      return lambda i, e: a[i] > 70
    if name == "close>prior high": a = df["prev_high"].values; return lambda i, e: c[i] > a[i]
    if name == "first up close":   return lambda i, e: i > 0 and c[i] > c[i - 1]

def strat(df, entry, ex, hold, extra=None, cash=None):
    col, x = ENTRIES[entry]
    sig = (df[col] < x) & (df["close"] > df["sma200"])
    if extra is not None:
        sig &= extra.reindex(df.index).fillna(False)
    d, t = run(df, sig, exit_rule(df, ex), max_hold=hold)
    if cash is not None:  # earn T-bill yield on days with no position open at prior close
        inpos = pd.Series(False, index=df.index)
        for _, r in t.iterrows():
            inpos[(df.index > r.entry) & (df.index <= r.exit)] = True
        d = d + np.where(inpos, 0, cash.reindex(df.index).ffill().fillna(0).values)
    return d, t

def evaluate(df, entry, ex, hold, **kw):
    d, t = strat(df, entry, ex, hold, **kw)
    return stats(f"{entry} | exit {ex} | max hold {hold or '-'}", d, t, df)

def fmt(rows):
    r = pd.DataFrame(rows)
    for k in ("cagr", "max_dd", "win_rate", "avg_trade", "worst_trade", "exposure"):
        r[k] = (r[k] * 100).round(2)
    cols = ["strategy", "cagr", "max_dd", "sharpe", "trades", "win_rate", "avg_trade",
            "worst_trade", "profit_factor", "avg_days", "exposure"]
    return r[cols].round(2).to_string(index=False)

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    pd.set_option("display.width", 250)
    spy = prep(load(f"{base}/SPY.json"))
    gspc = prep(load(f"{base}/%5EGSPC.json"))
    ny = prep(load(f"{base}/%5ENY.json"))
    vix = load(f"{base}/%5EVIX.json")["close"]
    tbill = load(f"{base}/%5EIRX.json")["close"] / 100 / 252

    ins = spy[spy.index < "2013-01-01"]
    tests = {"SPY 2013-2026 (OOS)": spy[spy.index >= "2013-01-01"],
             "S&P 500 1970-1992 (OOS)": gspc[(gspc.index >= "1970-01-01") & (gspc.index < "1993-01-01")],
             "NYSE U.S. 100 2004-2026 (OOS)": ny}

    grid = []
    for entry, ex, hold in itertools.product(ENTRIES, EXITS, HOLDS):
        s = evaluate(ins, entry, ex, hold)
        grid.append(s | dict(entry=entry, ex=ex, hold=hold))
    g = pd.DataFrame(grid)
    g = g[g.trades >= 80].sort_values("sharpe", ascending=False)
    g.to_csv(f"{base}/rsi2_grid_insample.csv", index=False)
    print("== In-sample SPY 1993-2012: top 12 of", len(grid), "by Sharpe (>=80 trades) ==")
    print(fmt(g.head(12).to_dict("records")))

    # Robustness: how does the neighbourhood of each parameter do (median Sharpe)?
    print("\n== Median in-sample Sharpe by parameter ==")
    for k in ("entry", "ex", "hold"):
        print(g.groupby(k, dropna=False)["sharpe"].median().sort_values(ascending=False).round(2).to_string(), "\n")

    best = g.iloc[0]
    picks = [("Original", "RSI2<10", "close>SMA5", None),
             ("Best in-sample", best.entry, best.ex, None if pd.isna(best.hold) else int(best.hold))]
    vix_up = vix > vix.rolling(10).mean()
    for name, df in [("SPY 1993-2012 (IS)", ins)] + list(tests.items()):
        rows = []
        for lbl, e, x, h in picks:
            rows.append(evaluate(df, e, x, h) | {"strategy": f"{lbl}: {e} / {x} / hold {h or '-'}"})
        e, x, h = picks[1][1:]
        rows.append(evaluate(df, e, x, h, extra=vix_up) | {"strategy": "Best + VIX above its 10d avg"})
        rows.append(evaluate(df, e, x, h, cash=tbill) | {"strategy": "Best + T-bill yield while flat"})
        bh = df["close"].pct_change().fillna(0)
        rows.append(stats("Buy & hold", bh, pd.DataFrame(columns=["ret", "days"]), df)
                    | {"exposure": 1.0, "win_rate": np.nan, "avg_trade": np.nan, "worst_trade": np.nan,
                       "profit_factor": np.nan, "avg_days": np.nan})
        print(f"\n== {name} ==")
        print(fmt(rows))
