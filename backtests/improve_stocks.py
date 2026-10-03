"""Test a market-regime filter and catastrophe stops on the stock-level RSI(2)
strategy. Develop on 2012-2018, test untouched on 2022-2026.

Base rules: buy at close when RSI(2) < 10 and close > 200-day SMA; sell at
close when close > 5-day SMA. Variants:
  regime : only open new trades when the S&P 500 index closes above its 200-day SMA
  stop   : sell at the close if it is X% or more below the entry price
Portfolio: up to 10 positions, 10% each, lowest RSI(2) first, 5 bps per side.
"""
import itertools, pickle, sys
import numpy as np, pandas as pd
from backtest import load, rsi
from stocks_rsi2 import summarize

COST = 5 / 1e4
SLOTS = 10
STOPS = [None, 0.08, 0.10, 0.15]
PERIODS = {"DEV 2012-2018": ("2012-01-01", "2018-12-31"), "TEST 2022-2026": ("2022-01-01", "2026-12-31")}

def build_panel(prices):
    p = {k: {} for k in ("close", "rsi2", "sma5", "sma200")}
    for sym, df in prices.items():
        c = df["close"].astype(float)
        p["close"][sym], p["rsi2"][sym] = c, rsi(c, 2)
        p["sma5"][sym], p["sma200"][sym] = c.rolling(5).mean(), c.rolling(200).mean()
    return {k: pd.DataFrame(v).sort_index() for k, v in p.items()}

def simulate(P, regime, stop, start, end, slots=SLOTS):
    """Event loop over days. Returns (equity curve, trades). slots=None -> unlimited
    (every signal traded, equal size) for per-trade statistics."""
    idx = P["close"].loc[start:end].index
    C, R, S5, S200 = (P[k].loc[idx].values for k in ("close", "rsi2", "sma5", "sma200"))
    reg = regime.reindex(idx).ffill().fillna(False).values
    syms = P["close"].columns
    cash, pos, eq, trades = 1.0, {}, [], []
    for d in range(len(idx)):
        px = C[d]
        for j in list(pos):
            if np.isnan(px[j]):
                continue
            sh, ep, ed = pos[j]
            if px[j] > S5[d, j] or (stop and px[j] <= ep * (1 - stop)):
                cash += sh * px[j] * (1 - COST)
                trades.append((syms[j], idx[ed], idx[d], px[j] / ep * (1 - COST) ** 2 - 1, d - ed))
                del pos[j]
        equity = cash + sum(sh * px[j] for j, (sh, _, _) in pos.items() if not np.isnan(px[j]))
        if reg[d]:
            ok = (R[d] < 10) & (px > S200[d])
            cand = [j for j in np.argsort(R[d]) if ok[j] and j not in pos]
            if slots is not None:
                cand = cand[:slots - len(pos)]
            for j in cand:
                alloc = equity / SLOTS if slots is not None else 0
                pos[j] = (alloc * (1 - COST) / px[j], px[j], d)
                cash -= alloc
        eq.append(cash + sum(sh * px[j] for j, (sh, _, _) in pos.items() if not np.isnan(px[j])))
    t = pd.DataFrame(trades, columns=["symbol", "entry", "exit", "ret", "days"])
    return pd.Series(eq, index=idx), t

def curve_stats(e):
    d = e.pct_change().fillna(0)
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return dict(cagr=e.iloc[-1] ** (1 / yrs) - 1, max_dd=(e / e.cummax() - 1).min(),
                sharpe=d.mean() / d.std() * np.sqrt(252))

if __name__ == "__main__":
    base = sys.argv[1]
    data = pickle.load(open(f"{base}/sp500.pkl", "rb"))
    P = build_panel(data["prices"])
    spx = load(f"{base}/%5EGSPC.json")["close"]
    regime_on = spx > spx.rolling(200).mean()
    regime_off = pd.Series(True, index=spx.index)
    spy = load(f"{base}/SPY.json")["close"]

    rows = []
    for (rname, reg), stop in itertools.product([("off", regime_off), ("on", regime_on)], STOPS):
        for pname, (s, e) in PERIODS.items():
            _, t = simulate(P, reg, stop, s, e, slots=None)        # every signal
            eqc, _ = simulate(P, reg, stop, s, e)                   # 10-slot portfolio
            ts, cs = summarize(t), curve_stats(eqc)
            rows.append(dict(period=pname, regime=rname, stop=f"{stop:.0%}" if stop else "none",
                             trades=int(ts.trades), win=ts.win_rate, exp=ts.expectancy, avg_win=ts.avg_win,
                             avg_loss=ts.avg_loss, pf=ts.profit_factor, worst=ts.worst,
                             p_cagr=cs["cagr"], p_maxdd=cs["max_dd"], p_sharpe=cs["sharpe"]))
    r = pd.DataFrame(rows)
    r.to_csv(f"{base}/improve_stocks.csv", index=False)
    show = r.copy()
    for k in ("win", "exp", "avg_win", "avg_loss", "worst", "p_cagr", "p_maxdd"):
        show[k] = (show[k] * 100).round(2)
    pd.set_option("display.width", 250)
    for pname in PERIODS:
        print(f"\n== {pname} ==")
        print(show[show.period == pname].drop(columns="period").round(2).to_string(index=False))
    for pname, (s, e) in PERIODS.items():
        x = spy.loc[s:e]; yrs = (x.index[-1] - x.index[0]).days / 365.25
        print(f"SPY buy & hold {pname}: CAGR {(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1:.1%}, maxDD {(x / x.cummax() - 1).min():.1%}")
    print("\nTEST by calendar year (portfolio return):")
    for rname, reg, stop in [("off", regime_off, None), ("on", regime_on, None), ("on", regime_on, 0.10), ("on", regime_on, 0.15)]:
        eqc, _ = simulate(P, reg, stop, *PERIODS["TEST 2022-2026"])
        yr = eqc.groupby(eqc.index.year).last() / eqc.groupby(eqc.index.year).first() - 1
        print(f"regime {rname:3} stop {str(stop):5}: " + "  ".join(f"{y}: {v:+.1%}" for y, v in yr.items()))
    x = spy.loc["2022":]; yr = x.groupby(x.index.year).last() / x.groupby(x.index.year).first() - 1
    print("SPY buy & hold      : " + "  ".join(f"{y}: {v:+.1%}" for y, v in yr.items()))
