"""
Kronos-style mean-reversion DCA backtest (US stocks, long only, no leverage).

Structure replicated from the Kronos marketing description:
  1. Mean-reversion entry  -> RSI(2) oversold (optionally only above 200-day SMA)
  2. Safety orders         -> add size at fixed % below first entry (capped ladder)
  3. Time / dynamic exit   -> take-profit on avg cost, or close > SMA(5), or max hold days
  4. Uncorrelation engine  -> max 15 names; new entries ranked by lowest correlation to book

Key output: REALIZED-only drawdown (what a closed-trade P&L display shows)
versus MARK-TO-MARKET drawdown (what your account actually goes through).

Usage:
  pip install yfinance pandas numpy matplotlib
  python kronos_style_backtest.py                 # real data via yfinance
  python kronos_style_backtest.py --synthetic     # offline smoke test
  python kronos_style_backtest.py --csv data/universe_ohlc.csv.gz   # offline real data
"""
import argparse
import numpy as np
import pandas as pd

# ------------------------------- parameters -------------------------------
P = dict(
    start="2021-01-04",
    end=None,                    # None = today
    capital=100_000.0,
    max_positions=15,
    base_pct=0.015,              # base order = 1.5% of equity
    so_drops=[0.04, 0.08, 0.13], # safety-order triggers below FIRST entry price
    so_mult=[1.0, 2.0, 3.0],     # safety-order size as multiple of base order
    take_profit=0.03,            # exit when price >= avg cost * (1+tp)
    dynamic_exit=True,           # also exit on close > SMA(5) if in profit
    max_hold=20,                 # time stop, trading days
    rsi_len=2, rsi_entry=10,
    trend_filter=True,           # only buy above 200-day SMA
    corr_lookback=60,
    commission_bps=1.0,          # slippage+fees per side, basis points
)

UNIVERSE = """AAPL MSFT AMZN GOOGL META NVDA TSLA BRK-B JPM V MA UNH HD PG JNJ XOM CVX
LLY ABBV MRK PFE KO PEP COST WMT MCD DIS NFLX ADBE CRM ORCL CSCO INTC AMD QCOM TXN
AVGO IBM BA CAT DE GE HON LMT RTX UPS UNP NKE SBUX LOW TGT BAC WFC GS MS C AXP BLK
SCHW T VZ CMCSA TMO ABT DHR MDT AMGN GILD BMY CVS LIN NEE DUK SO""".split()


# ------------------------------- data -------------------------------------
def load_real(tickers, start, end):
    import yfinance as yf
    warm = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    raw = yf.download(tickers + ["SPY"], start=warm, end=end,
                      auto_adjust=True, progress=False, group_by="column")
    o, h, l, c = (raw[k].ffill() for k in ("Open", "High", "Low", "Close"))
    return o, h, l, c


def load_csv(path, tickers, start, end):
    """Long-format CSV: Date,Symbol,Open,High,Low,Close[,Volume]."""
    df = pd.read_csv(path, parse_dates=["Date"])
    warm = pd.Timestamp(start) - pd.Timedelta(days=400)
    df = df[(df.Date >= warm) & ((end is None) | (df.Date <= pd.Timestamp(end or "2100")))]
    df = df[df.Symbol.isin(tickers + ["SPY"])]
    wide = {k: df.pivot(index="Date", columns="Symbol", values=k).ffill()
            for k in ("Open", "High", "Low", "Close")}
    missing = set(tickers + ["SPY"]) - set(wide["Close"].columns)
    if missing:
        print(f"warning: no data for {sorted(missing)}")
    return wide["Open"], wide["High"], wide["Low"], wide["Close"]


def load_synthetic(tickers, start, end, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(pd.Timestamp(start) - pd.Timedelta(days=400),
                         end or "2026-06-30")
    n, k = len(idx), len(tickers) + 1
    mkt = rng.normal(0.0004, 0.011, n)
    mkt[int(n * .35):int(n * .45)] -= 0.004          # a 2022-style bear leg
    rets = 0.8 * mkt[:, None] + rng.normal(0.0002, 0.015, (n, k))
    c = pd.DataFrame(100 * np.exp(np.cumsum(rets, 0)), idx, tickers + ["SPY"])
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, .004, c.shape))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, .006, c.shape)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, .006, c.shape)))
    return o, h, l, c


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# ------------------------------- engine -----------------------------------
def run(o, h, l, c, p):
    spy = c.pop("SPY"); o, h, l = o.drop(columns="SPY"), h.drop(columns="SPY"), l.drop(columns="SPY")
    r = rsi(c, p["rsi_len"])
    sma200, sma5 = c.rolling(200).mean(), c.rolling(5).mean()
    rets = c.pct_change()
    fee = p["commission_bps"] / 1e4

    dates = c.index[c.index >= p["start"]]
    cash = p["capital"]
    book = {}          # sym -> dict(shares, cost, first_px, next_so, days, base_usd)
    pending = []       # symbols to buy at next open
    trades, curve = [], []
    realized = 0.0

    for i, d in enumerate(dates):
        # ---- 1. fill pending entries at the open
        for s in pending:
            if s in book or len(book) >= p["max_positions"]:
                continue
            px = o.at[d, s]
            if not np.isfinite(px):
                continue
            eq = cash + sum(b["shares"] * c.at[dates[i - 1], k] for k, b in book.items())
            usd = min(eq * p["base_pct"], cash)
            if usd <= 0:
                continue
            sh = usd / px
            cash -= usd * (1 + fee)
            book[s] = dict(shares=sh, cost=usd, first_px=px, next_so=0,
                           days=0, base_usd=usd, entry=d, orders=1)
        pending = []

        # ---- 2. intraday: take-profit limits, safety-order limits
        for s in list(book):
            b = book[s]
            op, hi, lo = o.at[d, s], h.at[d, s], l.at[d, s]
            if not np.isfinite(op):
                continue
            avg = b["cost"] / b["shares"]
            tp = avg * (1 + p["take_profit"])
            if hi >= tp and b["days"] > 0:
                _close(book, s, max(op, tp), d, "take_profit", trades, fee)
                cash += trades[-1]["proceeds"]; realized += trades[-1]["pnl"]
                continue
            k = b["next_so"]
            if k < len(p["so_drops"]):
                trig = b["first_px"] * (1 - p["so_drops"][k])
                if lo <= trig:
                    fill = min(op, trig)
                    usd = min(b["base_usd"] * p["so_mult"][k], cash)
                    if usd > 0:
                        b["shares"] += usd / fill; b["cost"] += usd
                        cash -= usd * (1 + fee); b["orders"] += 1
                    b["next_so"] += 1

        # ---- 3. end of day: exits for tomorrow's open handled as close fills
        for s in list(book):
            b = book[s]; b["days"] += 1
            px = c.at[d, s]
            avg = b["cost"] / b["shares"]
            if b["days"] >= p["max_hold"]:
                _close(book, s, px, d, "time_stop", trades, fee)
            elif p["dynamic_exit"] and px > sma5.at[d, s] and px > avg:
                _close(book, s, px, d, "dynamic", trades, fee)
            else:
                continue
            cash += trades[-1]["proceeds"]; realized += trades[-1]["pnl"]

        # ---- 4. signals for tomorrow, ranked by low correlation to book
        slots = p["max_positions"] - len(book)
        if slots > 0 and i < len(dates) - 1:
            cand = [s for s in c.columns if s not in book
                    and r.at[d, s] < p["rsi_entry"]
                    and (not p["trend_filter"] or c.at[d, s] > sma200.at[d, s])]
            if cand:
                win = rets.loc[:d].tail(p["corr_lookback"])
                if book:
                    cm = win[cand + list(book)].corr()
                    score = {s: cm.loc[s, list(book)].mean() for s in cand}
                else:
                    score = {s: r.at[d, s] for s in cand}
                pending = sorted(cand, key=score.get)[:slots]

        mv = sum(b["shares"] * c.at[d, s] for s, b in book.items())
        curve.append(dict(date=d, mtm=cash + mv, realized=p["capital"] + realized,
                          exposure=mv / (cash + mv), n=len(book),
                          open_pnl=mv - sum(b["cost"] for b in book.values())))

    eq = pd.DataFrame(curve).set_index("date")
    eq["spy"] = p["capital"] * spy.reindex(eq.index) / spy.reindex(eq.index).iloc[0]
    return eq, pd.DataFrame(trades)


def _close(book, s, px, d, why, trades, fee):
    b = book.pop(s)
    proceeds = b["shares"] * px * (1 - fee)
    trades.append(dict(symbol=s, entry=b["entry"], exit=d, reason=why,
                       orders=b["orders"], cost=b["cost"], proceeds=proceeds,
                       pnl=proceeds - b["cost"], ret=proceeds / b["cost"] - 1,
                       days=b["days"]))


# ------------------------------- report -----------------------------------
def stats(series):
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / yrs) - 1
    dd = (series / series.cummax() - 1).min()
    dr = series.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else np.nan
    return cagr, dd, sharpe


def report(eq, tr, png="kronos_backtest.png"):
    print("\n=== Kronos-style DCA mean reversion ===")
    for name, col in [("Mark-to-market (real)", "mtm"),
                      ("Realized-only (closed P&L)", "realized"),
                      ("SPY buy & hold", "spy")]:
        cg, dd, sh = stats(eq[col])
        print(f"{name:28s} CAGR {cg:7.1%}   MaxDD {dd:7.1%}   Sharpe {sh:5.2f}")
    if len(tr):
        print(f"\nTrades {len(tr)} | win rate {(tr.pnl > 0).mean():.1%} | "
              f"avg win {tr.ret[tr.pnl > 0].mean():.2%} | avg loss {tr.ret[tr.pnl <= 0].mean():.2%}")
        print(f"Trades that used ALL safety orders: {(tr.orders == tr.orders.max()).mean():.1%}")
        print(f"Worst single trade: {tr.ret.min():.1%}   exit mix: "
              + ", ".join(f"{k} {v:.0%}" for k, v in tr.reason.value_counts(normalize=True).items()))
    print(f"Avg capital deployed {eq.exposure.mean():.1%} | peak {eq.exposure.max():.1%}")
    print(f"Worst open (unrealized) loss on book: ${eq.open_pnl.min():,.0f}")
    tr.to_csv("kronos_trades.csv", index=False)
    eq.to_csv("kronos_equity.csv")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 2, 1]))
    eq[["mtm", "realized", "spy"]].plot(ax=ax[0]); ax[0].set_title("Equity")
    ax[0].legend(["Mark-to-market", "Realized only", "SPY"])
    for col in ("mtm", "realized"):
        (eq[col] / eq[col].cummax() - 1).plot(ax=ax[1], label=col)
    ax[1].set_title("Drawdown"); ax[1].legend()
    eq.exposure.plot(ax=ax[2]); ax[2].set_title("Capital deployed")
    plt.tight_layout(); plt.savefig(png, dpi=110)
    print(f"\nSaved {png}, kronos_trades.csv, kronos_equity.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", help="offline long-format OHLC file (Date,Symbol,Open,High,Low,Close)")
    a = ap.parse_args()
    if a.csv:
        o, h, l, c = load_csv(a.csv, UNIVERSE, P["start"], P["end"])
    else:
        loader = load_synthetic if a.synthetic else load_real
        o, h, l, c = loader(UNIVERSE, P["start"], P["end"])
    eq, tr = run(o, h, l, c, P)
    report(eq, tr)
