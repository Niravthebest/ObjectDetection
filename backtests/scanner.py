"""Daily RSI(2) scanner (Connors rules as backtested in refine_rsi2.py).

ENTRY: RSI(2) < 10 and close > 200-day SMA  -> buy at the close
EXIT : close > 5-day SMA                     -> sell at the close

Run near 3:45pm ET: Yahoo's latest daily bar holds the live price, which is
used as today's close. Position status is rebuilt from the rules over the
past year, so no state file is needed.

Usage: python scanner.py [--out DIR] [--no-stocks] [--min-expectancy PCT]

Stocks are limited to those whose 1991-2026 backtest expectancy per trade
(stock_rsi2_per_symbol.csv, >= 30 trades) meets --min-expectancy (default 0.7%).

The S&P 500 list covers nearly every NYSE U.S. 100 member; NYSE-listed
stocks are flagged, and the NYSE U.S. 100 index itself is scanned.
"""
import argparse, csv, io, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np, pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
INDEX_SYMBOLS = {"SPY": "SPDR S&P 500 ETF", "^NY": "NYSE U.S. 100 index", "^GSPC": "S&P 500 index"}
WATCH_RSI = 20  # report near-misses below this RSI(2)
STATS_CSV = __file__.replace("scanner.py", "stock_rsi2_per_symbol.csv")  # from stocks_rsi2.py

def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)

def history(symbol):
    q = urllib.request.quote(symbol)
    d = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{q}?range=2y&interval=1d"))
    r = d["chart"]["result"][0]
    df = pd.DataFrame({"close": r["indicators"]["quote"][0]["close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True)
                      .tz_convert("America/New_York").normalize().tz_localize(None))
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df, r["meta"]

def rsi(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def next_day_levels(c):
    """Closing prices that would fire a signal on the NEXT session:
    buy_below : RSI(2) drops under 10 if the next close is below this
    trend_floor: the next close must stay above this to be over the 200-day SMA
    sell_above: an open trade exits if the next close is above this (next 5-day SMA)."""
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean().iloc[-1]
    last = c.iloc[-1]

    def rsi_at(p):
        u, v = 0.5 * up + 0.5 * max(p - last, 0), 0.5 * dn + 0.5 * max(last - p, 0)
        return 100.0 if v == 0 else 100 - 100 / (1 + u / v)

    lo, hi = last * 0.5, last * 1.5  # RSI rises with price: bisect for RSI == 10
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rsi_at(mid) < 10 else (lo, mid)
    return dict(buy_below=round(lo, 2), trend_floor=round(c.iloc[-199:].mean(), 2),
                sell_above=round(c.iloc[-4:].mean(), 2))

def evaluate(symbol, name=""):
    df, meta = history(symbol)
    c = df["close"]
    df["sma5"], df["sma200"], df["rsi2"] = c.rolling(5).mean(), c.rolling(200).mean(), rsi(c)
    df = df.dropna()
    if len(df) < 2:
        return None
    # Replay the rules over the last year to know whether a trade is open.
    pos, entry_date, entry_px = False, None, None
    for i, (dt, row) in enumerate(df.iloc[-252:].iterrows()):
        last = dt == df.index[-1]
        if pos and row.close > row.sma5:
            if last:
                break  # exit fires today; keep entry info for the report
            pos = False
        elif not pos and row.rsi2 < 10 and row.close > row.sma200:
            pos, entry_date, entry_px = True, dt, row.close
    t = df.iloc[-1]
    lv = next_day_levels(c)
    if pos and t.close > t.sma5:
        signal = "EXIT"
    elif pos and entry_date == df.index[-1]:
        signal = "ENTRY"
    elif pos:
        signal = "HOLD"
    elif t.rsi2 < WATCH_RSI and t.close > t.sma200:
        signal = "WATCH"
    else:
        signal = ""
    return dict(symbol=symbol, name=name, exchange=meta.get("exchangeName", ""), signal=signal,
                date=df.index[-1].date().isoformat(), close=round(t.close, 2), rsi2=round(t.rsi2, 1),
                sma5=round(t.sma5, 2), sma200=round(t.sma200, 2),
                pct_vs_sma200=round((t.close / t.sma200 - 1) * 100, 1),
                entry_date=entry_date.date().isoformat() if pos else "",
                entry_price=round(entry_px, 2) if pos else "",
                open_pnl_pct=round((t.close / entry_px - 1) * 100, 2) if pos else "",
                **lv, drop_needed_pct=round((lv["buy_below"] / t.close - 1) * 100, 1),
                buy_zone_ok=lv["buy_below"] > lv["trend_floor"])

def sp500():
    rows = list(csv.DictReader(io.StringIO(get(SP500_CSV).decode())))
    return {r["Symbol"].replace(".", "-"): r["Security"] for r in rows}

def safe(args):
    try:
        return evaluate(*args)
    except Exception as e:
        print(f"warn: {args[0]}: {e}", file=sys.stderr)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-stocks", action="store_true")
    ap.add_argument("--min-expectancy", type=float, default=0.7,
                    help="only scan stocks whose backtested expectancy per trade is at least this %% (0 = all)")
    a = ap.parse_args()

    universe = dict(INDEX_SYMBOLS)
    if not a.no_stocks:
        picks = sp500()
        if a.min_expectancy > 0:
            st = pd.read_csv(STATS_CSV, index_col=0)
            keep = st[(st.trades >= 30) & (st.expectancy * 100 >= a.min_expectancy)].index
            picks = {k: v for k, v in picks.items() if k in keep}
        universe |= picks
    with ThreadPoolExecutor(6) as ex:
        res = [r for r in ex.map(safe, universe.items()) if r]
    df = pd.DataFrame(res)
    df["is_index"] = df.symbol.isin(list(INDEX_SYMBOLS))
    df["nyse_listed"] = df.exchange.eq("NYQ")
    st = pd.read_csv(STATS_CSV, index_col=0)  # backtested per-stock stats, in %
    for k in ("win_rate", "avg_win", "avg_loss", "expectancy"):
        df[f"bt_{k}"] = (df.symbol.map(st[k]) * 100).round(1).fillna("")
    order = {"ENTRY": 0, "EXIT": 1, "HOLD": 2, "WATCH": 3, "": 4}
    df = df.sort_values(["is_index", "signal", "rsi2"], key=lambda s: s.map(order) if s.name == "signal" else
                        (~s if s.dtype == bool else s), ascending=True)

    now = datetime.now(timezone.utc).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    lines = [f"# RSI(2) scan — {now:%Y-%m-%d %H:%M} ET",
             f"Rules: buy when RSI(2) < 10 and close > 200-day SMA; sell when close > 5-day SMA. "
             f"Scanned {len(df)} symbols ({len(universe) - len(df)} skipped: no data or <200 days of history).", ""]

    def table(sub, cols):
        if sub.empty:
            return ["_none_", ""]
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(v) for v in r) + " |" for r in sub[cols].itertuples(index=False)]
        return out + [""]

    lines += ["bt_* columns = that stock's 1991-2026 backtest per trade, in % (win rate, avg win, avg loss, expectancy).", ""]
    lines += ["## Index signals (validated strategy)"]
    lines += table(df[df.is_index], ["symbol", "name", "signal", "close", "rsi2", "sma5", "sma200",
                                     "entry_date", "entry_price", "open_pnl_pct"])
    stocks = df[~df.is_index]
    if not stocks.empty:
        lines += [f"## Stocks: {len(stocks)} S&P 500 names" + (f" with backtested expectancy >= {a.min_expectancy}% per trade" if a.min_expectancy > 0 else "")]
        for sig, title, cols in [
            ("ENTRY", "New ENTRY triggers (buy at today's close)", ["symbol", "name", "close", "rsi2", "bt_win_rate", "bt_avg_win", "bt_avg_loss", "bt_expectancy"]),
            ("EXIT", "EXIT triggers (sell at today's close)", ["symbol", "name", "close", "entry_date", "open_pnl_pct", "bt_win_rate", "bt_avg_win", "bt_avg_loss", "bt_expectancy"]),
            ("HOLD", "Open per rules (waiting for close > 5-day SMA)", ["symbol", "close", "sma5", "entry_date", "open_pnl_pct", "bt_win_rate", "bt_avg_win", "bt_avg_loss", "bt_expectancy"])]:
            lines += [f"### {title}"] + table(stocks[stocks.signal == sig], cols)
        nxt = stocks[stocks.signal.isin(["", "WATCH", "EXIT"]) & stocks.buy_zone_ok
                     & (stocks.drop_needed_pct >= -3)].sort_values("drop_needed_pct", ascending=False)
        lines += ["### Next session: buys if it closes below buy_below (and above trend_floor)",
                  "drop_needed_pct = fall from the last close needed to trigger; within 3% shown."]
        lines += table(nxt, ["symbol", "name", "close", "rsi2", "buy_below", "drop_needed_pct", "trend_floor",
                             "bt_win_rate", "bt_avg_win", "bt_avg_loss", "bt_expectancy"])
        held = stocks[stocks.signal.isin(["ENTRY", "HOLD"])]
        lines += ["### Next session: open trades sell if they close above sell_above"]
        lines += table(held, ["symbol", "close", "sell_above", "entry_date", "entry_price", "open_pnl_pct"])
        lines += [f"### Watchlist (RSI(2) < {WATCH_RSI}, above 200-day SMA)"]
        lines += table(stocks[stocks.signal == "WATCH"].head(25), ["symbol", "close", "rsi2", "bt_win_rate", "bt_avg_win", "bt_avg_loss", "bt_expectancy"])

    report = "\n".join(lines)
    stamp = f"{now:%Y-%m-%d}"
    open(f"{a.out}/scan_{stamp}.md", "w").write(report)
    df.to_csv(f"{a.out}/scan_{stamp}.csv", index=False)
    print(report)
    n = df.signal.isin(["ENTRY", "EXIT"]).sum()
    print(f"\nTRIGGERS: {n} (index: {df[df.is_index].signal.isin(['ENTRY','EXIT']).sum()})")

if __name__ == "__main__":
    main()
