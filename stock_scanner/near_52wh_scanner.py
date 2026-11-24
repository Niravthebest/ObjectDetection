#!/usr/bin/env python3
"""
"Stock near 5% of 52-week high" scanner + backtest (port of a Chartink scan).

CONDITIONS (all on the latest daily bar):
    1. Close >= 0.95 * max(High, last 252 days)      within 5% of the 52-week high
    2. Close / Close 65 days ago > 1                  up over the last ~3 months
    3. Close > 100                                    (--min-price; Chartink used Rs.100)
    4. EMA(Close,20) > EMA(Close,50)                  short trend above medium trend
    5. Close > EMA(Close,21)
    6. Volume > 50,000                                (--min-volume)
    7. Yearly ROCE > 15%      ROCE = EBIT / (Total assets - Current liabilities)
    8. Yearly ROE  >= 15%     ROE  = Net income / Shareholders' equity  (Chartink "RoNW")

Fundamentals are point-in-time: a fiscal year's numbers are only used from
fiscal-year-end + 90 days (roughly when the 10-K is public), so the backtest
does not peek at numbers that weren't published yet. Yahoo only returns the last
~4 fiscal years, so conditions 7-8 can only be tested from early 2024 onward.

BACKTEST:
    "every day"  = every day a stock passes the scan (what you'd see running it daily)
    "fresh"      = first day it passes after not passing the day before (the entry signal)
    Forward returns at 5 / 10 / 20 trading days, close to close; win = return > 0.
    Baseline = every stock-day in the same universe and period.

USAGE:
    python near_52wh_scanner.py backtest [--min-price 100] [--universe both]
    python near_52wh_scanner.py scan     [--min-price 100] [--fresh-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import sma_cross_scanner as base  # universe + price download helpers

FUND_URL = "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{sym}"
FUND_TYPES = ["annualEBIT", "annualTotalAssets", "annualCurrentLiabilities",
              "annualNetIncome", "annualStockholdersEquity"]
REPORT_LAG_DAYS = 90
HORIZONS = (5, 10, 20)


# --------------------------------------------------------------------------- fundamentals
def fetch_fundamentals(sym: str, retries: int = 4) -> pd.DataFrame | None:
    """Annual ROCE / ROE indexed by the date the figures became public."""
    params = {"type": ",".join(FUND_TYPES), "period1": 1262304000, "period2": int(time.time()) + 86400}
    for attempt in range(retries):
        try:
            r = base.SESSION.get(FUND_URL.format(sym=sym.replace(".", "-")), params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            cols = {}
            for res in r.json()["timeseries"]["result"] or []:
                t = res["meta"]["type"][0]
                for x in res.get(t) or []:
                    if x and x.get("reportedValue"):
                        cols.setdefault(t, {})[x["asOfDate"]] = x["reportedValue"]["raw"]
            if not cols:
                return None
            f = pd.DataFrame(cols).sort_index()
            for c in FUND_TYPES:
                if c not in f:
                    f[c] = np.nan
            f["roce"] = f["annualEBIT"] / (f["annualTotalAssets"] - f["annualCurrentLiabilities"])
            f["roe"] = f["annualNetIncome"] / f["annualStockholdersEquity"]
            # negative equity / capital employed makes the ratio meaningless -> treat as failing
            f.loc[f["annualStockholdersEquity"] <= 0, "roe"] = np.nan
            f.loc[(f["annualTotalAssets"] - f["annualCurrentLiabilities"]) <= 0, "roce"] = np.nan
            f.index = pd.to_datetime(f.index) + pd.Timedelta(days=REPORT_LAG_DAYS)
            f.index.name = "avail"
            return f[["roce", "roe"]]
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_all_fundamentals(symbols: list[str], workers: int = 8) -> dict[str, pd.DataFrame]:
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_fundamentals, s): s for s in symbols}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if res is not None:
                out[futs[f]] = res
            if i % 200 == 0 or i == len(symbols):
                print(f"  fundamentals {i}/{len(symbols)} ({len(out)} ok)", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- conditions
def add_conditions(df: pd.DataFrame, fund: pd.DataFrame | None, min_price: float,
                   min_volume: float) -> pd.DataFrame:
    d = df.copy()
    px = d["adj"]
    d["high_adj"] = d["high"] * (d["adj"] / d["close"])      # split/dividend-consistent high
    d["hi252"] = d["high_adj"].rolling(252).max()
    ema = lambda n: px.ewm(span=n, adjust=False).mean()
    d["ema20"], d["ema21"], d["ema50"] = ema(20), ema(21), ema(50)

    d["c_near_high"] = px >= 0.95 * d["hi252"]
    d["c_3m_up"] = px / px.shift(65) > 1
    d["c_price"] = d["close"] > min_price
    d["c_ema_trend"] = d["ema20"] > d["ema50"]
    d["c_above_ema21"] = px > d["ema21"]
    d["c_volume"] = d["volume"] > min_volume
    d["technical"] = (d["c_near_high"] & d["c_3m_up"] & d["c_price"] & d["c_ema_trend"]
                      & d["c_above_ema21"] & d["c_volume"] & d["hi252"].notna())

    if fund is not None and len(fund):
        f = fund.reset_index().sort_values("avail")
        f["avail"] = f["avail"].astype("datetime64[ns]")
        left = d.reset_index(names="date")
        left["date"] = left["date"].astype("datetime64[ns]")
        m = pd.merge_asof(left.sort_values("date"), f,
                          left_on="date", right_on="avail", direction="backward")
        d["roce"], d["roe"] = m["roce"].values, m["roe"].values
    else:
        d["roce"] = d["roe"] = np.nan
    d["has_fund"] = d["roce"].notna() | d["roe"].notna()
    d["fundamental"] = (d["roce"] > 0.15) & (d["roe"] >= 0.15)
    d["pass_all"] = d["technical"] & d["fundamental"]
    for col in ("technical", "pass_all"):
        d[f"{col}_fresh"] = d[col] & ~d[col].shift(1, fill_value=False)
    for h in HORIZONS:
        d[f"fwd{h}"] = px.shift(-h) / px - 1
    return d


# --------------------------------------------------------------------------- stats
def stats(r: pd.DataFrame) -> dict:
    out = {"n": len(r)}
    for h in HORIZONS:
        x = r[f"fwd{h}"].dropna()
        out[f"win{h}"] = round(float((x > 0).mean()) * 100, 1) if len(x) else None
        out[f"avg{h}"] = round(float(x.mean()) * 100, 2) if len(x) else None
        out[f"med{h}"] = round(float(x.median()) * 100, 2) if len(x) else None
    return out


def cmd_backtest(args):
    os.makedirs(args.out_dir, exist_ok=True)
    uni = base.resolve_universe(args)
    syms = uni["symbol"].tolist()
    print(f"Downloading {args.years + 1}y prices ...", file=sys.stderr)
    prices = base.fetch_all(syms, f"{args.years + 1}y", args.workers)
    print("Downloading fundamentals ...", file=sys.stderr)
    funds = fetch_all_fundamentals(list(prices), args.workers)

    start = prices[next(iter(prices))].index[-1] - pd.DateOffset(years=args.years)
    frames = []
    for s, df in prices.items():
        d = add_conditions(df, funds.get(s), args.min_price, args.min_volume)
        d = d[(d.index >= start) & d["hi252"].notna()]
        keep = ["technical", "technical_fresh", "pass_all", "pass_all_fresh", "has_fund",
                "roce", "roe"] + [f"fwd{h}" for h in HORIZONS]
        frames.append(d[keep].assign(symbol=s).reset_index(names="date"))
    allx = pd.concat(frames, ignore_index=True)
    fund_start = allx.loc[allx["has_fund"], "date"].min()

    rows = []
    def add(label, period, sub):
        rows.append({"test": label, "period": period, **stats(sub)})

    full = f"{allx.date.min():%Y-%m-%d}..{allx.date.max():%Y-%m-%d}"
    add("Baseline: every stock-day", full, allx)
    add("Technical (1-6), every day", full, allx[allx.technical])
    add("Technical (1-6), fresh entry", full, allx[allx.technical_fresh])
    recent = allx[allx.date >= fund_start]
    rp = f"{fund_start:%Y-%m-%d}..{allx.date.max():%Y-%m-%d}"
    add("Baseline: every stock-day", rp, recent)
    add("Technical (1-6), every day", rp, recent[recent.technical])
    add("Technical (1-6), fresh entry", rp, recent[recent.technical_fresh])
    add("FULL scan (1-8), every day", rp, recent[recent.pass_all])
    add("FULL scan (1-8), fresh entry", rp, recent[recent.pass_all_fresh])
    summary = pd.DataFrame(rows)

    # year-by-year for the technical fresh entry vs baseline
    allx["year"] = allx["date"].dt.year
    yearly = []
    for y, g in allx.groupby("year"):
        b, t = stats(g), stats(g[g.technical_fresh])
        yearly.append({"year": y, "signals": t["n"], "win5": t["win5"], "base_win5": b["win5"],
                       "avg5": t["avg5"], "base_avg5": b["avg5"],
                       "win20": t["win20"], "base_win20": b["win20"],
                       "avg20": t["avg20"], "base_avg20": b["avg20"]})
    yearly = pd.DataFrame(yearly)

    # per-stock scores on technical fresh entries, 5-day horizon
    tf = allx[allx.technical_fresh].dropna(subset=["fwd5"])
    g = tf.groupby("symbol")["fwd5"]
    per = pd.DataFrame({"signals": g.size(), "wins": g.apply(lambda r: int((r > 0).sum())),
                        "avg_ret_5d": g.mean().round(5)})
    per["win_rate"] = (per.wins / per.signals).round(4)
    per["score"] = [round(base.wilson_lower(w, n), 4) for w, n in zip(per.wins, per.signals)]
    per = per.sort_values(["score", "win_rate"], ascending=False).reset_index()
    per = per.merge(uni[["symbol", "name", "index"]], on="symbol", how="left")

    pre = f"near52wh_p{int(args.min_price)}"
    summary.to_csv(os.path.join(args.out_dir, f"{pre}_summary.csv"), index=False)
    yearly.to_csv(os.path.join(args.out_dir, f"{pre}_yearly.csv"), index=False)
    per.to_csv(os.path.join(args.out_dir, f"{pre}_scores.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"\nmin price ${args.min_price}, {len(prices)} stocks, fundamentals for {len(funds)}")
    print(summary.to_string(index=False))
    print("\nTechnical fresh entries by year:")
    print(yearly.to_string(index=False))
    print(f"\nTop stocks (fresh entries, >= {args.min_signals} signals):")
    print(per[per.signals >= args.min_signals].head(args.top).to_string(index=False))


def cmd_scan(args):
    os.makedirs(args.out_dir, exist_ok=True)
    uni = base.resolve_universe(args)
    prices = base.fetch_all(uni["symbol"].tolist(), "2y", args.workers)
    # only pull fundamentals for stocks that pass the technical part (much faster)
    cand = {}
    for s, df in prices.items():
        d = add_conditions(df, None, args.min_price, args.min_volume)
        if args.asof:
            d = d.loc[:args.asof]
        if bool(d["technical"].iloc[-1]):
            cand[s] = df
    print(f"{len(cand)} pass the technical conditions; checking fundamentals ...", file=sys.stderr)
    funds = fetch_all_fundamentals(list(cand), args.workers)
    hits = []
    for s, df in cand.items():
        d = add_conditions(df, funds.get(s), args.min_price, args.min_volume)
        if args.asof:
            d = d.loc[:args.asof]
        last = d.iloc[-1]
        if not bool(last["pass_all"]) or (args.fresh_only and not bool(last["pass_all_fresh"])):
            continue
        hits.append({"symbol": s, "date": f"{d.index[-1]:%Y-%m-%d}", "close": round(last["close"], 2),
                     "pct_from_52wh": round((last["adj"] / last["hi252"] - 1) * 100, 2),
                     "ret_65d": round((last["adj"] / d["adj"].iloc[-66] - 1) * 100, 1),
                     "roce": round(last["roce"] * 100, 1), "roe": round(last["roe"] * 100, 1),
                     "fresh": bool(last["pass_all_fresh"])})
    res = pd.DataFrame(hits)
    if res.empty:
        print("No stocks pass all conditions.")
        return
    res = res.merge(uni[["symbol", "name", "index"]], on="symbol", how="left")
    res = res.sort_values(["fresh", "pct_from_52wh"], ascending=[False, False])
    out = os.path.join(args.out_dir, f"near52wh_scan_{res.date.max()}.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(res.drop(columns=["name"]).to_string(index=False))
    print(f"\n{len(res)} stocks ({int(res.fresh.sum())} new today). Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["backtest", "scan"])
    p.add_argument("--universe", choices=["both", "sp500", "nasdaq", "nasdaq100", "sp500+nasdaq100"],
                   default="both")
    p.add_argument("--list-source", choices=["auto", "slickcharts", "github"], default="github")
    p.add_argument("--nasdaq-min-cap", type=float, default=1e9)
    p.add_argument("--tickers", default="")
    p.add_argument("--min-price", type=float, default=100.0)
    p.add_argument("--min-volume", type=float, default=50_000)
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--min-signals", type=int, default=10)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--fresh-only", action="store_true", help="scan: only stocks newly passing today")
    p.add_argument("--asof", default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()
    {"backtest": cmd_backtest, "scan": cmd_scan}[args.command](args)


if __name__ == "__main__":
    main()
