#!/usr/bin/env python3
"""
SMA-14 reclaim scanner + backtest for S&P 500 and Nasdaq stocks.

SIGNAL (evaluated on daily closes, day t = "today"):
    1. Yesterday the close was BELOW its 14-day SMA      close[t-1] <  SMA14[t-1]
    2. Today the close moved ABOVE its 14-day SMA         close[t]   >  SMA14[t]
    3. Today the close is also above the 30-day SMA       close[t]   >  SMA30[t]
       (i.e. both the 14 and 30 day averages are below price)

BACKTEST:
    Every historical occurrence of the signal is "bought" at the signal-day close
    and "sold" at the close 5 trading days later. A trade is a WIN when the
    5-day return is > 0. Each stock is scored by its win rate.

    Score = Wilson lower bound (95%) of the win rate. This is the win rate
    shrunk toward lower values when a stock has few signals, so a stock that
    went 3/3 does not outrank one that went 40/55.

    Edge = signal win rate minus the stock's ordinary 5-day win rate over the
    same period (how much better than "buy on any random day").

USAGE:
    python sma_cross_scanner.py backtest            # build scores (run weekly/monthly)
    python sma_cross_scanner.py scan                # daily scan, uses saved scores
    python sma_cross_scanner.py scan --min-signals 10 --min-score 0.5

    Options common to both:
        --universe both|sp500|nasdaq|nasdaq100|sp500+nasdaq100   (default both)
                                       both = S&P 500 + all Nasdaq-listed >= --nasdaq-min-cap
        --list-source auto|slickcharts|github
                                       where index member lists come from. auto (default)
                                       tries slickcharts.com first, then falls back to GitHub.
        --nasdaq-min-cap 1e9           Nasdaq market-cap floor (drops micro caps/SPACs)
        --tickers AAPL,MSFT            explicit list, overrides --universe
        --out-dir results

Run `scan` after the close (after ~4:15pm ET). During market hours the last
bar is a partial day, so a "cross" may not hold by the close.
"""
from __future__ import annotations

import argparse
import io
import re
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

FAST, SLOW, HOLD = 14, 30, 5

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NASDAQ_URL = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json"
SLICK_URL = "https://www.slickcharts.com/{index}"   # sp500 | nasdaq100
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
UA = {"User-Agent": "Mozilla/5.0"}

SESSION = requests.Session()
SESSION.headers.update(UA)
_ca = "/root/.ccr/ca-bundle.crt"
if os.path.exists(_ca) and not os.environ.get("REQUESTS_CA_BUNDLE"):
    SESSION.verify = _ca


# --------------------------------------------------------------------------- universe
_SLICK_LINK = re.compile(r'href="/symbol/([A-Za-z0-9.\-]+)"[^>]*>\s*([^<]+?)\s*</a>')


def parse_slickcharts(html: str) -> pd.DataFrame:
    """Extract (symbol, company) pairs from a slickcharts index-weights page.
    Each row links the company name and the ticker to /symbol/<TICKER>."""
    names: dict[str, str] = {}
    for sym, text in _SLICK_LINK.findall(html):
        sym = sym.upper()
        names.setdefault(sym, "")
        if text.upper() != sym and not names[sym]:
            names[sym] = text
    return pd.DataFrame({"symbol": list(names), "name": list(names.values())})


def fetch_slickcharts(index: str) -> pd.DataFrame:
    r = SESSION.get(SLICK_URL.format(index=index), timeout=30)
    if r.status_code != 200 or "/symbol/" not in r.text:
        reason = "bot challenge (Cloudflare)" if "Just a moment" in r.text else f"HTTP {r.status_code}"
        raise RuntimeError(f"slickcharts {index}: {reason}")
    df = parse_slickcharts(r.text)
    lo = {"sp500": 480, "nasdaq100": 95}[index]
    if len(df) < lo:
        raise RuntimeError(f"slickcharts {index}: only {len(df)} symbols parsed")
    return df


def _nasdaq_listed() -> pd.DataFrame:
    r = SESSION.get(NASDAQ_URL, timeout=60)
    r.raise_for_status()
    nq = pd.DataFrame(r.json())
    nq["marketCap"] = pd.to_numeric(nq["marketCap"], errors="coerce").fillna(0)
    return nq


def _github_sp500() -> pd.DataFrame:
    r = SESSION.get(SP500_URL, timeout=30)
    r.raise_for_status()
    sp = pd.read_csv(io.StringIO(r.text))
    return pd.DataFrame({"symbol": sp["Symbol"], "name": sp["Security"], "sector": sp["GICS Sector"]})


def _github_nasdaq100_approx() -> pd.DataFrame:
    """Fallback only: 100 largest Nasdaq-listed non-financial companies (the
    Nasdaq-100's rule), one share class per company. Close to, not identical to, the index."""
    nq = _nasdaq_listed()
    nq = nq[(nq["sector"] != "Finance") & ~nq["symbol"].str.contains(r"[\^/ ]", regex=True)]
    nq = nq[~nq["name"].fillna("").str.contains(r"(?i)preferred|warrant|\bunits?\b|\brights?\b|notes due")]
    nq = nq.sort_values("marketCap", ascending=False)
    nq["co"] = nq["name"].str.replace(r"(?i)\s+class [a-c].*|\s+(common|capital|ordinary) stock.*", "", regex=True)
    nq = nq.drop_duplicates("co").head(100)
    return pd.DataFrame({"symbol": nq["symbol"], "name": nq["name"], "sector": nq["sector"]})


def load_index(index: str, source: str) -> pd.DataFrame:
    """index: 'sp500' or 'nasdaq100'. Returns symbol/name/sector."""
    if source in ("auto", "slickcharts"):
        try:
            df = fetch_slickcharts(index)
            df["sector"] = np.nan
            print(f"  {index}: {len(df)} symbols from slickcharts", file=sys.stderr)
            return df
        except Exception as e:
            if source == "slickcharts":
                raise
            print(f"  {e} -> falling back to GitHub list", file=sys.stderr)
    df = _github_sp500() if index == "sp500" else _github_nasdaq100_approx()
    note = "" if index == "sp500" else " (approximation: top 100 non-financial by market cap)"
    print(f"  {index}: {len(df)} symbols from GitHub{note}", file=sys.stderr)
    return df


def load_universe(universe: str, nasdaq_min_cap: float, min_price: float = 5.0,
                  source: str = "auto") -> pd.DataFrame:
    frames = []
    if universe in ("sp500", "both", "sp500+nasdaq100"):
        frames.append(load_index("sp500", source).assign(index="S&P500"))
    if universe in ("nasdaq100", "sp500+nasdaq100"):
        frames.append(load_index("nasdaq100", source).assign(index="NDX100"))
    if universe in ("nasdaq", "both"):
        nq = _nasdaq_listed()
        nq["marketCap"] = pd.to_numeric(nq["marketCap"], errors="coerce").fillna(0)
        nq["price"] = pd.to_numeric(nq["lastsale"].str.replace("$", "", regex=False), errors="coerce")
        nq = nq[(nq["marketCap"] >= nasdaq_min_cap) & (nq["price"] >= min_price)]
        # drop warrants / units / rights / preferreds that slip through
        nq = nq[~nq["symbol"].str.contains(r"[\^/ ]", regex=True)]
        junk = r"(?i)\b(?:preferred|notes? due|\d{4} notes|units?|warrants?|rights?|depositary shares,? each representing a (?:1/|fraction))\b"
        nq = nq[~nq["name"].fillna("").str.contains(junk, regex=True)]
        frames.append(pd.DataFrame({"symbol": nq["symbol"], "name": nq["name"],
                                    "sector": nq["sector"], "index": "Nasdaq"}))
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    # one row per symbol; mark membership in both
    df = (df.groupby("symbol", as_index=False)
            .agg(name=("name", "first"), sector=("sector", "first"),
                 index=("index", lambda s: "+".join(sorted(set(s))))))
    return df


# --------------------------------------------------------------------------- prices
def fetch_history(sym: str, rng: str, retries: int = 4) -> pd.DataFrame | None:
    ysym = sym.replace(".", "-")  # BRK.B -> BRK-B
    params = {"range": rng, "interval": "1d", "events": "div,split", "includeAdjustedClose": "true"}
    for attempt in range(retries):
        try:
            r = SESSION.get(CHART_URL.format(sym=ysym), params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            res = r.json()["chart"]["result"]
            if not res:
                return None
            res = res[0]
            ts = res.get("timestamp")
            if not ts:
                return None
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
            df = pd.DataFrame({"close": q["close"], "adj": adj, "volume": q["volume"]},
                              index=pd.to_datetime(ts, unit="s", utc=True)
                                      .tz_convert("America/New_York").normalize().tz_localize(None))
            df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
            df["adj"] = df["adj"].fillna(df["close"])
            return df
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_all(symbols: list[str], rng: str, workers: int = 8) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_history, s, rng): s for s in symbols}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            df = f.result()
            if df is not None and len(df) > SLOW + 2:
                out[s] = df
            if i % 100 == 0 or i == len(symbols):
                print(f"  downloaded {i}/{len(symbols)} ({len(out)} ok)", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- signal
def add_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Uses split/dividend-adjusted closes so splits don't create fake crosses."""
    px = df["adj"]
    df = df.copy()
    df["sma14"] = px.rolling(FAST).mean()
    df["sma30"] = px.rolling(SLOW).mean()
    below_yday = px.shift(1) < df["sma14"].shift(1)
    above_today = px > df["sma14"]
    above_slow = px > df["sma30"]
    df["signal"] = (below_yday & above_today & above_slow).fillna(False)
    df["fwd5"] = px.shift(-HOLD) / px - 1.0
    return df


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def backtest_one(sym: str, df: pd.DataFrame) -> dict | None:
    d = add_signal(df)
    valid = d.dropna(subset=["sma30", "fwd5"])
    trades = valid[valid["signal"]]
    n = len(trades)
    base = valid["fwd5"]
    if n == 0:
        return {"symbol": sym, "signals": 0}
    r = trades["fwd5"]
    wins = int((r > 0).sum())
    gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
    win_rate = wins / n
    base_wr = float((base > 0).mean())
    return {
        "symbol": sym,
        "signals": n,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "score": round(wilson_lower(wins, n), 4),
        "avg_ret_5d": round(float(r.mean()), 5),
        "median_ret_5d": round(float(r.median()), 5),
        "avg_win": round(float(r[r > 0].mean()), 5) if wins else 0.0,
        "avg_loss": round(float(r[r <= 0].mean()), 5) if wins < n else 0.0,
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else float("inf"),
        "base_win_rate_5d": round(base_wr, 4),
        "edge": round(win_rate - base_wr, 4),
        "last_signal": trades.index[-1].strftime("%Y-%m-%d"),
        "first_date": valid.index[0].strftime("%Y-%m-%d"),
        "last_date": valid.index[-1].strftime("%Y-%m-%d"),
    }


# --------------------------------------------------------------------------- commands
def resolve_universe(args) -> pd.DataFrame:
    if args.tickers:
        syms = [s.strip().upper() for s in args.tickers.split(",") if s.strip()]
        return pd.DataFrame({"symbol": syms, "name": "", "sector": "", "index": "custom"})
    print(f"Loading universe '{args.universe}' ...", file=sys.stderr)
    u = load_universe(args.universe, args.nasdaq_min_cap, source=args.list_source)
    print(f"  {len(u)} symbols", file=sys.stderr)
    return u


def cmd_backtest(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    uni = resolve_universe(args)
    print(f"Downloading {args.years}y of daily history ...", file=sys.stderr)
    data = fetch_all(uni["symbol"].tolist(), f"{args.years}y", args.workers)

    rows, all_trades = [], []
    for sym, df in data.items():
        res = backtest_one(sym, df)
        if res:
            rows.append(res)
        d = add_signal(df).dropna(subset=["sma30", "fwd5"])
        t = d[d["signal"]][["adj", "sma14", "sma30", "fwd5"]].copy()
        t.insert(0, "symbol", sym)
        all_trades.append(t.reset_index(names="date"))

    scores = pd.DataFrame(rows).merge(uni, on="symbol", how="left")
    scores = scores.sort_values(["score", "win_rate", "signals"], ascending=False)
    scores.insert(0, "rank", range(1, len(scores) + 1))
    scores_path = os.path.join(args.out_dir, "backtest_scores.csv")
    scores.to_csv(scores_path, index=False)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(os.path.join(args.out_dir, "backtest_trades.csv"), index=False)

    # aggregate summary
    tr = trades["fwd5"] if len(trades) else pd.Series(dtype=float)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "years": args.years,
        "symbols_tested": len(data),
        "total_signals": int(len(tr)),
        "overall_win_rate": round(float((tr > 0).mean()), 4) if len(tr) else None,
        "overall_avg_ret_5d": round(float(tr.mean()), 5) if len(tr) else None,
        "overall_median_ret_5d": round(float(tr.median()), 5) if len(tr) else None,
        "avg_base_win_rate_5d": round(float(scores["base_win_rate_5d"].mean()), 4),
    }
    with open(os.path.join(args.out_dir, "backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    show = scores[scores["signals"] >= args.min_signals].head(args.top)
    cols = ["rank", "symbol", "signals", "win_rate", "score", "avg_ret_5d", "edge", "profit_factor", "index"]
    print(f"\nTop {args.top} by score (min {args.min_signals} signals):")
    print(show[cols].to_string(index=False))
    print(f"\nSaved {scores_path}")


def cmd_scan(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    scores_path = os.path.join(args.out_dir, "backtest_scores.csv")
    scores = pd.read_csv(scores_path) if os.path.exists(scores_path) else None
    if scores is None:
        print("No backtest_scores.csv found - run `backtest` first for win-rate scores.", file=sys.stderr)

    uni = resolve_universe(args)
    print("Downloading recent history ...", file=sys.stderr)
    data = fetch_all(uni["symbol"].tolist(), "6mo", args.workers)

    hits = []
    for sym, df in data.items():
        d = add_signal(df)
        last = d.iloc[-1]
        if args.asof:
            d = d.loc[:args.asof]
            last = d.iloc[-1]
        if bool(last["signal"]):
            prev = d.iloc[-2]
            hits.append({
                "symbol": sym,
                "date": d.index[-1].strftime("%Y-%m-%d"),
                "close": round(float(last["close"]), 2),
                "prev_close": round(float(prev["close"]), 2),
                "chg_pct": round((float(last["adj"]) / float(prev["adj"]) - 1) * 100, 2),
                "sma14": round(float(last["sma14"]), 2),
                "sma30": round(float(last["sma30"]), 2),
                "pct_above_sma14": round((float(last["adj"]) / float(last["sma14"]) - 1) * 100, 2),
                "vol_vs_20d": round(float(last["volume"]) / float(d["volume"].tail(21).iloc[:-1].mean()), 2)
                               if d["volume"].tail(21).iloc[:-1].mean() else None,
            })

    res = pd.DataFrame(hits)
    if res.empty:
        print("No stocks triggered the signal on the latest bar.")
        return
    res = res.merge(uni[["symbol", "name", "index"]], on="symbol", how="left")
    if scores is not None:
        keep = ["symbol", "signals", "win_rate", "score", "avg_ret_5d", "edge", "profit_factor"]
        res = res.merge(scores[keep], on="symbol", how="left")
        res = res[(res["signals"].fillna(0) >= args.min_signals) & (res["score"].fillna(0) >= args.min_score)]
        res = res.sort_values(["score", "win_rate"], ascending=False)

    date = res["date"].max() if len(res) else datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(args.out_dir, f"scan_{date}.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(res)} stocks crossed above SMA14 (and above SMA30) on {date}"
          f" - top {min(args.top, len(res))}{' by backtest score' if scores is not None else ' (unscored)'}:\n")
    cols = [c for c in ["symbol", "close", "chg_pct", "pct_above_sma14", "vol_vs_20d", "signals",
                        "win_rate", "score", "avg_ret_5d", "edge", "index"] if c in res.columns]
    print(res[cols].head(args.top).to_string(index=False))
    print(f"\nSaved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["backtest", "scan"])
    p.add_argument("--universe", choices=["both", "sp500", "nasdaq", "nasdaq100", "sp500+nasdaq100"],
                   default="both")
    p.add_argument("--list-source", choices=["auto", "slickcharts", "github"], default="auto")
    p.add_argument("--nasdaq-min-cap", type=float, default=1e9)
    p.add_argument("--tickers", default="")
    p.add_argument("--years", type=int, default=5, help="backtest history length")
    p.add_argument("--min-signals", type=int, default=0, help="min historical signals to show")
    p.add_argument("--min-score", type=float, default=0.0, help="scan: min Wilson score to show")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--asof", default="", help="scan: evaluate signal as of YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()
    {"backtest": cmd_backtest, "scan": cmd_scan}[args.command](args)


if __name__ == "__main__":
    main()
