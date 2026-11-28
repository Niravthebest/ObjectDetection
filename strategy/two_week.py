"""One stock at a time, two-week holds: rule comparison vs SPY and a random-pick control.

Every 10 trading days: score all stocks at the close, buy the single top stock at the next
open, sell it 10 trading days later at the open (and buy the next pick). All-in, compounding,
0.10% cost per round trip. Stock prices are split-adjusted, no dividends; SPY is shown both
price-only and with dividends.

Optional MEMBERSHIP_CSV (ticker,start_date,end_date; e.g. fja05680/sp500
sp500_ticker_start_end.csv) restricts each pick to stocks in the S&P 500 on that date,
which removes the bias of only testing today's members.

Usage: python two_week.py DATA_DIR SPY_CSV [MEMBERSHIP_CSV]
"""
import glob, os, sys
import numpy as np
import pandas as pd

HOLD, COST = 10, 0.001
PERIODS = {"Design 2017-2021": ("2017-01-01", "2021-12-31"),
           "Test 2022-2026": ("2022-01-01", "2026-12-31"),
           "Full 2017-2026": ("2017-01-01", "2026-12-31")}


def load_panel(data_dir, spy_csv):
    spy = pd.read_csv(spy_csv, parse_dates=["date"], index_col="date")
    idx = spy.index
    fr = {}
    for p in glob.glob(os.path.join(data_dir, "*.csv")):
        s = os.path.basename(p)[:-4]
        if s in ("SPY", "KDP"):  # KDP: 2018 special dividend distorts prices
            continue
        df = pd.read_csv(p, parse_dates=["date"], index_col="date").reindex(idx)
        fr[s] = df
    O = pd.DataFrame({s: d.open for s, d in fr.items()})
    C = pd.DataFrame({s: d.close for s, d in fr.items()})
    return spy, O, C


# old ticker -> current ticker (price history is filed under the current ticker)
RENAMES = {"FB": "META", "ANTM": "ELV", "ABC": "COR", "BLL": "BALL", "SQ": "XYZ", "FISV": "FI",
           "PKI": "RVTY", "WLTW": "WTW", "CTL": "LUMN", "FLT": "CPAY", "HFC": "DINO", "RE": "EG",
           "COG": "CTRA", "TMK": "GL", "BRK.B": "BRK-B", "BF.B": "BF-B", "PEAK": "DOC"}


def membership(path, C):
    m = pd.read_csv(path)
    m["end_date"] = m.end_date.fillna("2099-12-31")
    M = pd.DataFrame(False, index=C.index, columns=C.columns)
    for t, a, b in m[["ticker", "start_date", "end_date"]].itertuples(index=False):
        t = RENAMES.get(t, t.replace(".", "-"))
        if t in M.columns:
            M.loc[(M.index >= a) & (M.index < b), t] = True
    return M


def scores(C, spy, member=None):
    """Each returns a DataFrame of scores (higher = pick)."""
    ok = (C.notna().rolling(252).sum() == 252) & (C > 5)
    if member is not None:
        ok &= member
    ma200 = C.rolling(200).mean()
    up = C > ma200
    r5 = C / C.shift(5) - 1
    d = C.diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    rsi2 = 100 - 100 / (1 + g.ewm(alpha=.5, adjust=False).mean() / l.ewm(alpha=.5, adjust=False).mean())
    spy_up = (spy.close > spy.close.rolling(200).mean()).reindex(C.index)
    S = {
        "Momentum 12-1 month": C.shift(21) / C.shift(252) - 1,
        "Momentum 6 month": C / C.shift(126) - 1,
        "Momentum 3 month": C / C.shift(63) - 1,
        "Nearest 52w high": C / C.rolling(252).max(),
        "Dip in uptrend (worst 5d, above 200dma)": (-r5).where(up),
        "RSI(2) dip in uptrend": (-rsi2).where(up),
    }
    S = {k: v.where(ok) for k, v in S.items()}
    # regime-filtered momentum: hold SPY when SPY is below its 200-day average
    S["Momentum 6 month + SPY trend filter"] = S["Momentum 6 month"].where(spy_up, axis=0)
    return S, ok, spy_up


def run(score, O, spy_open, start, end, rng=None, ok=None, fallback_spy=False):
    dates = O.index
    reb = [i for i in range(252, len(dates) - HOLD - 1, HOLD)
           if pd.Timestamp(start) <= dates[i] <= pd.Timestamp(end)]
    eq, curve, picks = 1.0, [], []
    for i in reb:
        e, x = i + 1, i + 1 + HOLD
        if rng is not None:
            cands = ok.columns[ok.iloc[i].to_numpy()]
            sym = rng.choice(cands)
        else:
            row = score.iloc[i].dropna()
            sym = row.idxmax() if len(row) else None
        if sym is None or np.isnan(O[sym].iloc[x]) or np.isnan(O[sym].iloc[e]):
            r = spy_open.iloc[x] / spy_open.iloc[e] - 1 if fallback_spy else 0.0
            sym = "SPY" if fallback_spy else "cash"
        else:
            r = O[sym].iloc[x] / O[sym].iloc[e] - 1 - COST
        eq *= 1 + r
        curve.append(eq)
        picks.append((dates[i], sym, r))
    c = pd.Series(curve)
    yrs = len(reb) * HOLD / 252
    mdd = (c / c.cummax() - 1).min()
    return eq, eq ** (1 / yrs) - 1, mdd, pd.DataFrame(picks, columns=["date", "symbol", "ret"])


if __name__ == "__main__":
    spy, O, C = load_panel(sys.argv[1], sys.argv[2])
    member = membership(sys.argv[3], C) if len(sys.argv) > 3 else None
    if member is not None:
        print(f"Point-in-time S&P 500 membership: {member.sum(axis=1).iloc[-1]} priced members today, "
              f"{member.sum(axis=1)['2017-01-03':].min()} at the lowest")
    S, ok, spy_up = scores(C, spy, member)
    spy_open = spy.open
    spy_tr_open = spy.open * spy.adjclose / spy.close
    ones = pd.DataFrame(1.0, index=O.index, columns=["SPY"])
    for name, (a, b) in PERIODS.items():
        print(f"\n=== {name}: $10,000 start ===")
        rows = []
        for lbl, px in (("SPY (price only)", spy_open), ("SPY (with dividends)", spy_tr_open)):
            eq, cagr, mdd, _ = run(ones, px.to_frame("SPY"), px, a, b)
            rows.append((lbl, eq, cagr, mdd))
        rnd = [run(None, O, spy_open, a, b, rng=np.random.default_rng(k), ok=ok)[0] for k in range(200)]
        for q in (50, 75, 90):
            v = np.percentile(rnd, q)
            rows.append((f"Random pick, {q}th pct of 200 runs", v, np.nan, np.nan))
        for k, sc in S.items():
            eq, cagr, mdd, p = run(sc, O, spy_open, a, b, fallback_spy="filter" in k)
            rows.append((k, eq, cagr, mdd))
            if name.startswith("Full"):
                p.to_csv(f"picks_{k.split()[0].lower()}_{len(k)}.csv", index=False)
        for lbl, eq, cagr, mdd in rows:
            cg = "" if np.isnan(cagr) else f"{100*cagr:6.1f}%/yr"
            md = "" if np.isnan(mdd) else f"maxDD {100*mdd:5.0f}%"
            print(f"{lbl:42s} ${10000*eq:>12,.0f}  {cg:>11s}  {md}")
