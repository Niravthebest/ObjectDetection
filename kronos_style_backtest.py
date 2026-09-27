"""
Kronos-style candlestick-forecast backtest.

Kronos (a foundation model for financial K-lines) forecasts future candles by
turning recent OHLCV windows into tokens and sampling many possible future
paths. This script is a lightweight stand-in that needs no GPU or model weights:

  1. Each day, the last LOOKBACK candles are turned into a normalized feature
     window (log returns, candle body/wicks, range, volume z-score).
  2. The K most similar windows from the *past* are found (analog forecasting).
     Their realized next-HORIZON returns act as K sampled forecast paths.
  3. From those samples we get an expected return and P(up).
  4. Go long at the close if expected return > threshold and P(up) > min_prob,
     otherwise stay in cash. Position is held for the next day's return.

Everything is walk-forward: a window is only usable as an analog once its
forward return is fully known, so there is no lookahead.

Usage:
    pip install yfinance pandas numpy matplotlib
    python kronos_style_backtest.py                      # SPY via yfinance
    python kronos_style_backtest.py --ticker QQQ --start 2015-01-01
    python kronos_style_backtest.py --csv data/SPY.csv   # offline data
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(ticker, start, end, csv_path):
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        print(f"Loaded {len(df)} bars from {csv_path}")
    else:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            fallback = os.path.join("data", f"{ticker}.csv")
            if os.path.exists(fallback):
                print(f"yfinance returned no data; falling back to {fallback}")
                return load_data(ticker, start, end, fallback)
            raise SystemExit("No data downloaded. Check the ticker or pass --csv.")
        print(f"Downloaded {len(df)} bars of {ticker} from Yahoo Finance")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# ----------------------------------------------------------------------------
# "Tokenizer": per-candle features
# ----------------------------------------------------------------------------
def candle_features(df):
    o, h, l, c, v = (df[k].values for k in ["Open", "High", "Low", "Close", "Volume"])
    prev_c = np.r_[np.nan, c[:-1]]
    ret = np.log(c / prev_c)
    body = (c - o) / prev_c
    upper = (h - np.maximum(o, c)) / prev_c
    lower = (np.minimum(o, c) - l) / prev_c
    rng = (h - l) / prev_c
    logv = np.log(v + 1)
    vol_z = (logv - pd.Series(logv).rolling(50).mean()) / pd.Series(logv).rolling(50).std()
    feats = np.column_stack([ret, body, upper, lower, rng, vol_z.values])
    return feats


def build_windows(feats, lookback):
    """Scale each feature by its trailing 250-day std (past data only), then
    flatten the last `lookback` candles into one pattern vector."""
    scale = pd.DataFrame(feats).rolling(250, min_periods=100).std().shift(1).values
    z = feats / scale
    z[:, -1] = feats[:, -1]  # volume is already a z-score
    n, f = feats.shape
    X = np.full((n, lookback * f), np.nan)
    for t in range(lookback - 1, n):
        w = z[t - lookback + 1 : t + 1]
        if not np.isnan(w).any():
            X[t] = w.ravel()
    return X


# ----------------------------------------------------------------------------
# Forecaster: sample K analog futures
# ----------------------------------------------------------------------------
def forecast(df, lookback, horizon, k, min_history):
    feats = candle_features(df)
    X = build_windows(feats, lookback)
    close = df["Close"].values
    fwd = np.full(len(df), np.nan)
    fwd[:-horizon] = np.log(close[horizon:] / close[:-horizon])

    exp_ret = np.full(len(df), np.nan)
    p_up = np.full(len(df), np.nan)
    valid = ~np.isnan(X).any(1)

    for t in range(len(df)):
        if not valid[t]:
            continue
        # analogs must have their forward return fully known by day t
        lib_end = t - horizon
        idx = np.where(valid[:lib_end + 1])[0] if lib_end >= 0 else np.array([], int)
        if len(idx) < min_history:
            continue
        d = np.linalg.norm(X[idx] - X[t], axis=1)
        pos = np.argpartition(d, k)[:k]
        samples = fwd[idx[pos]]
        w = 1.0 / (d[pos] + 1e-9)
        exp_ret[t] = np.average(samples, weights=w)
        p_up[t] = np.average(samples > 0, weights=w)
    return pd.DataFrame({"exp_ret": exp_ret, "p_up": p_up}, index=df.index)


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def metrics(ret, pos=None):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    years = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino = ret.mean() * 252 / downside if downside > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    out = {
        "Total return": f"{eq.iloc[-1] - 1:.1%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max drawdown": f"{dd:.1%}",
        "Calmar": f"{cagr / abs(dd):.2f}" if dd < 0 else "n/a",
    }
    if pos is not None:
        pos = pos.loc[ret.index]
        out["Exposure"] = f"{pos.mean():.1%}"
        out["Trades"] = int((pos.diff().abs() > 0).sum())
        held = ret[pos > 0]
        out["Hit rate (days in mkt)"] = f"{(held > 0).mean():.1%}" if len(held) else "n/a"
    return out


def run(args):
    df = load_data(args.ticker, args.start, args.end, args.csv)
    print(f"Range: {df.index[0].date()} -> {df.index[-1].date()}")
    print("Building walk-forward forecasts ...")
    fc = forecast(df, args.lookback, args.horizon, args.k, args.min_history)

    signal = ((fc["exp_ret"] > args.threshold) & (fc["p_up"] > args.min_prob)).astype(float)
    signal[fc["exp_ret"].isna()] = np.nan

    daily = df["Close"].pct_change()
    pos = signal.shift(1)  # decide at close t, earn return of t+1
    start = pos.first_valid_index()
    pos, daily = pos.loc[start:].fillna(0), daily.loc[start:]
    costs = pos.diff().abs().fillna(0) * args.cost_bps / 1e4
    strat = pos * daily - costs

    sma = (df["Close"] > df["Close"].rolling(200).mean()).astype(float).shift(1).loc[start:]
    sma_ret = sma * daily - sma.diff().abs().fillna(0) * args.cost_bps / 1e4

    results = pd.DataFrame(
        {
            "Kronos-style": metrics(strat, pos),
            "Buy & hold": metrics(daily),
            "SMA200 trend": metrics(sma_ret, sma),
        }
    )
    print(f"\nTest period: {start.date()} -> {df.index[-1].date()} ({len(daily)} days)")
    print(f"Params: lookback={args.lookback} horizon={args.horizon} k={args.k} "
          f"threshold={args.threshold} min_prob={args.min_prob} cost={args.cost_bps}bps\n")
    print(results.fillna("").to_string())

    # forecast skill: does exp_ret rank realized forward returns?
    fwd = np.log(df["Close"].shift(-args.horizon) / df["Close"])
    both = pd.concat([fc["exp_ret"], fwd], axis=1).dropna()
    ic = both.corr(method="spearman").iloc[0, 1]
    print(f"\nForecast rank IC (exp_ret vs realized {args.horizon}d return): {ic:.3f}")

    yearly = pd.DataFrame(
        {
            "Kronos-style": (1 + strat).groupby(strat.index.year).prod() - 1,
            "Buy & hold": (1 + daily).groupby(daily.index.year).prod() - 1,
        }
    )
    print("\nCalendar-year returns:")
    print(yearly.map(lambda x: f"{x:.1%}").to_string())

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for name, r in [("Kronos-style", strat), ("Buy & hold", daily), ("SMA200 trend", sma_ret)]:
        ax1.plot((1 + r).cumprod(), label=name)
    ax1.set_yscale("log")
    ax1.set_title(f"{args.ticker} Kronos-style analog forecast backtest")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    eq = (1 + strat).cumprod()
    ax2.fill_between(eq.index, eq / eq.cummax() - 1, 0, alpha=0.5, label="Kronos-style DD")
    bh = (1 + daily).cumprod()
    ax2.plot(bh / bh.cummax() - 1, lw=0.8, color="gray", label="Buy & hold DD")
    ax2.set_ylabel("Drawdown")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved chart to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="offline OHLCV CSV with a Date column")
    p.add_argument("--lookback", type=int, default=20, help="candles per pattern window")
    p.add_argument("--horizon", type=int, default=5, help="forecast horizon in days")
    p.add_argument("--k", type=int, default=50, help="number of sampled analog paths")
    p.add_argument("--threshold", type=float, default=0.0, help="min expected log return")
    p.add_argument("--min-prob", type=float, default=0.55, help="min P(up) to go long")
    p.add_argument("--min-history", type=int, default=500, help="analog library size before trading")
    p.add_argument("--cost-bps", type=float, default=5.0, help="cost per position change")
    p.add_argument("--out", default="kronos_backtest.png")
    run(p.parse_args())
