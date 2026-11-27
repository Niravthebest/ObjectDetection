"""Models what buying a ~40-delta call instead of the stock would have
done on every trade in an existing deep-capitulation trade log.

This is NOT real historical option data -- Robinhood's options history
goes back far enough, but for these names it's mostly flat/stale
illiquid quotes, and strike grids are too coarse to find a true 40-delta
strike on sub-$5 stocks. Instead this uses Black-Scholes:
  - volatility = the underlying's own trailing 20-trading-day realized
    vol (annualized), as of the entry date -- a stand-in for implied vol
  - time to expiry = calendar days from entry_date to exit_date (the
    SAME exit date as the stock trade's 30-trading-day hold), i.e. the
    call is modeled as expiring exactly when the stock position would
    have closed
  - strike is solved so the call's delta is exactly 0.40 at entry
  - the option is valued at entry via Black-Scholes, and at "exit" via
    its intrinsic value at expiry (max(S_exit - K, 0)) -- a European
    terminal payoff, not a mid-life resale
  - risk-free rate fixed at 3%; entry premium floored at $0.05/share to
    avoid absurd leverage artifacts from near-zero modeled premiums

Caveats this does NOT capture: real bid-ask spread, actual implied vol
(which for these biotech names moves a lot around catalysts -- this
model holds vol constant at the entry-day realized-vol estimate for
the option's whole life), early assignment, or the real strike grid
(it uses the exact solved strike, not the nearest listed one).

Usage:
    python3 options_overlay_backtest.py --trades results/trades_deep_hold30d_2017-01-01_2025-12-31.csv
"""
import argparse
import math

import numpy as np
import pandas as pd

from backtest import load_universe

R = 0.03  # flat risk-free rate assumption
TARGET_DELTA = 0.40
VOL_LOOKBACK = 20  # trading days
MIN_PREMIUM = 0.05  # per-share floor, avoids division blowups
VOL_CAP = 3.0  # 300% annualized -- caps a handful of extreme realized-vol
               # estimates (e.g. single-day news-driven crashes >90% in the
               # trailing window) that would otherwise produce absurd strikes


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_ppf(p):
    # Acklam's rational approximation, double-precision accurate
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def bs_call_price(s, k, t, r, sigma):
    if t <= 0:
        return max(s - k, 0.0)
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)


def solve_strike_for_delta(s, t, r, sigma, target_delta):
    d1_target = norm_ppf(target_delta)
    k = s * math.exp((r + 0.5 * sigma ** 2) * t - d1_target * sigma * math.sqrt(t))
    return k


def realized_vol(price_series, entry_idx, lookback):
    window = price_series.iloc[max(0, entry_idx - lookback):entry_idx]
    if len(window) < lookback // 2:
        return None
    log_rets = np.log(window / window.shift(1)).dropna()
    if log_rets.std() == 0 or len(log_rets) < 5:
        return None
    return min(log_rets.std() * math.sqrt(252), VOL_CAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/options_overlay_trades.csv")
    ap.add_argument("--delta", type=float, default=TARGET_DELTA)
    args = ap.parse_args()
    target_delta = args.delta

    trades = pd.read_csv(args.trades)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    universe = load_universe(args.data_dir)

    rows = []
    skipped = 0
    for _, row in trades.iterrows():
        sym = row["symbol"]
        df = universe.get(sym)
        if df is None:
            skipped += 1
            continue
        entry_rows = df.index[df["date"] == row["entry_date"]]
        if len(entry_rows) == 0:
            skipped += 1
            continue
        entry_idx = entry_rows[0]
        sigma = realized_vol(df["close"], entry_idx, VOL_LOOKBACK)
        if sigma is None or sigma <= 0:
            skipped += 1
            continue

        s0 = row["entry_price"]
        s_exit = row["exit_price"]
        t0_days = (row["exit_date"] - row["entry_date"]).days
        if t0_days <= 0:
            skipped += 1
            continue
        t0 = t0_days / 365.25

        k = solve_strike_for_delta(s0, t0, R, sigma, target_delta)
        entry_premium = max(bs_call_price(s0, k, t0, R, sigma), MIN_PREMIUM)
        exit_value = max(s_exit - k, 0.0)
        option_return = exit_value / entry_premium - 1.0

        rows.append({
            "symbol": sym,
            "signal_date": row["signal_date"],
            "entry_date": row["entry_date"],
            "exit_date": row["exit_date"],
            "stock_entry_price": s0,
            "stock_exit_price": s_exit,
            "stock_return": row["return"],
            "annualized_vol_used": sigma,
            "days_to_expiry": t0_days,
            "strike": k,
            "moneyness_pct": (k / s0 - 1.0) * 100,
            "option_entry_premium": entry_premium,
            "option_exit_value": exit_value,
            "option_return": option_return,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    n = len(out_df)
    print(f"Modeled {n} option trades ({skipped} skipped for missing data/history)")
    print(f"\n--- Stock version (for reference) ---")
    print(f"Win rate: {(trades['return']>0).mean():.1%}")
    print(f"Avg return: {trades['return'].mean():+.2%}  Median: {trades['return'].median():+.2%}")

    print(f"\n--- Modeled 40-delta call version ---")
    print(f"Win rate: {(out_df['option_return']>0).mean():.1%}")
    print(f"Avg return: {out_df['option_return'].mean():+.2%}  Median: {out_df['option_return'].median():+.2%}")
    print(f"Trades with option fully worthless (-100%): {(out_df['option_return']<=-0.999).sum()} ({(out_df['option_return']<=-0.999).mean():.1%})")
    print(f"Trades >=100% option return: {(out_df['option_return']>=1.0).sum()}")
    print(f"Avg moneyness (strike vs spot at entry): {out_df['moneyness_pct'].mean():+.1f}%")
    print(f"Avg days to expiry: {out_df['days_to_expiry'].mean():.1f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
