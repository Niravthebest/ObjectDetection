"""Exploratory condition-discovery sweep for biotech stocks.

Biotech moves are disproportionately driven by binary catalysts (trial
readouts, FDA decisions) rather than smooth technical continuation, so
this script frames "predicts a price move" as: does condition X raise
the odds of a LARGE subsequent move (either direction), and/or does it
tilt the direction of that move, versus the unconditional baseline for
this universe?

For every trading day of every symbol, compute a battery of candidate
technical conditions, then measure forward returns over several
horizons. Report each condition's lift over baseline, with a minimum
sample-size filter so thin-sample noise doesn't look like an edge.

Usage:
    python3 analyze_conditions.py --data-dir data --out-dir results
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from indicators import (
    sma, atr, natr, rsi, pgo, bollinger_width,
    pct_from_52w_high, pct_from_52w_low, volume_ratio, gap_pct,
    trend_up, trend_dn,
)

MIN_HISTORY_BARS = 260
HORIZONS = [3, 5, 10, 20]
MOVE_THRESHOLDS = [0.10, 0.15, 0.20, 0.30]  # |forward return| >= this counts as a "big move"
MIN_SAMPLE = 40  # minimum True-count for a condition to be reported


def load_universe(data_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.drop_duplicates(subset=["symbol", "date"])
    all_df = all_df.sort_values(["symbol", "date"])
    return {sym: g.reset_index(drop=True) for sym, g in all_df.groupby("symbol")}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low, open_, vol = out["close"], out["high"], out["low"], out["open"], out["volume"]

    out["natr10"] = natr(high, low, close, 10)
    out["natr20"] = natr(high, low, close, 20)
    out["natr50"] = natr(high, low, close, 50)
    out["natr_contraction"] = out["natr10"] / out["natr50"]  # <1 = short-term vol below long-run norm

    out["bb_width20"] = bollinger_width(close, 20)
    out["bb_width_pctile"] = out["bb_width20"].rolling(252, min_periods=60).rank(pct=True)  # squeeze = low pctile

    out["rsi7"] = rsi(close, 7)
    out["rsi14"] = rsi(close, 14)

    out["pgo20"] = pgo(high, low, close, 20)
    out["pgo50"] = pgo(high, low, close, 50)

    out["pct_from_52w_high"] = pct_from_52w_high(close)
    out["pct_from_52w_low"] = pct_from_52w_low(close)

    out["vol_ratio10"] = volume_ratio(vol, 10)
    out["vol_ratio20"] = volume_ratio(vol, 20)
    out["vol_ratio50"] = volume_ratio(vol, 50)

    out["gap_pct"] = gap_pct(open_, close)

    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["above_sma50"] = close > out["sma50"]
    out["above_sma200"] = close > out["sma200"]
    out["sma50_trend_up"] = trend_up(out["sma50"], 20)
    out["sma200_trend_dn"] = trend_dn(out["sma200"], 40)

    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    for h in HORIZONS:
        out[f"fwd_ret_{h}d"] = close.shift(-h) / close - 1.0

    return out


def build_condition_table(feat: pd.DataFrame) -> dict:
    """Return {condition_name: boolean Series} for every candidate
    condition, each a single, independently-interpretable screen."""
    c = {}
    c["vol_contraction_tight"] = feat["natr_contraction"] < 0.5
    c["vol_contraction_mild"] = feat["natr_contraction"] < 0.75
    c["bb_squeeze_tight"] = feat["bb_width_pctile"] < 0.10
    c["bb_squeeze_mild"] = feat["bb_width_pctile"] < 0.25

    c["rsi7_oversold"] = feat["rsi7"] < 30
    c["rsi7_overbought"] = feat["rsi7"] > 70
    c["rsi14_oversold"] = feat["rsi14"] < 30
    c["rsi14_overbought"] = feat["rsi14"] > 70

    c["near_52w_low"] = feat["pct_from_52w_low"] < 10
    c["near_52w_high"] = feat["pct_from_52w_high"] > -10

    c["volume_spike_2x"] = feat["vol_ratio10"] > 2.0
    c["volume_spike_3x"] = feat["vol_ratio10"] > 3.0
    c["volume_spike_5x"] = feat["vol_ratio10"] > 5.0
    c["volume_dryup"] = feat["vol_ratio20"] < 0.5

    c["gap_up_big"] = feat["gap_pct"] > 5
    c["gap_down_big"] = feat["gap_pct"] < -5
    c["gap_up_huge"] = feat["gap_pct"] > 10
    c["gap_down_huge"] = feat["gap_pct"] < -10

    c["above_sma50_200"] = feat["above_sma50"] & feat["above_sma200"]
    c["below_sma50_200"] = (~feat["above_sma50"]) & (~feat["above_sma200"])
    c["downtrend_200"] = feat["sma200_trend_dn"]
    c["uptrend_50"] = feat["sma50_trend_up"]

    c["momentum_strong_up_20d"] = feat["ret_20d"] > 0.30
    c["momentum_strong_dn_20d"] = feat["ret_20d"] < -0.30

    c["pgo_extreme_high"] = feat["pgo20"] > 3
    c["pgo_extreme_low"] = feat["pgo20"] < -3

    # combos suggested by typical catalyst-driven setups
    c["squeeze_AND_near_low"] = c["vol_contraction_mild"] & c["near_52w_low"]
    c["squeeze_AND_volume_pickup"] = c["bb_squeeze_mild"] & (feat["vol_ratio10"] > 1.5)
    c["gap_up_AND_volume_spike"] = c["gap_up_big"] & c["volume_spike_2x"]
    c["gap_down_AND_volume_spike"] = c["gap_down_big"] & c["volume_spike_2x"]
    c["oversold_AND_near_low"] = c["rsi14_oversold"] & c["near_52w_low"]
    c["downtrend_AND_oversold"] = c["downtrend_200"] & c["rsi7_oversold"]

    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    universe = load_universe(args.data_dir)
    print(f"Universe: {len(universe)} symbols")

    all_feat = []
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        feat = compute_features(df)
        feat["symbol"] = symbol
        all_feat.append(feat)
    print(f"Computed features for {len(all_feat)} symbols ({skipped} skipped for short history)")

    panel = pd.concat(all_feat, ignore_index=True)
    panel.to_parquet(os.path.join(args.out_dir, "feature_panel.parquet")) if False else None

    conditions = build_condition_table(panel)

    rows = []
    for horizon in HORIZONS:
        fwd_col = f"fwd_ret_{horizon}d"
        valid = panel[panel[fwd_col].notna()]
        baseline_mean = valid[fwd_col].mean()
        baseline_std = valid[fwd_col].std()
        n_total = len(valid)

        for thresh in MOVE_THRESHOLDS:
            baseline_big = (valid[fwd_col].abs() >= thresh).mean()
            baseline_big_up = (valid[fwd_col] >= thresh).mean()
            baseline_big_dn = (valid[fwd_col] <= -thresh).mean()

            for cond_name, cond_mask in conditions.items():
                cond_mask_aligned = cond_mask.reindex(panel.index, fill_value=False)
                sub = valid[cond_mask_aligned.reindex(valid.index, fill_value=False)]
                n = len(sub)
                if n < MIN_SAMPLE:
                    continue
                cond_mean = sub[fwd_col].mean()
                cond_big = (sub[fwd_col].abs() >= thresh).mean()
                cond_big_up = (sub[fwd_col] >= thresh).mean()
                cond_big_dn = (sub[fwd_col] <= -thresh).mean()
                # rough z-score for the big-move-rate lift (binomial approx)
                se = (baseline_big * (1 - baseline_big) / n) ** 0.5 if 0 < baseline_big < 1 else np.nan
                z = (cond_big - baseline_big) / se if se and se > 0 else np.nan

                rows.append({
                    "horizon_days": horizon,
                    "move_threshold": thresh,
                    "condition": cond_name,
                    "n": n,
                    "n_total": n_total,
                    "cond_mean_fwd_return": cond_mean,
                    "baseline_mean_fwd_return": baseline_mean,
                    "mean_return_lift": cond_mean - baseline_mean,
                    "cond_big_move_rate": cond_big,
                    "baseline_big_move_rate": baseline_big,
                    "big_move_lift": cond_big - baseline_big,
                    "big_move_lift_z": z,
                    "cond_big_up_rate": cond_big_up,
                    "baseline_big_up_rate": baseline_big_up,
                    "up_lift": cond_big_up - baseline_big_up,
                    "cond_big_dn_rate": cond_big_dn,
                    "baseline_big_dn_rate": baseline_big_dn,
                    "dn_lift": cond_big_dn - baseline_big_dn,
                })

    results = pd.DataFrame(rows)
    results = results.sort_values("big_move_lift_z", ascending=False)
    results.to_csv(os.path.join(args.out_dir, "condition_scan.csv"), index=False)
    print(f"Wrote {len(results)} condition/horizon/threshold rows to results/condition_scan.csv")

    print("\nTop 20 conditions by big-move-rate lift z-score (volatility-expansion predictors):")
    top = results[results["n"] >= 60].sort_values("big_move_lift_z", ascending=False).head(20)
    print(top[["horizon_days", "move_threshold", "condition", "n", "cond_big_move_rate",
               "baseline_big_move_rate", "big_move_lift", "big_move_lift_z"]].to_string(index=False))

    print("\nTop 15 conditions by directional UP lift (n>=60, 10d horizon, 15% threshold):")
    sub = results[(results["n"] >= 60) & (results["horizon_days"] == 10) & (results["move_threshold"] == 0.15)]
    print(sub.sort_values("up_lift", ascending=False).head(15)[
        ["condition", "n", "cond_big_up_rate", "baseline_big_up_rate", "up_lift"]].to_string(index=False))

    print("\nTop 15 conditions by directional DOWN lift (n>=60, 10d horizon, 15% threshold):")
    print(sub.sort_values("dn_lift", ascending=False).head(15)[
        ["condition", "n", "cond_big_dn_rate", "baseline_big_dn_rate", "dn_lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
