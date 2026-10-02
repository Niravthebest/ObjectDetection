"""
Jesse Stine "Insider Buy Superstocks"-style strategy + weekly backtester.

Stine's playbook (price/volume parts that can be tested mechanically):
  * Market timing: only buy when the broad market is healthy (here: SPY weekly
    close above its 40-week SMA) -- he sat in cash during hostile markets.
  * Setup: a stock coming out of a long base (multi-month consolidation) and
    breaking to a new 26-week high ("blue sky" breakout).
  * Confirmation: the breakout week prints a big volume surge (>= 2x the
    10-week average), the stock is above a rising 30-week SMA.
  * Concentration: a handful of positions (max 5), equal weight.
  * Selling: Stine sells INTO strength when a stock goes parabolic -- here a
    weekly close more than X% above the 30-week SMA, or a weekly RSI blow-off.
    Losers are cut fast: hard stop at -20% from entry, or a weekly close back
    below the 10-week SMA once the trade is under water / below the 30-week
    SMA at any time.

Not testable with free price data (documented, not modelled): insider
buying (Form 4 clusters), float size, short interest and sentiment readings.

Execution: signals on a weekly close, fills at the NEXT week's open, with a
per-side cost (commission + slippage).
"""
from __future__ import annotations

import dataclasses as dc
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dc.dataclass
class Params:
    base_weeks: int = 26          # breakout above highest high of prior N weeks
    max_base_depth: float = 0.60  # (base high - base low) / base high must be <= this
    vol_mult: float = 2.0         # breakout-week volume vs 10-week avg volume
    min_price: float = 3.0
    min_dollar_vol: float = 2e6   # avg weekly $ volume / 5 (approx daily), USD
    parabolic_pct: float = 0.80   # sell when close > (1+x) * 30-week SMA
    rsi_blowoff: float = 85.0     # or weekly RSI(14) above this
    hard_stop: float = 0.20       # -20% from entry, checked on weekly close
    max_positions: int = 5
    cost_per_side: float = 0.0025 # 0.25% commission+slippage per side
    regime_ma: int = 40           # SPY weekly SMA for market timing


# ---------------------------------------------------------------- indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    c, v = df["close"], df["volume"]
    df["sma10"] = c.rolling(10).mean()
    df["sma30"] = c.rolling(30).mean()
    df["sma30_rising"] = df["sma30"] > df["sma30"].shift(4)
    df["base_high"] = df["high"].shift(1).rolling(p.base_weeks).max()
    df["base_low"] = df["low"].shift(1).rolling(p.base_weeks).min()
    df["base_depth"] = (df["base_high"] - df["base_low"]) / df["base_high"]
    df["avg_vol10"] = v.shift(1).rolling(10).mean()
    df["dollar_vol"] = (c * v).rolling(10).mean() / 5
    df["rsi"] = rsi(c)
    df["entry_signal"] = (
        (c > df["base_high"])
        & (df["base_depth"] <= p.max_base_depth)
        & (v >= p.vol_mult * df["avg_vol10"])
        & (c > df["sma30"]) & df["sma30_rising"]
        & (c >= p.min_price)
        & (df["dollar_vol"] >= p.min_dollar_vol)
    )
    # strength score used to rank simultaneous breakouts: volume surge
    df["score"] = v / df["avg_vol10"]
    return df


# ---------------------------------------------------------------- backtest
def backtest(data: dict[str, pd.DataFrame], spy: pd.DataFrame, p: Params,
             start: str, end: str, capital: float = 100_000.0):
    """data: symbol -> weekly OHLCV DataFrame indexed by week-start date."""
    ind = {s: add_indicators(df, p) for s, df in data.items()}
    spy = spy.copy()
    spy["ma"] = spy["close"].rolling(p.regime_ma).mean()
    risk_on = spy["close"] > spy["ma"]

    weeks = spy.index[(spy.index >= start) & (spy.index <= end)]
    cash = capital
    pos: dict[str, dict] = {}          # sym -> {shares, entry, entry_date}
    pending_buys: list[str] = []
    pending_sells: dict[str, str] = {}  # sym -> reason
    trades, equity = [], []

    def bar(sym, wk):
        df = ind[sym]
        return df.loc[wk] if wk in df.index else None

    for wk in weeks:
        # 1) execute last week's orders at this week's open
        for sym, reason in list(pending_sells.items()):
            b = bar(sym, wk)
            if b is None or not np.isfinite(b["open"]):
                continue  # no bar (halt/delist) -> try next week
            px = b["open"] * (1 - p.cost_per_side)
            ps = pos.pop(sym)
            cash += ps["shares"] * px
            trades.append(dict(symbol=sym, entry_date=ps["entry_date"], entry=ps["entry"],
                               exit_date=wk, exit=px, ret=px / ps["entry"] - 1, reason=reason))
            del pending_sells[sym]

        # equity at open for sizing
        mv = sum(ps["shares"] * (bar(s, wk)["open"] if bar(s, wk) is not None else ps["last"])
                 for s, ps in pos.items())
        slots = p.max_positions - len(pos)
        for sym in pending_buys[:max(slots, 0)]:
            b = bar(sym, wk)
            if b is None or sym in pos:
                continue
            target = (cash + mv) / p.max_positions
            px = b["open"] * (1 + p.cost_per_side)
            alloc = min(target, cash)
            if alloc < 0.2 * target:
                break
            sh = alloc / px
            cash -= sh * px
            pos[sym] = dict(shares=sh, entry=px, entry_date=wk, last=b["open"])
            mv += sh * b["open"]
        pending_buys = []

        # 2) mark to market at the close, generate exits
        for sym, ps in pos.items():
            b = bar(sym, wk)
            if b is None:
                continue
            ps["last"] = b["close"]
            c = b["close"]
            reason = None
            if c > (1 + p.parabolic_pct) * b["sma30"]:
                reason = "parabolic_ext"
            elif b["rsi"] >= p.rsi_blowoff:
                reason = "rsi_blowoff"
            elif c <= ps["entry"] * (1 - p.hard_stop):
                reason = "hard_stop"
            elif c < b["sma30"]:
                reason = "below_30wk"
            elif c < b["sma10"] and c < ps["entry"]:
                reason = "below_10wk_loss"
            if reason and sym not in pending_sells:
                pending_sells[sym] = reason

        eq = cash + sum(ps["shares"] * ps["last"] for ps in pos.values())
        equity.append((wk, eq, len(pos)))

        # 3) new entries (only in a healthy market)
        if risk_on.get(wk, False):
            cands = []
            for sym, df in ind.items():
                if sym in pos or wk not in df.index:
                    continue
                r = df.loc[wk]
                if bool(r["entry_signal"]):
                    cands.append((r["score"], sym))
            pending_buys = [s for _, s in sorted(cands, reverse=True)]

    eq = pd.DataFrame(equity, columns=["week", "equity", "positions"]).set_index("week")
    # open positions at the end are marked to market
    for sym, ps in pos.items():
        trades.append(dict(symbol=sym, entry_date=ps["entry_date"], entry=ps["entry"],
                           exit_date=None, exit=ps["last"], ret=ps["last"] / ps["entry"] - 1,
                           reason="open"))
    return eq, pd.DataFrame(trades)


def stats(eq: pd.DataFrame, trades: pd.DataFrame) -> dict:
    e = eq["equity"]
    r = e.pct_change().dropna()
    yrs = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    total = e.iloc[-1] / e.iloc[0] - 1
    cagr = (1 + total) ** (1 / yrs) - 1 if yrs >= 1 else np.nan
    dd = (e / e.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else np.nan
    closed = trades[trades["reason"] != "open"] if len(trades) else trades
    win = (closed["ret"] > 0).mean() if len(closed) else np.nan
    pf = (closed.loc[closed.ret > 0, "ret"].sum() / -closed.loc[closed.ret < 0, "ret"].sum()
          if len(closed) and (closed.ret < 0).any() else np.nan)
    return dict(total_return=total, cagr=cagr, max_drawdown=dd, sharpe=sharpe,
                trades=int(len(trades)), win_rate=win, profit_factor=pf,
                avg_trade=closed["ret"].mean() if len(closed) else np.nan,
                exposure=(eq["positions"] > 0).mean())
