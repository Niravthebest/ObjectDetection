"""Command-line entry points:

    python -m strat_screener backtest --symbols-file symbols.txt --out strat_stats.json
    python -m strat_screener screen   --symbols-file symbols.txt --stats strat_stats.json
"""
import argparse
import json

from .backtest import run_backtest
from .data_sources import YFinanceDataSource
from .screener import run_screen

ROW_COLUMNS = ["symbol", "timeframe", "pattern", "direction", "score", "ftfc", "backtest_n", "close", "bar_date"]


def read_symbols(path: str):
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strat_screener", description="TheStrat multi-timeframe screener")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="scan history and write pattern edge stats")
    bt.add_argument("--symbols-file", required=True)
    bt.add_argument("--period", default="15y", help="yfinance history period, e.g. 10y, 15y, max")
    bt.add_argument("--horizon", type=int, default=6, help="bars forward (per timeframe) to measure outcomes")
    bt.add_argument("--out", default="strat_stats.json")

    sc = sub.add_parser("screen", help="rank current setups using backtested edge stats")
    sc.add_argument("--symbols-file", required=True)
    sc.add_argument("--stats", default="strat_stats.json")
    sc.add_argument("--period", default="15y")
    sc.add_argument("--top", type=int, default=25)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    symbols = read_symbols(args.symbols_file)

    if args.command == "backtest":
        data_source = YFinanceDataSource(period=args.period)
        stats, _frames, events = run_backtest(data_source, symbols, horizon=args.horizon)
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        print(f"Backtested {len(events)} historical setups across {len(symbols)} symbols.")
        print(f"Wrote pattern edge stats to {args.out}")

    elif args.command == "screen":
        try:
            with open(args.stats) as f:
                stats = json.load(f)
        except FileNotFoundError:
            print(f"No stats file at {args.stats!r}; run `backtest` first. Scoring with neutral priors for now.")
            stats = {}

        data_source = YFinanceDataSource(period=args.period)
        rows = run_screen(data_source, symbols, stats)[: args.top]

        print(" | ".join(col.upper() for col in ROW_COLUMNS))
        for row in rows:
            print(" | ".join(str(row[col]) for col in ROW_COLUMNS))


if __name__ == "__main__":
    main()
