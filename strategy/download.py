"""Download daily OHLCV (split-adjusted) for current S&P 500 constituents from Yahoo's chart API.

Writes one CSV per symbol (date,open,high,low,close,volume) for backtest.py.
Usage: python download.py OUT_DIR
"""
import csv, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

START, END = 1451606400, int(time.time())  # 2016-01-01 -> now
UA = {"User-Agent": "Mozilla/5.0"}
CONST = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()


def fetch(sym, out):
    path = os.path.join(out, f"{sym}.csv")
    if os.path.exists(path):
        return sym, "cached"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={START}&period2={END}&interval=1d"
    err = None
    for attempt in range(6):
        try:
            r = json.loads(get(url))["chart"]["result"][0]
            q = r["indicators"]["quote"][0]
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                for i, ts in enumerate(r["timestamp"]):
                    row = [q[k][i] for k in ("open", "high", "low", "close", "volume")]
                    if None not in row:
                        w.writerow([time.strftime("%Y-%m-%d", time.gmtime(ts))] + row)
            return sym, "ok"
        except urllib.error.HTTPError as e:
            err = e
            if e.code in (400, 404):  # delisted / unknown ticker: don't retry
                break
            time.sleep(2 ** attempt)
        except Exception as e:
            err = e
            time.sleep(2 ** attempt)
    return sym, f"fail {err}"


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(io.StringIO(get(CONST).decode())))
    syms = sorted({r["Symbol"].replace(".", "-") for r in rows})
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda s: fetch(s, out), syms))
    bad = [r for r in res if r[1] not in ("ok", "cached")]
    print(f"{len(syms)} symbols, {len(bad)} failed: {bad[:10]}")
