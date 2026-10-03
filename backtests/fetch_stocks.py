"""Download dividend/split-adjusted daily history for all current S&P 500 stocks
into a pickle cache: {symbol: DataFrame(open, close)}."""
import pickle, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, json
from scanner import get, sp500

def fetch(sym):
    q = sym.replace("^", "%5E")
    r = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{q}"
                       "?period1=631152000&period2=9999999999&interval=1d&events=div,split"))["chart"]["result"][0]
    qt = r["indicators"]["quote"][0]
    df = pd.DataFrame({"open": qt["open"], "close": qt["close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s").normalize())
    f = np.array(r["indicators"]["adjclose"][0]["adjclose"], dtype=float) / df["close"].values
    df = df.mul(f, axis=0)
    df = df[~df.index.duplicated(keep="last")].dropna()
    return sym, df[(df.open > 0) & (df.close > 0)].astype("float32")

if __name__ == "__main__":
    names = sp500()
    out = {}
    with ThreadPoolExecutor(6) as ex:
        for f in [ex.submit(fetch, s) for s in names]:
            try:
                s, df = f.result(); out[s] = df
            except Exception as e:
                print("warn", e, file=sys.stderr)
    pickle.dump({"prices": out, "names": names}, open(sys.argv[1], "wb"))
    print(len(out), "symbols")
