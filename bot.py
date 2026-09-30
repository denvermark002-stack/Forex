import os
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
MANUAL = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "Gold": "GC=F",
}
TFS = {"1m": "1d", "5m": "5d", "15m": "5d"}


def analyze(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + g / ls)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    sma, sd = c.rolling(20).mean(), c.rolling(20).std()
    pctb = (c - (sma - 2 * sd)) / (4 * sd)
    k = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    stoch = k.rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    price = float(c.iloc[-1])

    s = 0
    s += 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
    s += 1 if macd.iloc[-1] > sig.iloc[-1] else -1
    r = rsi.iloc[-1]
    s += 1 if r > 55 else (-1 if r < 45 else 0)
    b = pctb.iloc[-1]
    s += 1 if b < 0.15 else (-1 if b > 0.85 else 0)
    st_ = stoch.iloc[-1]
    s += 1 if st_ < 20 else (-1 if st_ > 80 else 0)
    if price - l.tail(30).min() < 0.5 * atr:
        s += 1
    elif h.tail(30).max() - price < 0.5 * atr:
        s -= 1
    return s, price, atr


def label(s):
    return "BUY" if s >= 2 else ("SELL" if s <= -2 else "WAIT")


def send(text):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )


lines = []
for name, sym in PAIRS.items():
    try:
        res = {}
        for tf, period in TFS.items():
            df = yf.download(sym, period=period, interval=tf, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 60:
                raise ValueError("kom data")
            res[tf] = analyze(df)
    except Exception:
        lines.append(f"{name}: data ashenai")
        continue

    lb = {tf: label(v[0]) for tf, v in res.items()}
    if all(x == "BUY" for x in lb.values()):
        final = "STRONG BUY"
    elif all(x == "SELL" for x in lb.values()):
        final = "STRONG SELL"
    else:
        final = "WAIT"

    _, p, a = res["5m"]
    detail = f"{name}: {final} | 1m {lb['1m']}, 5m {lb['5m']}, 15m {lb['15m']} | price {p:.5f}"
    if final == "STRONG BUY":
        detail += f"\nSL {p - 1.5 * a:.5f} | TP {p + 2 * a:.5f}"
    elif final == "STRONG SELL":
        detail += f"\nSL {p + 1.5 * a:.5f} | TP {p - 2 * a:.5f}"

    if final != "WAIT" or MANUAL:
        lines.append(detail)

if lines:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    send(f"Forex signal ({now})\n\n" + "\n\n".join(lines) + "\n\nEta 100% sure na. Demo te test koro.")
