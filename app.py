from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live Forex Signal", layout="centered")
st.title("Live Forex Signal")

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "Gold (XAU/USD)": "GC=F",
}
TFS = {"1m": "1d", "5m": "5d", "15m": "5d"}

pair = st.selectbox("Pair", list(PAIRS))
refresh = st.select_slider("Auto refresh (sec)", options=[10, 15, 30, 60], value=15)


@st.cache_data(ttl=10)
def load(sym, period, interval):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna()


def analyze(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()

    sma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    upper, lower = sma + 2 * sd, sma - 2 * sd
    pctb = (c - lower) / (upper - lower)

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - lo14) / (hi14 - lo14)
    stoch = k.rolling(3).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    support, resistance = l.tail(30).min(), h.tail(30).max()
    price, a = float(c.iloc[-1]), float(atr.iloc[-1])

    score, why = 0, []

    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 1; why.append("EMA20 > EMA50 (trend upore)")
    else:
        score -= 1; why.append("EMA20 < EMA50 (trend niche)")

    if macd.iloc[-1] > sig.iloc[-1]:
        score += 1; why.append("MACD bullish")
    else:
        score -= 1; why.append("MACD bearish")

    r = float(rsi.iloc[-1])
    if r > 55:
        score += 1; why.append(f"RSI {r:.0f} (buyer strong)")
    elif r < 45:
        score -= 1; why.append(f"RSI {r:.0f} (seller strong)")
    else:
        why.append(f"RSI {r:.0f} (neutral)")

    b = float(pctb.iloc[-1])
    if b < 0.15:
        score += 1; why.append("Bollinger lower band er kase (bounce somvabona)")
    elif b > 0.85:
        score -= 1; why.append("Bollinger upper band er kase (pullback somvabona)")
    else:
        why.append("Bollinger madhyokhane")

    s = float(stoch.iloc[-1])
    if s < 20:
        score += 1; why.append(f"Stochastic {s:.0f} (oversold)")
    elif s > 80:
        score -= 1; why.append(f"Stochastic {s:.0f} (overbought)")
    else:
        why.append(f"Stochastic {s:.0f} (neutral)")

    if price - support < 0.5 * a:
        score += 1; why.append("Support er kase")
    elif resistance - price < 0.5 * a:
        score -= 1; why.append("Resistance er kase")
    else:
        why.append("Support/Resistance theke dure")

    return {"score": score, "why": why, "price": price, "atr": a,
            "df": df.assign(EMA20=ema20, EMA50=ema50)}


def label(score):
    if score >= 2:
        return "BUY"
    if score <= -2:
        return "SELL"
    return "WAIT"


@st.fragment(run_every=refresh)
def live():
    results = {}
    for tf, period in TFS.items():
        try:
            df = load(PAIRS[pair], period, tf)
            if len(df) < 60:
                raise ValueError("kom data")
            results[tf] = analyze(df)
        except Exception:
            st.warning(f"{tf} data ashenai. Bazar bondho thakte pare (weekend).")
            return

    labels = {tf: label(r["score"]) for tf, r in results.items()}

    if all(v == "BUY" for v in labels.values()):
        final = "STRONG BUY"
    elif all(v == "SELL" for v in labels.values()):
        final = "STRONG SELL"
    elif labels["15m"] == "BUY" and "BUY" in (labels["1m"], labels["5m"]):
        final = "BUY"
    elif labels["15m"] == "SELL" and "SELL" in (labels["1m"], labels["5m"]):
        final = "SELL"
    else:
        final = "WAIT"

    if "BUY" in final:
        st.success(f"## {final}")
    elif "SELL" in final:
        st.error(f"## {final}")
    else:
        st.warning("## WAIT - timeframe gulo ekmot na")

    cols = st.columns(3)
    for col, (tf, r) in zip(cols, results.items()):
        col.metric(tf, labels[tf], f"score {r['score']:+d}/6", delta_color="off")

    p = results["5m"]["price"]
    a = results["5m"]["atr"]
    st.metric("Price", f"{p:.5f}")
    if "BUY" in final:
        st.write(f"**SL:** {p - 1.5 * a:.5f}  |  **TP:** {p + 2 * a:.5f}")
    elif "SELL" in final:
        st.write(f"**SL:** {p + 1.5 * a:.5f}  |  **TP:** {p - 2 * a:.5f}")

    for tf, r in results.items():
        with st.expander(f"{tf} analysis"):
            for w in r["why"]:
                st.write("- " + w)

    st.line_chart(results["1m"]["df"][["Close", "EMA20", "EMA50"]].tail(120))
    st.caption(
        "Last update: " + datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        + " | Data kichu minute deri hote pare. Kono signal 100% sure na."
    )


live()
st.caption("Shudhu shikkhar jonno. Age demo account e test koro, real taka risk koro na.")
