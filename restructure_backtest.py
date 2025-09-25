# app.py
import os, time, math, json
from datetime import datetime, timedelta
from functools import lru_cache

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# minimal Black-Scholes and implied vol utility
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------------------
# CONFIG / HELPERS
# ---------------------------
POLY_BASE = "https://api.polygon.io"
API_KEY = "6TeqTIlpZNcY4nVzdFq37cpMjUCe8ZYA"  # hardcoded

st.set_page_config(layout="wide", page_title="Options Event Backtester", initial_sidebar_state="expanded")

def bs_price(cp_flag, S, K, r, q, sigma, t):
    if t <= 0:
        return max(0.0, (S - K) if cp_flag=='c' else (K-S))
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    if cp_flag=='c':
        return S*math.exp(-q*t)*norm.cdf(d1) - K*math.exp(-r*t)*norm.cdf(d2)
    else:
        return K*math.exp(-r*t)*norm.cdf(-d2) - S*math.exp(-q*t)*norm.cdf(-d1)

def implied_vol(cp_flag, S, K, r, q, t, market_price, tol=1e-6, maxiter=100):
    if market_price <= 0:
        return 0.0
    func = lambda vol: bs_price(cp_flag, S, K, r, q, vol, t) - market_price
    try:
        return brentq(func, 1e-6, 5.0, maxiter=maxiter, xtol=tol)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def polygon_get(path, params=None, base=POLY_BASE):
    if params is None: params = {}
    params['apiKey'] = API_KEY
    url = f"{base}{path}"
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.warning(f"Polygon request failed: {r.status_code} {r.text}")
        return None
    return r.json()

# ---------------------------
# POLYGON ETL FUNCTIONS
# ---------------------------
def get_underlying_aggregates(ticker, from_dt, to_dt, timespan='1/day'):
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_dt}/{to_dt}"
    res = polygon_get(path)
    if not res or 'results' not in res:
        return pd.DataFrame()
    df = pd.DataFrame(res['results'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','vw':'vwap'})
    if 'vwap' not in df.columns:
        df['vwap'] = np.nan
    return df[['t','open','high','low','close','volume','vwap']]

@st.cache_data(show_spinner=False)
def get_option_contracts(ticker, expiration_gte=None, expiration_lte=None, limit=1000):
    path = "/v3/reference/options/contracts"
    params = {'underlying_ticker': ticker, 'limit': limit}
    if expiration_gte: params['expiration_date.gte'] = expiration_gte
    if expiration_lte: params['expiration_date.lte'] = expiration_lte
    res = polygon_get(path, params)
    if not res or 'results' not in res:
        return pd.DataFrame()
    return pd.DataFrame(res['results'])

@st.cache_data(show_spinner=False)
def get_option_trades(option_symbol, start_ms, end_ms, limit=5000):
    path = f"/v3/trades/{option_symbol}"
    params = {'timestamp.gte': start_ms, 'timestamp.lte': end_ms, 'limit': limit}
    res = polygon_get(path, params)
    if not res or 'results' not in res:
        return pd.DataFrame()
    df = pd.DataFrame(res['results'])
    return df

# ---------------------------
# BACKTEST ENGINE (simplified)
# ---------------------------
def simulate_simple_strategy(underlying_df, option_contracts_df, trades_df_by_symbol,
                             strategy='straddle', entry_date=None, exit_days=7,
                             capital=10000, fee_per_trade=1.0, slippage=0.01):
    S_row = underlying_df[underlying_df['t'].dt.date == entry_date.date()]
    if S_row.empty:
        S = underlying_df.iloc[-1]['close']
    else:
        S = float(S_row.iloc[0]['close'])
    option_contracts_df['expiration_date'] = pd.to_datetime(option_contracts_df['expiration_date'])
    target_exp = entry_date + timedelta(days=exit_days)
    candidate = option_contracts_df.iloc[((option_contracts_df['expiration_date'] - pd.Timestamp(entry_date)).abs()).argsort()][:200]
    candidate['strike_diff'] = abs(candidate['strike'] - S)
    atm = candidate.sort_values('strike_diff').iloc[0]
    atm_strike = atm['strike']
    call_sym = atm['symbol'] if atm['type']=='call' else candidate[(candidate['type']=='call') & (candidate['strike']==atm_strike)].iloc[0]['symbol']
    put_sym = atm['symbol'] if atm['type']=='put' else candidate[(candidate['type']=='put') & (candidate['strike']==atm_strike)].iloc[0]['symbol']
    def mid_at(symbol, ts_ms):
        if symbol in trades_df_by_symbol:
            df = trades_df_by_symbol[symbol]
            df['ts'] = pd.to_datetime(df['t'], unit='ms')
            near = df.iloc[(df['ts'] - pd.Timestamp(ts_ms, unit='ms')).abs().argsort()[:1]]
            if not near.empty:
                return float(near.iloc[0]['p'])
        return np.nan
    entry_ms = int(entry_date.timestamp() * 1000)
    exit_ms = int((entry_date + timedelta(days=exit_days)).timestamp() * 1000)
    call_entry_p = mid_at(call_sym, entry_ms) * (1 + slippage)
    put_entry_p  = mid_at(put_sym, entry_ms) * (1 + slippage)
    call_exit_p = mid_at(call_sym, exit_ms) * (1 - slippage)
    put_exit_p  = mid_at(put_sym, exit_ms) * (1 - slippage)
    paid = (call_entry_p + put_entry_p) * 100
    received = (call_exit_p + put_exit_p) * 100
    pnl = received - paid - fee_per_trade*2
    return {"pnl": pnl, "paid": paid, "received": received, "call_sym": call_sym, "put_sym": put_sym}

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Options Event Backtester — Event-driven strategies + cascading NN entry")

with st.sidebar:
    st.header("Config")
    ticker = st.text_input("Underlying ticker", value="PARA")
    event_date = st.date_input("Event anchor date", value=datetime(2025,8,7).date())
    before_days = st.slider("Days before event to pull", 1, 60, 14)
    after_days = st.slider("Days after event to pull", 7, 180, 90)
    strategy = st.selectbox("Strategy", ["straddle","long_call","long_put","strangle","vertical"])
    exit_days = st.selectbox("Exit horizon (days)", [1,3,7,14,30,90], index=2)
    capital = st.number_input("Capital per trade ($)", value=10000)
    run_btn = st.button("Fetch & Run Backtest")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Market & Options Data")
    if run_btn:
        start_dt = datetime.combine(event_date, datetime.min.time()) - timedelta(days=before_days)
        end_dt = datetime.combine(event_date, datetime.min.time()) + timedelta(days=after_days)
        st.info(f"Pulling underlying daily aggregates {start_dt.date()} → {end_dt.date()} (may take some seconds)")
        underlying = get_underlying_aggregates(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        st.write("Underlying samples:", underlying.shape)
        opt_df = get_option_contracts(ticker, expiration_gte=start_dt.strftime("%Y-%m-%d"),
                                      expiration_lte=end_dt.strftime("%Y-%m-%d"), limit=1000)
        st.write("Option contracts sample:", opt_df.shape)
        st.dataframe(opt_df.head(10))
        if 'open_interest' in opt_df.columns:
            top = opt_df.sort_values('open_interest', ascending=False).head(10)
        else:
            top = opt_df.head(10)
        trades = {}
        st.write("Fetching trade samples for top contracts (may be rate-limited)...")
        for sym in top['symbol'].unique():
            try:
                start_ms = int((datetime.combine(event_date, datetime.min.time()) - timedelta(hours=2)).timestamp()*1000)
                end_ms = int((datetime.combine(event_date, datetime.min.time()) + timedelta(hours=6)).timestamp()*1000)
                df_tr = get_option_trades(sym, start_ms, end_ms)
                if not df_tr.empty:
                    trades[sym] = df_tr
            except Exception as e:
                st.error(f"trade fetch error {e}")
        st.success(f"Fetched contracts: {len(opt_df)} and trades for {len(trades)} symbols.")
        st.session_state['underlying'] = underlying
        st.session_state['opt_df'] = opt_df
        st.session_state['trades'] = trades

    if 'underlying' in st.session_state:
        u = st.session_state['underlying']
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=u['t'], open=u['open'], high=u['high'], low=u['low'], close=u['close'], name=ticker))
        fig.update_layout(height=350, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Quick Strategy Run")
    if st.button("Quick ATM Straddle (event day)"):
        if 'underlying' not in st.session_state:
            st.error("Run fetch first")
        else:
            underlying = st.session_state['underlying']
            opt_df = st.session_state['opt_df']
            trades = st.session_state['trades']
            entry_dt = datetime.combine(event_date, datetime.min.time())
            result = simulate_simple_strategy(underlying, opt_df, trades, strategy="straddle", entry_date=entry_dt, exit_days=exit_days, capital=capital)
            st.metric("Simulated trade PnL (USD)", f"{result['pnl']:.2f}")
            st.json(result)

st.markdown("---")
st.subheader("Notes & next steps")
st.write("""
- This prototype uses trade-level data as a nearest-trade proxy for mid prices; to produce production-quality fills, use real NBBO bid/ask snapshots or orderbook-level data and incorporate slippage/market impact models.
- The cascading NN inference (not included inline for brevity) plugs into the 'Quick Strategy Run' stage.
- Persist cached contracts/trades for repeated runs and add a retraining scheduler for the cascade model.
""")