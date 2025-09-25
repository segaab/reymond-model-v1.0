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
API_KEY = "6TeqTIlpZNcY4nVzdFq37cpMjUCe8ZYA"  # hardcoded per request

st.set_page_config(layout="wide", page_title="Options Event Backtester", initial_sidebar_state="expanded")

def bs_price(cp_flag, S, K, r, q, sigma, t):
    """Black-Scholes price (European). cp_flag='c' or 'p' """
    if t <= 0:
        return max(0.0, (S - K) if cp_flag=='c' else (K-S))
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    if cp_flag=='c':
        return S*math.exp(-q*t)*norm.cdf(d1) - K*math.exp(-r*t)*norm.cdf(d2)
    else:
        return K*math.exp(-r*t)*norm.cdf(-d2) - S*math.exp(-q*t)*norm.cdf(-d1)

def implied_vol(cp_flag, S, K, r, q, t, market_price, tol=1e-6, maxiter=100):
    """Brent solver for implied vol"""
    if market_price <= 0 or np.isnan(market_price):
        return np.nan
    func = lambda vol: bs_price(cp_flag, S, K, r, q, vol, t) - market_price
    try:
        return brentq(func, 1e-6, 5.0, maxiter=maxiter, xtol=tol)
    except Exception:
        return np.nan

# caching wrapper for Polygon requests
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
    # use v2/aggs/ticker for daily aggregates
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_dt}/{to_dt}"
    res = polygon_get(path)
    if not res or 'results' not in res:
        return pd.DataFrame()
    df = pd.DataFrame(res['results'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    # normalize column names and ensure vwap exists
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','vw':'vwap'})
    if 'vwap' not in df.columns:
        df['vwap'] = np.nan
    # ensure we return consistent column set used downstream
    return df[['t','open','high','low','close','volume','vwap']]

@st.cache_data(show_spinner=False)
def get_option_contracts(ticker, expiration_gte=None, expiration_lte=None, limit=1000):
    # v3 reference options
    path = "/v3/reference/options/contracts"
    params = {'underlying_ticker': ticker, 'limit': limit}
    if expiration_gte: params['expiration_date.gte'] = expiration_gte
    if expiration_lte: params['expiration_date.lte'] = expiration_lte
    res = polygon_get(path, params)
    if not res or 'results' not in res:
        return pd.DataFrame()
    df = pd.DataFrame(res['results'])
    # keep raw schema but also ensure key columns are present for building a fallback symbol
    # common columns: 'symbol','expiration_date','strike','type','underlying_ticker','open_interest'
    return df

@st.cache_data(show_spinner=False)
def get_option_trades(option_symbol, start_ms, end_ms, limit=5000):
    # note: polygon returns paginated results; implement a simple loop with next_url if available
    path = f"/v3/trades/{option_symbol}"
    params = {'timestamp.gte': start_ms, 'timestamp.lte': end_ms, 'limit': limit}
    res = polygon_get(path, params)
    if not res or 'results' not in res:
        return pd.DataFrame()
    df = pd.DataFrame(res['results'])
    return df

# ---------------------------
# UTILITIES: robust column handling
# ---------------------------
def find_symbol_column(df: pd.DataFrame):
    """Return the name of a column that contains option contract symbol, or None."""
    candidates = ['symbol', 'contract', 'option_symbol', 'contract_symbol', 'option_contract', 'id']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: look for any column containing 'symbol' substring
    for c in df.columns:
        if 'symbol' in c.lower():
            return c
    return None

def build_synthetic_symbol(row, underlying_col_candidates=('underlying_ticker','underlying','root')):
    """Concoct a symbol from available fields: underlying + expiration + strike + type"""
    # expiration in YYYY-MM-DD or YYYYMMDD
    underlying = None
    for cand in underlying_col_candidates:
        if cand in row and pd.notna(row.get(cand, None)):
            underlying = str(row.get(cand))
            break
    if underlying is None and 'underlying' in row:
        underlying = str(row.get('underlying'))
    expiration = row.get('expiration_date', '')
    if pd.isna(expiration):
        expiration = ''
    else:
        expiration = pd.to_datetime(expiration).strftime('%Y%m%d') if not isinstance(expiration, str) else expiration.replace('-','')
    strike = row.get('strike', '')
    strike_s = f"{int(strike)}" if pd.notna(strike) and float(strike).is_integer() else (str(strike) if pd.notna(strike) else '')
    typ = row.get('type', row.get('option_type', ''))
    typ_short = ('C' if str(typ).lower().startswith('c') else 'P') if typ else ''
    parts = [p for p in [underlying, expiration, strike_s, typ_short] if p is not None and p != '']
    if parts:
        return "_".join(parts)
    return None

def get_option_symbol_list(opt_df: pd.DataFrame, top_n=10):
    """Return a list of option symbols to query trades for. Robust to missing 'symbol' column."""
    if opt_df is None or opt_df.empty:
        return []
    sym_col = find_symbol_column(opt_df)
    if sym_col:
        # sometimes values can be null; dropna and unique
        syms = opt_df[sym_col].dropna().unique().tolist()
        if syms:
            return syms[:top_n]
    # fallback: try to build synthetic symbols from columns
    syms = []
    for _, row in opt_df.iterrows():
        s = build_synthetic_symbol(row)
        if s:
            syms.append(s)
        if len(syms) >= top_n:
            break
    # ensure uniqueness
    return list(dict.fromkeys(syms))

# ---------------------------
# BACKTEST ENGINE (simplified)
# ---------------------------
def simulate_simple_strategy(underlying_df, option_contracts_df, trades_df_by_symbol,
                             strategy='straddle', entry_date=None, exit_days=7,
                             capital=10000, fee_per_trade=1.0, slippage=0.01):
    """
    Very conservative simulation:
    - Entry: take nearest trade price at entry timestamp for the option contracts of selected strikes/expiry
    - Exit: close at nearest trade price at exit timestamp
    - Uses robust column detection for trade payloads.
    """
    # select underlying spot S at entry_date (or last available)
    S_row = underlying_df[underlying_df['t'].dt.date == entry_date.date()]
    if S_row.empty:
        S = float(underlying_df.iloc[-1]['close'])
    else:
        S = float(S_row.iloc[0]['close'])

    # Ensure expiration_date exists in option_contracts_df
    opt_df = option_contracts_df.copy()
    if 'expiration_date' in opt_df.columns:
        opt_df['expiration_date'] = pd.to_datetime(opt_df['expiration_date'])
    else:
        # if not present, attempt to parse 'expiration' or leave as NaT
        for col in opt_df.columns:
            if 'expiration' in col.lower():
                opt_df['expiration_date'] = pd.to_datetime(opt_df[col], errors='coerce')
                break
        if 'expiration_date' not in opt_df.columns:
            opt_df['expiration_date'] = pd.NaT

    target_exp = entry_date + timedelta(days=exit_days)
    # sort by closeness to entry_date expiration
    candidate = opt_df.iloc[((opt_df['expiration_date'] - pd.Timestamp(entry_date)).abs()).argsort()][:200]

    # compute strike diff robustly
    if 'strike' not in candidate.columns:
        # try to find candidate column for strike
        for c in candidate.columns:
            if 'strike' in c.lower():
                candidate = candidate.rename(columns={c: 'strike'})
                break
    candidate['strike'] = pd.to_numeric(candidate.get('strike', np.nan), errors='coerce')
    candidate['strike_diff'] = abs(candidate['strike'] - S)
    
    # FIX: Check if candidate DataFrame is empty before attempting to sort and access rows
    sorted_candidates = candidate.sort_values('strike_diff', na_position='last')
    if sorted_candidates.empty:
        return {"pnl": 0.0, "paid": 0.0, "received": 0.0, 
                "call_sym": None, "put_sym": None, 
                "note": "no_suitable_options_found"}
                
    # pick ATM row
    atm_row = sorted_candidates.iloc[0]
    atm_strike = atm_row.get('strike', None)

    # find call/put symbols robustly
    sym_col = find_symbol_column(candidate)
    def find_leg_symbol(df, strike, typ):
        # try direct match by strike and type if symbol column exists
        if 'strike' in df.columns and 'type' in df.columns and sym_col:
            filt = (df['strike'] == strike) & (df['type'].str.lower().str.startswith(typ[0].lower()))
            subset = df[filt]
            if not subset.empty:
                return subset.iloc[0].get(sym_col)
        # otherwise, try to find any row with matching strike and prefer call/put words
        if 'strike' in df.columns:
            subset = df[df['strike'] == strike]
            if not subset.empty:
                # prefer a row where type matches, else return any symbol-like column
                if 'type' in subset.columns:
                    tmatch = subset[subset['type'].str.lower().str.startswith(typ[0].lower())]
                    if not tmatch.empty and sym_col:
                        return tmatch.iloc[0].get(sym_col)
                if sym_col:
                    return subset.iloc[0].get(sym_col)
        # fallback: try to build synthetic symbol from the row with same strike and type
        for _, r in df.iterrows():
            if pd.notna(r.get('strike')) and float(r.get('strike')) == float(strike):
                # build synthetic
                s = build_synthetic_symbol(r)
                if s:
                    return s
        return None

    call_sym = None
    put_sym = None
    # if atm_row already has symbol & type, use it for one leg, and search for the other
    if sym_col and pd.notna(atm_row.get(sym_col)):
        if str(atm_row.get('type','')).lower().startswith('c'):
            call_sym = atm_row.get(sym_col)
            put_sym = find_leg_symbol(candidate, atm_strike, 'put')
        elif str(atm_row.get('type','')).lower().startswith('p'):
            put_sym = atm_row.get(sym_col)
            call_sym = find_leg_symbol(candidate, atm_strike, 'call')

    # if still missing, try generic search
    if not call_sym:
        call_sym = find_leg_symbol(candidate, atm_strike, 'call')
    if not put_sym:
        put_sym = find_leg_symbol(candidate, atm_strike, 'put')

    # Defensive: if either leg still missing, return a neutral result (no trade)
    if not call_sym or not put_sym:
        return {"pnl": 0.0, "paid": 0.0, "received": 0.0, "call_sym": call_sym, "put_sym": put_sym, "note": "missing_leg"}

    # get mid price at entry and exit with robust field detection
    def mid_at(symbol, ts_ms):
        if symbol not in trades_df_by_symbol:
            return np.nan
        df = trades_df_by_symbol[symbol].copy()
        # find timestamp column (common names: 't', 'sip_timestamp', 'timestamp')
        if 't' in df.columns:
            df_ts = pd.to_datetime(df['t'], unit='ms', errors='coerce')
        elif 'timestamp' in df.columns:
            df_ts = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce') if df['timestamp'].dtype.kind in ('i','u') else pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'sip_timestamp' in df.columns:
            # some endpoints return ns
            try:
                df_ts = pd.to_datetime(df['sip_timestamp'], unit='ns', errors='coerce')
            except Exception:
                df_ts = pd.to_datetime(df['sip_timestamp'], errors='coerce')
        else:
            # try any column containing 'time'
            ts_cols = [c for c in df.columns if 'time' in c.lower() or 'ts'==c.lower()]
            if ts_cols:
                df_ts = pd.to_datetime(df[ts_cols[0]], errors='coerce')
            else:
                df_ts = pd.Series([pd.NaT]*len(df))

        df['__ts'] = df_ts
        # find price column
        price_col = None
        for c in ('p','price','px'):
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            # try to infer numeric column
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            # exclude size-like columns
            numeric_cols = [c for c in numeric_cols if c not in ('s','size')]
            price_col = numeric_cols[0] if numeric_cols else None
        if price_col is None:
            return np.nan

        # find nearest trade to timestamp
        if df['__ts'].isna().all():
            return np.nan
        target_ts = pd.to_datetime(ts_ms, unit='ms')
        df = df.dropna(subset=['__ts'])
        idx = (df['__ts'] - target_ts).abs().argsort()
        if len(idx) == 0:  # Defensive check
            return np.nan
        row = df.iloc[idx[0]]
        val = row.get(price_col, np.nan)
        try:
            return float(val)
        except Exception:
            return np.nan

    entry_ms = int(entry_date.timestamp() * 1000)
    exit_ms = int((entry_date + timedelta(days=exit_days)).timestamp() * 1000)

    call_entry_p = mid_at(call_sym, entry_ms)
    put_entry_p  = mid_at(put_sym, entry_ms)
    call_exit_p = mid_at(call_sym, exit_ms)
    put_exit_p  = mid_at(put_sym, exit_ms)

    # apply slippage and guard against NaN
    def apply_slippage_and_guard(px, add_slip):
        if px is None or pd.isna(px):
            return np.nan
        try:
            return float(px) * (1 + add_slip)
        except Exception:
            return np.nan

    call_entry_p = apply_slippage_and_guard(call_entry_p, slippage)
    put_entry_p  = apply_slippage_and_guard(put_entry_p, slippage)
    call_exit_p  = apply_slippage_and_guard(call_exit_p, -slippage)
    put_exit_p   = apply_slippage_and_guard(put_exit_p, -slippage)

    # if any price is nan, we cannot compute a reliable pnl
    if any(pd.isna(x) for x in [call_entry_p, put_entry_p, call_exit_p, put_exit_p]):
        return {"pnl": 0.0, "paid": 0.0, "received": 0.0, "call_sym": call_sym, "put_sym": put_sym, "note": "missing_prices"}

    paid = (call_entry_p + put_entry_p) * 100  # 100 multiplier
    received = (call_exit_p + put_exit_p) * 100
    pnl = received - paid - fee_per_trade*2
    return {"pnl": pnl, "paid": paid, "received": received, "call_sym": call_sym, "put_sym": put_sym, "note": "ok"}

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Options Event Backtester -- Event-driven strategies + cascading NN entry")

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
        # fetch option contracts for window
        opt_df = get_option_contracts(ticker, expiration_gte=start_dt.strftime("%Y-%m-%d"),
                                      expiration_lte=end_dt.strftime("%Y-%m-%d"), limit=1000)
        st.write("Option contracts sample:", opt_df.shape)
        st.dataframe(opt_df.head(10))
        # pick top contracts robustly
        symbol_list = get_option_symbol_list(opt_df, top_n=10)
        # if polygon returns contracts with an 'open_interest' column and a symbol column, prefer top by OI
        top_syms = []
        if 'open_interest' in opt_df.columns:
            # try to sort by open_interest and pull associated symbols (robust to missing symbol col)
            sorted_df = opt_df.sort_values('open_interest', ascending=False)
            sym_col = find_symbol_column(sorted_df)
            if sym_col:
                top_by_oi = sorted_df[sym_col].dropna().unique().tolist()
                top_syms = top_by_oi[:10]
        # merge fallbacks
        if not top_syms:
            top_syms = symbol_list

        trades = {}
        st.write("Fetching trade samples for top contracts (may be rate-limited)...")
        for sym in top_syms:
            try:
                # naive: pull trades for event date only (+some slack)
                start_ms = int((datetime.combine(event_date, datetime.min.time()) - timedelta(hours=2)).timestamp()*1000)
                end_ms = int((datetime.combine(event_date, datetime.min.time()) + timedelta(hours=6)).timestamp()*1000)
                df_tr = get_option_trades(sym, start_ms, end_ms)
                if not df_tr.empty:
                    trades[sym] = df_tr
                else:
                    # store an empty DataFrame to signal attempted symbol
                    trades[sym] = pd.DataFrame()
            except Exception as e:
                st.error(f"trade fetch error for {sym}: {e}")
        st.success(f"Fetched contracts: {len(opt_df)} and trades for {len([k for k,v in trades.items() if not v.empty])} symbols.")
        st.session_state['underlying'] = underlying
        st.session_state['opt_df'] = opt_df
        st.session_state['trades'] = trades

    # show preloaded data (if present)
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