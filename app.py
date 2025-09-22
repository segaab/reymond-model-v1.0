# app.py — Entry-Range Triangulation Dashboard (combined)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
import uuid
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Optional libs with graceful degradation
try:
    from sodapy import Socrata
except Exception:
    Socrata = None
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# ML libs (XGBoost + PyTorch)
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

# ---------------------------
# Streamlit Page Setup & Controls
# ---------------------------
st.set_page_config(page_title="Entry-Range Triangulation Dashboard", layout="wide")

st.title("Entry-Range Triangulation Dashboard — (HealthGauge → Candidates → Confirm)")

# Inputs
symbol = st.text_input("Symbol", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)

# Health thresholds (adjustable per your request)
st.sidebar.header("HealthGauge thresholds")
buy_threshold = st.sidebar.number_input("Buy threshold (HealthGauge)", min_value=0.0, max_value=1.0, value=0.55)
sell_threshold = st.sidebar.number_input("Sell threshold (HealthGauge)", min_value=0.0, max_value=1.0, value=0.45)

# Model training controls
st.sidebar.header("Model / Backtest Controls")
num_boost = st.sidebar.number_input("XGBoost num_boost_round", min_value=1, value=200)
early_stop = st.sidebar.number_input("XGBoost early_stopping_rounds", min_value=1, value=20)
test_size = st.sidebar.number_input("Test set fraction", min_value=0.0, max_value=1.0, value=0.2)

p_fast = st.sidebar.number_input("Threshold fast", min_value=0.0, max_value=1.0, value=0.6)
p_slow = st.sidebar.number_input("Threshold slow", min_value=0.0, max_value=1.0, value=0.55)
p_deep = st.sidebar.number_input("Threshold deep", min_value=0.0, max_value=1.0, value=0.45)

force_run = st.sidebar.checkbox("Force run even outside buy/sell band", value=False)
show_confusion = st.sidebar.checkbox("Show confusion matrix / classification report", value=True)
overlay_entries_on_price = st.sidebar.checkbox("Overlay entries on price chart", value=True)
include_health_as_feature = st.sidebar.checkbox("Include HealthGauge as feature", value=True)
save_feature_importance = st.sidebar.checkbox("Save feature importance on export", value=True)

# Breadth / sweep
st.sidebar.header("Breadth & Sweep")
run_breadth = st.sidebar.button("Run breadth modes (Low → Mid → High)")
run_sweep_btn = st.sidebar.button("Run grid sweep")
rr_vals = st.sidebar.multiselect("RR values", options=[1.5, 2.0, 2.5, 3.0], default=[2.0, 3.0])
sl_ranges_raw = st.sidebar.text_input("SL ranges (e.g. 0.5-1.0,1.0-2.0)", value="0.5-1.0,1.0-2.0")
session_modes = st.sidebar.multiselect("Session modes", options=["low", "mid", "high"], default=["low", "mid", "high"])
mpt_input = st.sidebar.text_input("Model prob thresholds (comma-separated)", value="0.6,0.7")
max_bars = int(st.sidebar.number_input("Max bars horizon (labels/sim)", value=60, step=1))

def parse_sl_ranges(s: str) -> List[Tuple[float,float]]:
    out = []
    for token in [t.strip() for t in s.split(",") if t.strip()]:
        try:
            a,b = token.split("-")
            out.append((float(a), float(b)))
        except Exception:
            continue
    return out

sl_ranges = parse_sl_ranges(sl_ranges_raw)
mpt_list = [float(x) for x in [t.strip() for t in mpt_input.split(",") if t.strip()]]

# Asset object placeholder
@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

asset_obj = Asset(name="Gold", cot_name="GOLD - COMMODITY EXCHANGE INC.", symbol=symbol)


# ---------------------------
# Section: fetch_data (yahooquery) + features + small helpers
# ---------------------------

def fetch_price(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """
    Use yahooquery to fetch OHLCV. Returns DataFrame with index as DatetimeIndex.
    """
    if YahooTicker is None:
        logger.error("yahooquery not available. Install yahooquery or adjust fetcher.")
        return pd.DataFrame()
    try:
        t = YahooTicker(symbol)
        # yahooquery history returns multiindex for multiple tickers; handle single symbol
        hist = t.history(start=start_date, end=end_date, interval=interval)
        if isinstance(hist, dict):
            # sometimes returns dict with 'symbol' key
            hist = pd.DataFrame(hist)
        if hist is None or hist.empty:
            return pd.DataFrame()
        # If multiindex (symbol, datetime) — reset
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index(level=0, drop=True)
        hist.index = pd.to_datetime(hist.index)
        hist = hist.sort_index()
        # ensure standard columns: open, high, low, close, volume
        rename_map = {}
        for c in hist.columns:
            lc = c.lower()
            if lc in ["open","high","low","close","volume","adjclose","adjclose"]:
                rename_map[c] = lc
        hist.rename(columns=rename_map, inplace=True)
        # prefer 'close' or 'adjclose'
        if "close" not in hist.columns and "adjclose" in hist.columns:
            hist["close"] = hist["adjclose"]
        # fill forward minimal
        hist = hist[~hist.index.duplicated(keep='first')]
        return hist
    except Exception as e:
        logger.error("fetch_price failed for %s: %s", symbol, e)
        return pd.DataFrame()

def init_socrata_client(timeout: int = 60, max_retries: int = 5, backoff_factor: float = 1.5):
    """
    Initialize Socrata client if available.
    """
    if Socrata is None:
        logger.warning("Socrata client not available (sodapy not installed). Returning None.")
        return None
    domain = "publicreporting.cftc.gov"
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    username = os.getenv("SOCRATA_USER")
    password = os.getenv("SOCRATA_PASS")
    client = None
    for attempt in range(1, max_retries + 1):
        try:
            client = Socrata(domain, app_token, username=username, password=password, timeout=timeout)
            logger.info("Socrata client initialized successfully.")
            return client
        except Exception as e:
            sleep_time = backoff_factor ** attempt
            logger.warning("Socrata init failed attempt %d/%d: %s — retrying in %.1f sec", attempt, max_retries, e, sleep_time)
            import time; time.sleep(sleep_time)
    logger.error("Failed to init Socrata client after retries.")
    return None

def fetch_cot(client, dataset_id: str = "6dca-aqww", report_date: Optional[str] = None, max_rows: int = 10000) -> pd.DataFrame:
    if client is None:
        logger.warning("fetch_cot called with no client — returning empty DataFrame.")
        return pd.DataFrame()
    where_clause = f"report_date_as_yyyy_mm_dd = '{report_date}'" if report_date else None
    retries = 3
    for attempt in range(1, retries+1):
        try:
            results = client.get(dataset_id, where=where_clause, limit=max_rows)
            df = pd.DataFrame.from_records(results)
            if not df.empty and "report_date_as_yyyy_mm_dd" in df.columns:
                df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
            return df
        except Exception as e:
            import time
            logger.warning("fetch_cot attempt %d failed: %s — retrying", attempt, e)
            time.sleep(2**attempt)
    logger.error("fetch_cot failed after retries")
    return pd.DataFrame()

# Features
def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        logger.warning("compute_rvol: 'volume' missing; returning 1.0 series.")
        return pd.Series(1.0, index=df.index)
    rolling_avg = df["volume"].rolling(window=lookback, min_periods=1).mean()
    rvol = df["volume"] / rolling_avg.replace(0, np.nan)
    rvol = rvol.fillna(1.0)
    return rvol

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame, rvol_col: str = "rvol", threshold: float = 1.5) -> pd.DataFrame:
    """
    Produces a DataFrame with index daily_bars.index and column 'health_gauge' 0..1 score.
    In this simplified version we mark 1 when rvol >= threshold else 0.
    """
    if daily_bars is None or daily_bars.empty:
        logger.warning("calculate_health_gauge: daily_bars empty; returning empty df.")
        return pd.DataFrame()
    db = daily_bars.copy()
    db['rvol'] = compute_rvol(db, lookback=20)
    score = (db['rvol'] >= threshold).astype(float)
    out = pd.DataFrame({'health_gauge': score}, index=db.index)
    return out

def ensure_no_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.duplicated().any():
        logger.warning("Duplicate timestamps found — keeping first occurrence.")
        df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df

# small helper: df to csv bytes for downloads
import io
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ---------------------------
# Section: fetch_data (yahooquery) + features + small helpers
# ---------------------------

def fetch_price(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """
    Use yahooquery to fetch OHLCV. Returns DataFrame with index as DatetimeIndex.
    """
    if YahooTicker is None:
        logger.error("yahooquery not available. Install yahooquery or adjust fetcher.")
        return pd.DataFrame()
    try:
        t = YahooTicker(symbol)
        # yahooquery history returns multiindex for multiple tickers; handle single symbol
        hist = t.history(start=start_date, end=end_date, interval=interval)
        if isinstance(hist, dict):
            # sometimes returns dict with 'symbol' key
            hist = pd.DataFrame(hist)
        if hist is None or hist.empty:
            return pd.DataFrame()
        # If multiindex (symbol, datetime) — reset
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index(level=0, drop=True)
        hist.index = pd.to_datetime(hist.index)
        hist = hist.sort_index()
        # ensure standard columns: open, high, low, close, volume
        rename_map = {}
        for c in hist.columns:
            lc = c.lower()
            if lc in ["open","high","low","close","volume","adjclose","adjclose"]:
                rename_map[c] = lc
        hist.rename(columns=rename_map, inplace=True)
        # prefer 'close' or 'adjclose'
        if "close" not in hist.columns and "adjclose" in hist.columns:
            hist["close"] = hist["adjclose"]
        # fill forward minimal
        hist = hist[~hist.index.duplicated(keep='first')]
        return hist
    except Exception as e:
        logger.error("fetch_price failed for %s: %s", symbol, e)
        return pd.DataFrame()

def init_socrata_client(timeout: int = 60, max_retries: int = 5, backoff_factor: float = 1.5):
    """
    Initialize Socrata client if available.
    """
    if Socrata is None:
        logger.warning("Socrata client not available (sodapy not installed). Returning None.")
        return None
    domain = "publicreporting.cftc.gov"
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    username = os.getenv("SOCRATA_USER")
    password = os.getenv("SOCRATA_PASS")
    client = None
    for attempt in range(1, max_retries + 1):
        try:
            client = Socrata(domain, app_token, username=username, password=password, timeout=timeout)
            logger.info("Socrata client initialized successfully.")
            return client
        except Exception as e:
            sleep_time = backoff_factor ** attempt
            logger.warning("Socrata init failed attempt %d/%d: %s — retrying in %.1f sec", attempt, max_retries, e, sleep_time)
            import time; time.sleep(sleep_time)
    logger.error("Failed to init Socrata client after retries.")
    return None

def fetch_cot(client, dataset_id: str = "6dca-aqww", report_date: Optional[str] = None, max_rows: int = 10000) -> pd.DataFrame:
    if client is None:
        logger.warning("fetch_cot called with no client — returning empty DataFrame.")
        return pd.DataFrame()
    where_clause = f"report_date_as_yyyy_mm_dd = '{report_date}'" if report_date else None
    retries = 3
    for attempt in range(1, retries+1):
        try:
            results = client.get(dataset_id, where=where_clause, limit=max_rows)
            df = pd.DataFrame.from_records(results)
            if not df.empty and "report_date_as_yyyy_mm_dd" in df.columns:
                df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
            return df
        except Exception as e:
            import time
            logger.warning("fetch_cot attempt %d failed: %s — retrying", attempt, e)
            time.sleep(2**attempt)
    logger.error("fetch_cot failed after retries")
    return pd.DataFrame()

# Features
def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        logger.warning("compute_rvol: 'volume' missing; returning 1.0 series.")
        return pd.Series(1.0, index=df.index)
    rolling_avg = df["volume"].rolling(window=lookback, min_periods=1).mean()
    rvol = df["volume"] / rolling_avg.replace(0, np.nan)
    rvol = rvol.fillna(1.0)
    return rvol

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame, rvol_col: str = "rvol", threshold: float = 1.5) -> pd.DataFrame:
    """
    Produces a DataFrame with index daily_bars.index and column 'health_gauge' 0..1 score.
    In this simplified version we mark 1 when rvol >= threshold else 0.
    """
    if daily_bars is None or daily_bars.empty:
        logger.warning("calculate_health_gauge: daily_bars empty; returning empty df.")
        return pd.DataFrame()
    db = daily_bars.copy()
    db['rvol'] = compute_rvol(db, lookback=20)
    score = (db['rvol'] >= threshold).astype(float)
    out = pd.DataFrame({'health_gauge': score}, index=db.index)
    return out

def ensure_no_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.duplicated().any():
        logger.warning("Duplicate timestamps found — keeping first occurrence.")
        df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df

# small helper: df to csv bytes for downloads
import io
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------
# labeling.py (generate candidates + labels)
# ---------------------------
import math
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def generate_candidates_and_labels(
    df: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars:int = 60,
    direction: str = "long"
) -> pd.DataFrame:
    """
    Generate triple-barrier-like candidates using ATR.
    Output columns include candidate_time, entry_price, atr, sl_price, tp_price, end_time, label, realized_return.
    """
    if df is None or df.empty:
        logger.warning("generate_candidates_and_labels: bars empty — returning empty df.")
        return pd.DataFrame()

    bars = df.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()
    if bars.index.duplicated().any():
        bars = bars[~bars.index.duplicated(keep='first')]

    required_cols = {"high","low","close"}
    if not required_cols.issubset(set(bars.columns)):
        raise KeyError(f"bars missing required columns: {required_cols - set(bars.columns)}")

    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(window=atr_window, min_periods=1).mean().fillna(method="ffill").fillna(0.0)

    candidates = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_price = float(bars["close"].iat[i])
        atr_t = float(bars["atr"].iat[i])
        if atr_t <= 0 or math.isnan(atr_t):
            continue
        if direction == "long":
            sl_price = entry_price - k_sl * atr_t
            tp_price = entry_price + k_tp * atr_t
        else:
            sl_price = entry_price + k_sl * atr_t
            tp_price = entry_price - k_tp * atr_t

        end_idx = min(i + max_bars, n-1)
        label = 0
        hit_idx = end_idx
        hit_price = float(bars["close"].iat[end_idx])

        for j in range(i+1, end_idx+1):
            px_high = float(bars["high"].iat[j])
            px_low = float(bars["low"].iat[j])
            if direction == "long":
                if px_high >= tp_price:
                    label = 1
                    hit_idx = j
                    hit_price = tp_price
                    break
                if px_low <= sl_price:
                    label = 0
                    hit_idx = j
                    hit_price = sl_price
                    break
            else:
                if px_low <= tp_price:
                    label = 1
                    hit_idx = j
                    hit_price = tp_price
                    break
                if px_high >= sl_price:
                    label = 0
                    hit_idx = j
                    hit_price = sl_price
                    break

        end_time = bars.index[hit_idx]
        realized_return = (hit_price - entry_price) / entry_price if direction == "long" else (entry_price - hit_price) / entry_price
        duration_min = (end_time - t).total_seconds() / 60.0

        candidates.append({
            "candidate_time": t,
            "entry_price": entry_price,
            "atr": atr_t,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "end_time": end_time,
            "label": int(label),
            "duration": float(duration_min),
            "realized_return": float(realized_return),
            "direction": direction
        })
    out_df = pd.DataFrame(candidates)
    logger.info("generate_candidates_and_labels: generated %d candidates", len(out_df))
    return out_df

# utils small: purged_train_test_split (kept from previous utils)
def purged_train_test_split(events_df: pd.DataFrame, purge_radius: int = 10):
    n = len(events_df)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    split_idx = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:split_idx] = True
    test_mask[split_idx:] = True
    return train_mask, test_mask



# ---------------------------
# backtest, sweep, breadth, summary
# ---------------------------
def simulate_limits(
    df: pd.DataFrame,
    bars: pd.DataFrame,
    label_col: str = "pred_label",
    symbol: str = "GC=F",
    rr: float = 2.0,
    sl: float = 0.01,
    tp: float = 0.02,
    max_holding: int = 20
) -> pd.DataFrame:
    logger.info("Starting simulate_limits for %s…", symbol)
    if df is None or df.empty:
        logger.warning("simulate_limits: input df empty")
        return pd.DataFrame()
    if bars is None or bars.empty:
        logger.error("simulate_limits: bars empty")
        return pd.DataFrame()

    trades = []
    for idx, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl):
            continue
        entry_time = row.get("candidate_time") or row.get("entry_time") or row.name
        try:
            entry_time = pd.to_datetime(entry_time)
        except Exception:
            pass
        if entry_time not in bars.index:
            logger.debug("simulate_limits: entry_time %s not in bars", entry_time)
            continue
        entry_price = float(bars.loc[entry_time, "close"])
        direction = int(np.sign(lbl))
        sl_price = entry_price * (1 - sl) if direction > 0 else entry_price * (1 + sl)
        tp_price = entry_price * (1 + tp) if direction > 0 else entry_price * (1 - tp)
        exit_time, exit_price, pnl = None, None, None
        holding_bars = bars.loc[entry_time:].head(max_holding)
        for t, b in holding_bars.iterrows():
            low = float(b.get("low", np.nan))
            high = float(b.get("high", np.nan))
            if direction > 0:
                if low <= sl_price:
                    exit_time, exit_price, pnl = t, sl_price, -sl
                    break
                if high >= tp_price:
                    exit_time, exit_price, pnl = t, tp_price, tp
                    break
            else:
                if high >= sl_price:
                    exit_time, exit_price, pnl = t, sl_price, -sl
                    break
                if low <= tp_price:
                    exit_time, exit_price, pnl = t, tp_price, tp
                    break
        if exit_time is None and not holding_bars.empty:
            exit_time = holding_bars.index[-1]
            exit_price = float(holding_bars.iloc[-1].get("close", entry_price))
            pnl = (exit_price - entry_price) / entry_price * direction
        if pnl is None:
            continue
        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": float(pnl)
        })
    overlay = pd.DataFrame(trades)
    if overlay.empty:
        logger.warning("simulate_limits: no trades generated.")
    else:
        logger.info("simulate_limits: %d trades generated avg pnl=%.4f", len(overlay), overlay["pnl"].mean())
    return overlay

def run_breadth_backtest(clean: pd.DataFrame, bars: pd.DataFrame, asset_obj: Asset=None, rr_vals: List[float]=None,
                         sl_ranges: List[Tuple[float,float]]=None, session_modes: List[str]=None, mpt_list: List[float]=None,
                         max_bars: int = 60, include_health: bool = True, health_df: pd.DataFrame = None,
                         model_train_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Runs training (per mode) and backtests, returns standardized dict:
    { "summary": [...rows...], "detailed_trades": {mode: DataFrame}, "diagnostics": [..] }
    """
    logger.info("run_breadth_backtest started.")
    diagnostics = []
    results_summary = []
    detailed = {}

    if clean is None or clean.empty:
        raise RuntimeError("run_breadth_backtest: clean dataset empty")

    # default parameters if None
    if rr_vals is None: rr_vals = [2.0]
    if sl_ranges is None: sl_ranges = [(0.5,1.0)]
    if session_modes is None: session_modes = ["low","mid","high"]
    if mpt_list is None: mpt_list = [0.6]

    # Example breadth modes -> we create simple parameterizations
    for mode in session_modes:
        diagnostics.append(f"Starting mode: {mode}")
        try:
            # For each mode, pick thresholds (these are example heuristics)
            if mode.lower() == "low":
                sell_th, buy_th = 5, 5
            elif mode.lower() == "mid":
                sell_th, buy_th = 4, 6
            else:
                sell_th, buy_th = 3, 7

            df = clean.copy()
            if "signal" not in df.columns:
                df["signal"] = (df.get("pred_prob", 0.0) * 10).round().astype(int)

            # Create different pred_label variants by model prob thresholds
            for mpt in mpt_list:
                di = f"{mode}_mpt_{mpt:.2f}"
                df_variant = df.copy()
                df_variant["pred_label"] = 0
                df_variant.loc[df_variant["signal"] > buy_th, "pred_label"] = 1
                df_variant.loc[df_variant["signal"] < sell_th, "pred_label"] = -1

                overlay = simulate_limits(df_variant, bars, label_col="pred_label", symbol=asset_obj.symbol if asset_obj else symbol, max_holding=max_bars)
                # summarize
                total_pnl = overlay["pnl"].sum() if overlay is not None and not overlay.empty else 0.0
                win_rate = (overlay["pnl"] > 0).sum() / len(overlay) if overlay is not None and len(overlay) > 0 else 0.0
                results_summary.append({
                    "mode": mode, "mpt": float(mpt), "sell_th": sell_th, "buy_th": buy_th,
                    "num_trades": int(len(overlay)) if overlay is not None else 0,
                    "total_pnl": float(total_pnl), "win_rate": float(win_rate)
                })
                detailed[f"{mode}_mpt_{mpt:.2f}"] = overlay
                diagnostics.append(f"Mode {mode} mpt {mpt:.2f}: trades={len(overlay)} pnl={total_pnl:.4f} win_rate={win_rate:.3f}")
        except Exception as e:
            logger.exception("run_breadth_backtest failed for mode %s: %s", mode, e)
            diagnostics.append(f"Mode {mode} failed: {e}")
            continue

    return {"summary": results_summary, "detailed_trades": detailed, "diagnostics": diagnostics}

def run_sweep(clean: pd.DataFrame, bars: pd.DataFrame, rr_vals: List[float]=None, sl_ranges: List[Tuple[float,float]]=None, mpt_list: List[float]=None, model_train_kwargs: Dict[str,Any]=None) -> Dict[str, Any]:
    """
    Run grid sweep over parameters (sell thresholds, buy thresholds) and return results dict.
    """
    logger.info("run_sweep started.")
    if clean is None or clean.empty:
        return {}
    if rr_vals is None: rr_vals = [2.0]
    if sl_ranges is None: sl_ranges = [(0.5,1.0)]
    if mpt_list is None: mpt_list = [0.6]

    results = {}
    for s_min, s_max in sl_ranges:
        for rr in rr_vals:
            for mpt in mpt_list:
                key = f"rr{rr}_sl{s_min}-{s_max}_mpt{mpt}"
                try:
                    df = clean.copy()
                    if "signal" not in df.columns:
                        df["signal"] = (df.get("pred_prob", 0.0) * 10).round().astype(int)
                    df["pred_label"] = 0
                    buy_th = int((s_max + s_min) / 2 + 1)
                    sell_th = int(s_min)
                    df.loc[df["signal"] > buy_th, "pred_label"] = 1
                    df.loc[df["signal"] < sell_th, "pred_label"] = -1
                    overlay = simulate_limits(df, bars, label_col="pred_label", symbol=symbol)
                    results[key] = {"trades": overlay.shape[0] if overlay is not None else 0, "overlay": overlay}
                    logger.info("run_sweep %s done: %d trades", key, results[key]["trades"])
                except Exception as e:
                    logger.exception("run_sweep iteration %s failed: %s", key, e)
                    results[key] = {"error": str(e)}
    return results

# summarization helpers
def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    summary = pd.DataFrame([{
        "total_trades": len(trades),
        "win_rate": (trades["pnl"] > 0).mean(),
        "avg_pnl": trades["pnl"].mean(),
        "median_pnl": trades["pnl"].median(),
        "total_pnl": trades["pnl"].sum(),
        "max_drawdown": trades["pnl"].cumsum().min(),
        "start_time": trades["entry_time"].min(),
        "end_time": trades["exit_time"].max()
    }])
    return summary

def combine_summaries(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = []
    for k, df in results.items():
        if df is None or df.empty:
            continue
        s = summarize_trades(df)
        s["mode"] = k
        combined.append(s)
    if not combined:
        return pd.DataFrame()
    return pd.concat(combined, ignore_index=True)


# ---------------------------
# model: xgboost wrapper and training
# ---------------------------
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import json
import time

class BoosterWrapper:
    def __init__(self, booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names
        self.best_iteration = getattr(booster, "best_iteration", None)

    def _dmatrix(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        Xp = X.reindex(columns=self.feature_names).fillna(0.0)
        return xgb.DMatrix(Xp, feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        dmat = self._dmatrix(X)
        if self.best_iteration is not None:
            raw = self.booster.predict(dmat, iteration_range=(0, int(self.best_iteration)+1))
        else:
            raw = self.booster.predict(dmat)
        raw = np.asarray(raw, dtype=float)
        # clip to [0,1]
        raw = np.clip(raw, 0.0, 1.0)
        probs = np.vstack([1.0 - raw, raw]).T
        return pd.Series(probs[:,1], index=X.index, name="confirm_proba")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save_model(self, filepath: str):
        joblib.dump({"booster": self.booster, "feature_names": self.feature_names}, filepath)
        logger.info("BoosterWrapper saved to %s", filepath)

    @classmethod
    def load_model(cls, filepath: str):
        data = joblib.load(filepath)
        return cls(booster=data["booster"], feature_names=data["feature_names"])

    def get_feature_importance(self) -> pd.DataFrame:
        try:
            importance = self.booster.get_score(importance_type="weight")
        except Exception:
            importance = {}
        df = pd.DataFrame([(k, importance.get(k, 0)) for k in self.feature_names], columns=["feature","importance"]).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)

def train_xgb_confirm(
    clean: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
    num_boost_round: int = 200,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[BoosterWrapper, List[str], Dict[str, Any]]:
    if xgb is None:
        raise RuntimeError("xgboost not installed or unavailable")

    missing = [c for c in feature_cols if c not in clean.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training data: {missing}")
    if label_col not in clean.columns:
        raise KeyError(f"Label column '{label_col}' missing in training data")

    X_all = clean[feature_cols].copy().apply(pd.to_numeric, errors="coerce")
    y_all = pd.to_numeric(clean[label_col], errors="coerce")

    mask_valid = X_all.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y_all.notnull() & np.isfinite(y_all)
    X_all = X_all.loc[mask_valid].reset_index(drop=True)
    y_all = y_all.loc[mask_valid].reset_index(drop=True)

    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    logger.info("train_xgb_confirm: pos=%d neg=%d", n_pos, n_neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both classes present to train")

    scale_pos_weight = float(n_neg) / max(1.0, float(n_pos))
    stratify = y_all if len(np.unique(y_all)) == 2 else None
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state, shuffle=True, stratify=stratify)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "verbosity": 1 if verbose else 0,
        "scale_pos_weight": scale_pos_weight,
    }
    evals = [(dval, "validation")]
    try:
        bst = xgb.train(params, dtrain, num_boost_round=int(num_boost_round), evals=evals, early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None, verbose_eval=verbose)
    except TypeError as te:
        logger.warning("xgb.train TypeError: %s — retrying without early_stopping_rounds", te)
        bst = xgb.train(params, dtrain, num_boost_round=int(num_boost_round), evals=evals, verbose_eval=verbose)

    wrapper = BoosterWrapper(bst, feature_cols)
    y_proba_val = wrapper.predict_proba(X_val)
    y_pred_val = (y_proba_val >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "accuracy": float(accuracy_score(y_val, y_pred_val)),
        "f1": float(f1_score(y_val, y_pred_val, zero_division=0)),
        "val_proba_mean": float(np.nanmean(y_proba_val)) if y_proba_val.size > 0 else 0.0,
    }
    if len(np.unique(y_val)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba_val))
        except Exception:
            metrics["roc_auc"] = np.nan

    logger.info("train_xgb_confirm metrics: %s", metrics)
    return wrapper, feature_cols, metrics

def predict_confirm_prob(model_wrapper, candidates: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    if candidates is None or candidates.empty:
        return pd.Series(dtype=float)
    missing = [c for c in feature_cols if c not in candidates.columns]
    if missing:
        logger.warning("predict_confirm_prob: missing features %s — filling zeros", missing)
        for m in missing:
            candidates[m] = 0.0
    X = candidates[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    try:
        if hasattr(model_wrapper, "predict_proba"):
            proba = model_wrapper.predict_proba(X)
        else:
            preds = model_wrapper.predict(X)
            proba = pd.Series(np.asarray(preds, dtype=float), index=candidates.index, name="confirm_proba")
    except Exception as exc:
        logger.error("predict_confirm_prob error: %s", exc)
        proba = pd.Series(np.zeros(len(X), dtype=float), index=candidates.index, name="confirm_proba")
    return proba

def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str,Any], model_basename: str, save_fi: bool = True):
    """
    Save as XGB native if possible, else save torch .pt wrapper or joblib.
    Returns dict of saved paths.
    """
    paths = {}
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_file = f"{model_basename}_{ts}.model"
    meta_file = f"{model_basename}_{ts}.json"
    fi_file = f"{model_basename}_{ts}_feature_importance.json"
    try:
        booster = getattr(model_wrapper, "booster", None)
        if booster is None:
            booster = getattr(model_wrapper, "model", None)
        if booster is None:
            # fallback: torch save wrapper
            if torch is not None:
                torch.save({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.pt")
                paths['pt'] = f"{model_basename}_{ts}.pt"
            else:
                joblib.dump({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.joblib")
                paths['joblib'] = f"{model_basename}_{ts}.joblib"
            with open(meta_file, "w") as f:
                json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
            paths['meta'] = meta_file
            return paths

        booster.save_model(model_file)
        paths['model'] = model_file
        fi = {}
        try:
            fi_raw = booster.get_score(importance_type="gain")
            fi = {f: float(fi_raw.get(f, 0.0)) for f in feature_list}
        except Exception:
            fi = {f: 0.0 for f in feature_list}
        with open(meta_file, "w") as f:
            json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
        paths['meta'] = meta_file
        if save_fi:
            with open(fi_file, "w") as f:
                json.dump(fi, f, indent=2)
            paths['feature_importance'] = fi_file
    except Exception as e:
        logger.exception("export_model_and_metadata failed: %s", e)
        # fallback single-file save
        if torch is not None:
            torch.save({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.pt")
            paths['pt'] = f"{model_basename}_{ts}.pt"
        else:
            joblib.dump({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.joblib")
            paths['joblib'] = f"{model_basename}_{ts}.joblib"
        with open(meta_file, "w") as f:
            json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
        paths['meta'] = meta_file
    return paths


# ---------------------------
# PyTorch models & multi-model training + bundling
# ---------------------------

if torch is not None:
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=2):
            super(SimpleMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            return self.layers(x)

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, output_dim=2, num_layers=1):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            # expects shape (batch, seq, features)
            _, (hn,_)= self.lstm(x)
            return self.fc(hn[-1])

    class SimpleCNN(nn.Module):
        def __init__(self, input_channels=1, output_dim=2):
            super(SimpleCNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.fc = nn.Linear(32 * 25, output_dim)  # note: depends on input length
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device="cpu"):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        model.to(device)
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_loss)
            # validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
                    correct += (preds.argmax(dim=1) == yb).sum().item()
                    total += yb.size(0)
            val_loss = val_loss / max(1, len(val_loader))
            val_acc = correct / max(1, total)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            logger.info("Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f", epoch+1, epochs, avg_loss, val_loss, val_acc)
        return model, history

    def train_and_bundle_models(train_loader, val_loader, input_dim, save_dir="saved_models", epochs=10):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []
        # candidate models
        candidates = {
            "mlp": SimpleMLP(input_dim=input_dim),
            "lstm": SimpleLSTM(input_dim=input_dim),
            # NOTE: CNN expects sequence dimension; user is responsible to provide shaped input if using CNN
            # "cnn": SimpleCNN(input_channels=1)
        }
        for name, model in candidates.items():
            logger.info("Training candidate model: %s", name)
            trained, history = train_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3, device=device)
            results.append({"name": name, "model": trained, "history": history})
        # rank by best validation accuracy
        results.sort(key=lambda x: max(x["history"]["val_acc"]) if x["history"]["val_acc"] else 0.0, reverse=True)
        top3 = results[:3]
        # bundle top3 into a .pt containing state_dicts + metadata
        bundle = {}
        bundle_meta = {"created_at": datetime.utcnow().isoformat(), "models": []}
        for entry in top3:
            name = entry["name"]
            model = entry["model"]
            bundle[name] = model.state_dict()
            bundle_meta["models"].append({"name": name, "best_val_acc": max(entry["history"]["val_acc"]) if entry["history"]["val_acc"] else 0.0, "history": entry["history"]})
        fname = save_path / f"top3_bundle_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pt"
        torch.save({"bundle": bundle, "meta": bundle_meta}, str(fname))
        logger.info("Saved top3 bundle to %s", fname)
        return top3, str(fname)
else:
    # torch not available: stub functions
    def train_and_bundle_models(*args, **kwargs):
        raise RuntimeError("PyTorch not installed. Multi-model training unavailable.")
