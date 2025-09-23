# app.py — Entry-Range Triangulation (Chunk 1/7: Imports + UI + configs + data fetcher)

import os
import io
import math
import uuid
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass

# ML / metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Optional libraries (graceful fallback)
try:
    from yahooquery import Ticker as YahooTicker
except ImportError:
    YahooTicker = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

# ---------------------------
# Streamlit page & UI inputs
# ---------------------------
st.set_page_config(page_title="Entry-Range Triangulation", layout="wide")
st.title("Entry-Range Triangulation Dashboard — Multi-level (3) Models")

# Basic symbol/time inputs
symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=90))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

# HealthGauge gating (sidebar)
st.sidebar.header("HealthGauge")
buy_threshold_global = st.sidebar.number_input("Global buy threshold", 0.0, 10.0, 5.5)
sell_threshold_global = st.sidebar.number_input("Global sell threshold", 0.0, 10.0, 4.5)
force_run = st.sidebar.checkbox("Force run even if gating fails", value=False)

# XGBoost / training parameters
st.sidebar.header("Training")
num_boost = int(st.sidebar.number_input("XGBoost rounds", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))

# Thresholds for confirm stage
p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

# ---------------------------
# Level-specific configurations
# ---------------------------
st.sidebar.header("Level 1 (Scope)")
lvl1_buy = st.sidebar.number_input("L1 buy threshold (signal)", 0.0, 10.0, 5.5, step=0.1)
lvl1_sell = st.sidebar.number_input("L1 sell threshold (signal)", 0.0, 10.0, 4.5, step=0.1)
lvl1_rr_min = st.sidebar.number_input("L1 RR min", 0.1, 10.0, 1.0, step=0.1)
lvl1_rr_max = st.sidebar.number_input("L1 RR max", 0.1, 10.0, 2.5, step=0.1)
lvl1_sl_min = st.sidebar.number_input("L1 SL min (pct)", 0.001, 0.5, 0.02, step=0.001)
lvl1_sl_max = st.sidebar.number_input("L1 SL max (pct)", 0.001, 0.5, 0.04, step=0.001)

st.sidebar.header("Level 2 (Focus)")
lvl2_buy = st.sidebar.number_input("L2 buy threshold (signal)", 0.0, 10.0, 6.0, step=0.1)
lvl2_sell = st.sidebar.number_input("L2 sell threshold (signal)", 0.0, 10.0, 4.0, step=0.1)
lvl2_rr_min = st.sidebar.number_input("L2 RR min", 0.1, 10.0, 2.0, step=0.1)
lvl2_rr_max = st.sidebar.number_input("L2 RR max", 0.1, 10.0, 3.5, step=0.1)
lvl2_sl_min = st.sidebar.number_input("L2 SL min (pct)", 0.001, 0.5, 0.01, step=0.001)
lvl2_sl_max = st.sidebar.number_input("L2 SL max (pct)", 0.001, 0.5, 0.03, step=0.001)

st.sidebar.header("Level 3 (Triangulation)")
lvl3_buy = st.sidebar.number_input("L3 buy threshold (signal)", 0.0, 10.0, 6.5, step=0.1)
lvl3_sell = st.sidebar.number_input("L3 sell threshold (signal)", 0.0, 10.0, 3.5, step=0.1)
lvl3_rr_min = st.sidebar.number_input("L3 RR min", 0.1, 10.0, 3.0, step=0.1)
lvl3_rr_max = st.sidebar.number_input("L3 RR max", 0.1, 10.0, 5.0, step=0.1)
lvl3_sl_min = st.sidebar.number_input("L3 SL min (pct)", 0.001, 0.5, 0.005, step=0.001)
lvl3_sl_max = st.sidebar.number_input("L3 SL max (pct)", 0.001, 0.5, 0.02, step=0.001)

# Breadth & sweep controls
st.sidebar.header("Breadth & Sweep")
run_breadth = st.sidebar.button("Run breadth backtest (3 levels)")
run_sweep_btn = st.sidebar.button("Run grid sweep")

# ---------------------------
# Price fetcher
# ---------------------------
def fetch_price(symbol: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    if YahooTicker is None:
        st.error("yahooquery not installed; install `yahooquery` to fetch price data.")
        return pd.DataFrame()
    try:
        tq = YahooTicker(symbol)
        raw = tq.history(start=start, end=end, interval=interval)
        if raw is None:
            return pd.DataFrame()
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        # Flatten multiindex if present
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        return raw[~raw.index.duplicated(keep="first")]
    except Exception as exc:
        logger.error("fetch_price failed: %s", exc)
        return pd.DataFrame()

# ---------------------------
# Chunk 2/7 — Feature Engineering + Labeling
# ---------------------------

@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

# Example assets list
ASSETS_LIST = [
    Asset(name="Gold", cot_name="GOLD - COMMODITY EXCHANGE INC.", symbol="GC=F"),
    Asset(name="Silver", cot_name="SILVER - COMMODITY EXCHANGE INC.", symbol="SI=F"),
    Asset(name="EuroFX", cot_name="EURO FX - CHICAGO MERCANTILE EXCHANGE", symbol="6E=F"),
    Asset(name="CrudeOil", cot_name="CRUDE OIL - NYMEX", symbol="CL=F")
]

# ---------------------------
# Feature Engineering Functions
# ---------------------------

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Realized volatility (rolling std dev of returns)
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    rvol = df["return"].rolling(lookback).std() * np.sqrt(252)  # annualized
    return rvol.fillna(0.0)

def compute_atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Average True Range
    """
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean()
    return atr.fillna(0.0)

def compute_health_gauge(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Example HealthGauge score: weighted combination of features
    weights: {"rvol": 0.5, "atr": 0.5} etc.
    """
    rvol = compute_rvol(df, lookback=weights.get("rvol_lookback", 20))
    atr = compute_atr(df, lookback=weights.get("atr_lookback", 14))
    score = rvol * weights.get("rvol", 0.5) + atr * weights.get("atr", 0.5)
    return score

# ---------------------------
# Label Generation
# ---------------------------

def generate_labels(df: pd.DataFrame, forward_window: int = 5, threshold: float = 0.01) -> pd.Series:
    """
    Generate binary labels for ML:
      - 1 = price rises more than threshold in forward_window bars
      - 0 = otherwise
    """
    df = df.copy()
    df["future_return"] = df["close"].shift(-forward_window) / df["close"] - 1.0
    df["label"] = (df["future_return"] > threshold).astype(int)
    return df["label"].fillna(0).astype(int)

# ---------------------------
# Feature + Label Pipeline
# ---------------------------

def prepare_features(df: pd.DataFrame, weights: Dict[str, float], forward_window: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    df = df.copy()
    df["rvol"] = compute_rvol(df, lookback=weights.get("rvol_lookback", 20))
    df["atr"] = compute_atr(df, lookback=weights.get("atr_lookback", 14))
    df["health_gauge"] = compute_health_gauge(df, weights)
    df["label"] = generate_labels(df, forward_window=forward_window, threshold=threshold)
    df = df.dropna()
    return df

# ---------------------------
# Example usage in Streamlit
# ---------------------------
if st.button("Compute Features + Labels"):
    df_price = fetch_price(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval=interval)
    if not df_price.empty:
        weights_example = {"rvol": 0.5, "atr": 0.5, "rvol_lookback": 20, "atr_lookback": 14}
        df_feat = prepare_features(df_price, weights_example)
        st.write(df_feat.tail())
    else:
        st.warning("Price data not available for this symbol.")

# ---------------------------
# Chunk 3/7 — Training + Model Saving
# ---------------------------

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# ---------------------------
# Model Training Function
# ---------------------------

def train_xgb_model(df: pd.DataFrame, feature_cols: list, label_col: str = "label", test_size: float = 0.2, random_state: int = 42, early_stopping_rounds: int = 10) -> tuple:
    """
    Train an XGBoost classifier with optional early stopping
    Returns:
        model: trained XGBClassifier
        metrics: dict of accuracy and F1
    """
    X = df[feature_cols]
    y = df[label_col]

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Initialize model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=True
    )

    # Predictions
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred)
    }

    return model, metrics

# ---------------------------
# Model Save / Load Functions
# ---------------------------

def save_model(model: XGBClassifier, path: str = "xgb_model.pkl"):
    joblib.dump(model, path)
    st.success(f"Model saved to {path}")

def load_model(path: str = "xgb_model.pkl") -> XGBClassifier:
    try:
        model = joblib.load(path)
        st.success(f"Model loaded from {path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------------------------
# Example usage in Streamlit
# ---------------------------

if st.button("Train XGBoost Model"):
    feature_cols = ["rvol", "atr", "health_gauge"]
    model, metrics = train_xgb_model(df_feat, feature_cols=feature_cols)
    save_model(model)
    st.write("Training Metrics:", metrics)

# ---------------------------
# Chunk 4/7 — Inference + Signal Generation
# ---------------------------

# Load model for inference
model = load_model("xgb_model.pkl")

# ---------------------------
# Signal Generation Function
# ---------------------------

def generate_signals(df: pd.DataFrame, model: XGBClassifier, feature_cols: list) -> pd.DataFrame:
    """
    Generate buy/sell/hold signals based on model predictions
    Adds 'prediction' and 'signal' columns to df
    """
    if model is None:
        st.error("Model not loaded. Cannot generate signals.")
        return df

    # Make predictions
    df["prediction"] = model.predict(df[feature_cols])
    
    # Map numeric predictions to signals
    df["signal"] = df["prediction"].map({0: "Hold", 1: "Buy", 2: "Sell"})

    return df

# ---------------------------
# Apply to current feature dataframe
# ---------------------------

if st.button("Generate Signals"):
    feature_cols = ["rvol", "atr", "health_gauge"]
    df_signals = generate_signals(df_feat, model, feature_cols)
    st.dataframe(df_signals[["name", "signal", "prediction"]])

# ---------------------------
# Chunk 5/7 — Real-Time Data Fetch + Preprocessing
# ---------------------------

from yahooquery import Ticker
from sodapy import Socrata

# --- Data Fetchers ---
def fetch_price(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception as e:
        logger.error(f"fetch_price failed for {symbol}: {e}")
    return pd.DataFrame()

def init_socrata_client() -> Socrata:
    domain = "publicreporting.cftc.gov"
    app_token = os.getenv("WSCaavlIcDgtLVZbJA1FKkq40")
    username = os.getenv("SEGAB120_EMAIL")
    password = os.getenv("SEGAB120_PASSWORD")
    return Socrata(domain, app_token, username=username, password=password)

def fetch_cot(client: Socrata, dataset_id: str = "6dca-aqww", report_date: str = None, max_rows: int = 10000) -> pd.DataFrame:
    where_clause = f"report_date_as_yyyy_mm_dd = '{report_date}'" if report_date else None
    try:
        results = client.get(dataset_id, where=where_clause, limit=max_rows)
        df = pd.DataFrame.from_records(results)
        if not df.empty and "report_date_as_yyyy_mm_dd" in df.columns:
            df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"fetch_cot failed: {e}")
        return pd.DataFrame()

# --- Feature Engineering ---
def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = df["close"].pct_change()
    rolling_vol = returns.rolling(window).std()
    return rolling_vol / rolling_vol.mean()

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()

def calculate_health_gauge(rvol: pd.Series, atr: pd.Series) -> pd.Series:
    return (rvol.rank(pct=True) + atr.rank(pct=True)) / 2.0

# --- Build Features DF ---
def build_feature_df(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    df_feat = price_df.copy()
    df_feat["rvol"] = compute_rvol(df_feat)
    df_feat["atr"] = compute_atr(df_feat)
    df_feat["health_gauge"] = calculate_health_gauge(df_feat["rvol"], df_feat["atr"])
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

# ---------------------------
# Chunk 6/7 — Breadth Backtest + Grid Sweep
# ---------------------------

def simulate_trade(entry_px: float, atr: float, direction: str, rr: float, sl_pct: float):
    """
    Simulate trade outcome with given parameters.
    """
    if direction == "long":
        tp_px = entry_px + rr * atr
        sl_px = entry_px - sl_pct * entry_px
        outcome = 1 if tp_px > entry_px else -1
    else:
        tp_px = entry_px - rr * atr
        sl_px = entry_px + sl_pct * entry_px
        outcome = 1 if tp_px < entry_px else -1

    return {"tp": tp_px, "sl": sl_px, "outcome": outcome}


def breadth_backtest(df_feat: pd.DataFrame, thresholds: dict, rr_scopes: dict, sl_scopes: dict):
    """
    Run breadth backtest for Levels 1 → 3 with given configs.
    Returns dict of results.
    """
    results = {}
    for lvl in ["Level1", "Level2", "Level3"]:
        buy_th, sell_th = thresholds[lvl]
        rr_low, rr_high = rr_scopes[lvl]
        sl_low, sl_high = sl_scopes[lvl]

        level_trades = []
        for _, row in df_feat.iterrows():
            if row["health_gauge"] >= buy_th:
                trade = simulate_trade(
                    row["close"], row["atr"], "long",
                    rr=np.random.uniform(rr_low, rr_high),
                    sl_pct=np.random.uniform(sl_low, sl_high)
                )
                level_trades.append(trade["outcome"])
            elif row["health_gauge"] <= sell_th:
                trade = simulate_trade(
                    row["close"], row["atr"], "short",
                    rr=np.random.uniform(rr_low, rr_high),
                    sl_pct=np.random.uniform(sl_low, sl_high)
                )
                level_trades.append(trade["outcome"])

        winrate = np.mean([1 if t == 1 else 0 for t in level_trades]) if level_trades else 0
        results[lvl] = {"n_trades": len(level_trades), "winrate": winrate}

    return results


def grid_sweep(df_feat: pd.DataFrame, thresholds: dict, rr_scopes: dict, sl_scopes: dict):
    """
    Run grid sweep across parameter space for Levels 1 → 3.
    Returns dict of sweep results.
    """
    sweep_results = {}
    for lvl in ["Level1", "Level2", "Level3"]:
        buy_th, sell_th = thresholds[lvl]
        rr_low, rr_high = rr_scopes[lvl]
        sl_low, sl_high = sl_scopes[lvl]

        outcomes = []
        for rr in np.linspace(rr_low, rr_high, num=3):
            for sl in np.linspace(sl_low, sl_high, num=3):
                trades = []
                for _, row in df_feat.iterrows():
                    if row["health_gauge"] >= buy_th:
                        t = simulate_trade(row["close"], row["atr"], "long", rr, sl)
                        trades.append(t["outcome"])
                    elif row["health_gauge"] <= sell_th:
                        t = simulate_trade(row["close"], row["atr"], "short", rr, sl)
                        trades.append(t["outcome"])
                winrate = np.mean([1 if t == 1 else 0 for t in trades]) if trades else 0
                outcomes.append({"rr": rr, "sl": sl, "winrate": winrate})

        sweep_results[lvl] = outcomes

    return sweep_results

# ---------------------------
# Chunk 7/7 — UI Integration + Results Display + Logging
# ---------------------------

st.subheader("Breadth Backtest & Grid Sweep")

if st.button("Run Breadth Backtest"):
    st.info("Running breadth backtest across Levels 1 → 3...")
    try:
        thresholds = {
            "Level1": (buy_th1, sell_th1),
            "Level2": (buy_th2, sell_th2),
            "Level3": (buy_th3, sell_th3),
        }
        rr_scopes = {
            "Level1": tuple(rr_scope1),
            "Level2": tuple(rr_scope2),
            "Level3": tuple(rr_scope3),
        }
        sl_scopes = {
            "Level1": tuple(sl_scope1),
            "Level2": tuple(sl_scope2),
            "Level3": tuple(sl_scope3),
        }

        breadth_results = breadth_backtest(clean, thresholds, rr_scopes, sl_scopes)
        st.write("### Breadth Results")
        st.json(breadth_results)

    except Exception as e:
        st.error(f"Breadth backtest failed: {e}")
        logger.error("Breadth backtest failed", exc_info=True)


if st.button("Run Grid Sweep"):
    st.info("Running grid sweep across Levels 1 → 3...")
    try:
        thresholds = {
            "Level1": (buy_th1, sell_th1),
            "Level2": (buy_th2, sell_th2),
            "Level3": (buy_th3, sell_th3),
        }
        rr_scopes = {
            "Level1": tuple(rr_scope1),
            "Level2": tuple(rr_scope2),
            "Level3": tuple(rr_scope3),
        }
        sl_scopes = {
            "Level1": tuple(sl_scope1),
            "Level2": tuple(sl_scope2),
            "Level3": tuple(sl_scope3),
        }

        sweep_results = grid_sweep(clean, thresholds, rr_scopes, sl_scopes)
        st.write("### Grid Sweep Results")

        for lvl, outcomes in sweep_results.items():
            st.write(f"**{lvl}**")
            df_outcomes = pd.DataFrame(outcomes)
            st.dataframe(df_outcomes)

            fig, ax = plt.subplots()
            pivot = df_outcomes.pivot(index="rr", columns="sl", values="winrate")
            cax = ax.matshow(pivot.values, cmap="viridis")
            plt.colorbar(cax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
            ax.set_xlabel("Stop Loss %")
            ax.set_ylabel("RR Scope")
            ax.set_title(f"{lvl} Winrate Heatmap")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Grid sweep failed: {e}")
        logger.error("Grid sweep failed", exc_info=True)

st.success("Dashboard loaded successfully ✅")