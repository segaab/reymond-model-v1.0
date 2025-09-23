# app.py — Entry-Range Triangulation (integrated single-file)
# Chunk 1/7: imports + Streamlit UI + level configs + data fetcher

import os
import io
import math
import time
import uuid
import json
import joblib
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass

# ML / metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Optional libs (graceful degradation)
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# Logging
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

# HealthGauge gating
st.sidebar.header("HealthGauge")
buy_threshold_global = st.sidebar.number_input("Global buy threshold (for gating)", 0.0, 10.0, 5.5)
sell_threshold_global = st.sidebar.number_input("Global sell threshold (for gating)", 0.0, 10.0, 4.5)
force_run = st.sidebar.checkbox("Force run even if gating fails", value=False)

# XGBoost / training
st.sidebar.header("Training")
num_boost = int(st.sidebar.number_input("XGBoost rounds", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))

# Confirm thresholds
p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

# Level-specific configs (user requested)
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


# fetch price: using yahooquery
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

# Chunk 2/7: features + labeling

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "volume" not in df.columns:
        return pd.Series(1.0, index=df.index)
    rolling = df["volume"].rolling(window=lookback, min_periods=1).mean()
    return (df["volume"] / rolling.replace(0, np.nan)).fillna(1.0)

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    if daily_bars is None or daily_bars.empty:
        return pd.DataFrame()
    db = daily_bars.copy()
    db["rvol"] = compute_rvol(db)
    db["health_gauge"] = (db["rvol"] >= threshold).astype(float)
    return db[["health_gauge"]]

def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

# Labeling helpers
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long"
) -> pd.DataFrame:
    """
    Generate candidates and triple-barrier-like labels. Defaults to 'long' logic as pipeline uses long entries.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = ensure_unique_index(bars)
    for col in ("high", "low", "close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")

    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(window=atr_window, min_periods=1).mean()

    recs: List[Dict[str, Any]] = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr = float(bars["atr"].iat[i])
        if atr <= 0 or math.isnan(atr):
            continue

        if direction == "long":
            sl_px = entry_px - k_sl * atr
            tp_px = entry_px + k_tp * atr
        else:
            sl_px = entry_px + k_sl * atr
            tp_px = entry_px - k_tp * atr

        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_px = float(bars["close"].iat[end_idx])

        for j in range(i + 1, end_idx + 1):
            hi = float(bars["high"].iat[j]); lo = float(bars["low"].iat[j])
            if direction == "long":
                if hi >= tp_px:
                    label, hit_idx, hit_px = 1, j, tp_px
                    break
                if lo <= sl_px:
                    label, hit_idx, hit_px = 0, j, sl_px
                    break
            else:
                if lo <= tp_px:
                    label, hit_idx, hit_px = 1, j, tp_px
                    break
                if hi >= sl_px:
                    label, hit_idx, hit_px = 0, j, sl_px
                    break

        end_t = bars.index[hit_idx]
        realized_return = (hit_px - entry_px) / entry_px if direction == "long" else (entry_px - hit_px) / entry_px
        dur_min = (end_t - t).total_seconds() / 60.0

        recs.append({
            "candidate_time": t,
            "entry_price": float(entry_px),
            "atr": float(atr),
            "sl_price": float(sl_px),
            "tp_price": float(tp_px),
            "end_time": end_t,
            "label": int(label),
            "duration": float(dur_min),
            "realized_return": float(realized_return),
            "direction": direction
        })
    return pd.DataFrame(recs)



def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str = "GC=F",
                    sl: float = 0.02,
                    tp: float = 0.04,
                    max_holding: int = 60) -> pd.DataFrame:
    if df is None or df.empty or bars is None or bars.empty:
        return pd.DataFrame()
    trades = []
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    for _, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl):
            continue
        entry_t = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_t not in bars.index:
            continue
        entry_px = float(bars.loc[entry_t, "close"])
        direction = 1 if lbl > 0 else -1
        sl_px = entry_px * (1 - sl) if direction > 0 else entry_px * (1 + sl)
        tp_px = entry_px * (1 + tp) if direction > 0 else entry_px * (1 - tp)
        exit_t, exit_px, pnl = None, None, None
        segment = bars.loc[entry_t:].head(max_holding)
        if segment.empty:
            continue
        for t, b in segment.iterrows():
            lo, hi = float(b["low"]), float(b["high"])
            if direction > 0:
                if lo <= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl
                    break
                if hi >= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp
                    break
            else:
                if hi >= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl
                    break
                if lo <= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp
                    break
        if exit_t is None:
            last_bar = segment.iloc[-1]
            exit_t = last_bar.name
            exit_px = float(last_bar["close"])
            pnl = (exit_px - entry_px) / entry_px * direction
        trades.append({
            "symbol": symbol,
            "entry_time": entry_t,
            "entry_price": entry_px,
            "direction": direction,
            "exit_time": exit_t,
            "exit_price": exit_px,
            "pnl": float(pnl),
            "rr_ratio": float(pnl/sl) if sl != 0 else 0.0  # Add RR ratio calculation
        })
    return pd.DataFrame(trades)

def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    total_trades = len(trades)
    win_rate = float((trades["pnl"] > 0).mean())
    avg_pnl = float(trades["pnl"].mean())
    median_pnl = float(trades["pnl"].median())
    total_pnl = float(trades["pnl"].sum())
    max_dd = float(trades["pnl"].cumsum().min())
    
    # New metrics for RR distribution
    rr_mean = float(trades["rr_ratio"].mean())
    rr_min = float(trades["rr_ratio"].min())
    rr_80th = float(np.percentile(trades["rr_ratio"], 80)) if len(trades) > 5 else float(rr_mean)
    
    # Annualized returns calculation (simplified)
    start_time = trades["entry_time"].min()
    end_time = trades["exit_time"].max()
    days_range = (end_time - start_time).days if start_time and end_time else 1
    days_range = max(1, days_range)
    annual_return = total_pnl * (365 / days_range)
    
    return pd.DataFrame([{
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "start_time": start_time,
        "end_time": end_time,
        "rr_mean": rr_mean,
        "rr_min": rr_min,
        "rr_80th_percentile": rr_80th,
        "annual_return": annual_return
    }])

def combine_summaries(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode, trades_df in results.items():
        if trades_df is None or trades_df.empty:
            continue
        s = summarize_trades(trades_df)
        if not s.empty:
            s["mode"] = mode
            rows.append(s)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# Chunk 4/7: model training (xgboost), prediction and export helpers

class BoosterWrapper:
    def __init__(self, booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names
        self.best_iteration = getattr(booster, "best_iteration", None)

    def _dmatrix(self, X: pd.DataFrame):
        Xp = X.reindex(columns=self.feature_names).fillna(0.0)
        return xgb.DMatrix(Xp, feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        d = self._dmatrix(X)
        if self.best_iteration is not None:
            raw = self.booster.predict(d, iteration_range=(0, int(self.best_iteration) + 1))
        else:
            raw = self.booster.predict(d)
        raw = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        return pd.Series(raw, index=X.index, name="confirm_proba")

    def save_model(self, path: str):
        try:
            self.booster.save_model(path)
        except Exception:
            joblib.dump(self.booster, path)

    def feature_importance(self) -> pd.DataFrame:
        imp = {}
        try:
            imp = self.booster.get_score(importance_type="gain")
        except Exception:
            pass
        return (pd.DataFrame([(f, float(imp.get(f, 0.0))) for f in self.feature_names],
                             columns=["feature", "gain"])
                .sort_values("gain", ascending=False).reset_index(drop=True))

def train_xgb_confirm(clean: pd.DataFrame,
                      feature_cols: List[str],
                      label_col: str = "label",
                      num_boost_round: int = 200,
                      early_stopping_rounds: int = 20,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[BoosterWrapper, Dict[str, Any]]:
    if xgb is None:
        raise RuntimeError("xgboost not installed")
    X = clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(clean[label_col], errors="coerce")
    mask = X.replace([np.inf, -np.inf], np.nan).notnull().all(1) & y.notnull()
    X, y = X[mask], y[mask]
    if len(X) < 10 or y.nunique() < 2:
        raise ValueError("Not enough data or not both classes present for training")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    params = {
        "objective": "binary:logistic",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "verbosity": 0,
        "scale_pos_weight": float((y == 0).sum()) / max(1, (y == 1).sum())
    }
    bst = xgb.train(params, dtr, num_boost_round, evals=[(dva, "val")],
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    wrap = BoosterWrapper(bst, feature_cols)
    y_proba = wrap.predict_proba(Xva)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "accuracy": float(accuracy_score(yva, y_pred)),
        "f1": float(f1_score(yva, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yva, y_proba))
    }
    return wrap, metrics

def predict_confirm_prob(model: BoosterWrapper, df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    missing = [c for c in feature_cols if c not in df.columns]
    for m in missing:
        df[m] = 0.0
    return model.predict_proba(df[feature_cols])

def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str,Any], model_basename: str, save_fi: bool = True):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    paths = {}
    try:
        if hasattr(model_wrapper, "booster"):
            model_file = f"{model_basename}_{ts}.model"
            model_wrapper.save_model(model_file)
            paths["model"] = model_file
        else:
            p = f"{model_basename}_{ts}.joblib"
            joblib.dump({"model": model_wrapper, "features": feature_list, "metrics": metrics}, p)
            paths["model"] = p
        meta_file = f"{model_basename}_{ts}.json"
        with open(meta_file, "w") as f:
            json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, default=str, indent=2)
        paths["meta"] = meta_file
        if save_fi and hasattr(model_wrapper, "feature_importance"):
            try:
                fi_df = model_wrapper.feature_importance()
                fi_path = f"{model_basename}_{ts}_fi.csv"
                fi_df.to_csv(fi_path, index=False)
                paths["feature_importance"] = fi_path
            except Exception:
                pass
    except Exception as exc:
        logger.exception("Failed to export model: %s", exc)
    return paths

def _compute_explicit_windows(levels_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Given levels_config with per-level buy_th and sell_th, return explicit non-overlapping windows:
      return { level: {"buy": (buy_min, buy_max), "sell": (sell_min, sell_max)} }
    buy windows: [buy_min, buy_max)  (upper exclusive); highest level => upper = +inf
    sell windows: (sell_min, sell_max] (lower exclusive); lowest level => lower = -inf
    """
    out = {}
    # Build buy windows (descending by buy_th)
    buy_list = sorted([(lvl, cfg.get("buy_th", 0.0)) for lvl, cfg in levels_config.items()], key=lambda x: x[1], reverse=True)
    # For descending order, the first is highest buy
    for i, (lvl, buy_th) in enumerate(buy_list):
        upper = buy_list[i - 1][1] if i - 1 >= 0 else np.inf
        lower = buy_th
        out.setdefault(lvl, {})["buy"] = (float(lower), float(upper))

    # Build sell windows (ascending by sell_th)
    sell_list = sorted([(lvl, cfg.get("sell_th", 0.0)) for lvl, cfg in levels_config.items()], key=lambda x: x[1])
    for i, (lvl, sell_th) in enumerate(sell_list):
        lower = sell_list[i - 1][1] if i - 1 >= 0 else -np.inf
        upper = sell_th
        out.setdefault(lvl, {})["sell"] = (float(lower), float(upper))

    return out

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    """
    For each level, apply explicit buy/sell windows (non-overlapping) and simulate trades.
    Returns standardized dict with per-level trades and a combined summary list.
    """
    out = {"summary": [], "detailed_trades": {}, "diagnostics": [], "metrics_by_level": {}}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out

    windows = _compute_explicit_windows(levels_config)

    # create numeric 'signal' if missing
    df_base = clean.copy()
    if "signal" not in df_base.columns:
        if "pred_prob" in df_base.columns:
            df_base["signal"] = (df_base["pred_prob"] * 10).round().astype(int)
        else:
            df_base["signal"] = 0

    # For each level, filter using explicit windows (buy and sell) and simulate
    for lvl_name, cfg in levels_config.items():
        try:
            w = windows.get(lvl_name, {})
            buy_min, buy_max = w.get("buy", (cfg.get("buy_th", 0.0), np.inf))
            sell_min, sell_max = w.get("sell", (-np.inf, cfg.get("sell_th", 0.0)))

            df = df_base.copy()
            df["assigned"] = False
            df["pred_label"] = 0
            df["level"] = lvl_name

            # Buy mask: signal in [buy_min, buy_max)
            buy_mask = (df["signal"] >= buy_min) & (df["signal"] < buy_max)
            # Sell mask: signal in (sell_min, sell_max]  => (df["signal"] > sell_min) & (df["signal"] <= sell_max)
            sell_mask = (df["signal"] > sell_min) & (df["signal"] <= sell_max)

            # Assign buys first, then sells (but masks are disjoint across levels by construction)
            df.loc[buy_mask & (~df["assigned"]), "pred_label"] = 1
            df.loc[buy_mask, "assigned"] = True
            df.loc[sell_mask & (~df["assigned"]), "pred_label"] = -1
            df.loc[sell_mask, "assigned"] = True

            # Count entries made on each level
            buy_entries = buy_mask.sum()
            sell_entries = sell_mask.sum()
            total_entries = buy_entries + sell_entries

            # Representative SL and TP from level config (mean)
            sl_pct = float((cfg.get("sl_min", 0.01) + cfg.get("sl_max", 0.02)) / 2.0)
            rr = float((cfg.get("rr_min", 1.0) + cfg.get("rr_max", 2.0)) / 2.0)
            tp_pct = rr * sl_pct

            trades = simulate_limits(df, bars, label_col="pred_label",
                                     sl=sl_pct, tp=tp_pct,
                                     max_holding=int(model_train_kwargs.get("max_bars", 60)))
            out["detailed_trades"][lvl_name] = trades
            
            # Enhanced metrics for this level
            level_metrics = {
                "total_entries": total_entries,
                "buy_entries": buy_entries,
                "sell_entries": sell_entries,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "rr_target": rr
            }
            
            summary = summarize_trades(trades)
            if not summary.empty:
                srow = summary.iloc[0].to_dict()
                srow.update({"mode": lvl_name, "sl_pct": sl_pct, "tp_pct": tp_pct})
                # Add the enhanced metrics
                level_metrics.update({
                    "win_rate": srow.get("win_rate", 0.0),
                    "total_trades": srow.get("total_trades", 0),
                    "total_pnl": srow.get("total_pnl", 0.0),
                    "avg_pnl": srow.get("avg_pnl", 0.0),
                    "rr_mean": srow.get("rr_mean", 0.0),
                    "rr_min": srow.get("rr_min", 0.0),
                    "rr_80th_percentile": srow.get("rr_80th_percentile", 0.0),
                    "annual_return": srow.get("annual_return", 0.0)
                })
                out["summary"].append(srow)
            
            out["metrics_by_level"][lvl_name] = level_metrics
            out["diagnostics"].append(f"{lvl_name}: buy_range=[{buy_min},{buy_max}), sell_range=({sell_min},{sell_max}], sl={sl_pct:.4f}, tp={tp_pct:.4f}, trades={len(trades)}, win_rate={level_metrics.get('win_rate', 0.0):.2f}")
        except Exception as exc:
            logger.exception("Breadth level %s failed: %s", lvl_name, exc)
            out["diagnostics"].append(f"{lvl_name} error: {exc}")

    return out

def run_grid_sweep(clean: pd.DataFrame,
                   bars: pd.DataFrame,
                   rr_vals: List[float],
                   sl_ranges: List[Tuple[float,float]],
                   mpt_list: List[float],
                   feature_cols: List[str],
                   model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    """
    Sweep RR x SL ranges x model probability thresholds (mpt_list).
    Returns dict keyed by config string with overlay and summary.
    """
    results = {}
    if clean is None or clean.empty:
        return results
    for rr in rr_vals:
        for sl_min, sl_max in sl_ranges:
            sl_pct = (sl_min + sl_max) / 2.0
            tp_pct = rr * sl_pct
            for mpt in mpt_list:
                key = f"rr{rr:.2f}_sl{sl_min:.3f}-{sl_max:.3f}_mpt{mpt:.2f}"
                try:
                    df = clean.copy()
                    if "pred_prob" not in df.columns:
                        if "signal" in df.columns:
                            df["pred_prob"] = (df["signal"] / 10.0).clip(0.0, 1.0)
                        else:
                            df["pred_prob"] = 0.0
                    df["pred_label"] = (df["pred_prob"] >= mpt).astype(int)
                    trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
                    summary = summarize_trades(trades)
                    
                    results[key] = {
                        "trades_count": len(trades), 
                        "overlay": trades,
                        "win_rate": float(summary["win_rate"].iloc[0]) if not summary.empty else 0.0,
                        "rr_mean": float(summary["rr_mean"].iloc[0]) if not summary.empty else 0.0,
                        "rr_80th": float(summary["rr_80th_percentile"].iloc[0]) if not summary.empty else 0.0,
                        "total_pnl": float(summary["total_pnl"].iloc[0]) if not summary.empty else 0.0
                    }
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results


class SupabaseLogger:
    def __init__(self):
        if create_client is None:
            raise RuntimeError("supabase client not installed")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE env vars missing")
        self.client = create_client(url, key)
        self.runs_tbl = "entry_runs"
        self.trades_tbl = "entry_trades"
        self.level_metrics_tbl = "level_metrics"  # New table for level-specific metrics

    def log_run(self, metrics: Dict, metadata: Dict, trades: Optional[List[Dict]] = None, level_metrics: Optional[Dict[str, Dict]] = None) -> str:
        run_id = metadata.get("run_id") or str(uuid.uuid4())
        metadata["run_id"] = run_id
        payload = {**metadata, "metrics": metrics}
        resp = self.client.table(self.runs_tbl).insert(payload).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to insert run: {resp.error}")
        
        # Log trades if provided
        if trades:
            for t in trades:
                t["run_id"] = run_id
            tr_resp = self.client.table(self.trades_tbl).insert(trades).execute()
            if getattr(tr_resp, "error", None):
                raise RuntimeError(f"Failed to insert trades: {tr_resp.error}")
        
        # Log level-specific metrics if provided
        if level_metrics:
            level_records = []
            for level_name, level_data in level_metrics.items():
                record = {
                    "run_id": run_id,
                    "level": level_name,
                    **level_data
                }
                level_records.append(record)
            
            if level_records:
                level_resp = self.client.table(self.level_metrics_tbl).insert(level_records).execute()
                if getattr(level_resp, "error", None):
                    raise RuntimeError(f"Failed to insert level metrics: {level_resp.error}")
        
        return run_id

    def fetch_runs(self, symbol: Optional[str] = None, limit: int = 50):
        q = self.client.table(self.runs_tbl)
        if symbol:
            q = q.eq("symbol", symbol)
        resp = q.order("start_date", desc=True).limit(limit).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch runs: {resp.error}")
        return getattr(resp, "data", [])

    def fetch_trades(self, run_id: str):
        resp = self.client.table(self.trades_tbl).select("*").eq("run_id", run_id).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch trades: {resp.error}")
        return getattr(resp, "data", [])
        
    def fetch_level_metrics(self, run_id: str):
        resp = self.client.table(self.level_metrics_tbl).select("*").eq("run_id", run_id).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch level metrics: {resp.error}")
        return getattr(resp, "data", [])

# Helpers
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def pick_top_runs_by_metrics(runs: List[Dict], top_n: int = 3):
    """
    Ranks runs using weightings:
      1) total_pnl (primary importance)
      2) win_rate (secondary)
    """
    scored = []
    for r in runs:
        metrics = r.get("metrics") if isinstance(r, dict) else r
        total_pnl = metrics.get("total_pnl", 0.0) if metrics else 0.0
        win_rate = metrics.get("win_rate", 0.0) if metrics else 0.0
        score = float(total_pnl) + float(win_rate) * 0.01
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_n]]


# small placeholders / defaults
feat_cols_default = ["atr", "rvol", "duration", "hg"]

def run_main_pipeline():
    st.info(f"Fetching price for {symbol} …")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars is None or bars.empty:
        st.error("No price data returned.")
        return

    bars = ensure_unique_index(bars)
    # daily for health
    try:
        daily = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    except Exception:
        daily = bars.copy().resample("1D").agg({"close":"last","volume":"sum"})
    health = calculate_health_gauge(None, daily)
    latest_health = float(health["health_gauge"].iloc[-1]) if not health.empty else 0.0
    st.metric("Latest HealthGauge", f"{latest_health:.3f}")

    if not (latest_health >= buy_threshold_global or latest_health <= sell_threshold_global or force_run):
        st.warning("Health gating prevented run. Use 'Force run' to override.")
        return

    # compute rvol and candidates
    bars["rvol"] = compute_rvol(bars, lookback=20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates generated.")
        return

    # add health-gauge to candidates
    if not health.empty:
        hg_map = health["health_gauge"].reindex(pd.to_datetime(health.index)).ffill().to_dict()
        cands["hg"] = cands["candidate_time"].dt.normalize().map(lambda t: hg_map.get(pd.Timestamp(t).normalize(), 0.0))
    else:
        cands["hg"] = 0.0

    feat_cols = feat_cols_default.copy()
    if "hg" not in feat_cols:
        feat_cols.append("hg")

    # training confirm-stage (one model for pipeline)
    st.info("Training confirm-stage XGBoost model…")
    try:
        model_wrap, metrics = train_xgb_confirm(cands, feat_cols, label_col="label",
                                                num_boost_round=num_boost,
                                                early_stopping_rounds=early_stop,
                                                test_size=test_size)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        st.error(f"Training failed: {exc}")
        return

    st.write("Training metrics:", metrics)

    # predict and simulate
    cands["pred_prob"] = predict_confirm_prob(model_wrap, cands, feat_cols)
    cands["pred_label"] = (cands["pred_prob"] >= p_fast).astype(int)

    st.info("Simulating trades using predicted labels…")
    trades = simulate_limits(cands, bars, label_col="pred_label", max_holding=60)
    st.write("Simulated trades:", len(trades))
    if not trades.empty:
        st.dataframe(trades.head())
        s = summarize_trades(trades)
        st.dataframe(s)

    # Auto-log to Supabase (if configured) and then save model
    if create_client is not None and st.button("Log run to Supabase and save model"):
        try:
            supa = SupabaseLogger()
            run_id = str(uuid.uuid4())
            metadata = {
                "run_id": run_id, "symbol": symbol, "start_date": str(start_date),
                "end_date": str(end_date), "interval": interval,
                "feature_cols": feat_cols, "training_params": {"num_boost": num_boost, "early_stop": early_stop, "test_size": test_size}
            }
            backtest_metrics = {
                "num_trades": int(len(trades)),
                "total_pnl": float(trades["pnl"].sum() if not trades.empty else 0.0),
                "win_rate": float((trades["pnl"] > 0).mean() if not trades.empty else 0.0),
            }
            combined = {}
            combined.update(metrics if isinstance(metrics, dict) else {})
            combined.update(backtest_metrics)
            trade_list = []
            if not trades.empty:
                for r in trades.to_dict(orient="records"):
                    trade_list.append({
                        "candidate_time": str(r.get("entry_time")),
                        "entry_price": float(r.get("entry_price") or 0.0),
                        "exit_time": str(r.get("exit_time")),
                        "ret": float(r.get("pnl") or 0.0)
                    })
            run_id_returned = supa.log_run(metrics=combined, metadata=metadata, trades=trade_list)
            st.success(f"Logged run {run_id_returned}")

            # After logging, fetch recent runs and pick top3 by weighting and offer saving
            runs = supa.fetch_runs(symbol=symbol, limit=20)
            top_runs = pick_top_runs_by_metrics(runs, top_n=3)
            st.subheader("Top 3 recent runs (by total_pnl then win_rate)")
            for rr in top_runs:
                st.write(rr)

            # Save model locally
            model_basename = f"confirm_model_{symbol.replace('=','_')}"
            saved_paths = export_model_and_metadata(model_wrap, feat_cols, combined, model_basename, save_fi=True)
            st.success(f"Saved model files: {saved_paths}")
        except Exception as exc:
            logger.exception("Logging/saving failed: %s", exc)
            st.error(f"Logging/saving failed: {exc}")

# Breadth handler
if run_breadth:
    st.info("Running breadth backtest across 3 levels...")
    levels = {
        "L1": {"buy_th": float(lvl1_buy), "sell_th": float(lvl1_sell), "rr_min": float(lvl1_rr_min), "rr_max": float(lvl1_rr_max), "sl_min": float(lvl1_sl_min), "sl_max": float(lvl1_sl_max)},
        "L2": {"buy_th": float(lvl2_buy), "sell_th": float(lvl2_sell), "rr_min": float(lvl2_rr_min), "rr_max": float(lvl2_rr_max), "sl_min": float(lvl2_sl_min), "sl_max": float(lvl2_sl_max)},
        "L3": {"buy_th": float(lvl3_buy), "sell_th": float(lvl3_sell), "rr_min": float(lvl3_rr_min), "rr_max": float(lvl3_rr_max), "sl_min": float(lvl3_sl_min), "sl_max": float(lvl3_sl_max)}
    }
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    bars["rvol"] = compute_rvol(bars, 20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates available for breadth run.")
    else:
        cands["pred_prob"] = cands.get("pred_prob", 0.0)
        try:
            res = run_breadth_backtest(clean=cands, bars=bars, levels_config=levels, feature_cols=feat_cols_default, model_train_kwargs={"max_bars": 60})
            
            # Display the enhanced metrics for each level
            st.subheader("Breadth Summary")
            summary_df = pd.DataFrame(res.get("summary", []))
            if not summary_df.empty:
                st.dataframe(summary_df)
            else:
                st.warning("Breadth run returned no summary rows.")
                
            # Display level-specific metrics in a more structured format
            st.subheader("Level-specific Metrics")
            for lvl, metrics in res.get("metrics_by_level", {}).items():
                with st.expander(f"{lvl} Metrics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2f}")
                        st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
                    with col2:
                        st.metric("RR Mean", f"{metrics.get('rr_mean', 0.0):.2f}")
                        st.metric("RR 80th Percentile", f"{metrics.get('rr_80th_percentile', 0.0):.2f}")
                    with col3:
                        st.metric("Total PnL", f"{metrics.get('total_pnl', 0.0):.2f}")
                        st.metric("Annual Return", f"{metrics.get('annual_return', 0.0):.2f}")
                    
                    st.write(f"Entries: Buy={metrics.get('buy_entries', 0)}, Sell={metrics.get('sell_entries', 0)}")
                    st.write(f"Target RR: {metrics.get('rr_target', 0.0):.2f} (SL: {metrics.get('sl_pct', 0.0):.4f}, TP: {metrics.get('tp_pct', 0.0):.4f})")
            
            # Display trades for each level
            detailed = res.get("detailed_trades", {})
            for lvl, df in detailed.items():
                with st.expander(f"{lvl} -- {len(df)} trades"):
                    st.dataframe(df.head(50))
            
            # Add a download button for the full results
            for lvl, df in detailed.items():
                if not df.empty:
                    st.download_button(f"Download {lvl} trades CSV", df_to_csv_bytes(df), f"{symbol}_{lvl}_trades.csv", "text/csv")
            
            # Log the breadth test results to Supabase if button is clicked
            if create_client is not None and st.button("Log breadth test to Supabase"):
                try:
                    supa = SupabaseLogger()
                    run_id = str(uuid.uuid4())
                    metadata = {
                        "run_id": run_id, 
                        "symbol": symbol, 
                        "start_date": str(start_date),
                        "end_date": str(end_date), 
                        "interval": interval,
                        "run_type": "breadth_backtest",
                        "levels": list(levels.keys())
                    }
                    
                    # Prepare the combined metrics
                    combined_metrics = {
                        "total_levels": len(res.get("metrics_by_level", {})),
                        "total_trades": sum(m.get("total_trades", 0) for m in res.get("metrics_by_level", {}).values()),
                        "avg_win_rate": np.mean([m.get("win_rate", 0.0) for m in res.get("metrics_by_level", {}).values()]) if res.get("metrics_by_level") else 0.0,
                        "total_pnl": sum(m.get("total_pnl", 0.0) for m in res.get("metrics_by_level", {}).values()),
                    }
                    
                    # Log with level-specific metrics
                    supa.log_run(metrics=combined_metrics, metadata=metadata, level_metrics=res.get("metrics_by_level", {}))
                    st.success(f"Logged breadth test with run_id {run_id}")
                except Exception as exc:
                    logger.exception("Logging breadth test failed: %s", exc)
                    st.error(f"Logging breadth test failed: {exc}")
            
            st.write("Diagnostics:")
            st.write(res.get("diagnostics", []))
        except Exception as exc:
            logger.exception("Breadth backtest failed: %s", exc)
            st.error(f"Breadth backtest failed: {exc}")

# Grid sweep handler
if run_sweep_btn:
    st.info("Running grid sweep (RR x SL x MPT)...")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    bars["rvol"] = compute_rvol(bars, 20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates for sweep.")
    else:
        rr_vals = [lvl1_rr_min, lvl1_rr_max, lvl2_rr_min, lvl2_rr_max, lvl3_rr_min, lvl3_rr_max]
        rr_vals = sorted(list(set([float(round(x, 2)) for x in rr_vals if x is not None])))
        sl_ranges = [(float(lvl1_sl_min), float(lvl1_sl_max)), (float(lvl2_sl_min), float(lvl2_sl_max)), (float(lvl3_sl_min), float(lvl3_sl_max))]
        mpt_list = [float(p_fast), float(p_slow), float(p_deep)]
        try:
            sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list, feature_cols=feat_cols_default, model_train_kwargs={"max_bars":60})
            # summarize best runs
            summary_rows = []
            for k, v in sweep_results.items():
                if isinstance(v, dict) and "overlay" in v and not v["overlay"].empty:
                    # Add enhanced metrics to the summary
                    summary_rows.append({
                        "config": k, 
                        "trades_count": v.get("trades_count", 0),
                        "win_rate": v.get("win_rate", 0.0),
                        "rr_mean": v.get("rr_mean", 0.0),
                        "rr_80th": v.get("rr_80th", 0.0),
                        "total_pnl": v.get("total_pnl", 0.0)
                    })
            if summary_rows:
                sdf = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
                st.subheader("Sweep summary (top configs)")
                
                # Create more informative display with columns for key metrics
                st.dataframe(sdf.head(20))
                
                # Visualization of top configs
                if len(sdf) > 5:
                    st.subheader("Top 5 Configurations Comparison")
                    top5 = sdf.head(5)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Win rate vs Total PnL
                    ax1.scatter(top5["win_rate"], top5["total_pnl"], s=100)
                    for i, row in top5.iterrows():
                        ax1.annotate(row["config"], (row["win_rate"], row["total_pnl"]))
                    ax1.set_xlabel("Win Rate")
                    ax1.set_ylabel("Total PnL")
                    ax1.set_title("Win Rate vs Total PnL")
                    
                    # RR Mean vs Trade Count
                    ax2.scatter(top5["rr_mean"], top5["trades_count"], s=100)
                    for i, row in top5.iterrows():
                        ax2.annotate(row["config"], (row["rr_mean"], row["trades_count"]))
                    ax2.set_xlabel("RR Mean")
                    ax2.set_ylabel("Trade Count")
                    ax2.set_title("RR Mean vs Trade Count")
                    
                    st.pyplot(fig)
                
                st.download_button("Download sweep summary CSV", df_to_csv_bytes(sdf), "sweep_summary.csv", "text/csv")
            else:
                st.warning("Sweep returned no runs with trades.")
        except Exception as exc:
            logger.exception("Sweep failed: %s", exc)
            st.error(f"Sweep failed: {exc}")

# Help text
st.sidebar.markdown("### Notes\n- Level thresholds now produce explicit non-overlapping buy/sell windows per level.\n- L3 is the narrowest/highest-confidence scope; L1 is the widest/scope level.\n- Use the Breadth run to simulate trades per level and ensure exclusivity.")
