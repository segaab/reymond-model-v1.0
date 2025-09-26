# app.py — Entry-Range Triangulation (integrated single-file)
# Chunk 1/8: imports + Streamlit UI + level configs + data fetcher + wrapper classes

import os
import io
import math
import time
import uuid
import json
import joblib
import logging
import traceback
import pickle
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

# Streamlit page & UI inputs
st.set_page_config(page_title="Entry-Range Triangulation", layout="wide")
st.title("Entry-Range Triangulation Dashboard — Multi-level (3) Models")

# Basic symbol/time inputs
symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=90))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

# HealthGauge gating
st.sidebar.header("HealthGauge")
buy_threshold_global = st.sidebar.number_input("Global buy threshold", 0.0, 10.0, 5.5)
sell_threshold_global = st.sidebar.number_input("Global sell threshold", 0.0, 10.0, 4.5)
force_run = st.sidebar.checkbox("Force run even if gating fails", value=False)

# XGBoost / training
st.sidebar.header("Training")
num_boost = int(st.sidebar.number_input("XGBoost rounds", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))

# Thresholds for confirm stage
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

# Top-level wrapper classes (module-level so they are picklable / exportable)
class L2Wrapper:
    """Wrapper for L2 model export (top-level so pickle works)."""
    def __init__(self):
        self.booster = None  # if XGBoost
        self.model = None    # if torch

    def feature_importance(self):
        try:
            if self.booster is not None:
                imp = self.booster.get_score(importance_type="gain")
                return pd.DataFrame([(k, float(imp.get(k, 0.0))) for k in sorted(imp.keys())],
                                    columns=["feature", "gain"]).sort_values("gain", ascending=False)
            elif self.model is not None and hasattr(self.model, "feature_importance"):
                return self.model.feature_importance()
        except Exception:
            pass
        return pd.DataFrame([{"feature": "none", "gain": 0.0}])

class L3Wrapper:
    """Wrapper for L3 model export (top-level so pickle works)."""
    def __init__(self):
        self.model = None

    def feature_importance(self):
        # L3 is usually an MLP — we provide a placeholder view
        return pd.DataFrame([{"feature": "l3_emb", "gain": 1.0}])

# Chunk 2/8: features + rvol + health gauge + helpers

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

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if df is None or df.empty or "volume" not in df.columns:
        return pd.Series(1.0, index=df.index if df is not None else [])
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
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# Chunk 3/8: candidate generation + ensure rvol mapping into candidates

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def generate_candidates_and_labels(bars: pd.DataFrame,
                                   lookback: int = 64,
                                   k_tp: float = 3.0,
                                   k_sl: float = 1.0,
                                   atr_window: int = 14,
                                   max_bars: int = 60,
                                   direction: str = "long") -> pd.DataFrame:
    """
    Generate ATR-based SL/TP candidates. Always returns a DataFrame with candidate_time etc.
    The caller must ensure rvol is present on 'bars' if rvol is desired as feature.
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

    recs = []
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

    candidates_df = pd.DataFrame(recs)
    # IMPORTANT: do not assume rvol exists in candidates — map it from bars if available
    return candidates_df

def attach_rvol_to_candidates(cands: pd.DataFrame, bars: pd.DataFrame, rvol_lookback: int = 20) -> pd.DataFrame:
    """
    Ensure candidates have an 'rvol' column by mapping bars['rvol'] at candidate times.
    If bars doesn't have rvol, compute and attach it first.
    """
    if cands is None or cands.empty:
        return cands
    if bars is None or bars.empty:
        cands["rvol"] = 1.0
        return cands

    bars = bars.copy()
    if "rvol" not in bars.columns:
        bars["rvol"] = compute_rvol(bars, lookback=rvol_lookback)

    # Map rvol by candidate_time to candidate rows (ffill for missing days)
    # Normalize timestamps to exact index values if needed
    try:
        rvol_series = bars["rvol"].reindex(bars.index)
        # Use forward-fill for missing and fill remaining with 1.0
        rvol_series = rvol_series.ffill().fillna(1.0)
        cand_times = pd.to_datetime(cands["candidate_time"])
        mapped = rvol_series.reindex(cand_times).ffill().fillna(1.0).values
        cands = cands.copy()
        cands["rvol"] = mapped
    except Exception as exc:
        logger.exception("attach_rvol_to_candidates failed: %s", exc)
        cands["rvol"] = 1.0
    return cands

# Chunk 4/8: backtest/simulate + summarization

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
            "pnl": float(pnl)
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
    start_time = trades["entry_time"].min()
    end_time = trades["exit_time"].max()
    return pd.DataFrame([{
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "start_time": start_time,
        "end_time": end_time
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

# Chunk 5/8: model wrapper, train_xgb_confirm, predict_confirm_prob, export_model_and_metadata

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
        self.booster.save_model(path)

    def feature_importance(self) -> pd.DataFrame:
        imp = self.booster.get_score(importance_type="gain")
        return (pd.DataFrame([(f, imp.get(f, 0.0)) for f in self.feature_names], columns=["feature", "gain"])
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

# Use the robust export_model_and_metadata function the user supplied (adapted slightly)
def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str,Any], model_basename: str, save_fi: bool = True):
    """
    Save model artifacts and a portable .pt bundle for each model_wrapper.
    Returns dict of saved paths.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    paths = {}
    try:
        # native xgb model file if booster exists
        if hasattr(model_wrapper, "booster") and getattr(model_wrapper, "booster") is not None:
            model_file = f"{model_basename}_{ts}.model"
            try:
                logger.info(f"Exporting native model file to {model_file}")
                model_wrapper.save_model(model_file)
                paths["model"] = model_file
                logger.info(f"Successfully saved native model file to {model_file}")
            except Exception as e:
                error_msg = f"Failed to save native booster file: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                paths["model_error"] = error_msg

        # metadata
        meta_file = f"{model_basename}_{ts}.json"
        try:
            logger.info(f"Exporting metadata to {meta_file}")
            with open(meta_file, "w") as f:
                json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, default=str, indent=2)
            paths["meta"] = meta_file
            logger.info(f"Successfully saved metadata to {meta_file}")
        except Exception as e:
            error_msg = f"Failed to save metadata file: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            paths["meta_error"] = error_msg

        # feature importance
        if save_fi and hasattr(model_wrapper, "feature_importance"):
            try:
                logger.info("Generating feature importance data")
                fi_df = model_wrapper.feature_importance()
                fi_path = f"{model_basename}_{ts}_fi.csv"
                fi_df.to_csv(fi_path, index=False)
                paths["feature_importance"] = fi_path
                logger.info(f"Successfully saved feature importance to {fi_path}")
            except Exception as e:
                error_msg = f"Could not save feature importance: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                paths["fi_error"] = error_msg

        # Save portable .pt as a bundle (joblib or torch.save)
        pt_path = f"{model_basename}_{ts}.pt"
        try:
            logger.info(f"Preparing to save .pt bundle to {pt_path}")
            payload = {"model_wrapper": model_wrapper, "features": feature_list, "metrics": metrics, "export_log": []}

            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Starting export of .pt bundle")

            if 'torch' in globals() and torch is not None:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Using torch.save for .pt export")
                try:
                    torch.save(payload, pt_path)
                except Exception:
                    # sometimes torch.save can't handle non-torch items; fallback to joblib
                    joblib.dump(payload, pt_path)
            else:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] torch not available, using joblib for .pt export")
                joblib.dump(payload, pt_path)

            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Successfully saved .pt bundle")
            paths["pt"] = pt_path
            logger.info(f"Successfully saved .pt bundle to {pt_path}")
        except Exception as e:
            error_msg = f"Failed to save .pt payload: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            paths["pt_error"] = error_msg

            # Try fallback with more detailed error logging
            fallback = f"{model_basename}_{ts}_fallback.pt"
            try:
                logger.info(f"Attempting joblib fallback to {fallback}")
                fallback_payload = {
                    "model_wrapper": model_wrapper,
                    "features": feature_list,
                    "metrics": metrics,
                    "export_error": error_msg,
                    "export_log": [
                        f"[{datetime.utcnow().isoformat()}] Primary .pt export failed",
                        f"[{datetime.utcnow().isoformat()}] Error: {str(e)}",
                        f"[{datetime.utcnow().isoformat()}] Attempting fallback with joblib"
                    ]
                }
                joblib.dump(fallback_payload, fallback)
                paths["pt_fallback"] = fallback
                logger.info(f"Successfully saved fallback .pt bundle to {fallback}")
            except Exception as fallback_e:
                fallback_error = f"Fallback also failed: {str(fallback_e)}\n{traceback.format_exc()}"
                logger.error(fallback_error)
                paths["pt_fallback_error"] = fallback_error
    except Exception as exc:
        error_msg = f"Failed to export model: {str(exc)}\n{traceback.format_exc()}"
        logger.exception(error_msg)
        paths["export_error"] = error_msg

    # Final validation of PT file
    if "pt" in paths:
        try:
            logger.info(f"Verifying .pt bundle at {paths['pt']}")
            pt_file_size = os.path.getsize(paths["pt"])
            if pt_file_size > 0:
                logger.info(f"Verified .pt bundle exists with size {pt_file_size} bytes")
                paths["pt_verification"] = f"File size: {pt_file_size} bytes"
            else:
                logger.warning(f".pt bundle has zero size")
                paths["pt_verification"] = "Warning: Zero file size"
        except Exception as e:
            verification_error = f"PT verification failed: {str(e)}"
            logger.error(verification_error)
            paths["pt_verification_error"] = verification_error

    return paths

# Chunk 6/8: breadth backtest (3-level exclusive scopes) + grid sweep

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    """
    Runs exclusive-level breadth backtest.
    Each level is exclusive: a candidate belongs only to the highest level whose thresholds it matches.
    """
    out = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out

    # Build signal if absent (scale pred_prob -> 0-10)
    df_base = clean.copy()
    if "signal" not in df_base.columns:
        if "pred_prob" in df_base.columns:
            df_base["signal"] = (df_base["pred_prob"] * 10).round().astype(int)
        else:
            df_base["signal"] = 0

    # Determine exclusive membership by descending level (3->1)
    # We'll annotate 'level' column: 3,2,1,0 (0 means none)
    df = df_base.copy()
    df["level"] = 0

    # Process levels in descending restrictiveness so higher level takes precedence
    level_order = sorted(levels_config.items(), key=lambda x: int(x[0].lstrip("L")), reverse=True)
    for lvl_name, cfg in level_order:
        buy_th = cfg.get("buy_th")
        sell_th = cfg.get("sell_th")
        mask_candidate = (df["level"] == 0) & (df["signal"] > buy_th)
        df.loc[mask_candidate, "level"] = int(lvl_name.lstrip("L"))

    # For each level simulate trades only for candidates assigned to that level
    for lvl_name, cfg in sorted(levels_config.items(), key=lambda x: int(x[0].lstrip("L"))):
        try:
            level_id = int(lvl_name.lstrip("L"))
            level_df = df[df["level"] == level_id].copy()
            if level_df.empty:
                out["detailed_trades"][lvl_name] = pd.DataFrame()
                out["diagnostics"].append(f"{lvl_name}: no candidates assigned.")
                continue

            sl_pct = float((cfg.get("sl_min") + cfg.get("sl_max")) / 2.0)
            rr = float((cfg.get("rr_min") + cfg.get("rr_max")) / 2.0)
            tp_pct = rr * sl_pct

            trades = simulate_limits(level_df, bars, label_col="pred_label" if "pred_label" in level_df.columns else "label",
                                     sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
            out["detailed_trades"][lvl_name] = trades
            summary = summarize_trades(trades)
            if not summary.empty:
                srow = summary.iloc[0].to_dict()
                srow["mode"] = lvl_name
                out["summary"].append(srow)
            out["diagnostics"].append(f"{lvl_name}: simulated {len(trades)} trades, sl={sl_pct:.4f}, tp={tp_pct:.4f}")
        except Exception as exc:
            logger.exception("Breadth level %s failed: %s", lvl_name, exc)
            out["diagnostics"].append(f"{lvl_name} error: {exc}")
    out["summary"] = out["summary"] or []
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
                    results[key] = {"trades_count": len(trades), "overlay": trades}
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results

# Chunk 7/8: supabase logger (light wrapper), helpers, pick_top_runs_by_metrics

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

    def log_run(self, metrics: Dict, metadata: Dict, trades: Optional[List[Dict]] = None) -> str:
        run_id = metadata.get("run_id") or str(uuid.uuid4())
        metadata["run_id"] = run_id
        payload = {**metadata, "metrics": metrics}
        resp = self.client.table(self.runs_tbl).insert(payload).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to insert run: {resp.error}")
        if trades:
            for t in trades:
                t["run_id"] = run_id
            tr_resp = self.client.table(self.trades_tbl).insert(trades).execute()
            if getattr(tr_resp, "error", None):
                raise RuntimeError(f"Failed to insert trades: {tr_resp.error}")
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

# Chunk 8/8: main pipeline + UI actions + breadth/sweep handlers (entrypoint)

feat_cols_default = ["atr", "rvol", "duration", "hg"]

def run_main_pipeline_and_export():
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

    # attach rvol to candidates (fix for missing rvol)
    cands = attach_rvol_to_candidates(cands, bars, rvol_lookback=20)

    # add health-gauge to candidates
    if not health.empty:
        hg_map = health["health_gauge"].reindex(pd.to_datetime(health.index)).ffill().to_dict()
        cands["hg"] = cands["candidate_time"].dt.normalize().map(lambda t: hg_map.get(pd.Timestamp(t).normalize(), 0.0))
    else:
        cands["hg"] = 0.0

    feat_cols = feat_cols_default.copy()
    for fc in feat_cols_default:
        if fc not in cands.columns:
            # ensure the feature exists (default 0/1 where appropriate)
            if fc == "rvol":
                cands["rvol"] = cands.get("rvol", 1.0)
            elif fc == "hg":
                cands["hg"] = cands.get("hg", 0.0)
            else:
                cands[fc] = cands.get(fc, 0.0)

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

    # Export models for levels: L1 not present here; export L2/L3 wrappers as discussed.
    out_dir = Path("./saved_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # create L2 / L3 wrappers top-level (so pickling works)
    l2_wrapper = L2Wrapper()
    if xgb is not None and hasattr(model_wrap, "booster"):
        # embed booster into wrapper for export
        try:
            l2_wrapper.booster = model_wrap.booster  # xgb Booster
        except Exception:
            l2_wrapper.model = model_wrap
    else:
        l2_wrapper.model = model_wrap

    # export L2 wrapper + metadata
    model_basename_l2 = str(out_dir / f"l2_confirm_{symbol.replace('=','_')}")
    paths_l2 = export_model_and_metadata(l2_wrapper, feat_cols, metrics, model_basename_l2, save_fi=True)
    st.write("L2 export paths:", paths_l2)

    # create a simple L3 wrapper stub (if you have L3 executor model, attach similarly)
    l3_wrapper = L3Wrapper()
    # We don't have a separate L3 trained here; if you later train L3 attach .model
    paths_l3 = export_model_and_metadata(l3_wrapper, feat_cols, metrics, str(out_dir / f"l3_stub_{symbol.replace('=','_')}"), save_fi=True)
    st.write("L3 export paths (stub):", paths_l3)

    # Allow download of .pt if exists
    if "pt" in paths_l2 and os.path.exists(paths_l2["pt"]):
        with open(paths_l2["pt"], "rb") as f:
            st.download_button("Download L2 .pt bundle", f, file_name=os.path.basename(paths_l2["pt"]))
    if "pt" in paths_l3 and os.path.exists(paths_l3["pt"]):
        with open(paths_l3["pt"], "rb") as f:
            st.download_button("Download L3 .pt bundle", f, file_name=os.path.basename(paths_l3["pt"]))

# UI button to run pipeline + export
if st.button("Run pipeline & export models"):
    try:
        run_main_pipeline_and_export()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        st.error(f"Pipeline failed: {exc}")

# Breadth handler (calls functions above)
if run_breadth:
    st.info("Running breadth backtest across 3 levels...")
    levels = {
        "L1": {"buy_th": float(lvl1_buy), "sell_th": float(lvl1_sell),
               "rr_min": float(lvl1_rr_min), "rr_max": float(lvl1_rr_max),
               "sl_min": float(lvl1_sl_min), "sl_max": float(lvl1_sl_max)},
        "L2": {"buy_th": float(lvl2_buy), "sell_th": float(lvl2_sell),
               "rr_min": float(lvl2_rr_min), "rr_max": float(lvl2_rr_max),
               "sl_min": float(lvl2_sl_min), "sl_max": float(lvl2_sl_max)},
        "L3": {"buy_th": float(lvl3_buy), "sell_th": float(lvl3_sell),
               "rr_min": float(lvl3_rr_min), "rr_max": float(lvl3_rr_max),
               "sl_min": float(lvl3_sl_min), "sl_max": float(lvl3_sl_max)}
    }
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    if bars is None or bars.empty:
        st.error("No bars for breadth run.")
    else:
        bars["rvol"] = compute_rvol(bars, 20)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        cands = attach_rvol_to_candidates(cands, bars, 20)
        if cands is None or cands.empty:
            st.error("No candidates available for breadth run.")
        else:
            cands["pred_prob"] = cands.get("pred_prob", 0.0)
            try:
                res = run_breadth_backtest(clean=cands, bars=bars, levels_config=levels, feature_cols=feat_cols_default, model_train_kwargs={"max_bars": 60})
                summary_df = pd.DataFrame(res.get("summary", []))
                st.subheader("Breadth Summary")
                if not summary_df.empty:
                    st.dataframe(summary_df)
                else:
                    st.warning("Breadth run returned no summary rows.")
                detailed = res.get("detailed_trades", {})
                for lvl, df in detailed.items():
                    st.subheader(f"{lvl} — {len(df)} trades")
                    st.dataframe(df.head(50))
            except Exception as exc:
                logger.exception("Breadth backtest failed: %s", exc)
                st.error(f"Breadth backtest failed: {exc}")

# Grid sweep handler
if run_sweep_btn:
    st.info("Running grid sweep (RR x SL x MPT)...")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    if bars is None or bars.empty:
        st.error("No bars for sweep.")
    else:
        bars["rvol"] = compute_rvol(bars, 20)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        cands = attach_rvol_to_candidates(cands, bars, 20)
        if cands is None or cands.empty:
            st.error("No candidates for sweep.")
        else:
            rr_vals = sorted(list(set([float(round(x, 2)) for x in [lvl1_rr_min, lvl1_rr_max, lvl2_rr_min, lvl2_rr_max, lvl3_rr_min, lvl3_rr_max] if x is not None])))
            sl_ranges = [(float(lvl1_sl_min), float(lvl1_sl_max)), (float(lvl2_sl_min), float(lvl2_sl_max)), (float(lvl3_sl_min), float(lvl3_sl_max))]
            mpt_list = [float(p_fast), float(p_slow), float(p_deep)]
            try:
                sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list, feature_cols=feat_cols_default, model_train_kwargs={"max_bars":60})
                summary_rows = []
                for k, v in sweep_results.items():
                    if isinstance(v, dict) and "overlay" in v and not v["overlay"].empty:
                        s = summarize_trades(v["overlay"])
                        if not s.empty:
                            r = s.iloc[0].to_dict()
                            r.update({"config": k, "trades_count": v.get("trades_count", 0)})
                            summary_rows.append(r)
                if summary_rows:
                    sdf = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
                    st.subheader("Sweep summary (top configs)")
                    st.dataframe(sdf.head(20))
                    st.download_button("Download sweep summary CSV", df_to_csv_bytes(sdf), "sweep_summary.csv", "text/csv")
                else:
                    st.warning("Sweep returned no runs with trades.")
            except Exception as exc:
                logger.exception("Sweep failed: %s", exc)
                st.error(f"Sweep failed: {exc}")

# Help text
st.sidebar.markdown("### Notes\n- The pipeline fetches price, computes features, generates candidates, trains a confirm model, simulates trades, and exports .pt bundles.\n- This script ensures rvol is attached to candidates (fixing the previous error).")