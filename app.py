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

st.sidebar.header("Level 2 (Aim)")
lvl2_buy = st.sidebar.number_input("L2 buy threshold (signal)", 0.0, 10.0, 6.0, step=0.1)
lvl2_sell = st.sidebar.number_input("L2 sell threshold (signal)", 0.0, 10.0, 4.0, step=0.1)
lvl2_rr_min = st.sidebar.number_input("L2 RR min", 0.1, 10.0, 2.0, step=0.1)
lvl2_rr_max = st.sidebar.number_input("L2 RR max", 0.1, 10.0, 3.5, step=0.1)
lvl2_sl_min = st.sidebar.number_input("L2 SL min (pct)", 0.001, 0.5, 0.01, step=0.001)
lvl2_sl_max = st.sidebar.number_input("L2 SL max (pct)", 0.001, 0.5, 0.03, step=0.001)

st.sidebar.header("Level 3 (Shoot)")
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

# For full pipeline
run_full_pipeline_btn = st.sidebar.button("Run full pipeline (fetch→train→export)")

# Chunk 2/7: features + labeling

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
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
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

# Labeling: generate candidates & triple-barrier-like labeling
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
    Returns DataFrame of candidates with candidate_time, entry_price, atr, sl_price, tp_price, end_time, label, duration, realized_return
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
        sl_px = entry_px - k_sl * atr if direction == "long" else entry_px + k_sl * atr
        # pipeline is long-oriented for candidates
        tp_px = entry_px + k_tp * atr
        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_px = float(bars["close"].iat[end_idx])
        for j in range(i + 1, end_idx + 1):
            hi = float(bars["high"].iat[j]); lo = float(bars["low"].iat[j])
            if hi >= tp_px:
                label, hit_idx, hit_px = 1, j, tp_px
                break
            if lo <= sl_px:
                label, hit_idx, hit_px = 0, j, sl_px
                break
        end_t = bars.index[hit_idx]
        realized_return = (hit_px - entry_px) / entry_px
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
            "direction": "long"
        })
    return pd.DataFrame(recs)


# Chunk 3/7: backtest + summarization

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
            # find the last bar <= entry_t
            locs = bars.index[bars.index <= entry_t]
            if len(locs) == 0:
                continue
            entry_t = locs[-1]
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
        raw = np.clip(raw, 0.0, 1.0)
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

# Use the export_model_and_metadata function the user provided (improved .pt export + logging)
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

            # Add initial log entry
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Starting export of .pt bundle")

            # Check torch availability
            if 'torch' in globals() and torch is not None:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Using torch.save for .pt export")
                torch.save(payload, pt_path)
            else:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] torch not available, using joblib for .pt export")
                joblib.dump(payload, pt_path)

            # Add success log entry
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
            # Check if file exists and has size > 0
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

# Chunk 5/7: breadth_backtest (multi-level) + sweep/grid search + helpers

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    """
    Per-level breadth backtest. Returns structured dict with summary + detailed trades per level.
    """
    out = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out
    for lvl_name, cfg in levels_config.items():
        try:
            buy_th = cfg.get("buy_th")
            sell_th = cfg.get("sell_th")
            rr_min = cfg.get("rr_min")
            rr_max = cfg.get("rr_max")
            sl_min = cfg.get("sl_min")
            sl_max = cfg.get("sl_max")

            df = clean.copy()
            if "signal" not in df.columns:
                if "pred_prob" in df.columns:
                    df["signal"] = (df["pred_prob"] * 10).round().astype(int)
                else:
                    df["signal"] = 0

            df["pred_label"] = 0
            df.loc[df["signal"] > buy_th, "pred_label"] = 1
            df.loc[df["signal"] < sell_th, "pred_label"] = -1

            sl_pct = float((sl_min + sl_max) / 2.0)
            rr = float((rr_min + rr_max) / 2.0)
            tp_pct = rr * sl_pct

            trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
            out["detailed_trades"][lvl_name] = trades
            summary = summarize_trades(trades)
            if not summary.empty:
                summary["mode"] = lvl_name
                out["summary"].append(summary.iloc[0].to_dict())
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
                    results[key] = {"trades_count": len(trades), "overlay": trades}
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results

# Helper to adapt the trader preds -> breadth levels (light wrapper)
def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame):
    """
    preds: DataFrame returned by trader.predict_batch with columns [t,p1,p2,p3,go2,go3,trade,size]
    cands: original candidates (includes candidate_time)
    bars: OHLC bars
    This function maps preds -> candidate rows and runs simulate_limits per level with configured UI thresholds.
    """
    levels_cfg = {
        "L1": {"buy_th": float(lvl1_buy), "sell_th": float(lvl1_sell), "rr_min": float(lvl1_rr_min), "rr_max": float(lvl1_rr_max), "sl_min": float(lvl1_sl_min), "sl_max": float(lvl1_sl_max)},
        "L2": {"buy_th": float(lvl2_buy), "sell_th": float(lvl2_sell), "rr_min": float(lvl2_rr_min), "rr_max": float(lvl2_rr_max), "sl_min": float(lvl2_sl_min), "sl_max": float(lvl2_sl_max)},
        "L3": {"buy_th": float(lvl3_buy), "sell_th": float(lvl3_sell), "rr_min": float(lvl3_rr_min), "rr_max": float(lvl3_rr_max), "sl_min": float(lvl3_sl_min), "sl_max": float(lvl3_sl_max)}
    }
    # Map preds to cands: preds.t are integer indices -> map to candidate_time
    # Build a clean DataFrame merging preds p-values into candidates
    cand_copy = cands.copy().reset_index(drop=True)
    cand_copy["pred_prob"] = 0.0
    if "t" in preds.columns:
        # build mapping index -> position and copy probabilities
        # assume preds.t corresponds to indices into bars used to create candidates
        # assign p3 (most selective) but keep pred_prob as p3 when present
        pred_map = {int(r.t): r for _, r in preds.iterrows()}
        # We do best-effort: for each candidate_time find its integer index in bars and then fetch pred row
        # The calling context should ensure this mapping is correct (run_full_pipeline does)
        # For simplicity, if preds has 'p3' use it, else 'p2' or 'p1'
        for idx, row in cand_copy.iterrows():
            # candidate_time -> find integer time index in preds by comparing timestamps (caller aligns)
            # if preds include 'candidate_time' this is straightforward; else we leave default
            pass
    # Fallback: if preds include 'p3' and length matches cands, copy array
    if "p3" in preds.columns and len(preds) == len(cand_copy):
        cand_copy["pred_prob"] = preds["p3"].values
    elif "p2" in preds.columns and len(preds) == len(cand_copy):
        cand_copy["pred_prob"] = preds["p2"].values
    elif "p1" in preds.columns and len(preds) == len(cand_copy):
        cand_copy["pred_prob"] = preds["p1"].values
    # Run breadth backtest
    out = run_breadth_backtest(clean=cand_copy, bars=bars,
                               levels_config=levels_cfg,
                               feature_cols=["atr","rvol","duration","hg"],
                               model_train_kwargs={"max_bars": 60})
    # convert summary to structured output
    return {"summary": out.get("summary", []), "detailed": out.get("detailed_trades", {}), "diagnostics": out.get("diagnostics", [])}

# Chunk 6/7: CascadeTrader class (condensed but consistent with previous cascade_trader.py design)
# Note: this is a compacted integration of the cascade code you provided earlier.

if torch is not None:
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = nn.Parameter(torch.zeros(1))

        def forward(self, logits):
            T = torch.exp(self.log_temp)
            return logits / T

        def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=500, lr=1e-2):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
            y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=device)
            opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
            bce = nn.BCEWithLogitsLoss()
            def closure():
                opt.zero_grad()
                scaled = self.forward(logits_t)
                loss = bce(scaled, y_t)
                loss.backward()
                return loss
            opt.step(closure)

        def transform(self, logits: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                device = next(self.parameters()).device
                logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
                scaled = self.forward(logits_t).cpu().numpy()
            return scaled

    # minimal ConvBlock and models (same as cascade_trader)
    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k, d, pdrop):
            super().__init__()
            pad = (k - 1) * d // 2
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
            self.bn = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(pdrop)
            self.res = (c_in == c_out)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
            out = self.drop(out)
            if self.res:
                out = out + x
            return out

    class Level1ScopeCNN(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            chs = [cfg.in_features] + list(cfg.channels)
            blocks = []
            # ensure channels length is at least 3 for original block logic
            for i in range(len(cfg.channels)):
                c_in = chs[i]
                c_out = chs[i+1]
                k = cfg.kernel_sizes[i] if i < len(cfg.kernel_sizes) else 3
                d = cfg.dilations[i] if i < len(cfg.dilations) else 1
                blocks.append(ConvBlock(c_in, c_out, k, d, cfg.dropout))
            self.blocks = nn.Sequential(*blocks)
            self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)

        def forward(self, x):
            z = self.blocks(x)
            z = self.project(z)
            z_pool = z.mean(dim=-1)
            logit = self.head(z_pool)
            return logit, z_pool

    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: List[int], out_dim: int = 1, dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
                last = h
            layers += [nn.Linear(last, out_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Level3ShootMLP(nn.Module):
        def __init__(self, in_dim: int, cfg):
            super().__init__()
            self.backbone = MLP(in_dim, cfg.hidden, out_dim=128, dropout=cfg.dropout)
            self.cls_head = nn.Linear(128, 1)
            self.reg_head = nn.Linear(128, 1) if getattr(cfg, "use_regression_head", False) else None

        def forward(self, x):
            h = self.backbone(x)
            logit = self.cls_head(h)
            ret = self.reg_head(h) if self.reg_head is not None else None
            return logit, ret

    # Minimal sequence + tab datasets and training utility (kept compact)
    from torch.utils.data import Dataset, DataLoader

    class SequenceDataset(Dataset):
        def __init__(self, X_seq: np.ndarray, y: np.ndarray):
            self.X = X_seq.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].transpose(1, 0)
            y = self.y[idx]
            return x, y

    class TabDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    # simplified training helpers (reuse previous training functions where appropriate)
    def _device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def train_torch_classifier(model: nn.Module, train_ds: Dataset, val_ds: Dataset,
                               lr: float, epochs: int, patience: int, pos_weight: float = 1.0,
                               device: str = "auto"):
        dev = _device(device)
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
        train_loader = DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 256), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        history = dict(train=[], val=[])
        for epoch in range(epochs):
            model.train()
            loss_sum, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                out = model(xb)
                if isinstance(out, tuple):
                    logit = out[0]
                else:
                    logit = out
                loss = bce(logit, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss.item()) * len(xb)
                n += len(xb)
            train_loss = loss_sum / max(n, 1)
            model.eval()
            vloss_sum, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    out = model(xb)
                    if isinstance(out, tuple):
                        logit = out[0]
                    else:
                        logit = out
                    loss = bce(logit, yb)
                    vloss_sum += float(loss.item()) * len(xb)
                    vn += len(xb)
            val_loss = vloss_sum / max(vn, 1)
            history['train'].append(train_loss); history['val'].append(val_loss)
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, {"best_val_loss": best_loss, "history": history}

    # CascadeTrader: simplified wrapper for fitting pipeline (keeps same API used by run_full_pipeline)
    class CascadeTrader:
        def __init__(self, l1_cfg=None, l2_cfg=None, l3_cfg=None, gate_cfg=None, fit_cfg=None):
            # default configs minimal
            from types import SimpleNamespace
            self.l1_cfg = l1_cfg or SimpleNamespace(seq_len=64, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1, lr=1e-3, batch_size=256, epochs=8, early_stop_patience=3, class_weight_pos=1.0)
            self.l2_cfg = l2_cfg or SimpleNamespace(backend="xgb", xgb_params=None, mlp_hidden=(128,64), mlp_dropout=0.1, lr=1e-3, batch_size=512, epochs=6, early_stop_patience=3)
            self.l3_cfg = l3_cfg or SimpleNamespace(hidden=(128,64), dropout=0.1, lr=1e-3, batch_size=512, epochs=8, early_stop_patience=3, use_regression_head=True)
            self.gate_cfg = gate_cfg or SimpleNamespace(th1=0.30, th2=0.55, th3=0.65)
            self.fit_cfg = fit_cfg or SimpleNamespace(device="auto", val_size=0.2, random_state=42, l1_seq_len=64, feature_windows=(5,10,20), lookahead=20, output_dir="./artifacts")
            self.device = _device(self.fit_cfg.device)
            self.l1 = Level1ScopeCNN(self.l1_cfg)
            self.l1_temp = TemperatureScaler()
            self.l2_backend = None
            self.l2_model = None
            self.l3 = None
            self.l3_temp = TemperatureScaler()
            self.scaler_seq = None
            self.scaler_tab = None
            self.tab_feature_names = []
            self.metadata = {}
            self._fitted = False

        def fit(self, df: pd.DataFrame, events: pd.DataFrame, l2_use_xgb=True, epochs_l1=None, epochs_l23=None):
            """
            Fit cascade given bars (df) and events (DataFrame with columns 't' integer indexes and 'y' labels).
            Simpler train/test split and minimal preprocessing to remain fast for demo.
            """
            # compute engineered features
            eng = compute_engineered_features(df, windows=self.fit_cfg.feature_windows) if 'compute_engineered_features' in globals() else pd.DataFrame()
            # build simple seq features: use close/open/high/low/volume and some engineered if available
            seq_cols = ['open','high','low','close','volume']
            micro_cols = [c for c in eng.columns[:6]] if not eng.empty else []
            use_cols = [c for c in seq_cols + micro_cols if c in df.columns or c in eng.columns]
            # create combined feature matrix for scalers (fallback to numeric casting)
            feat_seq = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[[c for c in use_cols if c in (list(df.columns) + list(eng.columns))]].replace([np.inf,-np.inf],0.0).fillna(0.0)
            feat_tab = eng.copy() if not eng.empty else pd.DataFrame(index=df.index)
            self.tab_feature_names = list(feat_tab.columns)
            # indices and labels
            idx = events['t'].astype(int).values
            y = events['y'].astype(int).values
            # train/val split (random for now)
            train_idx, val_idx = train_test_split(np.arange(len(idx)), test_size=self.fit_cfg.val_size, random_state=self.fit_cfg.random_state, stratify=y)
            idx_train, idx_val = idx[train_idx], idx[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # fit scalers
            X_seq_all = feat_seq.values
            from sklearn.preprocessing import StandardScaler
            self.scaler_seq = StandardScaler().fit(X_seq_all)
            X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
            X_tab_all = feat_tab.values if not feat_tab.empty else np.zeros((len(df), 0))
            self.scaler_tab = StandardScaler().fit(X_tab_all) if X_tab_all.shape[1] > 0 else None
            X_tab_all_scaled = self.scaler_tab.transform(X_tab_all) if self.scaler_tab is not None else np.zeros((len(df), 0))
            # build sequence arrays
            def to_sequences(features, indices, seq_len):
                X = []
                for t in indices:
                    t0 = t - seq_len + 1
                    if t0 < 0:
                        pad = np.repeat(features[[0]], repeats=(-t0), axis=0)
                        seq = np.concatenate([pad, features[0:t+1]], axis=0)
                    else:
                        seq = features[t0:t+1]
                    X.append(seq)
                return np.stack(X, axis=0)
            seq_len = self.l1_cfg.seq_len
            Xseq_train = to_sequences(X_seq_all_scaled, idx_train, seq_len)
            Xseq_val = to_sequences(X_seq_all_scaled, idx_val, seq_len)
            ds_l1_train = SequenceDataset(Xseq_train, y_train)
            ds_l1_val = SequenceDataset(Xseq_val, y_val)
            ds_l1_train.batch_size = self.l1_cfg.batch_size
            # train l1
            self.l1, l1_hist = train_torch_classifier(self.l1, ds_l1_train, ds_l1_val, lr=self.l1_cfg.lr, epochs=(epochs_l1 or self.l1_cfg.epochs), patience=self.l1_cfg.early_stop_patience, pos_weight=self.l1_cfg.class_weight_pos, device=self.fit_cfg.device)
            # infer L1 embeddings for train/val
            def _l1_infer_logits_emb_local(Xseq):
                self.l1.eval()
                logits, embeds = [], []
                with torch.no_grad():
                    for i in range(0, len(Xseq), 1024):
                        xb = torch.tensor(Xseq[i:i+1024].transpose(0,2,1), dtype=torch.float32, device=self.device) # [B,T,F] -> [B,F,T]
                        logit, emb = self.l1(xb)
                        logits.append(logit.detach().cpu().numpy())
                        embeds.append(emb.detach().cpu().numpy())
                logits = np.concatenate(logits, axis=0).reshape(-1,1)
                embeds = np.concatenate(embeds, axis=0)
                return logits, embeds
            l1_train_logits, l1_train_emb = _l1_infer_logits_emb_local(Xseq_train)
            l1_val_logits, l1_val_emb = _l1_infer_logits_emb_local(Xseq_val)
            self.l1_temp.fit(l1_val_logits, y_val if len(y_val.shape)>0 else np.array(y_val))
            # build L2 training matrices
            Xtab_train = X_tab_all_scaled[idx_train] if X_tab_all_scaled.size else np.zeros((len(idx_train), 0))
            Xtab_val = X_tab_all_scaled[idx_val] if X_tab_all_scaled.size else np.zeros((len(idx_val), 0))
            X_l2_train = np.hstack([l1_train_emb, Xtab_train]) if Xtab_train.size else l1_train_emb
            X_l2_val = np.hstack([l1_val_emb, Xtab_val]) if Xtab_val.size else l1_val_emb
            # Train L2
            use_xgb = (xgb is not None) and l2_use_xgb
            if use_xgb:
                self.l2_backend = "xgb"
                l2_params = dict(n_estimators=100, max_depth=4, learning_rate=0.05)  # simple fallback
                try:
                    model_x = xgb.XGBClassifier(**l2_params)
                    model_x.fit(X_l2_train, y_train, eval_set=[(X_l2_val, y_val)], verbose=False)
                    self.l2_model = model_x
                except Exception as e:
                    logger.exception("L2 xgb training failed: %s", e)
                    self.l2_backend = "mlp"
            if not use_xgb or self.l2_backend != "xgb":
                self.l2_backend = "mlp"
                in_dim = X_l2_train.shape[1]
                self.l2_model = MLP(in_dim, [128,64], out_dim=1, dropout=0.1)
                ds2_tr = TabDataset(X_l2_train, y_train)
                ds2_va = TabDataset(X_l2_val, y_val)
                self.l2_model, _ = train_torch_classifier(self.l2_model, ds2_tr, ds2_va, lr=1e-3, epochs=(epochs_l23 or 6), patience=3, device=self.fit_cfg.device)
            # Train L3
            X_l3_train = X_l2_train
            X_l3_val = X_l2_val
            self.l3 = Level3ShootMLP(X_l3_train.shape[1], self.l3_cfg)
            ds3_tr = TabDataset(X_l3_train, y_train)
            ds3_va = TabDataset(X_l3_val, y_val)
            self.l3, l3_hist = train_torch_classifier(self.l3, ds3_tr, ds3_va, lr=self.l3_cfg.lr, epochs=(epochs_l23 or 8), patience=self.l3_cfg.early_stop_patience, pos_weight=1.0, device=self.fit_cfg.device)
            l3_val_logits = self._l3_infer_logits(X_l3_val)
            self.l3_temp.fit(l3_val_logits, y_val)
            # metadata
            self.metadata = {"l1_hist": l1_hist, "l3_hist": l3_hist, "l2_backend": self.l2_backend, "tab_features": self.tab_feature_names, "fit_time_sec": 0}
            self._fitted = True
            return self

        def _l3_infer_logits(self, X):
            self.l3.eval()
            logits = []
            with torch.no_grad():
                for i in range(0, len(X), 2048):
                    xb = torch.tensor(X[i:i+2048], dtype=torch.float32, device=self.device)
                    out = self.l3(xb)
                    logit = out[0] if isinstance(out, tuple) else out
                    logits.append(logit.detach().cpu().numpy())
            return np.concatenate(logits, axis=0).reshape(-1,1)

        def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray) -> pd.DataFrame:
            assert self._fitted, "Call fit() first."
            # compute features and scaled arrays, then run through l1->l2->l3
            eng = compute_engineered_features(df, windows=self.fit_cfg.feature_windows) if 'compute_engineered_features' in globals() else pd.DataFrame()
            seq_cols = ['open','high','low','close','volume']
            micro_cols = [c for c in eng.columns[:6]] if not eng.empty else []
            use_cols = [c for c in seq_cols + micro_cols if c in (list(df.columns) + list(eng.columns))]
            feat_seq = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[[c for c in use_cols if c in (list(df.columns) + list(eng.columns))]].replace([np.inf,-np.inf],0.0).fillna(0.0)
            feat_tab = eng[self.tab_feature_names] if not eng.empty and len(self.tab_feature_names)>0 else pd.DataFrame(np.zeros((len(df),0)))
            X_seq_all_scaled = self.scaler_seq.transform(feat_seq.values)
            X_tab_all_scaled = self.scaler_tab.transform(feat_tab.values) if self.scaler_tab is not None else np.zeros((len(df),0))
            def to_sequences(features, indices, seq_len):
                X = []
                for t in indices:
                    t0 = t - seq_len + 1
                    if t0 < 0:
                        pad = np.repeat(features[[0]], repeats=(-t0), axis=0)
                        seq = np.concatenate([pad, features[0:t+1]], axis=0)
                    else:
                        seq = features[t0:t+1]
                    X.append(seq)
                return np.stack(X, axis=0)
            Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.l1_cfg.seq_len)
            # L1
            l1_logits, l1_emb = [], []
            self.l1.eval()
            with torch.no_grad():
                for i in range(0, len(Xseq), 1024):
                    xb = torch.tensor(Xseq[i:i+1024].transpose(0,2,1), dtype=torch.float32, device=self.device)
                    logit, emb = self.l1(xb)
                    l1_logits.append(logit.detach().cpu().numpy())
                    l1_emb.append(emb.detach().cpu().numpy())
            l1_logits = np.concatenate(l1_logits, axis=0).reshape(-1,1)
            l1_emb = np.concatenate(l1_emb, axis=0)
            l1_logits_scaled = self.l1_temp.transform(l1_logits)
            p1 = 1.0 / (1.0 + np.exp(-l1_logits_scaled)).reshape(-1)
            go2 = p1 >= float(getattr(self.gate_cfg, "th1", 0.3))
            # L2
            X_l2 = np.hstack([l1_emb, X_tab_all_scaled[t_indices]]) if X_tab_all_scaled.size else l1_emb
            if self.l2_backend == "xgb":
                p2 = self.l2_model.predict_proba(X_l2)[:,1]
            else:
                self.l2_model.eval()
                with torch.no_grad():
                    xb = torch.tensor(X_l2, dtype=torch.float32, device=self.device)
                    logit = self.l2_model(xb)
                    p2 = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            go3 = (p2 >= float(getattr(self.gate_cfg, "th2", 0.55))) & go2
            # L3
            p3 = np.zeros_like(p1)
            rhat = np.zeros_like(p1)
            if go3.any()


# Chunk 7/7: Streamlit actions including one-button run_full_pipeline

# session state containers
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

def fetch_and_prepare():
    st.info(f"Fetching {symbol} {interval} from YahooQuery …")
    bars = pd.DataFrame()
    if YahooTicker is None:
        st.error("yahooquery not installed -- cannot fetch price data.")
        return pd.DataFrame()
    try:
        tq = YahooTicker(symbol)
        raw = tq.history(start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        if raw is None:
            st.error("No data returned from yahooquery.")
            return pd.DataFrame()
        if isinstance(raw, dict): raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex): raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        bars = raw[~raw.index.duplicated(keep="first")]
        st.success(f"Fetched {len(bars)} bars.")
    except Exception as e:
        logger.exception("Fetch failed: %s", e)
        st.error(f"Fetch failed: {e}")
    return bars

def run_full_pipeline():
    # 1) fetch
    bars = fetch_and_prepare()
    if bars is None or bars.empty:
        return {"error": "No bars"}
    st.session_state.bars = bars
    bars = ensure_unique_index(bars)
    # 2) compute rvol and candidates
    try:
        bars["rvol"] = (bars["volume"] / bars["volume"].rolling(20, min_periods=1).mean()).fillna(1.0)
    except Exception:
        bars["rvol"] = 1.0
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.warning("No candidates generated.")
        st.session_state.cands = pd.DataFrame()
        return {"error": "No candidates"}
    # add placeholder pred_prob column for sim if needed
    cands["pred_prob"] = 0.0
    st.session_state.cands = cands
    st.success(f"Generated {len(cands)} candidates.")
    # 3) map candidate_time to integer positions for events
    bar_idx_map = {t: i for i, t in enumerate(bars.index)}
    cand_idx = []
    for t in cands["candidate_time"]:
        t0 = pd.Timestamp(t)
        if t0 in bar_idx_map:
            cand_idx.append(bar_idx_map[t0])
        else:
            locs = bars.index[bars.index <= t0]
            cand_idx.append(int(bar_idx_map[locs[-1]] if len(locs) else 0))
    events = pd.DataFrame({"t": np.array(cand_idx, dtype=int), "y": cands["label"].astype(int).values})
    # 4) train cascade
    if torch is None:
        st.error("Torch not installed -- cannot train cascade.")
        return {"error": "no_torch"}
    st.info("Training cascade (L1/L2/L3). This may take a while.")
    # instantiate trader with minimal config
    trader = CascadeTrader()
    try:
        trader.fit(bars, events, l2_use_xgb=(xgb is not None), epochs_l1=8, epochs_l23=8)
        st.success("Cascade trained.")
        st.session_state.trader = trader
    except Exception as e:
        logger.exception("Cascade training failed: %s", e)
        st.error(f"Cascade training failed: {e}")
        return {"error": "train_failed", "exception": str(e)}
    # 5) predict on candidate indices and breadth/backtest & optional sweep
    t_indices = np.array(cand_idx, dtype=int)
    preds = trader.predict_batch(bars, t_indices)
    st.write("Predictions head:")
    st.dataframe(preds.head(10))
    # breadth
    if run_breadth:
        st.info("Running breadth backtest")
        res = run_breadth_levels(preds, cands, bars)
        st.session_state.last_breadth = res
        if res["summary"]:
            st.subheader("Breadth summary")
            st.dataframe(pd.DataFrame(res["summary"]))
        else:
            st.warning("Breadth returned no summary rows.")
    # sweep
    if run_sweep_btn:
        st.info("Running grid sweep (light)")
        rr_vals = [2.0, 2.5, 3.0]  # small set for speed
        sl_ranges = [(0.02, 0.04), (0.01, 0.03), (0.005, 0.02)]
        mpt_list = [p_fast, p_slow, p_deep]
        sweep_results = run_grid_sweep(cands, bars, rr_vals, sl_ranges, mpt_list, feature_cols=["atr","rvol","duration","hg"], model_train_kwargs={"max_bars":60})
        st.session_state.last_sweep = sweep_results
        st.success("Sweep completed.")
    # 6) export artifacts & models (per-level)
    st.info("Exporting artifacts and .pt bundles")
    out_dir = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(out_dir, exist_ok=True)
    # Save L1/L3 Torch state_dicts + scalers + meta
    try:
        trader_path = os.path.join(out_dir, "cascade_trader")
        os.makedirs(trader_path, exist_ok=True)
        # L1
        torch.save(trader.l1.state_dict(), os.path.join(trader_path, "l1_state.pt"))
        torch.save(trader.l3.state_dict(), os.path.join(trader_path, "l3_state.pt"))
        # If L2 is xgb, save model; if mlp, save state_dict
        if trader.l2_backend == "xgb":
            try:
                trader.l2_model.save_model(os.path.join(trader_path, "l2_xgb.json"))
            except Exception:
                joblib.dump(trader.l2_model, os.path.join(trader_path, "l2_xgb.joblib"))
        else:
            torch.save(trader.l2_model.state_dict(), os.path.join(trader_path, "l2_mlp_state.pt"))
        # scalers and metadata
        with open(os.path.join(trader_path, "scaler_seq.pkl"), "wb") as f:
            pickle.dump(trader.scaler_seq, f)
        with open(os.path.join(trader_path, "scaler_tab.pkl"), "wb") as f:
            pickle.dump(trader.scaler_tab, f)
        with open(os.path.join(trader_path, "metadata.json"), "w") as f:
            json.dump(trader.metadata, f, default=str, indent=2)
    except Exception as e:
        logger.exception("Saving artifacts failed: %s", e)
        st.error(f"Saving artifacts failed: {e}")

    # Define proper picklable wrapper classes instead of using type()
    class L2Wrapper:
        def __init__(self, model=None, backend=None):
            self.booster = model
            self.backend = backend
        def feature_importance(self):
            return pd.DataFrame([{"feature": "dummy", "gain": 1.0}])
        def save_model(self, path):
            if hasattr(self.booster, "save_model"):
                self.booster.save_model(path)
            else:
                raise AttributeError("Model doesn't have save_model method")

    class L3Wrapper:
        def __init__(self, model=None):
            self.model = model
        def feature_importance(self):
            return pd.DataFrame([{"feature": "l3_emb", "gain": 1.0}])

    # use export_model_and_metadata to save a .pt bundle for L2 (as an example) and L3 wrapper
    export_paths = {}
    try:
        # Create proper wrapper instances with the models
        l2_wrapper = L2Wrapper(model=trader.l2_model, backend=trader.l2_backend)
        paths_l2 = export_model_and_metadata(l2_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2_model"), save_fi=True)
        export_paths["l2"] = paths_l2

        # export L3 as .pt bundle
        l3_wrapper = L3Wrapper(model=trader.l3)
        paths_l3 = export_model_and_metadata(l3_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l3_model"), save_fi=True)
        export_paths["l3"] = paths_l3
        st.success(f"Exported models to {out_dir}")
        st.write(export_paths)
        st.session_state.export_paths = export_paths
    except Exception as e:
        logger.exception("Export_model_and_metadata failed: %s", e)
        st.error(f"Export failed: {e}")
    return {"bars": bars, "cands": cands, "trader": trader, "export": export_paths}

# If user clicked the full pipeline button:
if run_full_pipeline_btn:
    with st.spinner("Running full pipeline (fetch→cands→train→export) … this may take several minutes"):
        result = run_full_pipeline()
        if result is None:
            st.error("Pipeline failed with no result.")
        elif "error" in result:
            st.error(f"Pipeline error: {result.get('error')} - {result.get('exception','')}")
        else:
            st.success("Full pipeline completed.")
            st.write("Artifacts saved:", st.session_state.export_paths)
            # show final trades if breadth run was executed
            if run_breadth and "last_breadth" in st.session_state:
                br = st.session_state.last_breadth
                st.subheader("Breadth detailed results")
                for lvl, df in br["detailed"].items():
                    st.write(lvl, len(df))
                    if not df.empty:
                        st.dataframe(df.head(20))