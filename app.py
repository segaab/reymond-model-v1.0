# app.py — Entry-Range Triangulation (integrated single-file)
# Chunk 1/7: imports + logging + Streamlit UI + redesigned level controls + fetch_price

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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

# Streamlit layout + page
st.set_page_config(page_title="Entry-Range Triangulation", layout="wide")
st.title("Entry-Range Triangulation Dashboard — Multi-level (3) Models")

# Basic symbol/time inputs
symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=90))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

# High level gating / training
st.sidebar.header("Global / Model")
buy_threshold_global = st.sidebar.number_input("Global buy threshold (not used directly)", 0.0, 10.0, 5.5)
sell_threshold_global = st.sidebar.number_input("Global sell threshold (not used directly)", 0.0, 10.0, 4.5)
force_run = st.sidebar.checkbox("Force run even if gating fails", value=False)

num_boost = int(st.sidebar.number_input("XGBoost rounds", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))

p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

# Redesigned Level controls: explicit buy_min/buy_max and sell_min/sell_max for each level
st.sidebar.header("Level Thresholds (explicit min / max)")

def level_ui_group(level_name: str, default):
    st.sidebar.subheader(level_name)
    buy_min = st.sidebar.number_input(f"{level_name} buy_min", 0.0, 10.0, default["buy_min"], step=0.1, key=f"{level_name}_buy_min")
    buy_max = st.sidebar.number_input(f"{level_name} buy_max", 0.0, 10.0, default["buy_max"], step=0.1, key=f"{level_name}_buy_max")
    sell_min = st.sidebar.number_input(f"{level_name} sell_min", 0.0, 10.0, default["sell_min"], step=0.1, key=f"{level_name}_sell_min")
    sell_max = st.sidebar.number_input(f"{level_name} sell_max", 0.0, 10.0, default["sell_max"], step=0.1, key=f"{level_name}_sell_max")
    rr_min = st.sidebar.number_input(f"{level_name} RR min", 0.1, 20.0, default["rr_min"], step=0.1, key=f"{level_name}_rr_min")
    rr_max = st.sidebar.number_input(f"{level_name} RR max", 0.1, 20.0, default["rr_max"], step=0.1, key=f"{level_name}_rr_max")
    sl_min = st.sidebar.number_input(f"{level_name} SL min (pct)", 0.000, 0.5, default["sl_min"], step=0.001, key=f"{level_name}_sl_min")
    sl_max = st.sidebar.number_input(f"{level_name} SL max (pct)", 0.000, 0.5, default["sl_max"], step=0.001, key=f"{level_name}_sl_max")
    return {
        "buy_min": float(buy_min), "buy_max": float(buy_max),
        "sell_min": float(sell_min), "sell_max": float(sell_max),
        "rr_min": float(rr_min), "rr_max": float(rr_max),
        "sl_min": float(sl_min), "sl_max": float(sl_max)
    }

# sensible defaults for L1/L2/L3
lvl1_default = {"buy_min":5.5, "buy_max":6.0, "sell_min":4.5, "sell_max":5.0, "rr_min":1.0, "rr_max":2.5, "sl_min":0.02, "sl_max":0.04}
lvl2_default = {"buy_min":6.0, "buy_max":6.5, "sell_min":4.0, "sell_max":4.9, "rr_min":2.0, "rr_max":3.5, "sl_min":0.01, "sl_max":0.03}
lvl3_default = {"buy_min":6.5, "buy_max":10.0, "sell_min":0.0, "sell_max":3.5, "rr_min":3.0, "rr_max":5.0, "sl_min":0.005, "sl_max":0.02}

st.sidebar.markdown("**Configure Level ranges** — ensure ranges do not overlap to keep exclusivity.")
L1 = level_ui_group("L1", lvl1_default)
L2 = level_ui_group("L2", lvl2_default)
L3 = level_ui_group("L3", lvl3_default)

# Breadth & sweep controls
st.sidebar.header("Breadth & Sweep")
run_breadth = st.sidebar.button("Run breadth backtest (3 levels)")
run_sweep_btn = st.sidebar.button("Run grid sweep")

# Fetch price helper (yahooquery then yfinance fallback)
def fetch_price(symbol: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """
    Robust fetcher: tries yahooquery, falls back to yfinance. Normalizes OHLC column names to lowercase.
    """
    # try yahooquery
    if YahooTicker is not None:
        try:
            tq = YahooTicker(symbol)
            raw = tq.history(start=start, end=end, interval=interval)
            if raw is not None and len(raw) > 0:
                if isinstance(raw, dict):
                    raw = pd.DataFrame(raw)
                if isinstance(raw.index, pd.MultiIndex):
                    raw = raw.reset_index(level=0, drop=True)
                raw.index = pd.to_datetime(raw.index)
                raw = raw.sort_index()
                raw.columns = [str(c).lower() for c in raw.columns]
                if "adjclose" in raw.columns and "close" not in raw.columns:
                    raw["close"] = raw["adjclose"]
                return raw[~raw.index.duplicated(keep="first")]
        except Exception:
            logger.exception("yahooquery fetch failed, will try fallback")

    # fallback to yfinance if available
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "adjclose" in df.columns and "close" not in df.columns:
            df["close"] = df["adjclose"]
        return df[~df.index.duplicated(keep="first")]
    except Exception:
        logger.exception("yfinance fallback failed")
    # return empty with consistent columns
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
# Chunk 2/7: features + robust candidate generation

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
    Tolerant candidate generator that adapts lookback for short series and logs diagnostics.
    """
    if bars is None or bars.empty:
        logger.info("generate_candidates_and_labels: bars empty")
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = ensure_unique_index(bars)

    # normalize columns
    colmap = {c.lower(): c for c in bars.columns}
    if "high" not in colmap or "low" not in colmap or "close" not in colmap:
        logger.warning("OHLC columns missing in input bars: %s", list(bars.columns))
        # try common alternatives
        for alt in ("h", "high_", "high."):
            if alt in colmap:
                colmap["high"] = colmap[alt]
        for alt in ("l", "low_", "low."):
            if alt in colmap:
                colmap["low"] = colmap[alt]
        for alt in ("adjclose", "close_", "price"):
            if alt in colmap:
                colmap["close"] = colmap[alt]
    if not set(["high", "low", "close"]).issubset(set(colmap.keys())):
        logger.warning("Unable to find required OHLC even after heuristics. cols=%s", list(bars.columns))
        return pd.DataFrame()

    df = pd.DataFrame(index=bars.index)
    df["high"] = bars[colmap["high"]].astype(float)
    df["low"] = bars[colmap["low"]].astype(float)
    df["close"] = bars[colmap["close"]].astype(float)
    df["volume"] = bars[colmap.get("volume")] if colmap.get("volume") in bars.columns else np.nan

    n = len(df)
    effective_lookback = int(min(lookback, max(5, n // 3)))
    if n <= effective_lookback + 2:
        logger.info("generate_candidates_and_labels: not enough bars for requested lookback (%d). n=%d", lookback, n)

    df["tr"] = _true_range(df["high"], df["low"], df["close"])
    df["atr"] = df["tr"].rolling(window=atr_window, min_periods=1).mean()

    recs = []
    for i in range(effective_lookback, n):
        t = df.index[i]
        entry_px = float(df["close"].iat[i])
        atr = float(df["atr"].iat[i])
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
        hit_px = float(df["close"].iat[end_idx])

        for j in range(i + 1, end_idx + 1):
            hi = float(df["high"].iat[j]); lo = float(df["low"].iat[j])
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

        end_t = df.index[hit_idx]
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
    out_df = pd.DataFrame(recs)
    logger.info("generate_candidates_and_labels: produced %d candidates (n_bars=%d, lookback=%d)", len(out_df), n, effective_lookback)
    return out_df
# Chunk 3/7: simulate_limits and summary utilities

def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str = "GC=F",
                    sl: float = 0.02,
                    tp: float = 0.04,
                    max_holding: int = 60) -> pd.DataFrame:
    """
    Light simulation using SL/TP percentages. Assumes df contains candidate_time (timestamp index).
    """
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

# Chunk 4/7: XGBoost wrapper, train/predict, export helpers

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


# Chunk 5/7: breadth_backtest (multi-level exclusive ranges) + grid sweep

def _validate_and_fix_level_ranges(levels: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Ensure exclusivity: buy ranges and sell ranges should not overlap across levels.
    If small overlaps exist we adjust boundaries slightly (with warnings).
    Returns (fixed_levels, diagnostics)
    """
    diags = []
    # Build sorted buy_min for checking
    # We'll primarily ensure buy ranges are increasing L1->L2->L3 and sell ranges decreasing accordingly if needed
    # Simple policy: enforce buy_min(L1) <= buy_max(L1) < buy_min(L2) <= buy_max(L2) < buy_min(L3)
    lvl_keys = list(levels.keys())
    fixed = {k: dict(v) for k, v in levels.items()}
    # Fix buy ranges
    prev_buy_max = -1e9
    for k in lvl_keys:
        bmin, bmax = fixed[k]["buy_min"], fixed[k]["buy_max"]
        if bmax < bmin:
            fixed[k]["buy_max"] = bmin
            diags.append(f"{k}: buy_max < buy_min, set buy_max=buy_min")
        if bmin <= prev_buy_max:
            # push bmin to prev_buy_max + small epsilon
            new_bmin = round(prev_buy_max + 0.01, 2)
            if new_bmin > bmax:
                fixed[k]["buy_max"] = new_bmin
                fixed[k]["buy_min"] = new_bmin
                diags.append(f"{k}: raised both buy_min & buy_max to {new_bmin} to enforce exclusivity")
            else:
                fixed[k]["buy_min"] = new_bmin
                diags.append(f"{k}: raised buy_min to {new_bmin} to avoid overlap")
        prev_buy_max = fixed[k]["buy_max"]
    # Fix sell ranges similarly (we allow sell ranges to be independent but require they don't overlap with others)
    prev_sell_min = 1e9
    for k in reversed(lvl_keys):
        smin, smax = fixed[k]["sell_min"], fixed[k]["sell_max"]
        if smax < smin:
            fixed[k]["sell_max"] = smin
            diags.append(f"{k}: sell_max < sell_min, set sell_max=sell_min")
        if smax >= prev_sell_min:
            new_smax = round(prev_sell_min - 0.01, 2)
            if new_smax < smin:
                fixed[k]["sell_min"] = new_smax
                fixed[k]["sell_max"] = new_smax
                diags.append(f"{k}: lowered both sell_min & sell_max to {new_smax} to enforce exclusivity")
            else:
                fixed[k]["sell_max"] = new_smax
                diags.append(f"{k}: lowered sell_max to {new_smax} to avoid overlap")
        prev_sell_min = fixed[k]["sell_min"]
    return fixed, diags

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    """
    Uses explicit level ranges: buy_min..buy_max and sell_min..sell_max.
    For each level we filter candidates whose 'signal' falls inside that level's buy range (or sell range)
    and simulate using average RR/SL from config.
    """
    out = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out

    # validate and fix ranges to ensure exclusivity
    fixed_levels, diags = _validate_and_fix_level_ranges(levels_config)
    out["diagnostics"].extend(diags)

    # Ensure signal column exists
    df_all = clean.copy()
    if "signal" not in df_all.columns:
        if "pred_prob" in df_all.columns:
            df_all["signal"] = (df_all["pred_prob"] * 10).round().astype(int)
        else:
            # fallback: scale realized_return to 0-10 for demonstration
            if "realized_return" in df_all.columns:
                scaled = (df_all["realized_return"] - df_all["realized_return"].min()) / max(1e-9, (df_all["realized_return"].max() - df_all["realized_return"].min()))
                df_all["signal"] = (scaled * 10).round().astype(int)
            else:
                df_all["signal"] = 0

    for lvl_name, cfg in fixed_levels.items():
        try:
            buy_min = cfg.get("buy_min")
            buy_max = cfg.get("buy_max")
            sell_min = cfg.get("sell_min")
            sell_max = cfg.get("sell_max")
            rr_min = cfg.get("rr_min")
            rr_max = cfg.get("rr_max")
            sl_min = cfg.get("sl_min")
            sl_max = cfg.get("sl_max")

            # Filter candidates for the level: signal within buy range OR within sell range
            df = df_all.copy()
            cond_buy = (df["signal"] >= buy_min) & (df["signal"] <= buy_max)
            cond_sell = (df["signal"] >= sell_min) & (df["signal"] <= sell_max)
            df_level = df[cond_buy | cond_sell].copy()
            if df_level.empty:
                out["detailed_trades"][lvl_name] = pd.DataFrame()
                out["diagnostics"].append(f"{lvl_name}: no candidates in specified ranges")
                continue

            # assign labels using ranges (buy -> 1, sell -> -1)
            df_level["pred_label"] = 0
            df_level.loc[cond_buy.loc[df_level.index], "pred_label"] = 1
            df_level.loc[cond_sell.loc[df_level.index], "pred_label"] = -1

            # represent sl/tp as mean values
            sl_pct = float((sl_min + sl_max) / 2.0)
            rr = float((rr_min + rr_max) / 2.0)
            tp_pct = rr * sl_pct

            trades = simulate_limits(df_level, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
            out["detailed_trades"][lvl_name] = trades
            s = summarize_trades(trades)
            if not s.empty:
                s_row = s.iloc[0].to_dict()
                s_row.update({"mode": lvl_name, "sl_pct": sl_pct, "tp_pct": tp_pct, "n_candidates": len(df_level)})
                out["summary"].append(s_row)
            out["diagnostics"].append(f"{lvl_name}: simulated {len(trades)} trades, sl={sl_pct:.4f}, tp={tp_pct:.4f}, candidates={len(df_level)}")
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
    Sweep RR x SL-range x model-prob-threshold (mpt_list).
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

# Chunk 6/7: Supabase logger (light wrapper) + helpers + pick_top_runs

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

# helpers
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def pick_top_runs_by_metrics(runs: List[Dict], top_n: int = 3):
    scored = []
    for r in runs:
        metrics = r.get("metrics") if isinstance(r, dict) else r
        total_pnl = metrics.get("total_pnl", 0.0) if metrics else 0.0
        win_rate = metrics.get("win_rate", 0.0) if metrics else 0.0
        score = float(total_pnl) + float(win_rate) * 0.01
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_n]]

# Chunk 7/7: main pipeline + UI handlers (breadth / sweep)

feat_cols_default = ["atr", "rvol", "duration", "hg"]

def run_main_pipeline():
    st.info(f"Fetching price for {symbol} …")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars is None or bars.empty:
        st.error("No price data returned.")
        return

    bars = ensure_unique_index(bars)
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

    bars["rvol"] = compute_rvol(bars, lookback=20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates generated.")
        return

    # add health gauge per candidate by mapping candidate_time -> daily health
    if not health.empty:
        hg_map = health["health_gauge"].reindex(pd.to_datetime(health.index)).ffill().to_dict()
        cands["hg"] = cands["candidate_time"].dt.normalize().map(lambda t: hg_map.get(pd.Timestamp(t).normalize(), 0.0))
    else:
        cands["hg"] = 0.0

    feat_cols = feat_cols_default.copy()
    if "hg" not in feat_cols:
        feat_cols.append("hg")

    # train confirm model
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

    cands["pred_prob"] = predict_confirm_prob(model_wrap, cands, feat_cols)
    cands["pred_label"] = (cands["pred_prob"] >= p_fast).astype(int)

    st.info("Simulating trades using predicted labels…")
    trades = simulate_limits(cands, bars, label_col="pred_label", max_holding=60)
    st.write("Simulated trades:", len(trades))
    if not trades.empty:
        st.dataframe(trades.head())
        s = summarize_trades(trades)
        st.dataframe(s)

    # save & log
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

            runs = supa.fetch_runs(symbol=symbol, limit=20)
            top_runs = pick_top_runs_by_metrics(runs, top_n=3)
            st.subheader("Top 3 recent runs (by total_pnl then win_rate)")
            for rr in top_runs:
                st.write(rr)

            model_basename = f"confirm_model_{symbol.replace('=','_')}"
            saved_paths = export_model_and_metadata(model_wrap, feat_cols, combined, model_basename, save_fi=True)
            st.success(f"Saved model files: {saved_paths}")
        except Exception as exc:
            logger.exception("Logging/saving failed: %s", exc)
            st.error(f"Logging/saving failed: {exc}")

# Breadth handler
if run_breadth:
    st.info("Running breadth backtest across 3 levels...")
    # gather level configs from UI variables L1/L2/L3
    levels = {"L1": L1, "L2": L2, "L3": L3}
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    if bars is None or bars.empty:
        st.error("No bars for breadth run.")
    else:
        bars["rvol"] = compute_rvol(bars, 20)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        if cands is None or cands.empty:
            st.warning("No candidates generated with default lookback. Trying fallback lookback=10...")
            cands = generate_candidates_and_labels(bars, lookback=10, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
            st.write("DEBUG: after fallback, candidates count:", 0 if cands is None else len(cands))
        if cands is None or cands.empty:
            st.error("Still no candidates available for breadth run.")
        else:
            cands["pred_prob"] = cands.get("pred_prob", 0.0)
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
            if res.get("diagnostics"):
                st.write("Diagnostics:")
                for d in res["diagnostics"]:
                    st.write("-", d)

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
        if cands is None or cands.empty:
            st.warning("No candidates for sweep.")
        else:
            rr_vals = sorted(list(set([L1["rr_min"], L1["rr_max"], L2["rr_min"], L2["rr_max"], L3["rr_min"], L3["rr_max"]])))
            sl_ranges = [(L1["sl_min"], L1["sl_max"]), (L2["sl_min"], L2["sl_max"]), (L3["sl_min"], L3["sl_max"])]
            mpt_list = [float(p_fast), float(p_slow), float(p_deep)]
            try:
                sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list, feature_cols=feat_cols_default, model_train_kwargs={"max_bars":60})
                summary_rows = []
                for k, v in sweep_results.items():
                    if isinstance(v, dict) and "overlay" in v and v["overlay"] is not None and not v["overlay"].empty:
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

# Sidebar help
st.sidebar.markdown("### Notes\n- Levels have explicit buy_min/buy_max and sell_min/sell_max. Keep ranges exclusive to avoid ambiguity.\n- Breadth uses level-specific RR/SL ranges to simulate trades.\n- Use fallback lookback when data is short or intraday.")