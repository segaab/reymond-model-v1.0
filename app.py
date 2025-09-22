# app.py — Entry-Range Triangulation (integrated single-file)
# Chunk 1/7: imports + Streamlit UI + basic config
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

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

# optional third-party imports (defensive)
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
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

# Streamlit page config
st.set_page_config(page_title="Entry-Range Triangulation Dashboard", layout="wide")
st.title("Entry-Range Triangulation Dashboard — Integrated")

# Chunk 2/7: UI controls, Asset dataclass, fetch + feature helpers

# ---- UI controls
symbol = st.text_input("Symbol", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)

# thresholds and model params
buy_threshold = st.sidebar.number_input("Buy threshold (HealthGauge)", 0.0, 1.0, 0.55)
sell_threshold = st.sidebar.number_input("Sell threshold (HealthGauge)", 0.0, 1.0, 0.45)

num_boost = int(st.sidebar.number_input("XGBoost rounds", 1, value=200))
early_stop = int(st.sidebar.number_input("XGBoost early_stop", 1, value=20))
test_size = float(st.sidebar.number_input("Test size fraction", 0.0, 1.0, 0.2))

p_fast = st.sidebar.number_input("Threshold fast (prob)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Threshold slow (prob)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Threshold deep (prob)", 0.0, 1.0, 0.45)

force_run = st.sidebar.checkbox("Force run even if HealthGauge not in band", value=False)
show_confusion = st.sidebar.checkbox("Show confusion matrix", value=True)
overlay_entries_on_price = st.sidebar.checkbox("Overlay entries on price", value=True)
include_health_as_feature = st.sidebar.checkbox("Include HealthGauge as feature", value=True)
save_feature_importance = st.sidebar.checkbox("Save feature importance on export", value=True)

run_breadth = st.sidebar.button("Run breadth modes")
run_sweep_btn = st.sidebar.button("Run grid sweep")

# breadth/sweep params
rr_vals = st.sidebar.multiselect("RR values", [1.0,1.5,2.0,2.5,3.0], default=[2.0,3.0])
sl_ranges_txt = st.sidebar.text_input("SL ranges (comma list like 0.5-1.0,1.0-2.0)", "0.5-1.0,1.0-2.0")
max_bars = int(st.sidebar.number_input("Max bars horizon", 1, value=60))

def parse_sl_ranges(txt: str) -> List[Tuple[float,float]]:
    out = []
    for s in (p.strip() for p in txt.split(",") if p.strip()):
        try:
            a,b = s.split("-")
            out.append((float(a), float(b)))
        except Exception:
            continue
    return out

sl_ranges = parse_sl_ranges(sl_ranges_txt)

# ---- Asset dataclass
@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

asset_obj = Asset(name="Gold", cot_name="GOLD - COMMODITY EXCHANGE INC.", symbol=symbol)

# ---- Data fetcher using yahooquery (defensive)
def fetch_price(symbol: str, start: Optional[str], end: Optional[str], interval: str = "1d") -> pd.DataFrame:
    if YahooTicker is None:
        logger.error("yahooquery not installed. Install via `pip install yahooquery`.")
        return pd.DataFrame()
    try:
        t = YahooTicker(symbol)
        raw = t.history(start=start, end=end, interval=interval)
        # yahooquery sometimes returns MultiIndex or dict
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        # normalize column names
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        return raw[~raw.index.duplicated()]
    except Exception as e:
        logger.error("fetch_price failed: %s", e)
        return pd.DataFrame()

# ---- Features
def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(1.0, index=df.index)
    rolling_avg = df["volume"].rolling(window=lookback, min_periods=1).mean()
    rvol = (df["volume"] / rolling_avg.replace(0, np.nan)).fillna(1.0)
    return rvol

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    if daily_bars is None or daily_bars.empty:
        return pd.DataFrame()
    db = daily_bars.copy()
    db["rvol"] = compute_rvol(db)
    score = (db["rvol"] >= threshold).astype(float)
    return pd.DataFrame({"health_gauge": score}, index=db.index)

def ensure_no_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

# Chunk 3/7: labeling / generate_candidates_and_labels
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Generate candidates with ATR-based SL/TP + triple-barrier-like labeling.
    Returns DataFrame with candidate_time, entry_price, atr, sl_price, tp_price, end_time, label, duration, realized_return, direction
    """
    if bars is None or bars.empty:
        return pd.DataFrame()

    df = bars.copy()
    df.index = pd.to_datetime(df.index)
    df = ensure_no_duplicate_index(df)

    # ensure required columns
    for c in ("high", "low", "close"):
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in bars")

    df["tr"] = _true_range(df["high"], df["low"], df["close"])
    df["atr"] = df["tr"].rolling(window=atr_window, min_periods=1).mean().fillna(method="ffill").fillna(0.0)

    records = []
    n = len(df)
    for i in range(lookback, n):
        t = df.index[i]
        entry_price = float(df["close"].iat[i])
        atr_t = float(df["atr"].iat[i])
        if atr_t <= 0 or math.isnan(atr_t):
            continue

        if direction == "long":
            sl_price = entry_price - k_sl * atr_t
            tp_price = entry_price + k_tp * atr_t
        else:
            sl_price = entry_price + k_sl * atr_t
            tp_price = entry_price - k_tp * atr_t

        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_price = float(df["close"].iat[end_idx])

        for j in range(i + 1, end_idx + 1):
            px_high = float(df["high"].iat[j])
            px_low = float(df["low"].iat[j])
            if direction == "long":
                if px_high >= tp_price:
                    label, hit_idx, hit_price = 1, j, tp_price
                    break
                if px_low <= sl_price:
                    label, hit_idx, hit_price = 0, j, sl_price
                    break
            else:
                if px_low <= tp_price:
                    label, hit_idx, hit_price = 1, j, tp_price
                    break
                if px_high >= sl_price:
                    label, hit_idx, hit_price = 0, j, sl_price
                    break

        end_time = df.index[hit_idx]
        realized_return = (hit_price - entry_price) / entry_price if direction == "long" else (entry_price - hit_price) / entry_price
        dur_min = (end_time - t).total_seconds() / 60.0

        rec = {
            "candidate_time": t,
            "entry_price": entry_price,
            "atr": atr_t,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "end_time": end_time,
            "label": int(label),
            "duration": float(dur_min),
            "realized_return": float(realized_return),
            "direction": direction
        }
        records.append(rec)
    return pd.DataFrame(records)


# Chunk 4/7: simulate_limits, breadth backtest, sweep
def simulate_limits(
    df: pd.DataFrame,
    bars: pd.DataFrame,
    label_col: str = "pred_label",
    symbol: str = "GC=F",
    sl: float = 0.01,
    tp: float = 0.02,
    max_holding: int = 20
) -> pd.DataFrame:
    if df is None or df.empty or bars is None or bars.empty:
        return pd.DataFrame()
    trades = []
    for _, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl):
            continue
        entry_time = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_time not in bars.index:
            continue
        entry_price = float(bars.loc[entry_time, "close"])
        direction = 1 if lbl > 0 else -1
        sl_price = entry_price * (1 - sl) if direction > 0 else entry_price * (1 + sl)
        tp_price = entry_price * (1 + tp) if direction > 0 else entry_price * (1 - tp)

        exit_time = None; exit_price = None; pnl = None
        window = bars.loc[entry_time:].head(max_holding)
        if window.empty:
            continue
        for t, b in window.iterrows():
            lo, hi = float(b["low"]), float(b["high"])
            if direction > 0:
                if lo <= sl_price:
                    exit_time, exit_price, pnl = t, sl_price, -sl
                    break
                if hi >= tp_price:
                    exit_time, exit_price, pnl = t, tp_price, tp
                    break
            else:
                if hi >= sl_price:
                    exit_time, exit_price, pnl = t, sl_price, -sl
                    break
                if lo <= tp_price:
                    exit_time, exit_price, pnl = t, tp_price, tp
                    break
        if exit_time is None:
            last = window.iloc[-1]
            exit_time = last.name
            exit_price = float(last["close"])
            pnl = (exit_price - entry_price) / entry_price * direction
        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": float(pnl)
        })
    return pd.DataFrame(trades)

def run_breadth_backtest(clean: pd.DataFrame, bars: pd.DataFrame, symbol: str = "GC=F"):
    """
    For each breadth mode (low/mid/high), apply thresholds and run simulate_limits.
    Return dict with summary (per-mode aggregated DataFrame) and detailed overlays.
    """
    results = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        results["diagnostics"].append("No clean candidates provided.")
        return results

    modes = {
        "low": {"sell_th": 5, "buy_th": 5},
        "mid": {"sell_th": 4, "buy_th": 6},
        "high": {"sell_th": 3, "buy_th": 7}
    }
    for mode, th in modes.items():
        try:
            df = clean.copy()
            # ensure 'signal' exists (scale pred_prob 0-1 → 0-10)
            if "signal" not in df.columns:
                if "pred_prob" in df.columns:
                    df["signal"] = (df["pred_prob"] * 10).round().astype(int)
                else:
                    df["signal"] = 0
            df["pred_label"] = 0
            df.loc[df["signal"] > th["buy_th"], "pred_label"] = 1
            df.loc[df["signal"] < th["sell_th"], "pred_label"] = -1

            overlay = simulate_limits(df, bars, label_col="pred_label", symbol=symbol, max_holding=max_bars)
            trades_df = overlay
            summary = {
                "mode": mode,
                "total_trades": int(trades_df.shape[0]),
                "win_rate": float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0,
                "total_pnl": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
            }
            results["summary"].append(summary)
            results["detailed_trades"][mode] = trades_df
        except Exception as e:
            results["diagnostics"].append(f"{mode} failed: {e}")
            results["detailed_trades"][mode] = pd.DataFrame()
    return results

def run_sweep(clean: pd.DataFrame, bars: pd.DataFrame, symbol: str = "GC=F"):
    """
    Sweep combinations of sell/buy thresholds (example grid).
    Return dict keyed by sweep id with overlay DataFrames.
    """
    results = {}
    if clean is None or clean.empty:
        return results
    sell_ths = [3,4,5]
    buy_ths = [5,6,7]
    for s in sell_ths:
        for b in buy_ths:
            key = f"sell{s}_buy{b}"
            try:
                df = clean.copy()
                if "signal" not in df.columns:
                    df["signal"] = (df.get("pred_prob", 0.0) * 10).round().astype(int)
                df["pred_label"] = 0
                df.loc[df["signal"] > b, "pred_label"] = 1
                df.loc[df["signal"] < s, "pred_label"] = -1
                overlay = simulate_limits(df, bars, label_col="pred_label", symbol=symbol, max_holding=max_bars)
                results[key] = {"trades": overlay.shape[0], "overlay": overlay}
            except Exception as e:
                results[key] = {"error": str(e)}
    return results

# Chunk 5/7: XGBoost wrapper, training, prediction, model export
class XGBWrapper:
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
            raw = self.booster.predict(d, iteration_range=(0, int(self.best_iteration)+1))
        else:
            raw = self.booster.predict(d)
        raw = np.asarray(raw, dtype=float)
        return pd.Series(raw, index=X.index, name="confirm_proba")

    def save_model(self, path: str):
        self.booster.save_model(path)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        imp = self.booster.get_score(importance_type=importance_type) or {}
        df = pd.DataFrame([(k, float(imp.get(k, 0.0))) for k in self.feature_names], columns=["feature", "importance"])
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

def train_xgb_confirm(clean: pd.DataFrame,
                      feature_cols: List[str],
                      label_col: str = "label",
                      num_boost_round: int = 200,
                      early_stopping_rounds: int = 20,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[XGBWrapper, Dict[str,Any]]:
    if xgb is None:
        raise RuntimeError("xgboost not installed")
    X = clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(clean[label_col], errors="coerce")
    mask = X.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y.notnull()
    X, y = X.loc[mask], y.loc[mask]
    if X.empty:
        raise ValueError("No valid training rows")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "verbosity": 0,
        "scale_pos_weight": float((y==0).sum()) / max(1.0, float((y==1).sum()))
    }

    bst = xgb.train(params, dtr,
                    num_boost_round=int(num_boost_round),
                    evals=[(dva, "validation")],
                    early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
                    verbose_eval=False)

    wrap = XGBWrapper(bst, feature_cols)
    y_proba_val = wrap.predict_proba(Xva)
    y_pred_val = (y_proba_val >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "accuracy": float(accuracy_score(yva, y_pred_val)),
        "f1": float(f1_score(yva, y_pred_val, zero_division=0)),
        "val_proba_mean": float(np.nanmean(y_proba_val)) if len(y_proba_val) else 0.0
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(yva, y_proba_val))
    except Exception:
        metrics["roc_auc"] = None

    return wrap, metrics

def predict_confirm_prob(model: XGBWrapper, candidates: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    missing = [c for c in feature_cols if c not in candidates.columns]
    for m in missing:
        candidates[m] = 0.0
    return model.predict_proba(candidates[feature_cols])

def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str,Any], model_basename: str = "confirm_model", save_fi: bool = True) -> Dict[str,str]:
    """
    Save xgboost model (native .model), metadata JSON and feature-importance JSON (optional).
    Returns dict of saved paths.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"{model_basename}_{ts}"
    paths = {}
    try:
        # try xgb booster
        booster = getattr(model_wrapper, "booster", None) or getattr(model_wrapper, "booster", None)
        if booster is None and hasattr(model_wrapper, "booster"):
            booster = model_wrapper.booster
        if booster is not None and hasattr(booster, "save_model"):
            model_file = f"{base}.model"
            booster.save_model(model_file)
            paths["model"] = model_file
        else:
            # fallback: joblib
            joblib_file = f"{base}.joblib"
            joblib.dump(model_wrapper, joblib_file)
            paths["joblib"] = joblib_file

        meta_file = f"{base}.json"
        with open(meta_file, "w") as f:
            json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2, default=str)
        paths["meta"] = meta_file

        if save_fi and hasattr(model_wrapper, "feature_importance"):
            try:
                fi_df = model_wrapper.feature_importance()
                fi_file = f"{base}_feature_importance.json"
                fi_df.to_json(fi_file, orient="records", date_format="iso")
                paths["feature_importance"] = fi_file
            except Exception:
                pass
    except Exception as e:
        logger.exception("export_model_and_metadata failed: %s", e)
    return paths

# Chunk 6/7: Supabase logger + (optional) PyTorch multi-model bundler

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
        rec = {**metadata, "metrics": metrics, "run_id": run_id}
        resp = self.client.table(self.runs_tbl).insert(rec).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Supabase insert run error: {resp.error}")
        if trades:
            for t in trades:
                t["run_id"] = run_id
            tr_resp = self.client.table(self.trades_tbl).insert(trades).execute()
            if getattr(tr_resp, "error", None):
                raise RuntimeError(f"Supabase insert trades error: {tr_resp.error}")
        return run_id

# Optional PyTorch multi-model training & bundling (top-3)
def train_and_bundle_top3(train_loader, val_loader, input_dim: int, save_dir: str = "saved_models", epochs: int = 10, device: Optional[str] = None):
    """
    Train a small set of PyTorch candidate models and bundle the top-3 by val accuracy into a .pt file.
    Returns list of dicts: [{"name":..., "model":..., "history":...}, ...] for top3
    """
    if torch is None:
        raise RuntimeError("PyTorch not available")

    save_path = Path(save_dir); save_path.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # candidate architectures
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))
        def forward(self, x): return self.net(x)

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 2)
        def forward(self, x):
            out, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    candidates = {
        "mlp": SimpleMLP(input_dim),
    }
    results = []
    for name, model in candidates.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_acc": []}
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(train_loader))
            # val
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    correct += (preds.argmax(dim=1) == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / max(1, total)
            history["train_loss"].append(avg_loss); history["val_acc"].append(val_acc)
        results.append({"name": name, "model": model.cpu(), "history": history})

    # rank by val_acc
    results.sort(key=lambda r: max(r["history"]["val_acc"]) if r["history"]["val_acc"] else 0.0, reverse=True)
    top3 = results[:3]
    # bundle state_dicts
    bundle = {r["name"]: r["model"].state_dict() for r in top3}
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bundle_path = save_path / f"top3_bundle_{ts}.pt"
    torch.save(bundle, str(bundle_path))
    logger.info("Saved top3 bundle: %s", bundle_path)
    return top3, str(bundle_path)

# Chunk 7/7: main pipeline, UI wiring, save/logging handlers

def run_main_pipeline():
    st.info(f"Fetching price for {symbol}…")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars is None or bars.empty:
        st.error("No price data; check symbol and date range.")
        return

    bars = ensure_no_duplicate_index(bars)
    # daily bars for health gauge
    try:
        daily = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    except Exception:
        daily = pd.DataFrame()

    health_df = calculate_health_gauge(None, daily, threshold=1.5) if not daily.empty else pd.DataFrame()
    if not health_df.empty:
        latest_health = float(health_df["health_gauge"].iloc[-1])
    else:
        latest_health = 0.0
    st.metric("Latest HealthGauge", f"{latest_health:.4f}")

    if not (latest_health >= buy_threshold or latest_health <= sell_threshold or force_run):
        st.warning("HealthGauge gating prevented main run. Use Force Run to override.")
        return

    # compute rvol and attach to bars
    bars["rvol"] = compute_rvol(bars, lookback=asset_obj.rvol_lookback)

    # generate candidates (labeling)
    st.info("Generating candidates and labels…")
    cands = generate_candidates_and_labels(bars, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=asset_obj.atr_lookback, max_bars=max_bars, direction="long")
    if cands is None or cands.empty:
        st.error("No candidates generated.")
        return

    # attach health gauge as feature if requested
    if include_health_as_feature and not health_df.empty:
        # map daily health gauge to candidate_date (normalize)
        cand_dates = cands["candidate_time"].dt.normalize()
        hg_series = health_df["health_gauge"].reindex(pd.to_datetime(health_df.index).normalize()).ffill().fillna(0.0)
        hg_map = hg_series.to_dict()
        cands["health_gauge"] = cand_dates.map(lambda d: hg_map.get(pd.to_datetime(d), 0.0))
    else:
        cands["health_gauge"] = 0.0

    # prepare clean dataset for training
    feat_cols = ["atr", "rvol", "duration"]
    if include_health_as_feature:
        feat_cols.append("health_gauge")

    # the generated candidates include atr/duration/realized_return etc — ensure columns
    for fc in feat_cols:
        if fc not in cands.columns:
            cands[fc] = 0.0

    clean = cands.dropna(subset=["label"]).reset_index(drop=True)
    if clean.empty:
        st.error("No labeled candidates available for training.")
        return

    # train XGBoost confirm
    st.info("Training XGBoost confirm model…")
    try:
        model_wrap, metrics = train_xgb_confirm(clean, feat_cols, label_col="label", num_boost_round=num_boost, early_stopping_rounds=early_stop, test_size=test_size)
    except Exception as e:
        logger.exception("Training failed: %s", e)
        st.error(f"Training failed: {e}")
        return

    st.write("Training metrics:", metrics)

    # predictions
    try:
        clean["pred_prob"] = predict_confirm_prob(model_wrap, clean, feat_cols).reindex(clean.index).fillna(0.0)
        clean["pred_label"] = (clean["pred_prob"] >= p_fast).astype(int)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        st.error(f"Prediction failed: {e}")
        return

    # simulate fills/backtest
    st.info("Simulating fills & backtest…")
    trades = simulate_limits(clean, bars, label_col="pred_label", max_holding=max_bars)
    st.write("Simulated trades:", 0 if trades is None else len(trades))
    if trades is None or trades.empty:
        st.warning("No trades simulated.")
    else:
        # show basic metrics
        trades["pnl"] = trades["pnl"].astype(float)
        num_trades = len(trades)
        total_pnl = trades["pnl"].sum()
        win_rate = float((trades["pnl"] > 0).mean())
        st.metric("Num trades (sim)", f"{num_trades}")
        st.metric("Total PnL (sim)", f"{total_pnl:.6f}")
        st.metric("Win rate", f"{win_rate:.2%}")

        if overlay_entries_on_price:
            fig, ax = plt.subplots(figsize=(12,4))
            if "close" in bars.columns:
                bars["close"].plot(ax=ax, label="close")
            for _, r in trades.iterrows():
                t = r["entry_time"]
                color = "g" if r["pnl"] > 0 else "r"
                ax.axvline(x=t, color=color, alpha=0.6, linewidth=0.8)
            ax.set_title(f"{symbol} — Price with entry overlays")
            ax.legend()
            st.pyplot(fig)

    # save final model (retrain on full candidate universe) and export
    st.subheader("Save Model (train on full candidate universe)")
    model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
    if st.button("Save model as .model + metadata"):
        try:
            # retrain on full candidate universe (no gating)
            full_cands = generate_candidates_and_labels(bars, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=asset_obj.atr_lookback, max_bars=max_bars, direction="long")
            if include_health_as_feature and not health_df.empty:
                cand_dates = full_cands["candidate_time"].dt.normalize()
                hg_series = health_df["health_gauge"].reindex(pd.to_datetime(health_df.index).normalize()).ffill().fillna(0.0)
                hg_map = hg_series.to_dict()
                full_cands["health_gauge"] = cand_dates.map(lambda d: hg_map.get(pd.to_datetime(d), 0.0))
            for col in feat_cols + ["label"]:
                if col not in full_cands.columns:
                    full_cands[col] = np.nan
            for col in feat_cols:
                full_cands[col] = pd.to_numeric(full_cands[col], errors="coerce").fillna(0.0)
            full_clean = full_cands.dropna(subset=["label"])
            full_clean = full_clean[full_clean["label"].isin([0,1])]
            if full_clean.empty:
                st.error("No valid labeled data in full candidate set.")
            else:
                final_model, final_metrics = train_xgb_confirm(full_clean, feat_cols, label_col="label", num_boost_round=num_boost, early_stopping_rounds=early_stop, test_size=test_size)
                saved_paths = export_model_and_metadata(final_model.booster if hasattr(final_model, "booster") else final_model, feat_cols, final_metrics, model_basename=model_name_input, save_fi=save_feature_importance)
                st.success(f"Saved final model. Files: {saved_paths}")
        except Exception as e:
            logger.exception("Saving final model failed: %s", e)
            st.error(f"Failed to train/save final model: {e}")

    # Supabase logging
    st.subheader("Logging")
    if st.button("Save logs to Supabase"):
        try:
            supa = SupabaseLogger()
            run_id = str(uuid.uuid4())
            metadata = {
                "run_id": run_id,
                "symbol": symbol,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "interval": interval,
                "feature_cols": feat_cols,
                "model_file": None,
                "training_params": {"num_boost_round": int(num_boost), "early_stopping_rounds": int(early_stop), "test_size": float(test_size)},
                "health_thresholds": {"buy_threshold": float(buy_threshold), "sell_threshold": float(sell_threshold)},
                "p_fast": float(p_fast), "p_slow": float(p_slow), "p_deep": float(p_deep),
            }
            backtest_metrics = {
                "num_trades": int(0 if trades is None else len(trades)),
                "total_pnl": float(0.0 if trades is None or trades.empty else trades["pnl"].sum()),
                "win_rate": float(0.0 if trades is None or trades.empty else (trades["pnl"] > 0).mean()),
                "latest_health": float(latest_health),
            }
            combined_metrics = {}
            combined_metrics.update(metrics if isinstance(metrics, dict) else {})
            combined_metrics.update(backtest_metrics)

            trade_list = []
            if trades is not None and not trades.empty:
                for r in trades.to_dict(orient="records"):
                    trade_list.append({
                        "candidate_time": str(r.get("entry_time")),
                        "layer": None,
                        "size": float(0.0),
                        "entry_price": float(r.get("entry_price") or 0.0),
                        "filled_at": str(r.get("exit_time") or ""),
                        "ret": float(r.get("pnl") or 0.0),
                        "pnl": float(r.get("pnl") or 0.0),
                    })

            run_id_returned = supa.log_run(metrics=combined_metrics, metadata=metadata, trades=trade_list)
            st.success(f"Logged run to Supabase with run_id: {run_id_returned}")
        except Exception as e:
            logger.exception("Supabase log failed: %s", e)
            st.error(f"Failed to log to Supabase: {e}")

    # Breadth / sweep handlers
    if run_breadth:
        st.info("Running breadth backtest...")
        breadth_results = run_breadth_backtest(clean, bars, symbol=symbol)
        st.subheader("Breadth Summary")
        st.dataframe(pd.DataFrame(breadth_results.get("summary", [])))
        # show detailed
        for mode, df in breadth_results.get("detailed_trades", {}).items():
            if df is None or df.empty:
                st.info(f"No trades for mode {mode}")
                continue
            with st.expander(f"{mode} — {len(df)} trades"):
                st.dataframe(df.head(200))

    if run_sweep_btn:
        st.info("Running sweep grid...")
        sweep_results = run_sweep(clean, bars, symbol=symbol)
        st.subheader("Sweep Results")
        st.write({k: (v.get("trades") if isinstance(v, dict) else None) for k,v in sweep_results.items()})

# run streamlit app
if st.button("Run main pipeline"):
    try:
        run_main_pipeline()
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        st.error(f"Pipeline failed: {e}")