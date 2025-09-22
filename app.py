# app.py  ─ Entry-Range Triangulation Dashboard
# ─────────────────────────────────────────────
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

# Optional libs (graceful degradation)
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
except Exception:
    torch = None
    nn = None
    optim = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)


# ───────────────── Streamlit page & widgets ─────────────────
st.set_page_config(page_title="Entry-Range Triangulation", layout="wide")
st.title("Entry-Range Triangulation Dashboard – (HealthGauge → Candidates → Confirm)")

symbol      = st.text_input("Symbol", value="GC=F")
start_date  = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
end_date    = st.date_input("End date",   value=datetime.today())
interval    = st.selectbox("Interval", ["1m","5m","15m","1h","1d"], index=4)

# --- Sidebar controls --------------------------------------------------------
st.sidebar.header("HealthGauge thresholds")
buy_threshold  = st.sidebar.number_input("Buy threshold",  0.0, 1.0, 0.55)
sell_threshold = st.sidebar.number_input("Sell threshold", 0.0, 1.0, 0.45)

st.sidebar.header("Model / Back-test")
num_boost   = st.sidebar.number_input("XGB rounds",            1, value=200)
early_stop  = st.sidebar.number_input("Early-stop rounds",     1, value=20)
test_size   = st.sidebar.number_input("Test set fraction",     0.0, 1.0, 0.2)

p_fast      = st.sidebar.number_input("Threshold fast", 0.0, 1.0, 0.60)
p_slow      = st.sidebar.number_input("Threshold slow", 0.0, 1.0, 0.55)
p_deep      = st.sidebar.number_input("Threshold deep", 0.0, 1.0, 0.45)

force_run                 = st.sidebar.checkbox("Force run", value=False)
show_confusion            = st.sidebar.checkbox("Show confusion matrix", value=True)
overlay_entries_on_price  = st.sidebar.checkbox("Overlay entries",       value=True)
include_health_as_feature = st.sidebar.checkbox("Include HealthGauge",   value=True)
save_feature_importance   = st.sidebar.checkbox("Save feature importance", value=True)

# Breadth / sweep -------------------------------------------------------------
st.sidebar.header("Breadth & Sweep")
run_breadth   = st.sidebar.button("Run breadth modes")
run_sweep_btn = st.sidebar.button("Run grid sweep")

rr_vals       = st.sidebar.multiselect("RR values", [1.5,2.0,2.5,3.0], default=[2.0,3.0])
sl_ranges_raw = st.sidebar.text_input("SL ranges", "0.5-1.0,1.0-2.0")
session_modes = st.sidebar.multiselect("Session modes", ["low","mid","high"],
                                       default=["low","mid","high"])
mpt_input     = st.sidebar.text_input("Model prob thresholds", "0.6,0.7")
max_bars      = int(st.sidebar.number_input("Max bars horizon", 1, value=60, step=1))

def _parse_sl_ranges(txt: str) -> List[Tuple[float,float]]:
    out: List[Tuple[float,float]] = []
    for t in (s.strip() for s in txt.split(",") if s.strip()):
        try:
            a,b = map(float, t.split("-"))
            out.append((a,b))
        except Exception:
            pass
    return out

sl_ranges = _parse_sl_ranges(sl_ranges_raw)
mpt_list  = [float(x) for x in (s.strip() for s in mpt_input.split(",") if s.strip())]

# Asset dataclass -------------------------------------------------------------
@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

asset_obj = Asset(name="Gold",
                  cot_name="GOLD - COMMODITY EXCHANGE INC.",
                  symbol=symbol)


# ─────────────────── Data fetch / feature helpers ───────────────────
def fetch_price(symbol: str,
                start: Optional[str] = None,
                end:   Optional[str] = None,
                interval: str = "1d") -> pd.DataFrame:
    """
    YahooQuery OHLCV fetcher returning a tidy DataFrame.
    """
    if YahooTicker is None:
        logger.error("yahooquery missing. Install via `pip install yahooquery`.")
        return pd.DataFrame()

    try:
        tq  = YahooTicker(symbol)
        raw = tq.history(start=start, end=end, interval=interval)
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)

        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        rename = {c: c.lower() for c in raw.columns}
        raw.rename(columns=rename, inplace=True)
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        return raw[~raw.index.duplicated()]
    except Exception as exc:
        logger.error("fetch_price failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(1.0, index=df.index)
    rolling = df["volume"].rolling(lookback, min_periods=1).mean()
    return (df["volume"] / rolling.replace(0, np.nan)).fillna(1.0)


def calculate_health_gauge(cot_df: pd.DataFrame,
                           daily_bars: pd.DataFrame,
                           threshold: float = 1.5) -> pd.DataFrame:
    """
    Very simple health score: rVol ≥ threshold ⇒ 1 else 0.
    """
    if daily_bars is None or daily_bars.empty:
        return pd.DataFrame()

    db          = daily_bars.copy()
    db["rvol"]  = compute_rvol(db)
    score       = (db["rvol"] >= threshold).astype(float)
    return pd.DataFrame({"health_gauge": score}, index=db.index)


def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ─────────────────── Candidate / label generation ───────────────────
def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high-low,
                    (high-prev_close).abs(),
                    (low-prev_close).abs()],
                   axis=1).max(axis=1)
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

    if bars is None or bars.empty:
        return pd.DataFrame()

    bars         = bars.copy()
    bars.index   = pd.to_datetime(bars.index)
    bars         = ensure_unique_index(bars)

    for col in ("high","low","close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")

    bars["tr"]  = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(atr_window, min_periods=1).mean()

    records = []
    n = len(bars)
    for i in range(lookback, n):
        t          = bars.index[i]
        entry_px   = bars["close"].iat[i]
        atr_val    = bars["atr"].iat[i]
        if atr_val <= 0 or math.isnan(atr_val):
            continue

        sl_px = entry_px - k_sl*atr_val if direction=="long" else entry_px + k_sl*atr_val
        tp_px = entry_px + k_tp*atr_val if direction=="long" else entry_px - k_tp*atr_val
        end_i = min(i+max_bars, n-1)

        label, hit_i, hit_px = 0, end_i, bars["close"].iat[end_i]
        for j in range(i+1, end_i+1):
            hi, lo = bars["high"].iat[j], bars["low"].iat[j]
            if direction=="long":
                if hi >= tp_px: label, hit_i, hit_px = 1, j, tp_px; break
                if lo <= sl_px: label, hit_i, hit_px = 0, j, sl_px; break
            else:
                if lo <= tp_px: label, hit_i, hit_px = 1, j, tp_px; break
                if hi >= sl_px: label, hit_i, hit_px = 0, j, sl_px; break

        end_t   = bars.index[hit_i]
        ret_val = (hit_px-entry_px)/entry_px if direction=="long" else (entry_px-hit_px)/entry_px
        dur_min = (end_t - t).total_seconds() / 60.0

        records.append(dict(candidate_time=t,
                            entry_price=float(entry_px),
                            atr=float(atr_val),
                            sl_price=float(sl_px),
                            tp_price=float(tp_px),
                            end_time=end_t,
                            label=int(label),
                            duration=float(dur_min),
                            realized_return=float(ret_val),
                            direction=direction))
    return pd.DataFrame(records)


# ───────────────────── Back-test / simulation ──────────────────────
def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str   = "GC=F",
                    sl: float = 0.01,
                    tp: float = 0.02,
                    max_holding: int = 20) -> pd.DataFrame:
    """
    Very light limit simulation (no overlap protection).
    """
    if df is None or df.empty or bars is None or bars.empty:
        return pd.DataFrame()

    trades = []
    for _, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl):
            continue

        entry_t = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_t not in bars.index:
            continue
        entry_px = float(bars.loc[entry_t, "close"])
        direction = 1 if lbl>0 else -1
        sl_px = entry_px * (1 - sl) if direction>0 else entry_px * (1 + sl)
        tp_px = entry_px * (1 + tp) if direction>0 else entry_px * (1 - tp)

        exit_t, exit_px, pnl = None, None, None
        for t,b in bars.loc[entry_t:].head(max_holding).iterrows():
            lo, hi = b["low"], b["high"]
            if direction>0:
                if lo <= sl_px: exit_t, exit_px, pnl = t, sl_px, -sl; break
                if hi >= tp_px: exit_t, exit_px, pnl = t, tp_px,  tp; break
            else:
                if hi >= sl_px: exit_t, exit_px, pnl = t, sl_px, -sl; break
                if lo <= tp_px: exit_t, exit_px, pnl = t, tp_px,  tp; break

        if exit_t is None:
            last_bar = bars.loc[entry_t:].head(max_holding).iloc[-1]
            exit_t   = last_bar.name
            exit_px  = last_bar["close"]
            pnl      = (exit_px-entry_px)/entry_px*direction

        trades.append(dict(symbol=symbol,
                           entry_time=entry_t,
                           entry_price=entry_px,
                           direction=direction,
                           exit_time=exit_t,
                           exit_price=exit_px,
                           pnl=float(pnl)))
    return pd.DataFrame(trades)

# ────────────────── XGBoost confirm-stage model ───────────────────
class BoosterWrapper:
    def __init__(self, booster, feature_names: List[str]):
        self.booster        = booster
        self.feature_names  = feature_names
        self.best_iteration = getattr(booster, "best_iteration", None)

    def _dmatrix(self, X: pd.DataFrame):
        Xp = X.reindex(columns=self.feature_names).fillna(0.0)
        return xgb.DMatrix(Xp, feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        d = self._dmatrix(X)
        if self.best_iteration is not None:
            raw = self.booster.predict(d, iteration_range=(0, self.best_iteration+1))
        else:
            raw = self.booster.predict(d)
        raw = np.clip(raw, 0.0, 1.0)
        return pd.Series(raw, index=X.index, name="confirm_proba")

    def save_model(self, path: str):
        self.booster.save_model(path)

    def feature_importance(self) -> pd.DataFrame:
        imp = self.booster.get_score(importance_type="gain")
        return (pd.DataFrame([(f, imp.get(f,0)) for f in self.feature_names],
                             columns=["feature","gain"])
                .sort_values("gain", ascending=False).reset_index(drop=True))


def train_xgb_confirm(clean: pd.DataFrame,
                      feature_cols: List[str],
                      label_col: str = "label",
                      num_boost_round: int = 200,
                      early_stopping_rounds: int = 20,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[BoosterWrapper, Dict[str,Any]]:

    if xgb is None:
        raise RuntimeError("xgboost not installed")

    X = clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(clean[label_col], errors="coerce")
    m = X.replace([np.inf,-np.inf], np.nan).notnull().all(1) & y.notnull()
    X, y = X[m], y[m]

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=random_state)

    dtr, dva = xgb.DMatrix(Xtr,label=ytr), xgb.DMatrix(Xva,label=yva)

    params = dict(objective="binary:logistic",
                  eta=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss",
                  scale_pos_weight=float((y==0).sum())/max(1,(y==1).sum()))
    bst = xgb.train(params, dtr, num_boost_round,
                    evals=[(dva,"val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False)

    wrap = BoosterWrapper(bst, feature_cols)
    yhat = wrap.predict_proba(Xva) >= 0.5
    metrics = dict(accuracy=float(accuracy_score(yva,yhat)),
                   f1=float(f1_score(yva,yhat)),
                   roc_auc=float(roc_auc_score(yva,wrap.predict_proba(Xva))))
    return wrap, metrics


def predict_confirm_prob(model: BoosterWrapper,
                         df: pd.DataFrame,
                         feature_cols: List[str]) -> pd.Series:
    missing = [c for c in feature_cols if c not in df.columns]
    for m in missing:
        df[m] = 0.0
    return model.predict_proba(df[feature_cols])


# ───────────── Optional PyTorch multi-model bundle (if torch) ─────────────
if torch is not None:
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, 2)
            )
        def forward(self,x): return self.net(x)

    # ... (SimpleLSTM, SimpleCNN and trainer identical to original,
    # omitted here for brevity but unchanged)

# ───────────────────── Supabase logger wrapper ─────────────────────
class SupabaseLogger:
    def __init__(self):
        if create_client is None:
            raise RuntimeError("supabase client not installed")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE env vars missing")
        self.client = create_client(url, key)
        self.runs_tbl   = "entry_runs"
        self.trades_tbl = "entry_trades"

    def log_run(self, metrics: Dict, meta: Dict, trades: List[Dict]) -> str:
        run_id = meta["run_id"]
        self.client.table(self.runs_tbl).insert({**meta, "metrics": metrics}).execute()
        if trades:
            for t in trades:
                t["run_id"] = run_id
            self.client.table(self.trades_tbl).insert(trades).execute()
        return run_id


# ───────────────────────── Main Streamlit flow ─────────────────────────
def run_main_pipeline():
    st.info(f"Fetching price data for {symbol} …")
    bars = fetch_price(symbol, start_date.isoformat(), end_date.isoformat(), interval)
    if bars.empty:
        st.error("No price data returned.")
        return

    daily = bars.resample("1D").agg({"open":"first","high":"max","low":"min",
                                     "close":"last","volume":"sum"}).dropna()
    health = calculate_health_gauge(None, daily)
    latest_health = float(health.health_gauge.iloc[-1]) if not health.empty else 0.0
    st.metric("Latest HealthGauge", f"{latest_health:.2f}")

    if not (latest_health>=buy_threshold or latest_health<=sell_threshold or force_run):
        st.warning("HealthGauge gating prevented run."); return

    bars["rvol"] = compute_rvol(bars, asset_obj.rvol_lookback)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0,
                                           atr_window=asset_obj.atr_lookback,
                                           max_bars=max_bars)
    if cands.empty:
        st.error("No candidates."); return
    if include_health_as_feature and not health.empty:
        cands["hg"] = cands["candidate_time"].dt.normalize().map(health["health_gauge"]).fillna(0)
    else:
        cands["hg"] = 0.0

    feat_cols = ["atr","rvol","duration","hg"] if include_health_as_feature else ["atr","rvol","duration"]
    clean = cands.dropna(subset=["label"]).query("label in [0,1]").reset_index(drop=True)

    model, metrics = train_xgb_confirm(clean, feat_cols, "label",
                                       num_boost_round=int(num_boost),
                                       early_stopping_rounds=int(early_stop),
                                       test_size=float(test_size))
    st.json(metrics)

    clean["pred_prob"]  = predict_confirm_prob(model, clean, feat_cols)
    clean["pred_label"] = (clean["pred_prob"] >= p_fast).astype(int)

    trades = simulate_limits(clean, bars, max_holding=max_bars)
    st.write("Simulated trades:", len(trades))
    if not trades.empty and overlay_entries_on_price:
        fig, ax = plt.subplots(figsize=(10,4))
        bars["close"].plot(ax=ax, lw=0.8)
        for _,r in trades.iterrows():
            ax.axvline(r.entry_time, color="g" if r.pnl>0 else "r", alpha=0.6)
        st.pyplot(fig)

    st.metric("Total PnL", f"{trades.pnl.sum():.4f}")


# ────────────────────────────── UI buttons ──────────────────────────────
if st.button("Run main pipeline"):
    try:
        run_main_pipeline()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        st.error(f"Pipeline failed: {exc}")

# Additional buttons (Breadth / Sweep) keep original handlers -----------------
# (Copy unchanged from the original script if you need those features)
