# app.py — Entry-Range Triangulation (Streamlit + Cascade) — Chunk 1/7
import os
import io
import math
import time
import uuid
import json
import joblib
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# optional libs
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

# Streamlit UI top-level
st.set_page_config(page_title="Cascade Entry-Range Triangulation", layout="wide")
st.title("Cascade Trader — Scope → Aim → Shoot (L1/L2/L3)")

# Basic inputs
symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=90))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

# Training & model controls
st.sidebar.header("Training controls")
num_boost = int(st.sidebar.number_input("XGB rounds (if used)", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))

# Confirm/prob thresholds for backtest labeling
st.sidebar.header("Confirm thresholds (for quick sim)")
p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

# Level gating UI: explicit min/max buy/sell thresholds for exclusivity
st.sidebar.header("Level gating (explicit min/max)")
st.sidebar.markdown("Set buy/sell signal ranges (signal scaled 0-10). Levels are exclusive by ranges.")
# Level 1
lvl1_buy_min = st.sidebar.number_input("L1 buy min", 0.0, 10.0, 5.5, step=0.1)
lvl1_buy_max = st.sidebar.number_input("L1 buy max", 0.0, 10.0, 9.9, step=0.1)
lvl1_sell_min = st.sidebar.number_input("L1 sell min", 0.0, 10.0, 0.0, step=0.1)
lvl1_sell_max = st.sidebar.number_input("L1 sell max", 0.0, 10.0, 4.5, step=0.1)
# Level 2
lvl2_buy_min = st.sidebar.number_input("L2 buy min", 0.0, 10.0, 6.0, step=0.1)
lvl2_buy_max = st.sidebar.number_input("L2 buy max", 0.0, 10.0, 9.9, step=0.1)
lvl2_sell_min = st.sidebar.number_input("L2 sell min", 0.0, 10.0, 0.0, step=0.1)
lvl2_sell_max = st.sidebar.number_input("L2 sell max", 0.0, 10.0, 4.0, step=0.1)
# Level 3
lvl3_buy_min = st.sidebar.number_input("L3 buy min", 0.0, 10.0, 6.5, step=0.1)
lvl3_buy_max = st.sidebar.number_input("L3 buy max", 0.0, 10.0, 9.9, step=0.1)
lvl3_sell_min = st.sidebar.number_input("L3 sell min", 0.0, 10.0, 0.0, step=0.1)
lvl3_sell_max = st.sidebar.number_input("L3 sell max", 0.0, 10.0, 3.5, step=0.1)

# Breadth & sweep controls
st.sidebar.header("Breadth & Sweep")
run_breadth = st.sidebar.button("Run breadth backtest (L1/L2/L3)")
run_sweep_btn = st.sidebar.button("Run grid sweep")

# Fetcher
def fetch_price(symbol: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    if YahooTicker is None:
        logger.error("yahooquery missing. Install via `pip install yahooquery`.")
        st.error("yahooquery not available in environment. Install to fetch price.")
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
        logger.exception("fetch_price failed: %s", exc)
        return pd.DataFrame()

# Chunk 2/7: features, sequences, candidate generation (fixed tp/sl logic)

def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1,w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = features.shape
    X = np.zeros((len(indices), seq_len, F), dtype=features.dtype)
    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1
        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0:t+1]])
        else:
            seq = features[t0:t+1]
        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])
        X[i] = seq[-seq_len:]
    return X

def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
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
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    # ensure high/low/close present
    for col in ("high","low","close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")
    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(atr_window, min_periods=1).mean()
    records = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr_val = float(bars["atr"].iat[i])
        if atr_val <= 0 or math.isnan(atr_val): continue
        sl_px = entry_px - k_sl*atr_val if direction=="long" else entry_px + k_sl*atr_val
        tp_px = entry_px + k_tp*atr_val if direction=="long" else entry_px - k_tp*atr_val
        end_i = min(i+max_bars, n-1)
        label, hit_i, hit_px = 0, end_i, float(bars["close"].iat[end_i])
        for j in range(i+1, end_i+1):
            hi, lo = float(bars["high"].iat[j]), float(bars["low"].iat[j])
            if direction=="long":
                if hi >= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px
                    break
                if lo <= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px
                    break
            else:
                if lo <= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px
                    break
                if hi >= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px
                    break
        end_t = bars.index[hit_i]
        ret_val = (hit_px - entry_px)/entry_px if direction=="long" else (entry_px - hit_px)/entry_px
        dur_min = (end_t - t).total_seconds()/60.0
        records.append(dict(candidate_time=t,
                            entry_price=entry_px,
                            atr=float(atr_val),
                            sl_price=float(sl_px),
                            tp_price=float(tp_px),
                            end_time=end_t,
                            label=int(label),
                            duration=float(dur_min),
                            realized_return=float(ret_val),
                            direction=direction))
    return pd.DataFrame(records)

# Chunk 3/7: simulate_limits, summarize_trades, simple XGB wrapper

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
        if segment.empty: continue
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
            pnl = (exit_px - entry_px)/entry_px * direction
        trades.append(dict(symbol=symbol,
                           entry_time=entry_t,
                           entry_price=entry_px,
                           direction=direction,
                           exit_time=exit_t,
                           exit_price=exit_px,
                           pnl=float(pnl)))
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

# Simple wrapper for XGBoost models for compatibility
class BoosterWrapper:
    def __init__(self, booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names
        self.best_iteration = getattr(booster, "best_iteration", None)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        Xp = X.reindex(columns=self.feature_names).fillna(0.0)
        if xgb is None:
            # fallback: check sklearn API
            raw = self.booster.predict_proba(Xp)[:,1]
        else:
            d = xgb.DMatrix(Xp, feature_names=self.feature_names)
            if self.best_iteration is not None:
                raw = self.booster.predict(d, iteration_range=(0, int(self.best_iteration)+1))
            else:
                raw = self.booster.predict(d)
        raw = np.clip(raw, 0.0, 1.0)
        return pd.Series(raw, index=X.index, name="confirm_proba")

    def save_model(self, path: str):
        if hasattr(self.booster, "save_model"):
            self.booster.save_model(path)
        else:
            joblib.dump(self.booster, path)

    def feature_importance(self) -> pd.DataFrame:
        if hasattr(self.booster, "get_score"):
            imp = self.booster.get_score(importance_type="gain")
            return (pd.DataFrame([(f, imp.get(f,0.0)) for f in self.feature_names], columns=["feature","gain"])
                    .sort_values("gain", ascending=False).reset_index(drop=True))
        return pd.DataFrame(columns=["feature","gain"])

# Chunk 4/7: CascadeTrader models & training helpers (Condensed but functional)

# Basic Conv and MLP blocks (L1/L2/L3)
if torch is None:
    raise RuntimeError("Torch is required for CascadeTrader. Install torch to proceed.")

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
        out = self.conv(x); out = self.bn(out); out = self.act(out); out = self.drop(out)
        if self.res: out = out + x
        return out

class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            blocks.append(ConvBlock(chs[i], chs[i+1], kernel_sizes[min(i,len(kernel_sizes)-1)], dilations[min(i,len(dilations)-1)], dropout))
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)
    @property
    def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
    def forward(self, x):
        z = self.blocks(x); z = self.project(z); z_pool = z.mean(dim=-1); logit = self.head(z_pool); return logit, z_pool

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=1, dropout=0.1):
        super().__init__(); layers=[]; last=in_dim
        for h in hidden: layers += [nn.Linear(last,h), nn.ReLU(), nn.Dropout(dropout)]; last=h
        layers += [nn.Linear(last, out_dim)]; self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Level3ShootMLP(nn.Module):
    def __init__(self, in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True):
        super().__init__()
        self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Linear(128, 1) if use_regression_head else None
    def forward(self, x):
        h = self.backbone(x); logit = self.cls_head(h); ret = self.reg_head(h) if self.reg_head is not None else None
        return logit, ret

# Training helpers (simple)
def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def train_torch_classifier(model: nn.Module, train_ds, val_ds, lr=1e-3, epochs=10, patience=3, pos_weight=1.0, device="auto"):
    dev = _device(device); model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor([pos_weight], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    tr = torch.utils.data.DataLoader(train_ds, batch_size=getattr(train_ds,"batch_size",256), shuffle=True)
    va = torch.utils.data.DataLoader(val_ds, batch_size=getattr(val_ds,"batch_size",1024), shuffle=False)
    best_loss = float("inf"); best_state=None; no_imp=0; history={"train":[],"val":[]}
    for ep in range(epochs):
        model.train(); ls_sum=0.0; n=0
        for xb,yb in tr:
            xb,yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); out = model(xb); logit = out[0] if isinstance(out,tuple) else out
            loss = bce(logit, yb); loss.backward(); opt.step()
            ls_sum += float(loss.item()) * len(xb); n += len(xb)
        train_loss = ls_sum / max(1,n)
        # val
        model.eval(); vls_sum=0.0; vn=0
        with torch.no_grad():
            for xb,yb in va:
                xb,yb = xb.to(dev), yb.to(dev)
                out = model(xb); logit = out[0] if isinstance(out,tuple) else out
                loss = bce(logit, yb)
                vls_sum += float(loss.item()) * len(xb); vn += len(xb)
        val_loss = vls_sum / max(1,vn)
        history['train'].append(train_loss); history['val'].append(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp += 1
            if no_imp >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "history": history}

# TemperatureScaler (condensed)
class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.log_temp = nn.Parameter(torch.zeros(1))
    def forward(self, logits): return logits / torch.exp(self.log_temp)
    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200, lr=1e-2):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.to(dev)
        logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=dev)
        y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=dev)
        opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        bce = nn.BCEWithLogitsLoss()
        def closure():
            opt.zero_grad(); scaled = self.forward(logits_t); loss = bce(scaled, y_t); loss.backward(); return loss
        opt.step(closure)
    def transform(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
            scaled = self.forward(logits_t).cpu().numpy()
        return scaled

# Chunk 5/7: CascadeTrader (fit, predict_batch, predict_step) + parallel training

from concurrent.futures import ThreadPoolExecutor, as_completed

class CascadeTrader:
    def __init__(self, seq_len=64, feat_windows=(5,10,20), device="auto"):
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.device = _device(device)
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.l1 = Level1ScopeCNN(in_features=5+len([f'vol_{feat_windows[0]}']))  # placeholder in_features calculation
        self.l1_temp = TemperatureScaler()
        self.l2_backend = None
        self.l2_model = None
        self.l3 = None
        self.l3_temp = TemperatureScaler()
        self.tab_feature_names = []
        self._fitted = False
        self.metadata = {}

    def fit(self, df: pd.DataFrame, events: pd.DataFrame, l2_use_xgb: bool = True):
        logger.info("CascadeTrader.fit starting")
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = ['ret1','tr', f'vol_{self.feat_windows[0]}']
        available = list(df.columns) + list(eng.columns)
        use_cols = [c for c in seq_cols + micro_cols if c in available]
        feat_seq = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1)[use_cols].fillna(0.0)
        feat_tab = eng.copy().fillna(0.0)
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        train_idx, val_idx = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=42, stratify=y)
        idx_train, idx_val = idx[train_idx], idx[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_seq_all = feat_seq.values
        self.scaler_seq.fit(X_seq_all[idx_train]); X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
        X_tab_all = feat_tab.values; self.tab_feature_names = list(feat_tab.columns)
        self.scaler_tab.fit(X_tab_all[idx_train]); X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)
        Xseq_train = to_sequences(X_seq_all_scaled, idx_train, seq_len=self.seq_len)
        Xseq_val = to_sequences(X_seq_all_scaled, idx_val, seq_len=self.seq_len)
        ds_l1_tr = type("D",(),{"__iter__":None})  # dummy to satisfy calls below if replaced; but we'll create SequenceDataset

        ds_l1_tr = __import__("torch").utils.data.DataLoader  # workaround - not used directly

        ds_tr = None
        # Build proper dataset objects
        from torch.utils.data import DataLoader as _DL
        ds_l1_train = type("Tmp",(),{})()
        ds_l1_train = __import__("__main__").SequenceDataset(Xseq_train, y_train, batch_size=128)
        ds_l1_val = __import__("__main__").SequenceDataset(Xseq_val, y_val, batch_size=256)

        # Re-init L1 with proper in_features according to feature dims
        in_features = Xseq_train.shape[2]
        self.l1 = Level1ScopeCNN(in_features=in_features, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1)
        self.l1, l1_hist = train_torch_classifier(self.l1, ds_l1_train, ds_l1_val, lr=1e-3, epochs=10, patience=3, pos_weight=1.0, device=str(self.device))
        # L1 embeddings for idx
        all_idx_seq = to_sequences(X_seq_all_scaled, idx, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(all_idx_seq)
        l1_train_emb = l1_emb[train_idx]; l1_val_emb = l1_emb[val_idx]
        Xtab_train = X_tab_all_scaled[idx_train]; Xtab_val = X_tab_all_scaled[idx_val]
        X_l2_train = np.hstack([l1_train_emb, Xtab_train]); X_l2_val = np.hstack([l1_val_emb, Xtab_val])

        # Train L2 (xgb if available + requested, else MLP) and L3 in parallel
        results = {}
        def train_l2_xgb():
            if xgb is None:
                raise RuntimeError("xgboost not available")
            clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_l2_train, y_train, eval_set=[(X_l2_val, y_val)], verbose=False)
            return ("xgb", clf)
        def train_l2_mlp():
            in_dim = X_l2_train.shape[1]
            m = MLP(in_dim, [128,64], out_dim=1, dropout=0.1)
            ds2_tr = __import__("__main__").TabDataset(X_l2_train, y_train, batch_size=256)
            ds2_va = __import__("__main__").TabDataset(X_l2_val, y_val, batch_size=512)
            m, hist = train_torch_classifier(m, ds2_tr, ds2_va, lr=1e-3, epochs=10, patience=3, pos_weight=1.0, device=str(self.device))
            return ("mlp", m)
        def train_l3():
            in_dim = X_l2_train.shape[1]
            m3 = Level3ShootMLP(in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True)
            ds3_tr = __import__("__main__").TabDataset(X_l2_train, y_train, batch_size=256)
            ds3_va = __import__("__main__").TabDataset(X_l2_val, y_val, batch_size=512)
            m3, hist3 = train_torch_classifier(m3, ds3_tr, ds3_va, lr=1e-3, epochs=12, patience=3, pos_weight=1.0, device=str(self.device))
            return ("l3", (m3, hist3))
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            if (xgb is not None) and l2_use_xgb:
                futures[ex.submit(train_l2_xgb)] = "l2"
            else:
                futures[ex.submit(train_l2_mlp)] = "l2"
            futures[ex.submit(train_l3)] = "l3"
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    logger.exception("Parallel train %s failed: %s", key, e); results[key] = None
        if results.get('l2') is not None:
            backend, model = results['l2']
            self.l2_backend = backend; self.l2_model = model
        if results.get('l3') is not None:
            self.l3, _ = results['l3'][1] if isinstance(results['l3'], tuple) else (results['l3'][1] if isinstance(results['l3'], tuple) else results['l3'])
            if isinstance(results['l3'], tuple):
                self.l3 = results['l3'][1][0]
        # calibrate temps (best-effort)
        try:
            self.l1_temp.fit(l1_logits.reshape(-1,1), y)
        except Exception as e:
            logger.warning("L1 temp scaling failed: %s", e)
        try:
            l3_val_logits = self._l3_infer_logits(X_l2_val)
            self.l3_temp.fit(l3_val_logits.reshape(-1,1), y_val)
        except Exception as e:
            logger.warning("L3 temp scaling failed: %s", e)

        self.metadata = {"l1_hist": l1_hist if 'l1_hist' in locals() else None, "fit_time": time.time()}
        self._fitted = True
        logger.info("Cascade fit completed")
        return self

    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        self.l1.eval(); logits=[]; emb=[]
        batch = 256
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32, device=self.device)
                logit, e = self.l1(xb)
                logits.append(logit.detach().cpu().numpy()); emb.append(e.detach().cpu().numpy())
        return np.vstack(logits).reshape(-1,1), np.vstack(emb)

    def _l3_infer_logits(self, X: np.ndarray):
        self.l3.eval(); logits=[]
        batch = 2048
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1)

    def _mlp_predict_proba(self, model, X: np.ndarray):
        model.eval(); probs=[]
        batch=4096
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit = model(xb)
                p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray):
        assert self._fitted, "Call fit() first"
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']; micro_cols=['ret1','tr',f'vol_{self.feat_windows[0]}']
        available = list(df.columns) + list(eng.columns)
        use_cols = [c for c in seq_cols + micro_cols if c in available]
        feat_seq = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1)[use_cols].fillna(0.0)
        feat_tab = eng[self.tab_feature_names].fillna(0.0)
        X_seq_all_scaled = self.scaler_seq.transform(feat_seq.values)
        X_tab_all_scaled = self.scaler_tab.transform(feat_tab.values)
        Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        l1_logits_scaled = self.l1_temp.transform(l1_logits.reshape(-1,1)).reshape(-1)
        p1 = 1.0/(1.0+np.exp(-l1_logits_scaled))
        go2 = p1 >= 0.3
        X_l2 = np.hstack([l1_emb, X_tab_all_scaled[t_indices]])
        if self.l2_backend == "xgb":
            try:
                p2 = self.l2_model.predict_proba(X_l2)[:,1]
            except Exception:
                p2 = np.zeros(len(X_l2))
        else:
            p2 = self._mlp_predict_proba(self.l2_model, X_l2)
        go3 = (p2 >= 0.55) & go2
        p3 = np.zeros_like(p1); rhat = np.zeros_like(p1)
        if go3.any() and self.l3 is not None:
            X_l3 = X_l2[go3]; l3_logits = self._l3_infer_logits(X_l3)
            l3_logits_scaled = self.l3_temp.transform(l3_logits.reshape(-1,1)).reshape(-1)
            p3_vals = 1.0/(1.0+np.exp(-l3_logits_scaled))
            p3[go3] = p3_vals; rhat[go3] = p3_vals - 0.5
        trade = (p3 >= 0.65) & go3
        size = np.clip(rhat, 0.0, None) * trade.astype(float)
        return pd.DataFrame({"t": t_indices, "p1": p1, "p2": p2, "p3": p3, "go2": go2.astype(int), "go3": go3.astype(int), "trade": trade.astype(int), "size": size})

# Chunk 6/7: export_model_and_metadata (user-provided logic integrated) + breadth/sweep helpers

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

# Breadth backtest logic applying level-exclusive ranges
def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame):
    """
    preds: DataFrame with p1/p2/p3 and 't' indices aligned to candidate rows
    cands: original candidates (must have candidate_time)
    returns dict of trades per level and summaries
    """
    out = {"detailed": {}, "summary": []}
    # Build a signal from pred_prob or p3 — we map p3*10 -> signal scale 0-10
    df = cands.copy().reset_index(drop=True)
    # align preds by index t -> candidate_time index mapping:
    # preds['t'] contains index positions; map preds.t -> candidate_time in df rows order
    # We'll assume preds rows correspond to df indices; otherwise align by time
    # create a 'signal' scaled 0-10 from p3 for selection
    preds_indexed = preds.set_index('t')
    df['signal'] = (preds_indexed.reindex(df.index)['p3'].fillna(0.0) * 10).values

    # apply exclusivity ranges per level
    def filter_and_sim(level_name, buy_min, buy_max, sell_min, sell_max, sl_pct, rr):
        sel = df[(df['signal'] >= buy_min) & (df['signal'] <= buy_max)]
        # remove those that fall into higher-level ranges by being exclusive — caller must pass non-overlapping ranges
        sel = sel.copy()
        tp_pct = rr * sl_pct
        sel['pred_label'] = 0
        sel.loc[sel['signal'] >= buy_min, 'pred_label'] = 1
        trades = simulate_limits(sel, bars, label_col='pred_label', sl=sl_pct, tp=tp_pct, max_holding=60)
        out['detailed'][level_name] = trades
        s = summarize_trades(trades)
        if not s.empty:
            row = s.iloc[0].to_dict(); row['mode'] = level_name; out['summary'].append(row)
        return trades

    # L1
    trade_l1 = filter_and_sim("L1", lvl1_buy_min, lvl1_buy_max, lvl1_sell_min, lvl1_sell_max, sl_pct= (lvl1_sl_min+lvl1_sl_max)/2.0 if 'lvl1_sl_min' in globals() else 0.02, rr=(lvl1_rr_min+lvl1_rr_max)/2.0 if 'lvl1_rr_min' in globals() else 2.0)
    trade_l2 = filter_and_sim("L2", lvl2_buy_min, lvl2_buy_max, lvl2_sell_min, lvl2_sell_max, sl_pct= (lvl2_sl_min+lvl2_sl_max)/2.0, rr=(lvl2_rr_min+lvl2_rr_max)/2.0)
    trade_l3 = filter_and_sim("L3", lvl3_buy_min, lvl3_buy_max, lvl3_sell_min, lvl3_sell_max, sl_pct= (lvl3_sl_min+lvl3_sl_max)/2.0, rr=(lvl3_rr_min+lvl3_rr_max)/2.0)
    out['summary'] = out['summary'] or []
    return out

# Chunk 7/7: Streamlit actions: run pipeline, fit cascade, breadth/sweep, export models

st.sidebar.header("Run")
run_fetch = st.sidebar.button("Fetch & preview data")
train_cascade_btn = st.sidebar.button("Train Cascade (L1/L2/L3)")
export_models_btn = st.sidebar.button("Export trained models (.pt/.model)")
run_backtest_btn = st.sidebar.button("Run quick backtest using cascade preds")

# global container for trained trader across reruns (not persistent between processes)
if "trader" not in st.session_state:
    st.session_state.trader = None
if "bars" not in st.session_state:
    st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state:
    st.session_state.cands = pd.DataFrame()

# Fetch data
if run_fetch:
    st.info(f"Fetching price data for {symbol} …")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars.empty:
        st.error("No price data returned.")
    else:
        st.session_state.bars = bars
        st.success(f"Fetched {len(bars)} bars.")
        st.dataframe(bars.tail(20))

# Prepare candidates button (helper)
if st.button("Generate candidates (k_tp=2,k_sl=1)"):
    bars = st.session_state.bars
    if bars is None or bars.empty:
        st.error("Fetch price data first.")
    else:
        bars = bars.copy()
        bars["rvol"] = (bars["volume"] / bars["volume"].rolling(20, min_periods=1).mean()).fillna(1.0)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        if cands.empty:
            st.warning("No candidates generated.")
        else:
            # attach a simple pred_prob placeholder (we will replace after training)
            cands["pred_prob"] = 0.0
            st.session_state.cands = cands
            st.success(f"Generated {len(cands)} candidates.")
            st.dataframe(cands.head(50))

# Train cascade
if train_cascade_btn:
    bars = st.session_state.bars
    cands = st.session_state.cands
    if bars is None or bars.empty:
        st.error("Fetch price data first.")
    elif cands is None or cands.empty:
        st.error("Generate candidates first.")
    else:
        # Build events for cascade.fit: events are candidates indices with label=original label
        # Map candidate_time -> integer index within bars for sequence builder
        bars = ensure_unique_index(bars) if 'ensure_unique_index' in globals() else bars
        # Ensure index alignment
        cand_idx = []
        # map candidate_time to integer positions
        bar_idx_map = {t: i for i, t in enumerate(bars.index)}
        for t in cands["candidate_time"]:
            t0 = pd.Timestamp(t)
            # find closest exact match; if not present, set to nearest previous
            if t0 in bar_idx_map:
                cand_idx.append(bar_idx_map[t0])
            else:
                # find previous index
                locs = bars.index[bars.index <= t0]
                if len(locs) == 0:
                    cand_idx.append(0)
                else:
                    cand_idx.append(int(bar_idx_map[locs[-1]]))
        events = pd.DataFrame({"t": np.array(cand_idx, dtype=int), "y": cands["label"].astype(int).values})
        st.info("Training Cascade (this may take a while)… logs in server console")
        trader = CascadeTrader(seq_len=64, feat_windows=(5,10,20), device="auto")
        try:
            trader.fit(bars, events, l2_use_xgb=(xgb is not None))
            st.session_state.trader = trader
            st.success("Cascade trained and ready.")
        except Exception as e:
            logger.exception("Cascade training failed: %s", e)
            st.error(f"Training failed: {e}")

# Backtest using trained cascade predictions
if run_backtest_btn:
    trader = st.session_state.trader
    bars = st.session_state.bars
    cands = st.session_state.cands
    if trader is None:
        st.error("Train the cascade first.")
    elif bars is None or bars.empty or cands is None or cands.empty:
        st.error("Ensure bars and candidates exist.")
    else:
        # compute preds for candidate indices
        # construct t_indices same as in training mapping
        bar_idx_map = {t: i for i, t in enumerate(bars.index)}
        t_indices = []
        for t in cands["candidate_time"]:
            t0 = pd.Timestamp(t)
            if t0 in bar_idx_map:
                t_indices.append(bar_idx_map[t0])
            else:
                locs = bars.index[bars.index <= t0]
                t_indices.append(int(bar_idx_map[locs[-1]] if len(locs)>0 else 0))
        t_indices = np.array(t_indices, dtype=int)
        preds = trader.predict_batch(bars, t_indices)
        st.write("Predictions head:")
        st.dataframe(preds.head(20))
        # Map p3 into cands.signal and run breadth/exclusive samplings
        # We'll reuse run_breadth_levels
        res = run_breadth_levels(preds, cands, bars)
        st.subheader("Breadth Summary")
        if res["summary"]:
            st.dataframe(pd.DataFrame(res["summary"]))
        else:
            st.warning("No trades in breadth run.")
        for lvl, df in res["detailed"].items():
            st.subheader(f"{lvl} — {len(df)} trades")
            if not df.empty:
                st.dataframe(df.head(50))
                s = summarize_trades(df)
                st.write(s)

# Export models (train artifacts)
if export_models_btn:
    trader = st.session_state.trader
    if trader is None:
        st.error("Train the cascade first.")
    else:
        out_dir = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        os.makedirs(out_dir, exist_ok=True)
        # Use export_artifacts helper embedded in CascadeTrader earlier?
        try:
            # export stateful artifacts
            export_artifacts = globals().get("export_artifacts", None)
            if export_artifacts:
                export_artifacts(trader, out_dir)
                st.success(f"Saved artifacts to {out_dir}")
            # also save a .pt bundle using the robust function for the L3 model wrapper
            # create a simple wrapper object to pass to export_model_and_metadata
            model_wrapper = type("MW", (), {})()
            # attach booster if a xgb model present
            if trader.l2_backend == "xgb":
                model_wrapper.booster = trader.l2_model.get_booster() if hasattr(trader.l2_model, "get_booster") else trader.l2_model
            else:
                model_wrapper.booster = None
            # attach feature_importance if possible (not necessary)
            if hasattr(trader, "l1"):
                def fi_stub(): return pd.DataFrame([{"feature":"l1_emb","gain":1.0}])
                model_wrapper.feature_importance = fi_stub
            paths = export_model_and_metadata(model_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time")}, os.path.join(out_dir, "cascade_model"), save_fi=True)
            st.write("Export paths:", paths)
        except Exception as e:
            logger.exception("Export failed: %s", e)
            st.error(f"Export failed: {e}")

# Grid sweep (simple)
if run_sweep_btn:
    st.info("Running light grid sweep across RR x SL x MPT using current cands")
    bars = st.session_state.bars
    cands = st.session_state.cands
    trader = st.session_state.trader
    if bars is None or bars.empty or cands is None or cands.empty:
        st.error("Fetch data and generate candidates first.")
    else:
        # form rr candidates from UI level min/max values
        rr_vals = sorted(list(set([float(round(x,2)) for x in [lvl1_rr_min if 'lvl1_rr_min' in globals() else 2.0, lvl2_rr_min if 'lvl2_rr_min' in globals() else 2.5, lvl3_rr_min if 'lvl3_rr_min' in globals() else 3.0]])))
        sl_ranges = [(lvl1_sl_min if 'lvl1_sl_min' in globals() else 0.02, lvl1_sl_max if 'lvl1_sl_max' in globals() else 0.04),
                     (lvl2_sl_min if 'lvl2_sl_min' in globals() else 0.01, lvl2_sl_max if 'lvl2_sl_max' in globals() else 0.03),
                     (lvl3_sl_min if 'lvl3_sl_min' in globals() else 0.005, lvl3_sl_max if 'lvl3_sl_max' in globals() else 0.02)]
        mpt_list = [p_fast, p_slow, p_deep]
        # produce preds if trader exists else simulate with pred_prob=0
        bar_idx_map = {t: i for i, t in enumerate(bars.index)}
        t_indices = []
        for t in cands["candidate_time"]:
            t0 = pd.Timestamp(t)
            if t0 in bar_idx_map:
                t_indices.append(bar_idx_map[t0])
            else:
                locs = bars.index[bars.index <= t0]
                t_indices.append(int(bar_idx_map[locs[-1]] if len(locs) else 0))
        t_indices = np.array(t_indices, dtype=int)
        if trader is None:
            # no trader, use cands.pred_prob or zeros
            cands["pred_prob"] = cands.get("pred_prob", 0.0)
            df_for_sweep = cands.copy()
        else:
            preds = trader.predict_batch(bars, t_indices)
            df_for_sweep = cands.copy().reset_index(drop=True)
            preds_indexed = preds.set_index('t')
            df_for_sweep['pred_prob'] = preds_indexed.reindex(df_for_sweep.index)['p3'].fillna(0.0).values

        sweep_results = {}
        for rr in rr_vals:
            for sl_min, sl_max in sl_ranges:
                sl_pct = (sl_min + sl_max)/2.0
                tp_pct = rr * sl_pct
                for mpt in mpt_list:
                    key = f"rr{rr:.2f}_sl{sl_min:.3f}-{sl_max:.3f}_mpt{mpt:.2f}"
                    try:
                        df_tmp = df_for_sweep.copy()
                        df_tmp["pred_label"] = (df_tmp["pred_prob"] >= mpt).astype(int)
                        trades = simulate_limits(df_tmp, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
                        sweep_results[key] = {"trades_count": len(trades), "trades": trades}
                    except Exception as e:
                        logger.exception("Sweep config %s failed: %s", key, e)
                        sweep_results[key] = {"error": str(e)}
        # summarize top
        summary_rows = []
        for k, v in sweep_results.items():
            if isinstance(v, dict) and "trades" in v and not v["trades"].empty:
                s = summarize_trades(v["trades"])
                if not s.empty:
                    r = s.iloc[0].to_dict(); r.update({"config": k, "trades_count": v.get("trades_count", 0)}); summary_rows.append(r)
        if summary_rows:
            sdf = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
            st.subheader("Sweep summary (top configs)")
            st.dataframe(sdf.head(20))
        else:
            st.warning("No sweep configs produced trades.")