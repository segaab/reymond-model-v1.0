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
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ML / metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Optional libs (graceful degradation)
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cascade_trader_app")
logger.setLevel(logging.INFO)

# Streamlit UI
st.set_page_config(page_title="Cascade Trader — Scope→Aim→Shoot", layout="wide")
st.title("Cascade Trader — Scope → Aim → Shoot (L1/L2/L3)")

# Main panel inputs
with st.sidebar:
    st.header("Training & Model")
    num_boost = int(st.number_input("XGBoost rounds (if used)", min_value=1, value=200))
    early_stop = int(st.number_input("Early stopping rounds", min_value=1, value=20))
    test_size = float(st.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.20))

    st.markdown("---")
    st.header("L1 / Seq + Device")
    l1_seq_len = int(st.number_input("L1 sequence length", min_value=8, max_value=1024, value=64))
    device_opt = st.selectbox("Device", ["auto", "cpu", "cuda"] if torch is not None else ["cpu"], index=0)

    st.markdown("---")
    st.header("Quick sim thresholds")
    p_fast = st.number_input("Confirm threshold (fast)", min_value=0.0, max_value=1.0, value=0.60)
    p_slow = st.number_input("Confirm threshold (slow)", min_value=0.0, max_value=1.0, value=0.55)
    p_deep = st.number_input("Confirm threshold (deep)", min_value=0.0, max_value=1.0, value=0.45)

    st.markdown("---")
    st.header("Level gating (exclusive ranges) — signal scale 0..10")
    st.markdown("Define explicit buy/sell ranges per level. Ranges must be exclusive (L3 inside L2 inside L1).")

    l1_buy_min = st.number_input("L1 buy min", 0.0, 10.0, 5.5, step=0.1)
    l1_buy_max = st.number_input("L1 buy max", 0.0, 10.0, 9.9, step=0.1)
    l1_sell_min = st.number_input("L1 sell min", 0.0, 10.0, 0.0, step=0.1)
    l1_sell_max = st.number_input("L1 sell max", 0.0, 10.0, 4.5, step=0.1)

    l2_buy_min = st.number_input("L2 buy min", 0.0, 10.0, 6.0, step=0.1)
    l2_buy_max = st.number_input("L2 buy max", 0.0, 10.0, 9.9, step=0.1)
    l2_sell_min = st.number_input("L2 sell min", 0.0, 10.0, 0.0, step=0.1)
    l2_sell_max = st.number_input("L2 sell max", 0.0, 10.0, 4.0, step=0.1)

    l3_buy_min = st.number_input("L3 buy min", 0.0, 10.0, 6.5, step=0.1)
    l3_buy_max = st.number_input("L3 buy max", 0.0, 10.0, 9.9, step=0.1)
    l3_sell_min = st.number_input("L3 sell min", 0.0, 10.0, 0.0, step=0.1)
    l3_sell_max = st.number_input("L3 sell max", 0.0, 10.0, 3.5, step=0.1)

    st.markdown("---")
    st.header("Breadth & Sweep")
    run_breadth = st.button("Run breadth backtest (3 levels)")
    run_sweep_btn = st.button("Run grid sweep (light)")

# main inputs
symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=480))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

run_pipeline = st.button("Run full pipeline (fetch → train → sweep → export)")

# Helper: enforce exclusivity of ranges (L3 ⊂ L2 ⊂ L1)
def ranges_valid() -> Tuple[bool, str]:
    try:
        # buy side
        if not (l1_buy_min <= l2_buy_min <= l3_buy_min and l3_buy_max <= l2_buy_max <= l1_buy_max):
            return False, "Buy-side ranges must be nested: L3 inside L2 inside L1 (min increasing, max decreasing permitted)."
        # sell side: smaller sell_max for deeper levels
        if not (l1_sell_min <= l2_sell_min <= l3_sell_min and l3_sell_max <= l2_sell_max <= l1_sell_max):
            return False, "Sell-side ranges must be nested: L3 inside L2 inside L1."
        return True, ""
    except Exception as e:
        return False, f"Range validation error: {e}"

valid_ranges, vr_msg = ranges_valid()
if not valid_ranges:
    st.warning("Level ranges invalid: " + vr_msg)

# Fetch price function using yahooquery
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
        logger.exception("fetch_price failed: %s", exc)
        return pd.DataFrame()

# Chunk 2/7: features + labeling + rvol + health

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if df is None or "volume" not in df.columns:
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

# true range
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
            "direction": direction
        })
    return pd.DataFrame(recs)

# Engineered features used by cascade (we will reuse this function in CascadeTrader as well)
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df['volume'].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)
    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean().fillna(0.0) - v.rolling(max(1, w*3)).mean().fillna(0.0)).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        f[f'kurt_{w}'] = ret1.rolling(w).kurt().fillna(0.0).replace([np.inf,-np.inf],0.0)
        f[f'skew_{w}'] = ret1.rolling(w).skew().fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    f = f.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return f

# Chunk 2/7: features + labeling + rvol + health

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if df is None or "volume" not in df.columns:
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

# true range
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
            "direction": direction
        })
    return pd.DataFrame(recs)

# Engineered features used by cascade (we will reuse this function in CascadeTrader as well)
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df['volume'].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)
    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean().fillna(0.0) - v.rolling(max(1, w*3)).mean().fillna(0.0)).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        f[f'kurt_{w}'] = ret1.rolling(w).kurt().fillna(0.0).replace([np.inf,-np.inf],0.0)
        f[f'skew_{w}'] = ret1.rolling(w).skew().fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    f = f.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return f

# Chunk 3/7: Models, Datasets, Temperature scaler (Cascade core)

if torch is None:
    st.warning("PyTorch not available; cascade CNN/L3 training will fail. Please install torch if you want to train deep models.")

# Datasets
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = X_seq.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].transpose(1, 0)  # [T,F] -> [F,T]
        y = self.y[idx]
        return x, y

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Small NN blocks
if torch is not None:
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
        def __init__(self, in_features: int = 12, channels: List[int] = (32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                k = kernel_sizes[i] if i < len(kernel_sizes) else 3
                d = dilations[i] if i < len(dilations) else 1
                blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
            self.blocks = nn.Sequential(*blocks)
            self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)
        @property
        def embedding_dim(self):
            return self.blocks[-1].conv.out_channels
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
        def __init__(self, in_dim: int, hidden=(128,64), dropout=0.1, use_regression_head=True):
            super().__init__()
            self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
            self.cls_head = nn.Linear(128, 1)
            self.reg_head = nn.Linear(128, 1) if use_regression_head else None
        def forward(self, x):
            h = self.backbone(x)
            logit = self.cls_head(h)
            ret = self.reg_head(h) if self.reg_head is not None else None
            return logit, ret

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = nn.Parameter(torch.zeros(1))
        def forward(self, logits):
            T = torch.exp(self.log_temp)
            return logits / T
        def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200, lr=1e-2):
            if torch is None:
                return
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
            if torch is None:
                return logits
            with torch.no_grad():
                device = next(self.parameters()).device
                logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
                scaled = self.forward(logits_t).cpu().numpy()
            return scaled
else:
    # placeholders if torch not available
    Level1ScopeCNN = None
    Level3ShootMLP = None
    TemperatureScaler = None
    MLP = None

# Chunk 4/7: training helpers + CascadeTrader skeleton + threaded L2/L3 train

from sklearn.preprocessing import StandardScaler

def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    return name

def train_torch_classifier(model: nn.Module,
                           train_ds: Dataset,
                           val_ds: Dataset,
                           lr: float,
                           epochs: int,
                           patience: int,
                           pos_weight: float = 1.0,
                           device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("torch not installed")
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
    train_loader = DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 256), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(xb); n += len(xb)
        train_loss = loss_sum / max(n, 1)
        model.eval()
        vloss_sum, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                vloss_sum += float(loss.item()) * len(xb); vn += len(xb)
        val_loss = vloss_sum / max(vn, 1)
        history['train'].append(train_loss); history['val'].append(val_loss)
        if val_loss + 1e-8 < best_loss:
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

def train_tab_mlp(model: nn.Module,
                  train_ds: Dataset,
                  val_ds: Dataset,
                  lr: float,
                  epochs: int,
                  patience: int,
                  device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    return train_torch_classifier(model, train_ds, val_ds, lr, epochs, patience, pos_weight=1.0, device=device)

# CascadeTrader class (core)
class CascadeTrader:
    def __init__(self, device="auto", seq_len=64, feature_windows=(5,10,20)):
        self.device = _device(device)
        self.seq_len = seq_len
        self.feature_windows = feature_windows
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.l1_model = None
        self.l2_model = None
        self.l2_backend = None
        self.l3_model = None
        self.l1_temp = TemperatureScaler() if torch is not None else None
        self.l3_temp = TemperatureScaler() if torch is not None else None
        self.tab_feature_names: List[str] = []
        self.embed_dim = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, events: pd.DataFrame, l1_cfg: Dict=None, l2_cfg: Dict=None, l3_cfg: Dict=None):
        """
        Fit L1 -> produce embeddings -> fit L2 & L3 (L2/L3 training runs in threads after embeddings produced)
        events: DataFrame with columns 't' (int index) and 'y' (0/1)
        """
        t0 = time.time()
        logger.info("Starting cascade fit")
        eng = compute_engineered_features(df, windows=self.feature_windows)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = ['ret1','tr','vol_5','vol_10','mom_5','chanpos_10','chanpos_20']
        use_cols = [c for c in seq_cols + micro_cols if c in (list(df.columns) + list(eng.columns))]
        feat_seq = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[use_cols].replace([np.inf,-np.inf],0.0).fillna(0.0)
        feat_tab = eng.copy()
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        # train/val split (time-aware split recommended in prod)
        train_idx, val_idx = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=42, stratify=y)
        idx_train, idx_val = idx[train_idx], idx[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # scalers
        X_seq_all = feat_seq.values
        self.scaler_seq.fit(X_seq_all)
        X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
        X_tab_all = feat_tab.values
        self.tab_feature_names = list(feat_tab.columns)
        self.scaler_tab.fit(X_tab_all)
        X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)
        # build seq arrays
        def to_sequences_local(features, indices, seq_len):
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
        Xseq_train = to_sequences_local(X_seq_all_scaled, idx_train, seq_len=self.seq_len)
        Xseq_val = to_sequences_local(X_seq_all_scaled, idx_val, seq_len=self.seq_len)
        ds_l1_tr = SequenceDataset(Xseq_train, y_train)
        ds_l1_va = SequenceDataset(Xseq_val, y_val)
        ds_l1_tr.batch_size = 256
        # init L1
        in_f = Xseq_train.shape[2]
        l1_channels = (32, 64, 128)
        self.l1_model = Level1ScopeCNN(in_features=in_f, channels=l1_channels, kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1)
        device_str = self.device
        logger.info("Training L1 CNN on device %s", device_str)
        self.l1_model, l1_hist = train_torch_classifier(self.l1_model, ds_l1_tr, ds_l1_va, lr=1e-3, epochs=10, patience=3, pos_weight=1.0, device=device_str)
        logger.info("L1 trained. Extracting embeddings.")
        # get embeddings for all indices
        l1_train_logits, l1_train_emb = self._l1_infer_logits_emb(Xseq_train)
        l1_val_logits, l1_val_emb = self._l1_infer_logits_emb(Xseq_val)
        self.embed_dim = l1_train_emb.shape[1]
        # build tabular matrices
        Xtab_train = X_tab_all_scaled[idx_train]
        Xtab_val = X_tab_all_scaled[idx_val]
        X_l2_train = np.hstack([l1_train_emb, Xtab_train])
        X_l2_val = np.hstack([l1_val_emb, Xtab_val])
        # Train L2 and L3 in separate threads (they don't modify L1)
        def train_l2_thread():
            try:
                if XGBOOST_AVAILABLE:
                    logger.info("Training L2 with XGBoost")
                    self.l2_backend = "xgb"
                    self.l2_model = xgb.XGBClassifier(n_estimators=num_boost, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
                    self.l2_model.fit(X_l2_train, y_train, eval_set=[(X_l2_val, y_val)], verbose=False)
                else:
                    logger.info("Training L2 with MLP fallback")
                    self.l2_backend = "mlp"
                    in_dim = X_l2_train.shape[1]
                    mlp = MLP(in_dim, [128,64], out_dim=1, dropout=0.1)
                    ds2_tr = TabDataset(X_l2_train, y_train)
                    ds2_va = TabDataset(X_l2_val, y_val)
                    self.l2_model, _ = train_tab_mlp(mlp, ds2_tr, ds2_va, lr=1e-3, epochs=10, patience=3, device=device_str)
                logger.info("L2 training finished.")
            except Exception as e:
                logger.exception("L2 training error: %s", e)
        def train_l3_thread():
            try:
                logger.info("Training L3 executor")
                in_dim = X_l2_train.shape[1]
                self.l3_model = Level3ShootMLP(in_dim, hidden=(128,64), dropout=0.1, use_regression_head=False)
                ds3_tr = TabDataset(X_l2_train, y_train)
                ds3_va = TabDataset(X_l2_val, y_val)
                self.l3_model, _ = train_torch_classifier(self.l3_model, ds3_tr, ds3_va, lr=1e-3, epochs=10, patience=3, pos_weight=1.0, device=device_str)
                logger.info("L3 training finished.")
            except Exception as e:
                logger.exception("L3 training error: %s", e)
        t2 = threading.Thread(target=train_l2_thread, daemon=True)
        t3 = threading.Thread(target=train_l3_thread, daemon=True)
        t2.start(); t3.start()
        # join threads
        t2.join(); t3.join()
        # calibrate temps (if available)
        try:
            if self.l1_temp is not None:
                self.l1_temp.fit(l1_val_logits, y_val)
            if self.l3_temp is not None:
                l3_val_logits = self._l3_infer_logits(X_l2_val)
                self.l3_temp.fit(l3_val_logits, y_val)
        except Exception:
            logger.exception("Temperature calibration failed")
        self.fitted = True
        self.metadata = {"fit_time_sec": round(time.time() - t0, 2)}
        logger.info("Cascade fit complete in %.2f sec", time.time() - t0)
        return self

    # internal helper: L1 infer
    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        if torch is None:
            raise RuntimeError("torch required")
        self.l1_model.eval()
        dev = torch.device(self.device)
        logits, embeds = [], []
        with torch.no_grad():
            for i in range(0, len(Xseq), 512):
                xb = torch.tensor(Xseq[i:i+512].transpose(0,2,1), dtype=torch.float32, device=dev)
                out = self.l1_model(xb)
                logit = out[0].detach().cpu().numpy()
                emb = out[1].detach().cpu().numpy()
                logits.append(logit)
                embeds.append(emb)
        logits = np.concatenate(logits, axis=0).reshape(-1,1)
        embeds = np.concatenate(embeds, axis=0)
        return logits, embeds

    def _l3_infer_logits(self, X: np.ndarray):
        if torch is None:
            raise RuntimeError("torch required")
        self.l3_model.eval()
        dev = torch.device(self.device)
        logits = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.tensor(X[i:i+2048], dtype=torch.float32, device=dev)
                out = self.l3_model(xb)
                logit = out[0].detach().cpu().numpy()
                logits.append(logit)
        return np.concatenate(logits, axis=0).reshape(-1,1)

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray, gate_cfg: Dict = None):
        assert self.fitted, "Cascade not fitted"
        eng = compute_engineered_features(df, windows=self.feature_windows)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = ['ret1','tr','vol_5','vol_10','mom_5','chanpos_10','chanpos_20']
        use_cols = [c for c in seq_cols + micro_cols if c in (list(df.columns) + list(eng.columns))]
        feat_seq = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[use_cols].replace([np.inf,-np.inf],0.0).fillna(0.0)
        feat_tab = eng[self.tab_feature_names].replace([np.inf,-np.inf],0.0).fillna(0.0)
        X_seq_scaled = self.scaler_seq.transform(feat_seq.values)
        X_tab_scaled = self.scaler_tab.transform(feat_tab.values)
        # sequences
        def to_sequences_local(features, indices, seq_len):
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
        Xseq = to_sequences_local(X_seq_scaled, t_indices, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        try:
            l1_logits_scaled = self.l1_temp.transform(l1_logits)
        except Exception:
            l1_logits_scaled = l1_logits
        p1 = 1.0 / (1.0 + np.exp(-l1_logits_scaled)).reshape(-1)
        go2 = p1 >= (gate_cfg.get("th1", 0.3) if gate_cfg else 0.3)
        X_l2 = np.hstack([l1_emb, X_tab_scaled[t_indices]])
        if self.l2_backend == "xgb":
            p2 = self.l2_model.predict_proba(X_l2)[:,1]
        else:
            p2 = self._mlp_predict_proba(self.l2_model, X_l2)
        go3 = (p2 >= (gate_cfg.get("th2", 0.55) if gate_cfg else 0.55)) & go2
        p3 = np.zeros_like(p1); rhat = np.zeros_like(p1)
        if go3.any():
            X_l3 = X_l2[go3]
            l3_logits = self._l3_infer_logits(X_l3)
            try:
                l3_logits_scaled = self.l3_temp.transform(l3_logits)
            except Exception:
                l3_logits_scaled = l3_logits
            p3_vals = 1.0 / (1.0 + np.exp(-l3_logits_scaled)).reshape(-1)
            p3[go3] = p3_vals
            rhat_vals = p3_vals - 0.5
            rhat[go3] = rhat_vals
        trade = (p3 >= (gate_cfg.get("th3", 0.65) if gate_cfg else 0.65)) & go3
        size = np.clip(rhat, 0, None) * trade.astype(float)
        out = pd.DataFrame({
            "t": t_indices,
            "p1": p1, "p2": p2, "p3": p3,
            "go2": go2.astype(int), "go3": go3.astype(int),
            "trade": trade.astype(int), "size": size
        })
        return out

    def _mlp_predict_proba(self, model, X: np.ndarray):
        if isinstance(model, nn.Module):
            model.eval()
            dev = torch.device(self.device)
            probs = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    xb = torch.tensor(X[i:i+4096], dtype=torch.float32, device=dev)
                    logit = model(xb)
                    p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                    probs.append(p)
            return np.concatenate(probs, axis=0)
        else:
            # fallback: xgboost or sklearn-like
            try:
                return model.predict_proba(X)[:,1]
            except Exception:
                return np.zeros(X.shape[0])


# Chunk 5/7: backtest simulate_limits + summarize + breadth & sweep functions

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
    bars = bars.copy(); bars.index = pd.to_datetime(bars.index)
    for _, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl):
            continue
        entry_t = pd.to_datetime(row.get("candidate_time", row.get("t")))
        if entry_t not in bars.index:
            # try align by nearest index
            try:
                entry_t = bars.index[bars.index.get_indexer([entry_t], method="nearest")[0]]
            except Exception:
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
    if trades:
        return pd.DataFrame(trades)
    return pd.DataFrame()

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

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out
    # Build signal from pred_prob if present, else use scaling of realized_return as proxy (for demo)
    df = clean.copy()
    if "signal" not in df.columns:
        if "pred_prob" in df.columns:
            df["signal"] = (df["pred_prob"] * 10.0).round(2)
        else:
            df["signal"] = (np.clip(df.get("realized_return", 0.0) * 100.0, 0.0, 10.0)).round(2)
    for lvl_name, cfg in levels_config.items():
        try:
            buy_min, buy_max = cfg["buy_min"], cfg["buy_max"]
            sell_min, sell_max = cfg["sell_min"], cfg["sell_max"]
            rr_min, rr_max = cfg["rr_min"], cfg["rr_max"]
            sl_min, sl_max = cfg["sl_min"], cfg["sl_max"]
            dd = df.copy()
            # level-exclusive filter: only signals within this window
            dd["pred_label"] = 0
            mask_buy = (dd["signal"] >= buy_min) & (dd["signal"] <= buy_max)
            mask_sell = (dd["signal"] >= sell_min) & (dd["signal"] <= sell_max)
            dd.loc[mask_buy, "pred_label"] = 1
            dd.loc[mask_sell, "pred_label"] = -1
            sl_pct = float((sl_min + sl_max) / 2.0)
            rr = float((rr_min + rr_max) / 2.0)
            tp_pct = rr * sl_pct
            trades = simulate_limits(dd, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
            out["detailed_trades"][lvl_name] = trades
            summary = summarize_trades(trades)
            if not summary.empty:
                s = summary.iloc[0].to_dict()
                s["mode"] = lvl_name
                out["summary"].append(s)
            out["diagnostics"].append(f"{lvl_name}: simulated {len(trades)} trades, sl={sl_pct:.4f}, tp={tp_pct:.4f}")
        except Exception as e:
            logger.exception("Breadth level %s failed: %s", lvl_name, e)
            out["diagnostics"].append(f"{lvl_name} error: {e}")
    return out

def run_grid_sweep(clean: pd.DataFrame, bars: pd.DataFrame, rr_vals: List[float], sl_ranges: List[Tuple[float,float]], mpt_list: List[float]) -> Dict[str, Any]:
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
                    trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
                    results[key] = {"trades_count": len(trades), "overlay": trades}
                except Exception as e:
                    logger.exception("Sweep config %s failed: %s", key, e)
                    results[key] = {"error": str(e)}
    return results

# Chunk 6/7: export_model_and_metadata (serializable) + helpers

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def export_model_and_metadata_serializable(l1_model, l2_model, l3_model, cascade: CascadeTrader, feature_list: List[str], metrics: Dict[str,Any], model_basename: str, save_fi: bool = True):
    """
    Save model artifacts and a portable .pt bundle for each model wrapper.
    We save serializable pieces: torch state_dicts, xgb model file if present, scalers via joblib, metadata json.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = f"{model_basename}_{ts}"
    os.makedirs(outdir, exist_ok=True)
    result_paths = {}
    try:
        # L1 state dict
        if l1_model is not None and torch is not None:
            try:
                p = os.path.join(outdir, f"l1_state_{ts}.pt")
                torch.save(l1_model.state_dict(), p)
                result_paths["l1_state"] = p
            except Exception as e:
                result_paths["l1_state_error"] = str(e) + "\n" + traceback.format_exc()

        # L2: if xgboost booster, ask booster to save native model
        if l2_model is not None:
            try:
                if XGBOOST_AVAILABLE and hasattr(l2_model, "save_model"):
                    l2_path = os.path.join(outdir, f"l2_xgb_{ts}.json")
                    l2_model.save_model(l2_path)
                    result_paths["l2_model"] = l2_path
                else:
                    # If it's a torch module, save state_dict
                    if torch is not None and isinstance(l2_model, nn.Module):
                        p = os.path.join(outdir, f"l2_state_{ts}.pt")
                        torch.save(l2_model.state_dict(), p)
                        result_paths["l2_state"] = p
                    else:
                        # fallback: try sklearn-like picklable model via joblib
                        p = os.path.join(outdir, f"l2_joblib_{ts}.joblib")
                        try:
                            joblib.dump(l2_model, p)
                            result_paths["l2_joblib"] = p
                        except Exception as dump_e:
                            result_paths["l2_joblib_error"] = str(dump_e)
            except Exception as e:
                result_paths["l2_error"] = str(e) + "\n" + traceback.format_exc()

        # L3
        if l3_model is not None and torch is not None:
            try:
                p = os.path.join(outdir, f"l3_state_{ts}.pt")
                torch.save(l3_model.state_dict(), p)
                result_paths["l3_state"] = p
            except Exception as e:
                result_paths["l3_state_error"] = str(e) + "\n" + traceback.format_exc()

        # scalers
        try:
            scaler_seq_p = os.path.join(outdir, f"scaler_seq_{ts}.joblib")
            scaler_tab_p = os.path.join(outdir, f"scaler_tab_{ts}.joblib")
            joblib.dump(cascade.scaler_seq, scaler_seq_p)
            joblib.dump(cascade.scaler_tab, scaler_tab_p)
            result_paths["scaler_seq"] = scaler_seq_p
            result_paths["scaler_tab"] = scaler_tab_p
        except Exception as e:
            result_paths["scalers_error"] = str(e) + "\n" + traceback.format_exc()

        # metadata
        try:
            meta = {
                "features": feature_list,
                "metrics": metrics,
                "saved_at": ts,
                "cascade_metadata": getattr(cascade, "metadata", {}),
                "tab_features": getattr(cascade, "tab_feature_names", []),
                "embed_dim": getattr(cascade, "embed_dim", None)
            }
            meta_file = os.path.join(outdir, f"{model_basename}_{ts}.json")
            with open(meta_file, "w") as f:
                json.dump(meta, f, default=str, indent=2)
            result_paths["meta"] = meta_file
        except Exception as e:
            result_paths["meta_error"] = str(e) + "\n" + traceback.format_exc()

        # create portable .pt bundle (dictionary of state_dicts + scalers path + meta)
        try:
            pt_payload = {"meta": meta, "l1_state": None, "l2_type": None, "l2_path": None, "l3_state": None}
            if "l1_state" in result_paths:
                pt_payload["l1_state"] = result_paths["l1_state"]
            if "l2_model" in result_paths:
                pt_payload["l2_type"] = "xgb"
                pt_payload["l2_path"] = result_paths["l2_model"]
            elif "l2_state" in result_paths:
                pt_payload["l2_type"] = "torch"
                pt_payload["l2_path"] = result_paths["l2_state"]
            elif "l2_joblib" in result_paths:
                pt_payload["l2_type"] = "joblib"
                pt_payload["l2_path"] = result_paths["l2_joblib"]
            if "l3_state" in result_paths:
                pt_payload["l3_state"] = result_paths["l3_state"]
            pt_path = os.path.join(outdir, f"{model_basename}_{ts}.pt")
            # Save payload as joblib to avoid class pickling issues
            joblib.dump(pt_payload, pt_path)
            result_paths["pt"] = pt_path
        except Exception as e:
            result_paths["pt_error"] = str(e) + "\n" + traceback.format_exc()
            # fallback: attempt to save minimal meta only
            try:
                fallback = os.path.join(outdir, f"{model_basename}_{ts}_meta_only.json")
                with open(fallback, "w") as f:
                    json.dump({"meta": meta}, f, indent=2, default=str)
                result_paths["pt_fallback_meta"] = fallback
            except Exception as fe:
                result_paths["pt_fallback_error"] = str(fe) + "\n" + traceback.format_exc()

        # verification
        try:
            for k, p in result_paths.items():
                if isinstance(p, str) and os.path.exists(p):
                    size = os.path.getsize(p)
                    result_paths[f"{k}_size"] = size
        except Exception:
            pass

    except Exception as exc:
        result_paths["export_error"] = str(exc) + "\n" + traceback.format_exc()
    return outdir, result_paths

# Chunk 7/7: main pipeline + UI actions + breadth/sweep handlers + run button

# helper feature columns
feat_cols_default = ["atr", "rvol", "duration", "hg"]

def run_full_pipeline_once():
    st.info(f"Fetching {symbol} {interval} from YahooQuery …")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars is None or bars.empty:
        st.error("No price data returned.")
        return
    st.success(f"Fetched {len(bars)} bars.")
    bars = ensure_unique_index(bars)
    # daily health
    try:
        daily = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    except Exception:
        daily = bars.copy().resample("1D").agg({"close":"last","volume":"sum"})
    health = calculate_health_gauge(None, daily)
    latest_health = float(health["health_gauge"].iloc[-1]) if not health.empty else 0.0
    st.metric("Latest HealthGauge", f"{latest_health:.3f}")

    # generate candidates
    bars["rvol"] = compute_rvol(bars, lookback=20)
    cands = generate_candidates_and_labels(bars, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates generated.")
        return
    st.success(f"Generated {len(cands)} candidates.")
    # attach health gauge for each candidate day
    if not health.empty:
        hg_map = health["health_gauge"].reindex(pd.to_datetime(health.index)).ffill().to_dict()
        cands["hg"] = cands["candidate_time"].dt.normalize().map(lambda t: hg_map.get(pd.Timestamp(t).normalize(), 0.0))
    else:
        cands["hg"] = 0.0

    # add features
    cands["rvol"] = cands.get("atr", 0.0) * 0 + cands.get("rvol", 1.0)  # ensure rvol column exists if present
    # craft events DataFrame for cascade: t indexes must be integers referencing bar row positions
    # map candidate_time to integer index in bars
    idx_map = {t: i for i, t in enumerate(bars.index)}
    cand_indices = []
    for t in cands["candidate_time"]:
        # find nearest index in bars
        try:
            pos = bars.index.get_loc(pd.to_datetime(t))
        except Exception:
            pos = bars.index.get_indexer([pd.to_datetime(t)], method="nearest")[0]
        cand_indices.append(int(pos))
    # create events: label using label column generated earlier
    events = pd.DataFrame({"t": cand_indices, "y": cands["label"].astype(int).values})
    # instantiate cascade and fit
    cascade = CascadeTrader(device=device_opt, seq_len=l1_seq_len, feature_windows=(5,10,20))
    st.info("Training cascade (L1/L2/L3). This may take a while.")
    try:
        cascade.fit(bars, events)
        st.success("Cascade trained.")
    except Exception as e:
        st.error(f"Cascade training failed: {e}")
        logger.exception("Cascade training failed: %s", e)
        return

    # predictions head
    try:
        test_idx = np.array(events['t'].values)
        preds = cascade.predict_batch(bars, test_idx, gate_cfg={"th1": p_fast, "th2": p_slow, "th3": p_deep})
        st.subheader("Predictions head")
        st.dataframe(preds.head(20))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)

    # run grid sweep (light)
    st.info("Running grid sweep (light)")
    rr_vals = sorted(list(set([1.0, 2.0, 3.0, 4.0])))
    sl_ranges = [(float(l1_sell_min/100 if l1_sell_min>1 else l1_sell_min), float(l1_sell_max/100 if l1_sell_max>1 else l1_sell_max)),
                 (float(l2_sell_min/100 if l2_sell_min>1 else l2_sell_min), float(l2_sell_max/100 if l2_sell_max>1 else l2_sell_max)),
                 (float(l3_sell_min/100 if l3_sell_min>1 else l3_sell_min), float(l3_sell_max/100 if l3_sell_max>1 else l3_sell_max))]
    # mpt list from UI
    mpt_list = [p_fast, p_slow, p_deep]
    try:
        sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list)
        st.success("Sweep completed.")
    except Exception as e:
        st.error(f"Sweep failed: {e}")
        logger.exception("Sweep failed: %s", e)
        sweep_results = {}

    # Export artifacts
    st.info("Exporting artifacts and .pt bundles")
    model_basename = f"cascade_{symbol.replace('=','_')}"
    metrics = {"num_candidates": len(cands)}
    outdir, export_paths = export_model_and_metadata_serializable(
        l1_model=cascade.l1_model if hasattr(cascade, "l1_model") else None,
        l2_model=cascade.l2_model if hasattr(cascade, "l2_model") else None,
        l3_model=cascade.l3_model if hasattr(cascade, "l3_model") else None,
        cascade=cascade,
        feature_list=cascade.tab_feature_names if hasattr(cascade, "tab_feature_names") else [],
        metrics=metrics,
        model_basename=model_basename
    )
    st.success(f"Exported models to {outdir}")
    st.json(export_paths)
    # provide downloads for pt bundle if exists
    if "pt" in export_paths:
        st.download_button("Download portable bundle (.joblib)", data=open(export_paths["pt"], "rb").read(), file_name=os.path.basename(export_paths["pt"]), mime="application/octet-stream")
    # show breadth if requested
    if run_breadth:
        st.info("Running breadth backtest across 3 levels...")
        levels = {
            "L1": {"buy_min": l1_buy_min, "buy_max": l1_buy_max, "sell_min": l1_sell_min, "sell_max": l1_sell_max,
                   "rr_min": l1_buy_min, "rr_max": l1_buy_max, "sl_min": l1_sell_min, "sl_max": l1_sell_max},
            "L2": {"buy_min": l2_buy_min, "buy_max": l2_buy_max, "sell_min": l2_sell_min, "sell_max": l2_sell_max,
                   "rr_min": l2_buy_min, "rr_max": l2_buy_max, "sl_min": l2_sell_min, "sl_max": l2_sell_max},
            "L3": {"buy_min": l3_buy_min, "buy_max": l3_buy_max, "sell_min": l3_sell_min, "sell_max": l3_sell_max,
                   "rr_min": l3_buy_min, "rr_max": l3_buy_max, "sl_min": l3_sell_min, "sl_max": l3_sell_max},
        }
        breadth_res = run_breadth_backtest(clean=cands, bars=bars, levels_config=levels)
        st.subheader("Breadth Summary")
        if breadth_res.get("summary"):
            st.dataframe(pd.DataFrame(breadth_res["summary"]))
        else:
            st.warning("Breadth returned no summary.")
        for lvl, df in breadth_res.get("detailed_trades", {}).items():
            st.write(f"{lvl} — {0 if df is None else len(df)} trades")
            if df is not None and not df.empty:
                st.dataframe(df.head(50))
    st.success("Full pipeline completed.")
    return

# Run via button
if run_pipeline:
    if not valid_ranges:
        st.error("Fix level ranges before running pipeline: " + vr_msg)
    else:
        run_full_pipeline_once()