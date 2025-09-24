# app.py — Entry-Range Triangulation: Cascade Trader Streamlit (Chunk 1/7)
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass, asdict

# ML libs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    nn = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cascade_trader_app")
logger.setLevel(logging.INFO)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Cascade Trader — Scope→Aim→Shoot", layout="wide")
st.title("Cascade Trader — Scope → Aim → Shoot (L1/L2/L3)")

# top inputs
col1, col2 = st.columns([2,1])
with col1:
    symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
    start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=450))
    end_date = st.date_input("End date", value=datetime.today())
    interval = st.selectbox("Interval", ["1d","1h","15m","5m","1m"], index=0)
with col2:
    st.markdown("### Training & Model")
    num_boost = int(st.number_input("XGBoost rounds (if used)", min_value=1, value=200))
    early_stop = int(st.number_input("Early stopping rounds", min_value=1, value=20))
    test_size = float(st.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.20))
    l1_seq_len = int(st.number_input("L1 sequence length", min_value=8, value=64, step=8))
    device_choice = st.selectbox("Device", ["auto","cpu","cuda"], index=0)

st.sidebar.header("Quick sim thresholds")
p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

st.sidebar.markdown("### Level gating (exclusive ranges) — signal scale 0..10")
st.sidebar.markdown("Define explicit buy/sell ranges per level. Ranges should be exclusive (L3 inside L2 inside L1).")

# Level ranges - we enforce exclusivity in logic later
lvl1_buy_min = st.sidebar.number_input("L1 buy min", 0.0, 10.0, 5.50, step=0.01)
lvl1_buy_max = st.sidebar.number_input("L1 buy max", 0.0, 10.0, 9.90, step=0.01)
lvl1_sell_min = st.sidebar.number_input("L1 sell min", 0.0, 10.0, 0.00, step=0.01)
lvl1_sell_max = st.sidebar.number_input("L1 sell max", 0.0, 10.0, 4.50, step=0.01)

lvl2_buy_min = st.sidebar.number_input("L2 buy min", 0.0, 10.0, 6.00, step=0.01)
lvl2_buy_max = st.sidebar.number_input("L2 buy max", 0.0, 10.0, 9.90, step=0.01)
lvl2_sell_min = st.sidebar.number_input("L2 sell min", 0.0, 10.0, 0.00, step=0.01)
lvl2_sell_max = st.sidebar.number_input("L2 sell max", 0.0, 10.0, 4.00, step=0.01)

lvl3_buy_min = st.sidebar.number_input("L3 buy min", 0.0, 10.0, 6.50, step=0.01)
lvl3_buy_max = st.sidebar.number_input("L3 buy max", 0.0, 10.0, 9.90, step=0.01)
lvl3_sell_min = st.sidebar.number_input("L3 sell min", 0.0, 10.0, 0.00, step=0.01)
lvl3_sell_max = st.sidebar.number_input("L3 sell max", 0.0, 10.0, 3.50, step=0.01)

st.sidebar.header("Breadth & Sweep")
run_breadth_btn = st.sidebar.button("Run breadth backtest (3 levels)")
run_sweep_btn = st.sidebar.button("Run grid sweep (light)")

autofocus = st.sidebar.checkbox("Enable autofocus tuning", value=True)
threads_training = int(st.sidebar.number_input("Training threads (L2/L3 concurrency)", min_value=1, max_value=8, value=2))

st.sidebar.markdown("### Export")
save_pt = st.sidebar.checkbox("Save .pt bundles", value=True)

# Fetcher (YahooQuery)
def fetch_price(symbol: str, start: Optional[str]=None, end: Optional[str]=None, interval: str="1d") -> pd.DataFrame:
    if YahooTicker is None:
        logger.error("yahooquery missing. Install via `pip install yahooquery`.")
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

# Chunk 2/7: features + health gauge + candidate generator
def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(1.0, index=df.index)
    rolling = df["volume"].rolling(lookback, min_periods=1).mean()
    return (df["volume"] / rolling.replace(0, np.nan)).fillna(1.0)

def calculate_health_gauge(daily_bars: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
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
def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr

def generate_candidates_and_labels(bars: pd.DataFrame,
                                   lookback: int = 64,
                                   k_tp: float = 3.0,
                                   k_sl: float = 1.0,
                                   atr_window: int = 14,
                                   max_bars: int = 60,
                                   direction: str = "long") -> pd.DataFrame:
    """
    Triple-barrier style candidate generation similar to your earlier code.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = ensure_unique_index(bars)
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
        atr = float(bars["atr"].iat[i])
        if atr <= 0 or math.isnan(atr):
            continue
        sl_px = entry_px - k_sl*atr if direction=="long" else entry_px + k_sl*atr
        tp_px = entry_px + k_tp*atr if direction=="long" else entry_px - k_tp*atr
        end_i = min(i+max_bars, n-1)
        label, hit_i, hit_px = 0, end_i, float(bars["close"].iat[end_i])
        for j in range(i+1, end_i+1):
            hi, lo = float(bars["high"].iat[j]), float(bars["low"].iat[j])
            if direction=="long":
                if hi >= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px; break
                if lo <= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px; break
            else:
                if lo <= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px; break
                if hi >= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px; break
        end_t = bars.index[hit_i]
        ret_val = (hit_px-entry_px)/entry_px if direction=="long" else (entry_px-hit_px)/entry_px
        dur_min = (end_t - t).total_seconds() / 60.0
        records.append(dict(candidate_time=t,
                            entry_price=float(entry_px),
                            atr=float(atr),
                            sl_price=float(sl_px),
                            tp_price=float(tp_px),
                            end_time=end_t,
                            label=int(label),
                            duration=float(dur_min),
                            realized_return=float(ret_val),
                            direction=direction))
    return pd.DataFrame(records)

# Chunk 3/7: backtest + summarization + sweep helpers
def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str   = "GC=F",
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
        entry_t = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_t not in bars.index:
            continue
        entry_px = float(bars.loc[entry_t, "close"])
        direction = 1 if lbl>0 else -1
        sl_px = entry_px * (1 - sl) if direction>0 else entry_px * (1 + sl)
        tp_px = entry_px * (1 + tp) if direction>0 else entry_px * (1 - tp)
        exit_t, exit_px, pnl = None, None, None
        segment = bars.loc[entry_t:].head(max_holding)
        if segment.empty:
            continue
        for t, b in segment.iterrows():
            lo, hi = float(b["low"]), float(b["high"])
            if direction>0:
                if lo <= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl; break
                if hi >= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp; break
            else:
                if hi >= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl; break
                if lo <= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp; break
        if exit_t is None:
            last_bar = segment.iloc[-1]
            exit_t = last_bar.name
            exit_px = float(last_bar["close"])
            pnl = (exit_px-entry_px)/entry_px * direction
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

def run_grid_sweep(clean: pd.DataFrame,
                   bars: pd.DataFrame,
                   rr_vals: List[float],
                   sl_ranges: List[Tuple[float,float]],
                   mpt_list: List[float],
                   feature_cols: List[str],
                   model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    results = {}
    if clean is None or clean.empty:
        return results
    for rr in rr_vals:
        for sl_min, sl_max in sl_ranges:
            sl_pct = (sl_min + sl_max) / 2.0
            tp_pct = rr * sl_pct
            for mpt in mpt_list:
                key = f"rr{rr:.2f}_sl{sl_min:.4f}-{sl_max:.4f}_mpt{mpt:.2f}"
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


# Chunk 4/7: models, training helpers, temperature scaler, cascade wrapper
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
        def __init__(self, in_features:int=12, channels:List[int]=[32,64,128], kernel_sizes=[5,3,3], dilations=[1,2,4], dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            self.blocks = nn.Sequential(
                ConvBlock(chs[0], chs[1], kernel_sizes[0], dilations[0], dropout),
                ConvBlock(chs[1], chs[2], kernel_sizes[1], dilations[1], dropout),
                ConvBlock(chs[2], chs[2], kernel_sizes[2], dilations[2], dropout)
            )
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

    # Training helpers (small, robust)
    from torch.utils.data import Dataset, DataLoader

    class SequenceDataset(Dataset):
        def __init__(self, X_seq: np.ndarray, y: np.ndarray):
            self.X = X_seq.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1,1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].transpose(1,0)  # [F,T]
            return x, self.y[idx]

    class TabDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1,1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def _device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def train_torch_classifier(model: nn.Module,
                               train_ds: Dataset,
                               val_ds: Dataset,
                               lr: float,
                               epochs: int,
                               patience: int,
                               pos_weight: float = 1.0,
                               device: str = "auto") -> (nn.Module, Dict[str,Any]):
        dev = _device(device)
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
        train_loader = DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 256), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
        best_loss = float("inf"); best_state = None; no_improve = 0; history = {"train":[], "val":[]}
        for epoch in range(epochs):
            model.train()
            loss_sum, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                loss.backward(); opt.step()
                loss_sum += float(loss.item()) * len(xb); n += len(xb)
            train_loss = loss_sum / max(n,1)
            model.eval()
            vloss_sum, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    out = model(xb)
                    logit = out[0] if isinstance(out, tuple) else out
                    loss = bce(logit, yb)
                    vloss_sum += float(loss.item()) * len(xb); vn += len(xb)
            val_loss = vloss_sum / max(vn,1)
            history["train"].append(train_loss); history["val"].append(val_loss)
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, {"best_val_loss": best_loss, "history": history}

# We will implement a small CascadeTrainer wrapper with the essential fit/predict/export hooks
class SimpleCascade:
    def __init__(self, device="auto"):
        self.device = _device(device) if torch is not None else None
        self.l1 = None
        self.l1_temp = None
        self.l2_model = None
        self.l2_backend = None
        self.l3 = None
        self.l3_temp = None
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.tab_feature_names = []
        self._fitted = False
        self.metadata = {}

# Chunk 5/7: Cascade fitting, inference, autofocus, parallel L2/L3 training
if torch is not None:
    def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
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

    def _l1_infer_logits_emb_batch(model, Xseq, device):
        model.eval()
        logits, embeds = [], []
        with torch.no_grad():
            for i in range(0, len(Xseq), 1024):
                chunk = Xseq[i:i+1024]
                xb = torch.tensor(chunk.transpose(0,2,1), dtype=torch.float32, device=device)  # [B,F,T] -> model expects [B,F,T]
                logit, emb = model(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        logits = np.concatenate(logits, axis=0).reshape(-1,1)
        embeds = np.concatenate(embeds, axis=0)
        return logits, embeds

    # Integrate into SimpleCascade as methods
    def cascade_fit(self, df: pd.DataFrame, events: pd.DataFrame, l1_seq_len: int = 64,
                    xgb_rounds: int = 200, early_stop_rounds: int = 20,
                    val_size: float = 0.2, device: str = "auto", threads: int = 2):
        """
        Fit L1 (CNN), then L2 (XGB or MLP), then L3 (MLP). L2 and L3 are trained in parallel threads.
        events: DataFrame with columns 't' (index positions into df) and 'y' {0,1}
        """
        t0 = time.time()
        logger.info("Starting cascade_fit")
        # 1) engineered features (reuse compute_engineered_features pattern)
        eng = compute_engineered_features_for_cascade(df)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = [c for c in eng.columns if c.startswith(("ret1","tr","vol_","mom_","chanpos_"))]
        use_cols = [c for c in seq_cols + micro_cols if c in df.columns.union(eng.columns)]
        feat_seq_df = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[use_cols].replace([np.inf,-np.inf],0.0).fillna(0.0)
        feat_tab_df = eng.copy()

        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values

        # train/val split (time-aware would be better; quick stratified version here)
        train_idx, val_idx = train_test_split(np.arange(len(idx)), test_size=val_size, random_state=42, stratify=y)
        idx_train, idx_val = idx[train_idx], idx[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # scaleers — fit on full train slices
        X_seq_all = feat_seq_df.values
        self.scaler_seq.fit(X_seq_all)
        X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)

        X_tab_all = feat_tab_df.values
        self.tab_feature_names = list(feat_tab_df.columns)
        self.scaler_tab.fit(X_tab_all)
        X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)

        # Build L1 datasets
        Xseq_train = to_sequences(X_seq_all_scaled, idx_train, seq_len=l1_seq_len)
        Xseq_val = to_sequences(X_seq_all_scaled, idx_val, seq_len=l1_seq_len)
        ds_l1_train = SequenceDataset(Xseq_train, y_train); ds_l1_train.batch_size = 128
        ds_l1_val = SequenceDataset(Xseq_val, y_val)

        # Instantiate L1
        in_features = Xseq_train.shape[2]
        self.l1 = Level1ScopeCNN(in_features=in_features)
        self.l1_temp = TemperatureScaler()

        self.l1, l1_hist = train_torch_classifier(
            self.l1, ds_l1_train, ds_l1_val,
            lr=1e-3, epochs=20, patience=5, pos_weight=1.0, device=device)

        # L1 embeddings for L2/L3
        l1_train_logits, l1_train_emb = _l1_infer_logits_emb_batch(self.l1, Xseq_train, self.device)
        l1_val_logits, l1_val_emb = _l1_infer_logits_emb_batch(self.l1, Xseq_val, self.device)
        # temperature scaling
        try:
            self.l1_temp.fit(l1_val_logits, y_val)
        except Exception as e:
            logger.warning("L1 temp fit failed: %s", e)

        # Build tabular inputs for L2/L3
        Xtab_train = X_tab_all_scaled[idx_train]
        Xtab_val = X_tab_all_scaled[idx_val]
        X_l2_train = np.hstack([l1_train_emb, Xtab_train])
        X_l2_val = np.hstack([l1_val_emb, Xtab_val])

        # Train L2 & L3 in parallel (XGBoost or MLP)
        results = {}
        def train_l2():
            try:
                if XGBOOST_AVAILABLE:
                    clf = xgb.XGBClassifier(n_estimators=xgb_rounds if (xgb_rounds:=200) else 200, max_depth=5, use_label_encoder=False, eval_metric="logloss")
                    clf.fit(X_l2_train, y_train, eval_set=[(X_l2_val, y_val)], verbose=False)
                    return ("xgb", clf, {})
                else:
                    mlp = MLP(X_l2_train.shape[1], [128,64], out_dim=1, dropout=0.1)
                    ds2_tr = TabDataset(X_l2_train, y_train); ds2_va = TabDataset(X_l2_val, y_val)
                    mlp, hist = train_torch_classifier(mlp, ds2_tr, ds2_va, lr=1e-3, epochs=15, patience=4, device=device)
                    return ("mlp", mlp, hist)
            except Exception as e:
                logger.exception("L2 training failed: %s", e)
                return ("error", None, {"error": str(e)})

        def train_l3():
            try:
                mlp = MLP(X_l2_train.shape[1], [128,64], out_dim=1, dropout=0.1)
                ds3_tr = TabDataset(X_l2_train, y_train); ds3_va = TabDataset(X_l2_val, y_val)
                mlp, hist = train_torch_classifier(mlp, ds3_tr, ds3_va, lr=1e-3, epochs=20, patience=5, device=device)
                return ("mlp", mlp, hist)
            except Exception as e:
                logger.exception("L3 training failed: %s", e)
                return ("error", None, {"error": str(e)})

        with ThreadPoolExecutor(max_workers=max(1,threads)) as ex:
            futs = {"l2": ex.submit(train_l2), "l3": ex.submit(train_l3)}
            for name, fut in futs.items():
                try:
                    res = fut.result()
                    results[name] = res
                except Exception as e:
                    results[name] = ("error", None, {"error": str(e)})
        # Unpack
        l2_kind, l2_model, l2_meta = results.get("l2", ("error", None, {}))
        l3_kind, l3_model, l3_meta = results.get("l3", ("error", None, {}))
        self.l2_backend = l2_kind if l2_kind in ("xgb","mlp") else None
        self.l2_model = l2_model
        self.l3 = l3_model
        self.l3_temp = TemperatureScaler()
        # calibrate l3
        try:
            l3_val_logits = _l3_infer_logits(self.l3, X_l2_val, device=self.device) if hasattr(self, "l3") and self.l3 is not None else np.zeros((len(y_val),1))
            self.l3_temp.fit(l3_val_logits, y_val)
        except Exception as e:
            logger.warning("L3 temp fit failed: %s", e)

        # metadata
        self.metadata = {
            "l1_hist": l1_hist,
            "l2_meta": l2_meta,
            "l3_meta": l3_meta,
            "fit_time_sec": round(time.time()-t0,2),
            "tab_features": self.tab_feature_names,
            "embed_dim": l1_train_emb.shape[1] if len(l1_train_emb.shape)>=2 else 0,
            "l1_seq_len": l1_seq_len
        }
        self._fitted = True
        logger.info("Cascade fit completed")
        return self

    def _l3_infer_logits(self, model, X, device):
        model.eval()
        logits = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.tensor(X[i:i+2048], dtype=torch.float32, device=device)
                logit = model(xb)
                if isinstance(logit, tuple): logit = logit[0]
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1)

    # attach methods to SimpleCascade
    SimpleCascade.fit = cascade_fit
    SimpleCascade._l3_infer_logits = _l3_infer_logits
    SimpleCascade._l1_infer_logits_emb_batch = _l1_infer_logits_emb_batch

# autofocus update (keeps the Gate adjustments separate, simplified)
def autofocus_update_placeholder():
    # NOTE: lightweight stub — you can plug your autofocus search here
    logger.info("Autofocus update placeholder called")

# Chunk 6/7: robust export_model_and_metadata (uses your provided logic but avoids pickling model objects)
def export_model_and_metadata_safe(cascade: SimpleCascade, out_dir_base: str, save_fi: bool = True) -> Dict[str, Any]:
    """
    Save models and artifacts in out_dir and produce a portable .pt bundle that contains only
    serializable metadata and file paths. This avoids trying to pickle live model objects.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"{out_dir_base}_{ts}"
    os.makedirs(base, exist_ok=True)
    paths = {}
    try:
        # L1: save state_dict
        try:
            l1_file = os.path.join(base, f"l1_state_{ts}.pt")
            torch.save(cascade.l1.state_dict(), l1_file)
            paths["l1_state"] = l1_file
        except Exception as e:
            logger.exception("Failed to save l1 state: %s", e)
            paths["l1_state_error"] = str(e)

        # L1 temp
        try:
            l1_temp_file = os.path.join(base, f"l1_temp_{ts}.pkl")
            joblib.dump(cascade.l1_temp.state_dict(), l1_temp_file)
            paths["l1_temp"] = l1_temp_file
        except Exception as e:
            logger.exception("Failed to save l1_temp: %s", e)
            paths["l1_temp_error"] = str(e)

        # L2: if xgboost native, save model via its own API; if MLP, save state_dict
        try:
            if cascade.l2_backend == "xgb" and hasattr(cascade.l2_model, "save_model"):
                l2_file = os.path.join(base, f"l2_xgb_{ts}.json")
                cascade.l2_model.save_model(l2_file)
                paths["l2_model"] = l2_file
            else:
                # MLP or other: save state_dict if torch module
                if torch is not None and isinstance(cascade.l2_model, nn.Module):
                    l2_file = os.path.join(base, f"l2_mlp_{ts}.pt")
                    torch.save(cascade.l2_model.state_dict(), l2_file)
                    paths["l2_model"] = l2_file
                else:
                    # else, try joblib-safe serializable repr (e.g., XGB sklearn wrapper names)
                    l2_meta_file = os.path.join(base, f"l2_model_{ts}.json")
                    joblib.dump({"note": "non-serializable model; inspect cascade.l2_model"}, l2_meta_file)
                    paths["l2_model_note"] = l2_meta_file
        except Exception as e:
            logger.exception("Failed to save l2 model: %s", e)
            paths["l2_model_error"] = str(e)

        # L3: save state_dict
        try:
            if torch is not None and isinstance(cascade.l3, nn.Module):
                l3_file = os.path.join(base, f"l3_state_{ts}.pt")
                torch.save(cascade.l3.state_dict(), l3_file)
                paths["l3_state"] = l3_file
            else:
                paths["l3_note"] = "no l3 torch model to save"
        except Exception as e:
            logger.exception("Failed to save l3: %s", e)
            paths["l3_state_error"] = str(e)

        # L3 temp
        try:
            if cascade.l3_temp is not None:
                l3_temp_file = os.path.join(base, f"l3_temp_{ts}.pkl")
                joblib.dump(cascade.l3_temp.state_dict(), l3_temp_file)
                paths["l3_temp"] = l3_temp_file
        except Exception as e:
            logger.exception("Failed to save l3_temp: %s", e)
            paths["l3_temp_error"] = str(e)

        # scalers and metadata
        try:
            scalers_file = os.path.join(base, f"scalers_{ts}.joblib")
            joblib.dump({"scaler_seq": cascade.scaler_seq, "scaler_tab": cascade.scaler_tab}, scalers_file)
            paths["scalers"] = scalers_file
        except Exception as e:
            logger.exception("Failed to save scalers: %s", e)
            paths["scalers_error"] = str(e)

        try:
            meta_file = os.path.join(base, f"metadata_{ts}.json")
            with open(meta_file, "w") as f:
                json.dump(cascade.metadata, f, default=str, indent=2)
            paths["metadata"] = meta_file
        except Exception as e:
            logger.exception("Failed to save metadata: %s", e)
            paths["metadata_error"] = str(e)

        # Feature importance placeholders: for XGB try to extract importance
        try:
            if cascade.l2_backend == "xgb" and hasattr(cascade.l2_model, "get_booster"):
                booster = cascade.l2_model.get_booster()
                fi = booster.get_score(importance_type="gain")
                fi_items = [{"feature":k, "gain":float(v)} for k,v in fi.items()]
                fi_df = pd.DataFrame(fi_items).sort_values("gain", ascending=False)
                fi_path = os.path.join(base, f"l2_fi_{ts}.csv")
                fi_df.to_csv(fi_path, index=False)
                paths["l2_feature_importance"] = fi_path
        except Exception as e:
            logger.exception("Failed to save l2 feature importance: %s", e)
            paths["l2_fi_error"] = str(e)

        # Build portable .pt bundle: contain only file paths + simple metadata (no live objects)
        try:
            bundle = {
                "artifact_paths": paths,
                "metadata": cascade.metadata,
                "created_at": datetime.utcnow().isoformat(),
            }
            pt_path = os.path.join(base, f"cascade_bundle_{ts}.pt")
            # Use joblib to store bundle metadata (safe, no model objects)
            joblib.dump(bundle, pt_path)
            paths["pt"] = pt_path
        except Exception as e:
            logger.exception("Failed to create pt bundle: %s", e)
            paths["pt_error"] = str(e)

    except Exception as exc:
        logger.exception("Failed to export model artifacts: %s", exc)
        paths["export_error"] = str(exc)
    # Final verification
    try:
        if "pt" in paths:
            sz = os.path.getsize(paths["pt"])
            paths["pt_verification"] = f"{sz} bytes"
    except Exception as e:
        paths["pt_verification_error"] = str(e)
    return paths

# Chunk 7/7: Streamlit main pipeline + handlers
st.markdown("## Run pipeline")
run_pipeline = st.button("Run Full Pipeline: Fetch → Train → Sweep → Export")

log_container = st.empty()
log_buf = []

def log(msg, level="info"):
    ts = datetime.utcnow().isoformat()
    log_buf.append(f"[{ts}] {msg}")
    log_container.text("\n".join(log_buf[-200:]))

if run_pipeline:
    try:
        log(f"Fetching {symbol} {interval} from YahooQuery …")
        bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        if bars is None or bars.empty:
            st.error("No price data returned.")
            raise SystemExit()
        log(f"Fetched {len(bars)} bars.")
        bars = ensure_unique_index(bars)

        daily = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        health = calculate_health_gauge(daily)
        log("Computed HealthGauge.")

        # candidates
        bars["rvol"] = compute_rvol(bars, lookback=20)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        log(f"Generated {len(cands)} candidates.")
        if cands is None or cands.empty:
            st.error("No candidates generated.")
            raise SystemExit()

        # attach a lightweight 'signal' column scaled 0..10 using a simple heuristic (here pred_prob placeholder)
        # For now create a proxy signal: atr-normalized inverse (small ATR => stronger signal)
        cands["signal"] = ((1.0 / (1.0 + cands["atr"])) * 10).clip(0,10)

        # apply exclusive-level selection: mark each candidate level membership
        def assign_level(sig):
            if lvl3_buy_min <= sig <= lvl3_buy_max and not (lvl3_sell_min <= sig <= lvl3_sell_max):
                return "L3"
            if lvl2_buy_min <= sig <= lvl2_buy_max and not (lvl2_sell_min <= sig <= lvl2_sell_max):
                return "L2"
            if lvl1_buy_min <= sig <= lvl1_buy_max and not (lvl1_sell_min <= sig <= lvl1_sell_max):
                return "L1"
            return "OUT"
        cands["level"] = cands["signal"].apply(assign_level)
        log("Assigned candidate levels (exclusive).")
        st.write("Candidates head:", cands.head())

        # build events DataFrame for training (simple: use candidate label)
        events = cands.reset_index(drop=True).reset_index().rename(columns={"index":"t"})
        events = events[["t","label"]].rename(columns={"label":"y"})
        log(f"Prepared events (N={len(events)}) for training.")

        # instantiate cascade and fit
        cascade = SimpleCascade(device=device_choice)
        st.info("Training cascade (L1/L2/L3). This may take a while.")
        log("Starting training...")
        cascade.fit(df=bars, events=events, l1_seq_len=l1_seq_len, xgb_rounds=num_boost, early_stop_rounds=early_stop, val_size=test_size, device=device_choice, threads=threads_training)
        log("Cascade trained.")
        st.success("Cascade trained.")

        # Predict head (sample)
        try:
            sample_idx = events["t"].values[:10]
            # We'll do inference in a simplified manner (not full predict_batch here)
            log("Producing sample predictions (L1 logits)...")
            # produce sequences and call L1 inference helper
            # reuse code from chunk but call l1 inference directly
            # For brevity show just shapes
            st.write("Model metadata:", cascade.metadata)
        except Exception:
            logger.exception("Prediction sample failed")

        # Run a light grid sweep
        st.info("Running grid sweep (light)")
        rr_vals = [lvl1_buy_min/1.0, lvl2_buy_min/1.0, lvl3_buy_min/1.0]
        rr_vals = sorted(list(set([float(round(x,2)) for x in rr_vals if x is not None])))
        sl_ranges = [(float(lvl1_sell_min), float(lvl1_sell_max)),
                     (float(lvl2_sell_min), float(lvl2_sell_max)),
                     (float(lvl3_sell_min), float(lvl3_sell_max))]
        mpt_list = [float(p_fast), float(p_slow), float(p_deep)]
        sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list, feature_cols=["atr","rvol","duration"], model_train_kwargs={"max_bars":60})
        st.success("Sweep completed.")
        # summarize sweep
        summary_rows = []
        for k,v in sweep_results.items():
            if isinstance(v, dict) and "overlay" in v and not v["overlay"].empty:
                s = summarize_trades(v["overlay"])
                if not s.empty:
                    r = s.iloc[0].to_dict(); r.update({"config":k, "trades_count": v.get("trades_count",0)}); summary_rows.append(r)
        if summary_rows:
            sdf = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
            st.dataframe(sdf.head(20))
        else:
            st.warning("Sweep returned no runs with trades.")

        # Export artifacts
        st.info("Exporting artifacts and .pt bundles")
        out_base = f"artifacts"
        out_paths = export_model_and_metadata_safe(cascade, out_base, save_fi=save_pt)
        st.write("Exported models to", os.path.dirname(list(out_paths.get("pt", out_paths.get("metadata","artifacts")))))
        st.json(out_paths)

        st.success("Full pipeline completed.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        st.error(f"Pipeline failed: {exc}")