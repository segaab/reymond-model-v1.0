import os
import sys
import gc
import logging
import joblib
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ML imports
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Data fetching
from yahooquery import Ticker

# Supabase wrapper
from supabase_client_wrapper import SupabaseClientWrapper

# -----------------------------------------------------
# Logging setup
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cascade_trader")

# -----------------------------------------------------
# Supabase client initialization
# -----------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dzddytphimhoxeccxqsw.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client = None
if SUPABASE_SERVICE_ROLE_KEY:
    supabase_client = SupabaseClientWrapper(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    logger.info("Supabase client initialized for stats logging")
else:
    logger.warning("Supabase service role key not set. Stats logging disabled.")

# -----------------------------------------------------
# Streamlit App UI - Sidebar Controls
# -----------------------------------------------------
st.set_page_config(page_title="Cascade Trader", layout="wide")

st.sidebar.header("Cascade Trader Controls")

# Date selection
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=180))

# Asset ticker
symbol = st.sidebar.text_input("Asset Ticker", "AAPL")

# Interval
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "30m", "15m", "5m", "1m"],
    index=0
)

# Buy/Sell thresholds per layer
st.sidebar.subheader("Layer Thresholds (Buy)")
lvl1_buy_min = st.sidebar.slider("L1 Buy Min", 0.0, 10.0, 3.0, 0.1)
lvl1_buy_max = st.sidebar.slider("L1 Buy Max", 0.0, 10.0, 10.0, 0.1)
lvl2_buy_min = st.sidebar.slider("L2 Buy Min", 0.0, 10.0, 5.5, 0.1)
lvl2_buy_max = st.sidebar.slider("L2 Buy Max", 0.0, 10.0, 10.0, 0.1)
lvl3_buy_min = st.sidebar.slider("L3 Buy Min", 0.0, 10.0, 6.5, 0.1)
lvl3_buy_max = st.sidebar.slider("L3 Buy Max", 0.0, 10.0, 10.0, 0.1)

st.sidebar.subheader("Layer Thresholds (Sell)")
lvl1_sell_min = st.sidebar.slider("L1 Sell Min", 0.0, 10.0, 0.0, 0.1)
lvl1_sell_max = st.sidebar.slider("L1 Sell Max", 0.0, 10.0, 7.0, 0.1)
lvl2_sell_min = st.sidebar.slider("L2 Sell Min", 0.0, 10.0, 0.0, 0.1)
lvl2_sell_max = st.sidebar.slider("L2 Sell Max", 0.0, 10.0, 5.0, 0.1)
lvl3_sell_min = st.sidebar.slider("L3 Sell Min", 0.0, 10.0, 0.0, 0.1)
lvl3_sell_max = st.sidebar.slider("L3 Sell Max", 0.0, 10.0, 4.0, 0.1)

# Pipeline control buttons
st.sidebar.subheader("Pipeline Execution")
run_full_pipeline_btn = st.sidebar.button("Run Full Pipeline (Fetch → Train → Export)")

# ---------------- Chunk 2/5 ----------------
# engineered-feature helpers

def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute a compact set of engineered features from OHLCV."""
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)

    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build sequences ending at each index t: [t-seq_len+1, ..., t].
    Returns shape [N, seq_len, F]
    """
    Nrows, F = features.shape
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


# torch Datasets (only if torch is available)
if torch is not None:
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        def __init__(self, X_seq: np.ndarray, y: np.ndarray):
            self.X = X_seq.astype(np.float32)  # [N, T, F]
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].transpose(1,0)  # [F, T]
            y = self.y[idx]
            return x, y

    class TabDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
else:
    SequenceDataset = None
    TabDataset = None


# utilities: ensure unique index
def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


# candidate generation
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
        tp_px = entry_px + k_tp * atr if direction == "long" else entry_px - k_tp * atr
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


# simulate limits + summarization
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
        lbl = int(row.get(label_col, 0))
        if lbl == 0:
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

# ---------------- Chunk 3/5 ----------------
# Model blocks (if torch available)
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
        def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                k = kernel_sizes[min(i, len(kernel_sizes)-1)]
                d = dilations[min(i, len(dilations)-1)]
                blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
            self.blocks = nn.Sequential(*blocks)
            self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)
        @property
        def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
        def forward(self, x):
            z = self.blocks(x)
            z = self.project(z)
            z_pool = z.mean(dim=-1)
            logit = self.head(z_pool)
            return logit, z_pool

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim=1, dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
                last = h
            layers += [nn.Linear(last, out_dim)]
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    class Level3ShootMLP(nn.Module):
        def __init__(self, in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True):
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
            y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=device)
            opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
            bce = nn.BCEWithLogitsLoss()
            def closure():
                opt.zero_grad()
                scaled = self.forward(logits_t)
                loss = bce(scaled, y_t)
                loss.backward()
                return loss
            try:
                opt.step(closure)
            except Exception as e:
                logger.warning("Temp scaler LBFGS failed: %s", e)
        def transform(self, logits: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                device = next(self.parameters()).device
                logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
                scaled = self.forward(logits_t).cpu().numpy()
            return scaled.reshape(-1)


# ------------ Module-level simple wrapper classes for export (fixes pickling) ------------
class L2ExportWrapper:
    """Simple export wrapper for L2 models (xgb or mlp)."""
    def __init__(self):
        self.booster = None
    def save_model(self, path: str):
        if hasattr(self.booster, "save_model"):
            return self.booster.save_model(path)
        joblib.dump(self.booster, path)
    def feature_importance(self):
        try:
            if hasattr(self.booster, "get_booster"):
                imp = self.booster.get_booster().get_score(importance_type="gain")
            elif hasattr(self.booster, "get_score"):
                imp = self.booster.get_score(importance_type="gain")
            else:
                return pd.DataFrame([{"feature":"none","gain":0.0}])
            return (pd.DataFrame([(k, imp.get(k,0.0)) for k in sorted(imp.keys())], columns=["feature","gain"])
                    .sort_values("gain", ascending=False).reset_index(drop=True))
        except Exception:
            return pd.DataFrame([{"feature":"none","gain":0.0}])


class L3ExportWrapper:
    """Export wrapper for L3 torch models."""
    def __init__(self):
        self.model = None
    def feature_importance(self):
        # placeholder; real importance should be computed separately
        return pd.DataFrame([{"feature":"l3_emb","gain":1.0}])

# ---------------- Chunk 4/5 ----------------
# training helpers
def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def train_torch_classifier(model: nn.Module,
                           train_ds,
                           val_ds,
                           lr: float = 1e-3,
                           epochs: int = 10,
                           patience: int = 3,
                           pos_weight: float = 1.0,
                           device: str = "auto",
                           st_progress: Optional[st.delta_generator.Progress] = None,
                           progress_offset: int = 0,
                           progress_total: int = 100):
    """
    Train with optional Streamlit progress updater.
    progress_total is total units to represent; the function will update progress_offset..progress_offset+progress_total
    """
    dev = _device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor([pos_weight], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 128), shuffle=True)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=getattr(val_ds, "batch_size", 1024), shuffle=False)
    best_loss = float("inf"); best_state = None; no_imp = 0
    history = {"train": [], "val": []}
    total_steps = epochs
    for ep in range(epochs):
        model.train()
        running_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / max(1, n)
        # val
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                vloss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
        val_loss = vloss / max(1, vn)
        history["train"].append(train_loss); history["val"].append(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
        # update progress if available
        if st_progress is not None:
            try:
                fraction = int(progress_offset + (ep + 1) / total_steps * progress_total)
                st_progress.progress(min(100, max(0, fraction)))
            except Exception:
                pass
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "history": history}


# CascadeTrader: fit / predict
class CascadeTrader:
    def __init__(self, seq_len: int = 64, feat_windows=(5,10,20), device: str = "auto"):
        if torch is None:
            raise RuntimeError("Torch required for CascadeTrader")
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.device = _device(device)
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.l1 = None
        self.l1_temp = TemperatureScaler()
        self.l2_backend = None
        self.l2_model = None
        self.l3 = None
        self.l3_temp = TemperatureScaler()
        self.tab_feature_names = []
        self._fitted = False
        self.metadata = {}

    def fit(self, df: pd.DataFrame, events: pd.DataFrame, l2_use_xgb: bool = True, epochs_l1: int = 10, epochs_l23: int = 10, st_progress: Optional[st.delta_generator.Progress] = None):
        t0 = time.time()
        df = ensure_unique_index(df)
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = [c for c in ['ret1','tr','vol_5','mom_5','chanpos_10'] if c in eng.columns or c in df.columns]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1, sort=False).fillna(0.0)
        feat_tab_df = eng.fillna(0.0)
        self.tab_feature_names = list(feat_tab_df.columns)
        # prepare events
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        tr_idx, va_idx = train_test_split(np.arange(len(idx)), test_size=test_size, random_state=42, stratify=y)
        idx_tr, idx_va = idx[tr_idx], idx[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        # scalers and sequences
        X_seq_all = feat_seq_df.values
        self.scaler_seq.fit(X_seq_all[idx_tr])
        X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
        X_tab_all = feat_tab_df.values
        self.scaler_tab.fit(X_tab_all[idx_tr])
        X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)
        Xseq_tr = to_sequences(X_seq_all_scaled, idx_tr, seq_len=self.seq_len)
        Xseq_va = to_sequences(X_seq_all_scaled, idx_va, seq_len=self.seq_len)
        ds_l1_tr = SequenceDataset(Xseq_tr, y_tr); ds_l1_va = SequenceDataset(Xseq_va, y_va)
        ds_l1_tr.batch_size = 128; ds_l1_va.batch_size = 256
        in_features = Xseq_tr.shape[2]
        self.l1 = Level1ScopeCNN(in_features=in_features, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1)
        self.l1, l1_hist = train_torch_classifier(self.l1, ds_l1_tr, ds_l1_va, lr=1e-3, epochs=epochs_l1, patience=3, pos_weight=1.0, device=str(self.device), st_progress=st_progress, progress_offset=0, progress_total=30)
        # embeddings for events
        all_idx_seq = to_sequences(X_seq_all_scaled, idx, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(all_idx_seq)
        l1_emb_tr, l1_emb_va = l1_emb[tr_idx], l1_emb[va_idx]
        Xtab_tr = X_tab_all_scaled[idx_tr]; Xtab_va = X_tab_all_scaled[idx_va]
        X_l2_tr = np.hstack([l1_emb_tr, Xtab_tr]); X_l2_va = np.hstack([l1_emb_va, Xtab_va])
        # train L2 & L3 in parallel
        results = {}
        def do_l2_xgb():
            if xgb is None:
                raise RuntimeError("xgboost not installed")
            clf = xgb.XGBClassifier(n_estimators=num_boost, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_l2_tr, y_tr, eval_set=[(X_l2_va, y_va)], verbose=False)
            return ("xgb", clf)
        def do_l2_mlp():
            in_dim = X_l2_tr.shape[1]
            m = MLP(in_dim, [128,64], out_dim=1, dropout=0.1)
            ds2_tr = TabDataset(X_l2_tr, y_tr); ds2_va = TabDataset(X_l2_va, y_va)
            ds2_tr.batch_size = 256; ds2_va.batch_size = 512
            m, hist = train_torch_classifier(m, ds2_tr, ds2_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device), st_progress=st_progress, progress_offset=30, progress_total=20)
            return ("mlp", m)
        def do_l3():
            in_dim = X_l2_tr.shape[1]
            m3 = Level3ShootMLP(in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True)
            ds3_tr = TabDataset(X_l2_tr, y_tr); ds3_va = TabDataset(X_l2_va, y_va)
            ds3_tr.batch_size = 256; ds3_va.batch_size = 512
            m3, hist3 = train_torch_classifier(m3, ds3_tr, ds3_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device), st_progress=st_progress, progress_offset=50, progress_total=50)
            return ("l3", (m3, hist3))
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if (xgb is not None) and l2_use_xgb:
                futures[ex.submit(do_l2_xgb)] = "l2"
            else:
                futures[ex.submit(do_l2_mlp)] = "l2"
            futures[ex.submit(do_l3)] = "l3"
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results[label] = fut.result()
                except Exception as e:
                    logger.exception("Parallel train error for %s: %s", label, e)
                    results[label] = None
        if results.get("l2") is not None:
            self.l2_backend, self.l2_model = results["l2"]
        if results.get("l3") is not None:
            self.l3 = results["l3"][1][0] if isinstance(results["l3"][1], tuple) else results["l3"][1]
        # calibrate temp scalers
        try:
            self.l1_temp.fit(l1_logits.reshape(-1,1), y)
        except Exception as e:
            logger.warning("L1 temp scaling failed: %s", e)
        try:
            l3_val_logits = self._l3_infer_logits(X_l2_va)
            self.l3_temp.fit(l3_val_logits.reshape(-1,1), y_va)
        except Exception as e:
            logger.warning("L3 temp scaling failed: %s", e)
        self.metadata = {"l1_hist": l1_hist, "fit_time_sec": round(time.time() - t0, 2), "l2_backend": self.l2_backend}
        self._fitted = True
        logger.info("CascadeTrader.fit finished in %.2fs", time.time() - t0)
        return self

    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        self.l1.eval()
        logits = []; embeds = []
        batch = 256
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32, device=self.device)
                logit, emb = self.l1(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1), np.concatenate(embeds, axis=0)

    def _l3_infer_logits(self, X: np.ndarray):
        self.l3.eval()
        logits = []
        batch = 2048
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1)

    def _mlp_predict_proba(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        probs = []
        batch = 4096
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit = model(xb)
                p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray) -> pd.DataFrame:
        assert self._fitted, "CascadeTrader not fitted"
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']; micro_cols=['ret1','tr','vol_5','mom_5','chanpos_10']
        use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(eng.columns)]
        feat_seq = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1)[use_cols].fillna(0.0)
        feat_tab = eng[self.tab_feature_names].fillna(0.0)
        X_seq_all_scaled = self.scaler_seq.transform(feat_seq.values)
        X_tab_all_scaled = self.scaler_tab.transform(feat_tab.values)
        Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        l1_logits_scaled = self.l1_temp.transform(l1_logits.reshape(-1,1)).reshape(-1)
        p1 = 1.0/(1.0+np.exp(-l1_logits_scaled))
        go2 = p1 >= 0.30
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
            X_l3 = X_l2[go3]
            l3_logits = self._l3_infer_logits(X_l3)
            l3_logits_scaled = self.l3_temp.transform(l3_logits.reshape(-1,1)).reshape(-1)
            p3_vals = 1.0/(1.0+np.exp(-l3_logits_scaled))
            p3[go3] = p3_vals
            rhat[go3] = p3_vals - 0.5
        trade = (p3 >= 0.65) & go3
        size = np.clip(rhat, 0.0, None) * trade.astype(float)
        return pd.DataFrame({"t": t_indices, "p1": p1, "p2": p2, "p3": p3, "go2": go2.astype(int), "go3": go3.astype(int), "trade": trade.astype(int), "size": size})


# complete run_grid_sweep (replaces the earlier cut-off version)
def run_grid_sweep(clean: pd.DataFrame, bars: pd.DataFrame, rr_vals: List[float], sl_ranges: List[Tuple[float,float]], mpt_list: List[float]) -> Dict[str,Any]:
    """
    Run a light grid sweep over reward/risk ratios, stop loss ranges, and minimum probability thresholds.
    Returns a dictionary with a DataFrame under "results".
    """
    results = []
    if clean is None or clean.empty:
        return {"results": pd.DataFrame()}
    sl_step = 0.005
    for i, (sl_min, sl_max) in enumerate(sl_ranges):
        sl_vals = np.arange(sl_min, sl_max + sl_step/2, sl_step)
        for sl in sl_vals:
            for rr in rr_vals:
                tp = sl * rr
                # pick a threshold from mpt_list (safeguard index)
                mpt = mpt_list[min(i, len(mpt_list)-1)]
                df = clean.copy()
                # clean expected "signal" column (old code uses signal scaled 0..10)
                if "signal" not in df.columns:
                    # attempt to recreate from pred_prob if present (0..1)
                    if "pred_prob" in df.columns:
                        df["signal"] = df["pred_prob"] * 10.0
                    else:
                        df["signal"] = 0.0
                df['pred_label'] = (df['signal'] >= mpt*10).astype(int)
                trades = simulate_limits(df, bars, label_col='pred_label', sl=sl, tp=tp, max_holding=60)
                if trades is None or trades.empty:
                    continue
                s = summarize_trades(trades)
                if s is None or s.empty:
                    continue
                row = s.iloc[0].to_dict()
                row.update({
                    "sl": float(sl),
                    "tp": float(tp),
                    "rr": float(rr),
                    "mpt": float(mpt),
                    "level": int(i+1)
                })
                results.append(row)
    return {"results": pd.DataFrame(results) if results else pd.DataFrame()}


# breadth helper (unchanged logic)
def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame) -> Dict[str,Any]:
    out = {"detailed": {}, "summary": []}
    if preds is None or preds.empty or cands is None or cands.empty:
        return out
    df = cands.copy().reset_index(drop=True)
    preds_indexed = preds.set_index('t')
    signals = (preds_indexed.reindex(df.index)['p3'].fillna(0.0) * 10).values
    df['signal'] = signals
    def run_level(name, buy_min, buy_max, sl_pct, rr):
        sel = df[(df['signal'] >= buy_min) & (df['signal'] <= buy_max)].copy()
        tp_pct = rr * sl_pct
        if sel.empty:
            out['detailed'][name] = pd.DataFrame()
            return
        sel['pred_label'] = (sel['signal'] >= buy_min).astype(int)
        trades = simulate_limits(sel, bars, label_col='pred_label', sl=sl_pct, tp=tp_pct, max_holding=60)
        out['detailed'][name] = trades
        s = summarize_trades(trades)
        if not s.empty:
            row = s.iloc[0].to_dict(); row['mode'] = name; out['summary'].append(row)
    l1_sl = float((globals().get("lvl1_sl_min",0.02) + globals().get("lvl1_sl_max",0.04))/2.0)
    l1_rr = float((globals().get("lvl1_rr_min",2.0) + globals().get("lvl1_rr_max",2.5))/2.0)
    l2_sl = float((globals().get("lvl2_sl_min",0.01) + globals().get("lvl2_sl_max",0.03))/2.0)
    l2_rr = float((globals().get("lvl2_rr_min",2.0) + globals().get("lvl2_rr_max",3.5))/2.0)
    l3_sl = float((globals().get("lvl3_sl_min",0.005) + globals().get("lvl3_sl_max",0.02))/2.0)
    l3_rr = float((globals().get("lvl3_rr_min",3.0) + globals().get("lvl3_rr_max",5.0))/2.0)
    run_level("L1", lvl1_buy_min, lvl1_buy_max, l1_sl, l1_rr)
    run_level("L2", lvl2_buy_min, lvl2_buy_max, l2_sl, l2_rr)
    run_level("L3", lvl3_buy_min, lvl3_buy_max, l3_sl, l3_rr)
    out["summary"] = out["summary"] or []
    return out


# export_model_and_metadata (unchanged content; keep the detailed function you provided earlier)
# Ensure function exists in this chunk; assume it's present exactly as in your previous version.
# (To avoid duplication in this message, we assume the same export_model_and_metadata is already pasted in Chunk 1 or earlier.)

# ---------------- Chunk 5/5 ----------------
# Supabase summary saver
def save_layer_summary(sb: Optional[SupabaseClientWrapper],
                       symbol: str,
                       summary_rows: List[Dict[str,Any]]) -> None:
    """
    Push the per-layer summary rows returned by run_breadth_levels() into the `layer_summary` table on Supabase.
    """
    if sb is None or not summary_rows:
        return
    now = datetime.utcnow().isoformat()
    rows = []
    for r in summary_rows:
        rows.append({
            "symbol":       symbol,
            "mode":         r.get("mode",""),
            "total_trades": int(r.get("total_trades") or 0),
            "win_rate":     float(r.get("win_rate") or 0.0),
            "avg_pnl":      float(r.get("avg_pnl") or 0.0),
            "median_pnl":   float(r.get("median_pnl") or 0.0),
            "total_pnl":    float(r.get("total_pnl") or 0.0),
            "max_drawdown": float(r.get("max_drawdown") or 0.0),
            "start_time":   str(r.get("start_time")) if r.get("start_time") is not None else None,
            "end_time":     str(r.get("end_time")) if r.get("end_time") is not None else None,
            "created_at":   now
        })
    try:
        resp = sb.insert_data("layer_summary", rows)
        if not resp.get("success"):
            logger.warning("Supabase insert returned non-success: %s", resp)
    except Exception as e:
        logger.exception("Failed to save layer summary to Supabase: %s", e)


# small helper: checksum file
import hashlib, zipfile
try:
    import psutil
except Exception:
    psutil = None

def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# session state defaults
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

# show latest metrics from Supabase in sidebar
if sb_client:
    latest = sb_client.fetch_records("layer_summary", order_by="created_at", descending=True, limit=25)
    if latest:
        st.sidebar.subheader("Latest layer metrics")
        st.sidebar.dataframe(pd.DataFrame(latest))

# fetch function
def fetch_and_prepare():
    st.info(f"Fetching {symbol} {interval} from Yahoo …")
    if YahooTicker is None:
        st.error("yahooquery not installed."); return pd.DataFrame()
    try:
        tq = YahooTicker(symbol)
        raw = tq.history(start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            st.error("No data returned."); return pd.DataFrame()
        if isinstance(raw, dict): raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex): raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index); raw = raw.sort_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw and "adjclose" in raw: raw["close"] = raw["adjclose"]
        bars = raw[~raw.index.duplicated(keep="first")]
        st.success(f"Fetched {len(bars)} bars.")
        return bars
    except Exception as e:
        logger.exception("Fetch failed: %s", e); st.error(str(e)); return pd.DataFrame()

# run full pipeline with better error handling and progress
def run_full_pipeline():
    progress = st.progress(0)
    try:
        # 1 fetch
        progress.text("Fetching price data...")
        bars = fetch_and_prepare()
        progress.progress(5)
        if bars.empty:
            return {"error":"no bars"}
        bars = ensure_unique_index(bars)
        st.session_state.bars = bars
        # optional memory report
        if psutil:
            mem = psutil.virtual_memory()
            st.sidebar.text(f"RAM: {mem.percent}% used")
        # rvol
        try:
            bars["rvol"] = (bars["volume"] / bars["volume"].rolling(20, min_periods=1).mean()).fillna(1.0)
        except Exception:
            bars["rvol"] = 1.0
        progress.progress(15)
        # 2 candidates
        progress.text("Generating candidates...")
        cands = generate_candidates_and_labels(bars)
        st.session_state.cands = cands
        progress.progress(20)
        if cands.empty:
            st.warning("No candidates generated."); return {"error":"nocand"}
        # map candidate_time -> indices
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
        progress.progress(25)
        # 3 train
        if torch is None:
            st.error("Torch not installed — cannot train.")
            return {"error":"no_torch"}
        st.info("Training cascade (L1/L2/L3)… (this may take some time)")
        # progress widget for training
        train_progress = st.empty()
        train_bar = st.progress(0)
        trader = CascadeTrader(seq_len=seq_len, feat_windows=(5,10,20), device=device_choice)
        try:
            trader.fit(bars, events, l2_use_xgb=(xgb is not None), epochs_l1=8, epochs_l23=8, st_progress=train_bar)
            st.session_state.trader = trader
        except Exception as e:
            logger.exception("Training failed: %s", e)
            with st.expander("Training traceback"):
                st.text(traceback.format_exc())
            return {"error":"train_failed", "exception": str(e)}
        progress.progress(70)
        # 4 preds
        t_indices = np.array(cand_idx, dtype=int)
        preds = trader.predict_batch(bars, t_indices)
        st.write("Predictions (head):"); st.dataframe(preds.head(10))
        progress.progress(75)
        # 5 breadth/sweep
        if run_breadth:
            st.info("Running breadth backtest")
            res = run_breadth_levels(preds, cands, bars)
            st.session_state.last_breadth = res
            if res["summary"]:
                st.subheader("Breadth summary"); st.dataframe(pd.DataFrame(res["summary"]))
                try:
                    save_layer_summary(sb_client, symbol, res["summary"])
                except Exception as e:
                    logger.exception("Supabase save failed: %s", e)
            else:
                st.warning("Breadth returned no summary rows.")
        progress.progress(85)
        if run_sweep:
            st.info("Running grid sweep (light)")
            rr_vals = [2.0, 2.5, 3.0]
            sl_ranges = [(0.02,0.04),(0.01,0.03),(0.005,0.02)]
            mpt_list = [p_fast, p_slow, p_deep]
            sweep_results = run_grid_sweep(cands, bars, rr_vals, sl_ranges, mpt_list)
            st.session_state.last_sweep = sweep_results
            if sweep_results and isinstance(sweep_results.get("results"), pd.DataFrame):
                st.write("Sweep results (head):")
                st.dataframe(sweep_results["results"].head(10))
            st.success("Sweep completed.")
        progress.progress(90)
        # 6 export artifacts
        st.info("Exporting artifacts")
        out_dir = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        os.makedirs(out_dir, exist_ok=True)
        trader_path = os.path.join(out_dir, "cascade_trader")
        os.makedirs(trader_path, exist_ok=True)
        try:
            # L1
            torch.save(trader.l1.state_dict(), os.path.join(trader_path, "l1_state.pt"))
            # L3
            if trader.l3 is not None:
                torch.save(trader.l3.state_dict(), os.path.join(trader_path, "l3_state.pt"))
            # L2
            if trader.l2_backend == "xgb":
                try:
                    trader.l2_model.save_model(os.path.join(trader_path, "l2_xgb.json"))
                except Exception:
                    joblib.dump(trader.l2_model, os.path.join(trader_path, "l2_xgb.joblib"))
            else:
                if trader.l2_model is not None:
                    torch.save(trader.l2_model.state_dict(), os.path.join(trader_path, "l2_mlp_state.pt"))
            # scalers + metadata
            with open(os.path.join(trader_path, "scaler_seq.pkl"), "wb") as f:
                pickle.dump(trader.scaler_seq, f)
            with open(os.path.join(trader_path, "scaler_tab.pkl"), "wb") as f:
                pickle.dump(trader.scaler_tab, f)
            with open(os.path.join(trader_path, "metadata.json"), "w") as f:
                json.dump(trader.metadata, f, default=str, indent=2)
        except Exception as e:
            logger.exception("Saving artifacts failed: %s", e); st.error(f"Saving artifacts failed: {e}")
        # export bundles using export_model_and_metadata
        export_paths = {}
        try:
            l2_wrap = L2ExportWrapper()
            if trader.l2_backend == "xgb":
                l2_wrap.booster = trader.l2_model
            else:
                l2_wrap.booster = getattr(trader.l2_model, "state_dict", None)
            export_paths["l2"] = export_model_and_metadata(l2_wrap, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2_model"), save_fi=True)
            l3_wrap = L3ExportWrapper()
            l3_wrap.model = getattr(trader.l3, "state_dict", None)
            export_paths["l3"] = export_model_and_metadata(l3_wrap, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l3_model"), save_fi=True)
            st.session_state.export_paths = export_paths
            st.success(f"Exported artifacts to {out_dir}")
        except Exception as e:
            logger.exception("Export bundles failed: %s", e); st.error(f"Export bundles failed: {e}")
        # zip up artifacts and provide download + checksums
        try:
            zip_path = os.path.join(out_dir, "artifacts.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(trader_path):
                    for fn in files:
                        path = os.path.join(root, fn)
                        arcname = os.path.relpath(path, out_dir)
                        zf.write(path, arcname=arcname)
            csum = file_checksum(zip_path)
            with open(zip_path, "rb") as f:
                st.download_button("Download artifacts.zip", f.read(), file_name="artifacts.zip")
            st.write("Artifacts checksum (sha256):", csum)
        except Exception as e:
            logger.exception("Packaging artifacts failed: %s", e)
        progress.progress(100)
        return {"ok": True, "out_dir": out_dir}
    except Exception as e:
        logger.exception("Pipeline top-level failure: %s", e)
        with st.expander("Pipeline traceback"):
            st.text(traceback.format_exc())
        return {"error": "pipeline_failed", "exception": str(e)}