# app.py – Cascade Trading Application with Multi-Model Integration and Risk Manager
# Regenerated in 5 sequential chunks for clarity and manageability

import os
import io
import sys
import gc
import math
import json
import joblib
import torch
import xgboost as xgb
import traceback
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# ==============================
# GLOBAL CONFIG
# ==============================
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# Define paths for saving models, logs, and checkpoints
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================
# UTILS
# ==============================
def save_json(data: Dict[str, Any], filename: str):
    """Save dictionary to JSON file."""
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved JSON: {filepath}")


def load_json(filename: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    filepath = os.path.join(LOG_DIR, filename)
    if not os.path.exists(filepath):
        logging.warning(f"JSON file not found: {filepath}")
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


def memory_cleanup():
    """Free up memory by clearing cache and collecting garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Memory cleanup executed.")

# -----------------------------
# Chunk 2/5: data handling, features, candidates, datasets, model blocks
# -----------------------------

# Optional Yahoo fetcher - uses yahooquery if installed.
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

def fetch_price(symbol: str, start: Optional[str], end: Optional[str], interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV using yahooquery if available. Returns DataFrame with lowercase columns."""
    if YahooTicker is None:
        logging.warning("yahooquery not installed; fetch_price will return empty DataFrame.")
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
        df = raw[~raw.index.duplicated(keep="first")]
        logging.info("Fetched %d bars for %s", len(df), symbol)
        return df
    except Exception as exc:
        logging.exception("fetch_price failed: %s", exc)
        return pd.DataFrame()

# Engineered features (compact)
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    df = df.copy()
    f = pd.DataFrame(index=df.index)
    if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        return f
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)
    ret1 = c.pct_change().fillna(0.0)
    f["ret1"] = ret1
    f["logret1"] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f["tr"] = tr.fillna(0.0)
    for w in windows:
        f[f"rmean_{w}"] = c.pct_change(w).fillna(0.0)
        f[f"vol_{w}"] = ret1.rolling(w).std().fillna(0.0)
        f[f"tr_mean_{w}"] = tr.rolling(w).mean().fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method="bfill")
        roll_min = c.rolling(w).min().fillna(method="bfill")
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f"chanpos_{w}"] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# Candidate generator (triple-barrier-like simplified)
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
    if not {"high", "low", "close"}.issubset(bars.columns):
        raise KeyError("bars must contain high/low/close")
    tr = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = tr.rolling(atr_window, min_periods=1).mean()
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
            "direction": direction
        })
    return pd.DataFrame(recs)

# Torch dataset wrappers (if torch available)
if torch is not None:
    from torch.utils.data import Dataset, DataLoader

    class SequenceDataset(Dataset):
        def __init__(self, X_seq: np.ndarray, y: np.ndarray):
            self.X = X_seq.astype(np.float32)   # [N, T, F]
            self.y = y.astype(np.float32).reshape(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = self.X[idx].transpose(1, 0)  # [F, T]
            return x, self.y[idx]

    class TabDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
else:
    SequenceDataset = None
    TabDataset = None

# Simple conv blocks and small CNN (L1), MLP (L2/L3)
if torch is not None:
    class ConvBlock(torch.nn.Module):
        def __init__(self, c_in, c_out, k, d, pdrop):
            super().__init__()
            pad = (k - 1) * d // 2
            self.conv = torch.nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
            self.bn = torch.nn.BatchNorm1d(c_out)
            self.act = torch.nn.ReLU()
            self.drop = torch.nn.Dropout(pdrop)
            self.res = (c_in == c_out)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
            out = self.drop(out)
            if self.res:
                out = out + x
            return out

    class Level1ScopeCNN(torch.nn.Module):
        def __init__(self, in_features: int, channels=(32, 64, 128), kernel_sizes=(5, 3, 3), dilations=(1, 2, 4), dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
                d = dilations[min(i, len(dilations) - 1)]
                blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
            self.blocks = torch.nn.Sequential(*blocks)
            self.project = torch.nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = torch.nn.Linear(chs[-1], 1)

        def forward(self, x):
            z = self.blocks(x)
            z = self.project(z)
            z_pool = z.mean(dim=-1)
            logit = self.head(z_pool)
            return logit, z_pool

    class MLP(torch.nn.Module):
        def __init__(self, in_dim: int, hidden: List[int], out_dim: int = 1, dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [torch.nn.Linear(last, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
                last = h
            layers += [torch.nn.Linear(last, out_dim)]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Level3ShootMLP(torch.nn.Module):
        def __init__(self, in_dim: int, hidden=(128, 64), dropout=0.1, use_reg_head=True):
            super().__init__()
            self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
            self.cls_head = torch.nn.Linear(128, 1)
            self.reg_head = torch.nn.Linear(128, 1) if use_reg_head else None

        def forward(self, x):
            h = self.backbone(x)
            logit = self.cls_head(h)
            ret = self.reg_head(h) if self.reg_head is not None else None
            return logit, ret

    class TemperatureScaler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = torch.nn.Parameter(torch.zeros(1))

        def forward(self, logits):
            T = torch.exp(self.log_temp)
            return logits / T

        def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200, lr=1e-2):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            logits_t = torch.tensor(logits.reshape(-1, 1), dtype=torch.float32, device=device)
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
            opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
            bce = torch.nn.BCEWithLogitsLoss()

            def closure():
                opt.zero_grad()
                scaled = self.forward(logits_t)
                loss = bce(scaled, y_t)
                loss.backward()
                return loss

            try:
                opt.step(closure)
            except Exception as e:
                logging.warning("Temperature scaling failed: %s", e)


# -----------------------------
# Chunk 3/5: training helpers and CascadeTrader
# -----------------------------

def _device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def train_torch_classifier(model: torch.nn.Module,
                           train_ds,
                           val_ds,
                           lr: float = 1e-3,
                           epochs: int = 10,
                           patience: int = 3,
                           pos_weight: float = 1.0,
                           device: str = "auto"):
    dev = _device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_tensor = torch.tensor([pos_weight], device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 128), shuffle=True)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=getattr(val_ds, "batch_size", 512), shuffle=False)

    best_loss = float("inf")
    best_state = None
    no_imp = 0
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        train_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = train_loss / max(1, n)
        model.eval()
        val_loss = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = loss_fn(logit, yb)
                val_loss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
        val_loss = val_loss / max(1, vn)
        history["train"].append(train_loss); history["val"].append(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "history": history}

class CascadeTrader:
    def __init__(self, seq_len: int = 64, feat_windows=(5, 10, 20), device: str = "auto"):
        if torch is None:
            raise RuntimeError("Torch is required for CascadeTrader.")
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

    def fit(self, df: pd.DataFrame, events: pd.DataFrame, l2_use_xgb: bool = True, epochs_l1: int = 8, epochs_l23: int = 8):
        """Fit L1 (CNN) → generate embeddings → L2 (xgb or MLP) + L3 (MLP) trained in parallel."""
        t0 = time.time()
        df = ensure_unique_index(df)
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open', 'high', 'low', 'close', 'volume']
        micro_cols = [c for c in ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10'] if c in eng.columns or c in df.columns]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1, sort=False).fillna(0.0)
        feat_tab_df = eng.fillna(0.0)
        self.tab_feature_names = list(feat_tab_df.columns)
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        tr_idx, va_idx = train_test_split(np.arange(len(idx)), test_size=test_size, random_state=42, stratify=y)
        idx_tr, idx_va = idx[tr_idx], idx[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
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
        self.l1 = Level1ScopeCNN(in_features=in_features, channels=(32, 64, 128), kernel_sizes=(5, 3, 3), dilations=(1, 2, 4), dropout=0.1)
        self.l1, l1_hist = train_torch_classifier(self.l1, ds_l1_tr, ds_l1_va, lr=1e-3, epochs=epochs_l1, patience=3, pos_weight=1.0, device=str(self.device))
        # infer embeddings for all event indices
        all_idx_seq = to_sequences(X_seq_all_scaled, idx, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(all_idx_seq)
        l1_emb_tr, l1_emb_va = l1_emb[tr_idx], l1_emb[va_idx]
        Xtab_tr = X_tab_all_scaled[idx_tr]; Xtab_va = X_tab_all_scaled[idx_va]
        X_l2_tr = np.hstack([l1_emb_tr, Xtab_tr]); X_l2_va = np.hstack([l1_emb_va, Xtab_va])

        # train L2 & L3 in parallel
        results = {}
        def l2_xgb_job():
            if xgb is None:
                raise RuntimeError("xgboost not installed")
            clf = xgb.XGBClassifier(n_estimators=num_boost, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_l2_tr, y_tr, eval_set=[(X_l2_va, y_va)], verbose=False)
            return ("xgb", clf)

        def l2_mlp_job():
            in_dim = X_l2_tr.shape[1]
            m = MLP(in_dim, [128, 64], out_dim=1, dropout=0.1)
            ds2_tr = TabDataset(X_l2_tr, y_tr); ds2_va = TabDataset(X_l2_va, y_va)
            ds2_tr.batch_size = 256; ds2_va.batch_size = 512
            m, _ = train_torch_classifier(m, ds2_tr, ds2_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device))
            return ("mlp", m)

        def l3_job():
            in_dim = X_l2_tr.shape[1]
            m3 = Level3ShootMLP(in_dim, hidden=(128, 64), dropout=0.1, use_reg_head=True)
            ds3_tr = TabDataset(X_l2_tr, y_tr); ds3_va = TabDataset(X_l2_va, y_va)
            ds3_tr.batch_size = 256; ds3_va.batch_size = 512
            m3, hist3 = train_torch_classifier(m3, ds3_tr, ds3_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device))
            return ("l3", (m3, hist3))

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if (xgb is not None) and l2_use_xgb:
                futures[ex.submit(l2_xgb_job)] = "l2"
            else:
                futures[ex.submit(l2_mlp_job)] = "l2"
            futures[ex.submit(l3_job)] = "l3"
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results[label] = fut.result()
                except Exception as e:
                    logging.exception("Parallel train error for %s: %s", label, e)
                    results[label] = None

        if results.get("l2") is not None:
            self.l2_backend, self.l2_model = results["l2"]
        if results.get("l3") is not None:
            self.l3 = results["l3"][1][0] if isinstance(results["l3"][1], tuple) else results["l3"][1]

        # calibrate temps (best-effort)
        try:
            self.l1_temp.fit(l1_logits.reshape(-1, 1), y)
        except Exception as e:
            logging.warning("L1 temp scale failed: %s", e)
        try:
            l3_val_logits = self._l3_infer_logits(X_l2_va)
            self.l3_temp.fit(l3_val_logits.reshape(-1, 1), y_va)
        except Exception as e:
            logging.warning("L3 temp scale failed: %s", e)

        self.metadata = {"l1_hist": l1_hist, "fit_time_sec": round(time.time() - t0, 2), "l2_backend": self.l2_backend}
        self._fitted = True
        logging.info("Cascade fit completed in %.2fs", time.time() - t0)
        return self

    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        self.l1.eval()
        logits = []; embeds = []
        batch = 256
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0, 2, 1), dtype=torch.float32, device=self.device)
                logit, emb = self.l1(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1, 1), np.concatenate(embeds, axis=0)

    def _l3_infer_logits(self, X: np.ndarray):
        self.l3.eval()
        logits = []
        batch = 2048
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1, 1)

    def _mlp_predict_proba(self, model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
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

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray):
        assert self._fitted, "Call fit() first."
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open', 'high', 'low', 'close', 'volume']
        micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']
        use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(eng.columns)]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1)[use_cols].fillna(0.0)
        feat_tab_df = eng[self.tab_feature_names].fillna(0.0)
        X_seq_all_scaled = self.scaler_seq.transform(feat_seq_df.values)
        X_tab_all_scaled = self.scaler_tab.transform(feat_tab_df.values)
        Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        l1_logits_scaled = self.l1_temp.forward(torch.tensor(l1_logits, dtype=torch.float32)).numpy().reshape(-1)
        p1 = 1.0 / (1.0 + np.exp(-l1_logits_scaled))
        go2 = p1 >= 0.30
        X_l2 = np.hstack([l1_emb, X_tab_all_scaled[t_indices]])
        if self.l2_backend == "xgb":
            try:
                p2 = self.l2_model.predict_proba(X_l2)[:, 1]
            except Exception:
                p2 = np.zeros(len(X_l2))
        else:
            p2 = self._mlp_predict_proba(self.l2_model, X_l2)
        go3 = (p2 >= 0.55) & go2
        p3 = np.zeros_like(p1); rhat = np.zeros_like(p1)
        if go3.any() and self.l3 is not None:
            X_l3 = X_l2[go3]
            l3_logits = self._l3_infer_logits(X_l3)
            l3_logits_scaled = self.l3_temp.forward(torch.tensor(l3_logits, dtype=torch.float32)).numpy().reshape(-1)
            p3_vals = 1.0 / (1.0 + np.exp(-l3_logits_scaled))
            p3[go3] = p3_vals
            rhat[go3] = p3_vals - 0.5
        trade = (p3 >= 0.65) & go3
        size = np.clip(rhat, 0.0, None) * trade.astype(float)
        return pd.DataFrame({"t": t_indices, "p1": p1, "p2": p2, "p3": p3, "go2": go2.astype(int), "go3": go3.astype(int), "trade": trade.astype(int), "size": size})

# -----------------------------
# Chunk 4/5: export_model_and_metadata, breadth/sweep helpers
# -----------------------------

def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str,Any], model_basename: str, save_fi: bool = True):
    """
    Save model artifacts and a portable .pt bundle for each model_wrapper.
    Returns dict of saved paths.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    paths = {}
    try:
        # native xgb model file if booster exists (support both booster attr or direct model)
        try:
            if hasattr(model_wrapper, "booster") and getattr(model_wrapper, "booster") is not None:
                model_file = f"{model_basename}_{ts}.model"
                try:
                    logging.info(f"Exporting native model file to {model_file}")
                    # try save_model method first (xgboost Booster)
                    if hasattr(model_wrapper.booster, "save_model"):
                        model_wrapper.booster.save_model(model_file)
                    else:
                        joblib.dump(model_wrapper.booster, model_file)
                    paths["model"] = model_file
                    logging.info(f"Successfully saved native model file to {model_file}")
                except Exception as e:
                    error_msg = f"Failed to save native booster file: {str(e)}\n{traceback.format_exc()}"
                    logging.error(error_msg)
                    paths["model_error"] = error_msg
        except Exception:
            pass

        # metadata
        meta_file = f"{model_basename}_{ts}.json"
        try:
            logging.info(f"Exporting metadata to {meta_file}")
            with open(meta_file, "w") as f:
                json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, default=str, indent=2)
            paths["meta"] = meta_file
            logging.info(f"Successfully saved metadata to {meta_file}")
        except Exception as e:
            error_msg = f"Failed to save metadata file: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            paths["meta_error"] = error_msg

        # feature importance
        if save_fi and hasattr(model_wrapper, "feature_importance"):
            try:
                logging.info("Generating feature importance data")
                fi_df = model_wrapper.feature_importance()
                fi_path = f"{model_basename}_{ts}_fi.csv"
                fi_df.to_csv(fi_path, index=False)
                paths["feature_importance"] = fi_path
                logging.info(f"Successfully saved feature importance to {fi_path}")
            except Exception as e:
                error_msg = f"Could not save feature importance: {str(e)}\n{traceback.format_exc()}"
                logging.error(error_msg)
                paths["fi_error"] = error_msg

        # Save portable .pt as a bundle (joblib or torch.save)
        pt_path = f"{model_basename}_{ts}.pt"
        try:
            logging.info(f"Preparing to save .pt bundle to {pt_path}")
            payload = {"model_wrapper": model_wrapper, "features": feature_list, "metrics": metrics, "export_log": []}
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Starting export of .pt bundle")
            if 'torch' in globals() and torch is not None:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Using torch.save for .pt export")
                # torch.save may fail on local or non-picklable objects; try and fallback
                try:
                    torch.save(payload, pt_path)
                except Exception:
                    joblib.dump(payload, pt_path)
            else:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] torch not available, using joblib for .pt export")
                joblib.dump(payload, pt_path)
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Successfully saved .pt bundle")
            paths["pt"] = pt_path
            logging.info(f"Successfully saved .pt bundle to {pt_path}")
        except Exception as e:
            error_msg = f"Failed to save .pt payload: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            paths["pt_error"] = error_msg
            # fallback
            fallback = f"{model_basename}_{ts}_fallback.pt"
            try:
                logging.info(f"Attempting joblib fallback to {fallback}")
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
                logging.info(f"Successfully saved fallback .pt bundle to {fallback}")
            except Exception as fallback_e:
                fallback_error = f"Fallback also failed: {str(fallback_e)}\n{traceback.format_exc()}"
                logging.error(fallback_error)
                paths["pt_fallback_error"] = fallback_error
    except Exception as exc:
        error_msg = f"Failed to export model: {str(exc)}\n{traceback.format_exc()}"
        logging.exception(error_msg)
        paths["export_error"] = error_msg

    # verify
    if "pt" in paths:
        try:
            logging.info(f"Verifying .pt bundle at {paths['pt']}")
            pt_file_size = os.path.getsize(paths["pt"])
            if pt_file_size > 0:
                logging.info(f"Verified .pt bundle exists with size {pt_file_size} bytes")
                paths["pt_verification"] = f"File size: {pt_file_size} bytes"
            else:
                logging.warning(f".pt bundle has zero size")
                paths["pt_verification"] = "Warning: Zero file size"
        except Exception as e:
            verification_error = f"PT verification failed: {str(e)}"
            logging.error(verification_error)
            paths["pt_verification_error"] = verification_error

    return paths

# Breadth levels runner (applies exclusive ranges, simulates per-level)
def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame) -> Dict[str, Any]:
    out = {"detailed": {}, "summary": []}
    if preds is None or preds.empty or cands is None or cands.empty:
        return out
    # Try to align by candidate ordering: preds.t corresponds to candidate index mapping in pipeline.
    # Build a simple 'signal' in 0-10 from p3 if available else p2/p1 fallback
    if "p3" in preds.columns:
        sig = (preds["p3"].fillna(0.0) * 10.0).values
    elif "p2" in preds.columns:
        sig = (preds["p2"].fillna(0.0) * 10.0).values
    else:
        sig = (preds["p1"].fillna(0.0) * 10.0).values
    df = cands.copy().reset_index(drop=True)
    # Ensure length alignment
    L = min(len(df), len(sig))
    df = df.iloc[:L].copy()
    df["signal"] = sig[:L]
    def run_level(name, buy_min, buy_max, sl_pct, rr):
        sel = df[(df["signal"] >= buy_min) & (df["signal"] <= buy_max)].copy()
        if sel.empty:
            out["detailed"][name] = pd.DataFrame()
            return
        tp_pct = rr * sl_pct
        sel["pred_label"] = (sel["signal"] >= buy_min).astype(int)
        trades = simulate_limits(sel, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
        out["detailed"][name] = trades
        s = summarize_trades(trades)
        if not s.empty:
            row = s.iloc[0].to_dict(); row["mode"] = name
            out["summary"].append(row)
    # choose default sl/rr if not defined in environment (safe fallback)
    l1_sl = float(globals().get("lvl1_sl_min", 0.02) + globals().get("lvl1_sl_max", 0.04)) / 2.0 if globals().get("lvl1_sl_min", None) is not None else 0.02
    l1_rr = float(globals().get("lvl1_rr_min", 1.0) + globals().get("lvl1_rr_max", 2.5)) / 2.0 if globals().get("lvl1_rr_min", None) is not None else 1.5
    l2_sl = float(globals().get("lvl2_sl_min", 0.01) + globals().get("lvl2_sl_max", 0.03)) / 2.0 if globals().get("lvl2_sl_min", None) is not None else 0.02
    l2_rr = float(globals().get("lvl2_rr_min", 2.0) + globals().get("lvl2_rr_max", 3.5)) / 2.0 if globals().get("lvl2_rr_min", None) is not None else 2.5
    l3_sl = float(globals().get("lvl3_sl_min", 0.005) + globals().get("lvl3_sl_max", 0.02)) / 2.0 if globals().get("lvl3_sl_min", None) is not None else 0.01
    l3_rr = float(globals().get("lvl3_rr_min", 3.0) + globals().get("lvl3_rr_max", 5.0)) / 2.0 if globals().get("lvl3_rr_min", None) is not None else 3.5
    run_level("L1", lvl1_buy_min, lvl1_buy_max, l1_sl, l1_rr)
    run_level("L2", lvl2_buy_min, lvl2_buy_max, l2_sl, l2_rr)
    run_level("L3", lvl3_buy_min, lvl3_buy_max, l3_sl, l3_rr)
    return out

# grid sweep helper
def run_grid_sweep(clean: pd.DataFrame, bars: pd.DataFrame, rr_vals: List[float], sl_ranges: List[Tuple[float, float]], mpt_list: List[float]):
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
                    df["pred_prob"] = df.get("pred_prob", 0.0)
                    df["pred_label"] = (df["pred_prob"] >= mpt).astype(int)
                    trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
                    results[key] = {"trades_count": len(trades), "trades": trades}
                except Exception as exc:
                    logging.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results

# -----------------------------
# Chunk 5/5: Streamlit actions, pipeline, supabase summary saving, SQL schema
# -----------------------------

# Supabase client wrapper optional (user-provided file)
try:
    from supabase_client_wrapper import SupabaseClientWrapper
except Exception:
    SupabaseClientWrapper = None

# Instantiate supabase client if the env is present and wrapper available
sb_client = None
if SupabaseClientWrapper is not None and SUPABASE_URL and SUPABASE_KEY:
    try:
        sb_client = SupabaseClientWrapper(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logging.warning("Supabase init failed: %s", e)
        sb_client = None

def save_layer_summary_to_supabase(sb: Optional[SupabaseClientWrapper], symbol: str, summary_rows: List[Dict[str, Any]]):
    if sb is None or not summary_rows:
        return
    payload = []
    now = datetime.utcnow().isoformat()
    for r in summary_rows:
        payload.append({
            "symbol": symbol,
            "mode": r.get("mode", ""),
            "total_trades": int(r.get("total_trades") or 0),
            "win_rate": float(r.get("win_rate") or 0.0),
            "avg_pnl": float(r.get("avg_pnl") or 0.0),
            "median_pnl": float(r.get("median_pnl") or 0.0),
            "total_pnl": float(r.get("total_pnl") or 0.0),
            "max_drawdown": float(r.get("max_drawdown") or 0.0),
            "start_time": str(r.get("start_time")) if r.get("start_time") is not None else None,
            "end_time": str(r.get("end_time")) if r.get("end_time") is not None else None,
            "created_at": now
        })
    try:
        resp = sb.insert_data("layer_summary", payload)
        logging.info("Supabase insert response: %s", resp)
    except Exception as e:
        logging.exception("Failed to insert to supabase: %s", e)

# Fetch prepared for dashboard load (non-blocking / quick)
with st.spinner("Fetching latest price snapshot..."):
    bars_preview = fetch_price(symbol, start=(datetime.today() - timedelta(days=7)).isoformat(), end=datetime.today().isoformat(), interval=interval)
    if not bars_preview.empty:
        st.sidebar.metric("Latest close", f"{bars_preview['close'].iloc[-1]:.2f}")
        if "volume" in bars_preview.columns:
            st.sidebar.metric("Latest volume", f"{bars_preview['volume'].iloc[-1]:.0f}")
    else:
        st.sidebar.write("No preview data available (yahooquery missing or fetch failed).")

def run_full_pipeline():
    # 1) fetch full dataset
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars.empty:
        st.error("No bars fetched — aborting pipeline.")
        return {"error": "no_bars"}
    st.session_state.bars = bars
    bars = ensure_unique_index(bars)
    # rvol
    try:
        bars["rvol"] = (bars["volume"] / bars["volume"].rolling(20, min_periods=1).mean()).fillna(1.0)
    except Exception:
        bars["rvol"] = 1.0
    # 2) candidates
    cands = generate_candidates_and_labels(bars, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.warning("No candidates generated for given window/params.")
        st.session_state.cands = pd.DataFrame()
        return {"error": "no_cands"}
    st.session_state.cands = cands
    st.success(f"Generated {len(cands)} candidates.")
    # 3) map candidate times to indices for events
    bar_index_map = {t: i for i, t in enumerate(bars.index)}
    cand_idx = []
    for t in cands["candidate_time"]:
        t0 = pd.Timestamp(t)
        if t0 in bar_index_map:
            cand_idx.append(bar_index_map[t0])
        else:
            # use nearest past bar
            locs = bars.index[bars.index <= t0]
            cand_idx.append(int(bar_index_map[locs[-1]]) if len(locs) else 0)
    events = pd.DataFrame({"t": np.array(cand_idx, dtype=int), "y": cands["label"].astype(int).values})
    # 4) train cascade
    if torch is None:
        st.error("Torch not available in environment. Install torch to train models.")
        return {"error": "no_torch"}
    st.info("Training cascade (L1/L2/L3). This may take time.")
    trader = CascadeTrader(seq_len=seq_len, feat_windows=(5,10,20), device=device_choice)
    try:
        trader.fit(bars, events, l2_use_xgb=(xgb is not None), epochs_l1=8, epochs_l23=8)
        st.success("Cascade training complete.")
        st.session_state.trader = trader
    except Exception as e:
        logging.exception("Training failed: %s", e)
        st.error(f"Training failed: {e}")
        return {"error": "train_failed", "exception": str(e)}
    # 5) predictions for candidates & breadth/backtest
    t_indices = np.array(cand_idx, dtype=int)
    preds = trader.predict_batch(bars, t_indices)
    st.write("Prediction head (first 10):")
    st.dataframe(preds.head(10))
    # breadth
    if run_breadth:
        st.info("Running breadth backtest for L1/L2/L3")
        res = run_breadth_levels(preds, cands, bars)
        st.session_state.last_breadth = res
        if res["summary"]:
            st.subheader("Breadth summary")
            st.dataframe(pd.DataFrame(res["summary"]))
            # push to supabase if configured
            save_layer_summary_to_supabase(sb_client, symbol, res["summary"])
        else:
            st.warning("Breadth returned no summary rows.")
    # sweep
    if run_sweep:
        st.info("Running light grid sweep (may be slow)")
        rr_vals = [2.0, 2.5, 3.0]
        sl_ranges = [(0.02, 0.04), (0.01, 0.03), (0.005, 0.02)]
        mpt_list = [p_fast, p_slow, p_deep]
        sweep = run_grid_sweep(cands, bars, rr_vals, sl_ranges, mpt_list)
        st.session_state.last_sweep = sweep
        st.success("Grid sweep finished.")
    # 6) export artifacts
    out_dir = os.path.join(MODEL_DIR, f"cascade_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        # Save torch state_dicts
        torch.save(trader.l1.state_dict(), os.path.join(out_dir, "l1_state.pt"))
        if trader.l3 is not None:
            torch.save(trader.l3.state_dict(), os.path.join(out_dir, "l3_state.pt"))
        # L2 save
        if trader.l2_backend == "xgb" and trader.l2_model is not None:
            try:
                trader.l2_model.save_model(os.path.join(out_dir, "l2_xgb.json"))
            except Exception:
                joblib.dump(trader.l2_model, os.path.join(out_dir, "l2_xgb.joblib"))
        elif trader.l2_model is not None:
            try:
                torch.save(trader.l2_model.state_dict(), os.path.join(out_dir, "l2_mlp_state.pt"))
            except Exception:
                joblib.dump(trader.l2_model, os.path.join(out_dir, "l2_mlp.joblib"))
        # scalers & metadata
        with open(os.path.join(out_dir, "scaler_seq.pkl"), "wb") as f:
            pickle.dump(trader.scaler_seq, f)
        with open(os.path.join(out_dir, "scaler_tab.pkl"), "wb") as f:
            pickle.dump(trader.scaler_tab, f)
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(trader.metadata, f, default=str, indent=2)
        st.success(f"Saved artifacts to {out_dir}")
        # Export .pt bundles using export_model_and_metadata with simple wrappers (module-level classes recommended)
        export_paths = {}
        # L2 wrapper
        class L2Wrapper:
            def __init__(self, booster=None):
                self.booster = booster
            def feature_importance(self):
                try:
                    if hasattr(self.booster, "get_score"):
                        imp = self.booster.get_score(importance_type="gain")
                        return pd.DataFrame([(k, imp.get(k, 0.0)) for k in (trader.tab_feature_names)], columns=["feature", "gain"])
                except Exception:
                    return pd.DataFrame([{"feature": "none", "gain": 0.0}])
        l2w = L2Wrapper(getattr(trader, "l2_model", None))
        export_paths["l2"] = export_model_and_metadata(l2w, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2"), save_fi=True)
        # L3 wrapper
        class L3Wrapper:
            def __init__(self, model=None):
                self.model = model
            def feature_importance(self):
                return pd.DataFrame([{"feature": "l3_emb", "gain": 1.0}])
        l3w = L3Wrapper(getattr(trader, "l3", None))
        export_paths["l3"] = export_model_and_metadata(l3w, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l3"), save_fi=True)
        st.session_state.export_paths = export_paths
        st.write("Export paths:", export_paths)
    except Exception as e:
        logging.exception("Export failed: %s", e)
        st.error(f"Export failed: {e}")
    return {"ok": True, "out_dir": out_dir}

# Pipeline button handler
if run_full_pipeline_btn:
    with st.spinner("Running full pipeline... this can take several minutes"):
        res = run_full_pipeline()
        if res.get("error"):
            st.error(f"Pipeline error: {res.get('error')}")
        else:
            st.success("Pipeline finished successfully.")
            if "export_paths" in st.session_state:
                st.write("Exports:", st.session_state.export_paths)

# SQL schema helper for the summary table (displayed for convenience)
st.sidebar.markdown("#### Supabase SQL (layer_summary table)")
st.sidebar.code("""\
create extension if not exists "uuid-ossp";

create table public.layer_summary (
  id             uuid primary key default uuid_generate_v4(),
  symbol         text,
  mode           text,
  total_trades   integer,
  win_rate       numeric,
  avg_pnl        numeric,
  median_pnl     numeric,
  total_pnl      numeric,
  max_drawdown   numeric,
  start_time     timestamptz,
  end_time       timestamptz,
  created_at     timestamptz default now()
);

create index if not exists layer_summary_created_idx
  on public.layer_summary (created_at desc);
""", language="sql")