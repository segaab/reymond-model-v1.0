# Chunk 5/8: CascadeTrader and models (from your cascade_trader.py content)

# Config dataclasses for the cascade
@dataclass
class Level1Config:
    seq_len: int = 64
    in_features: int = 12
    channels: Tuple[int, ...] = (32, 64, 128)
    kernel_sizes: Tuple[int, ...] = (5, 3, 3)
    dilations: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    early_stop_patience: int = 5
    class_weight_pos: float = 1.0

@dataclass
class Level2Config:
    backend: str = "xgb"
    xgb_params: Dict[str, Any] = None
    mlp_hidden: Tuple[int, ...] = (128, 64)
    mlp_dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 15
    early_stop_patience: int = 4

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist"
            )

@dataclass
class Level3Config:
    hidden: Tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 20
    early_stop_patience: int = 5
    use_regression_head: bool = True

@dataclass
class GateConfig:
    th1: float = 0.30
    th2: float = 0.55
    th3: float = 0.65
    auto_target_precision: Optional[float] = None
    compute_budget_frac: Optional[float] = None

@dataclass
class FitConfig:
    device: str = "auto"
    val_size: float = 0.2
    random_state: int = 42
    l1_seq_len: int = 64
    feature_windows: Tuple[int, ...] = (5, 10, 20)
    lookahead: int = 20
    output_dir: str = "./artifacts"

# Utilities (engineered features)
def compute_engineered_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df.get('volume', pd.Series(0, index=df.index)).astype(float)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(w*3).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        f[f'kurt_{w}'] = ret1.rolling(w).kurt().fillna(0.0).replace([np.inf,-np.inf],0.0)
        f[f'skew_{w}'] = ret1.rolling(w).skew().fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)

    f = f.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return f

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

# Datasets
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = X_seq.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].transpose(1, 0)  # [T, F] -> [F, T]
        y = self.y[idx]
        return x, y

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Models (ConvBlock, Level1, MLP, Level3)
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
    def __init__(self, cfg: Level1Config):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.in_features] + list(cfg.channels)
        blocks = []
        # build blocks defensively
        ks = list(cfg.kernel_sizes) + [cfg.kernel_sizes[-1]] * (len(chs)-2)
        ds = list(cfg.dilations) + [cfg.dilations[-1]] * (len(chs)-2)
        for i in range(1, len(chs)):
            blocks.append(ConvBlock(chs[i-1], chs[i], ks[i-1], ds[i-1], cfg.dropout))
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)

    @property
    def embedding_dim(self):
        return self.cfg.channels[-1]

    def forward(self, x):
        # x: [B, F, T]
        z = self.blocks(x)
        z = self.project(z)          # [B, C, T]
        z_pool = z.mean(dim=-1)      # [B, C]
        logit = self.head(z_pool)    # [B, 1]
        return logit, z_pool         # logits + embedding

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
    def __init__(self, in_dim: int, cfg: Level3Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLP(in_dim, list(cfg.hidden), out_dim=128, dropout=cfg.dropout)
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Linear(128, 1) if cfg.use_regression_head else None

    def forward(self, x):
        h = self.backbone(x)
        logit = self.cls_head(h)
        ret = self.reg_head(h) if self.reg_head is not None else None
        return logit, ret

# Temperature scaler
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

# Chunk 6/8: Training helpers, L2 fallback MLP, and CascadeTrader wrapper

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
                           device: str = "auto") -> Tuple[nn.Module, Dict[str, Any]]:
    dev = _device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
    train_loader = DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 256), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)

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
        history['train'].append(train_loss)
        history['val'].append(val_loss)

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

# Level2 MLP fallback & trainer
class Level2MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, out_dim=1, dropout=dropout)

    def forward(self, x):
        return self.mlp(x)

def train_tab_mlp(model: nn.Module,
                  train_ds: Dataset,
                  val_ds: Dataset,
                  lr: float,
                  epochs: int,
                  patience: int,
                  device: str = "auto") -> Tuple[nn.Module, Dict[str, Any]]:
    dev = _device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    tl = DataLoader(train_ds, batch_size=512, shuffle=True)
    vl = DataLoader(val_ds, batch_size=1024, shuffle=False)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    history = dict(train=[], val=[])

    for epoch in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logit = model(xb)
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(xb); n += len(xb)
        train_loss = loss_sum / max(n, 1)

        model.eval()
        vloss_sum, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(dev), yb.to(dev)
                logit = model(xb)
                loss = bce(logit, yb)
                vloss_sum += float(loss.item()) * len(xb); vn += len(xb)
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

# CascadeTrader wrapper (simplified fit/predict based on your cascade)
class CascadeTrader:
    def __init__(self,
                 l1_cfg: Level1Config = Level1Config(),
                 l2_cfg: Level2Config = Level2Config(),
                 l3_cfg: Level3Config = Level3Config(),
                 gate_cfg: GateConfig = GateConfig(),
                 fit_cfg: FitConfig = FitConfig()):
        self.l1_cfg = l1_cfg
        self.l2_cfg = l2_cfg
        self.l3_cfg = l3_cfg
        self.gate_cfg = gate_cfg
        self.fit_cfg = fit_cfg

        self.device = _device(fit_cfg.device)
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()

        self.l1 = Level1ScopeCNN(self.l1_cfg)
        self.l1_temp = TemperatureScaler()

        self.l2_backend = None
        self.l2_model = None

        self.l3 = None
        self.l3_temp = TemperatureScaler()

        self.embed_dim = self.l1_cfg.channels[-1]
        self.tab_feature_names: List[str] = []
        self.autofocus_buffer = deque(maxlen=2000)
        self._fitted = False
        self.metadata: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, events: pd.DataFrame):
        t0 = time.time()
        eng = compute_engineered_features(df, windows=self.fit_cfg.feature_windows)

        seq_cols = ['open','high','low','close','volume']
        micro_cols = ['ret1','tr','vol_5','vol_10','mom_5','chanpos_10','chanpos_20']
        use_cols = [c for c in seq_cols + micro_cols if c in (list(df.columns) + list(eng.columns))]

        feat_seq = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[use_cols].replace([np.inf,-np.inf],0.0).fillna(0.0)
        feat_tab = eng.copy()

        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values

        train_idx, val_idx = train_test_split(np.arange(len(idx)), test_size=self.fit_cfg.val_size,
                                             random_state=self.fit_cfg.random_state, stratify=y)
        idx_train, idx_val = idx[train_idx], idx[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_seq_all = feat_seq.values
        self.scaler_seq.fit(X_seq_all)
        X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)

        X_tab_all = feat_tab.values
        self.tab_feature_names = list(feat_tab.columns)
        self.scaler_tab.fit(X_tab_all)
        X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)

        Xseq_train = to_sequences(X_seq_all_scaled, idx_train, seq_len=self.l1_cfg.seq_len)
        Xseq_val = to_sequences(X_seq_all_scaled, idx_val, seq_len=self.l1_cfg.seq_len)
        ds_l1_train = SequenceDataset(Xseq_train, y_train)
        ds_l1_val = SequenceDataset(Xseq_val, y_val)
        ds_l1_train.batch_size = self.l1_cfg.batch_size

        # Train L1
        self.l1, l1_hist = train_torch_classifier(
            self.l1, ds_l1_train, ds_l1_val,
            lr=self.l1_cfg.lr,
            epochs=self.l1_cfg.epochs,
            patience=self.l1_cfg.early_stop_patience,
            pos_weight=self.l1_cfg.class_weight_pos,
            device=self.fit_cfg.device
        )

        l1_val_logits, _ = self._l1_infer_logits_emb(Xseq_val)
        try:
            self.l1_temp.fit(l1_val_logits, y_val)
        except Exception:
            pass

        l1_train_logits, l1_train_emb = self._l1_infer_logits_emb(Xseq_train)
        l1_val_logits, l1_val_emb = self._l1_infer_logits_emb(Xseq_val)

        Xtab_train = X_tab_all_scaled[idx_train]
        Xtab_val = X_tab_all_scaled[idx_val]

        X_l2_train = np.hstack([l1_train_emb, Xtab_train])
        X_l2_val = np.hstack([l1_val_emb, Xtab_val])

        # L2 training
        self.l2_backend = self.l2_cfg.backend if (xgb is not None and self.l2_cfg.backend == "xgb") else "mlp"
        if self.l2_backend == "xgb":
            self.l2_model = xgb.XGBClassifier(**self.l2_cfg.xgb_params)
            self.l2_model.fit(X_l2_train, y_train, eval_set=[(X_l2_val, y_val)], verbose=False)
        else:
            in_dim = X_l2_train.shape[1]
            self.l2_model = Level2MLP(in_dim, list(self.l2_cfg.mlp_hidden), self.l2_cfg.mlp_dropout)
            ds2_tr = TabDataset(X_l2_train, y_train)
            ds2_va = TabDataset(X_l2_val, y_val)
            self.l2_model, _ = train_tab_mlp(self.l2_model, ds2_tr, ds2_va,
                                             lr=self.l2_cfg.lr, epochs=self.l2_cfg.epochs,
                                             patience=self.l2_cfg.early_stop_patience,
                                             device=self.fit_cfg.device)

        # L3 training
        X_l3_train = X_l2_train
        X_l3_val = X_l2_val
        self.l3 = Level3ShootMLP(X_l3_train.shape[1], self.l3_cfg)
        ds3_tr = TabDataset(X_l3_train, y_train)
        ds3_va = TabDataset(X_l3_val, y_val)
        self.l3, l3_hist = train_torch_classifier(
            self.l3, ds3_tr, ds3_va,
            lr=self.l3_cfg.lr,
            epochs=self.l3_cfg.epochs,
            patience=self.l3_cfg.early_stop_patience,
            pos_weight=1.0,
            device=self.fit_cfg.device
        )
        l3_val_logits = self._l3_infer_logits(X_l3_val)
        try:
            self.l3_temp.fit(l3_val_logits, y_val)
        except Exception:
            pass

        self.metadata = {
            "l1_hist": l1_hist,
            "l3_hist": l3_hist,
            "l2_backend": self.l2_backend,
            "feature_windows": self.fit_cfg.feature_windows,
            "seq_len": self.l1_cfg.seq_len,
            "embed_dim": self.embed_dim,
            "tab_features": self.tab_feature_names,
            "gate_cfg": asdict(self.gate_cfg),
            "fit_time_sec": round(time.time() - t0, 2)
        }
        self._fitted = True
        return self

    # ---------- inference helpers ----------
    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        self.l1.eval()
        dev = self.device
        logits, embeds = [], []
        with torch.no_grad():
            for i in range(0, len(Xseq), 1024):
                xb = torch.tensor(Xseq[i:i+1024].transpose(0,2,1), dtype=torch.float32, device=dev)
                logit, emb = self.l1(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        logits = np.concatenate(logits, axis=0).reshape(-1, 1)
        embeds = np.concatenate(embeds, axis=0)
        return logits, embeds

    def _l3_infer_logits(self, X: np.ndarray):
        self.l3.eval()
        dev = self.device
        logits = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.tensor(X[i:i+2048], dtype=torch.float32, device=dev)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1, 1)

    def _mlp_predict_proba(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        dev = self.device
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.tensor(X[i:i+4096], dtype=torch.float32, device=dev)
                logit = model(xb)
                p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray) -> pd.DataFrame:
        assert self._fitted, "Call fit() first."
        eng = compute_engineered_features(df, windows=self.fit_cfg.feature_windows)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = ['ret1','tr','vol_5','vol_10','mom_5','chanpos_10','chanpos_20']
        use_cols = [c for c in seq_cols + micro_cols if c in (list(df.columns) + list(eng.columns))]

        feat_seq = pd.concat([df[seq_cols], eng[micro_cols]], axis=1)[use_cols].replace([np.inf,-np.inf],0.0).fillna(0.0)
        feat_tab = eng[self.tab_feature_names].replace([np.inf,-np.inf],0.0).fillna(0.0)

        X_seq_all_scaled = self.scaler_seq.transform(feat_seq.values)
        X_tab_all_scaled = self.scaler_tab.transform(feat_tab.values)

        Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.l1_cfg.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        try:
            l1_logits_scaled = self.l1_temp.transform(l1_logits)
            p1 = 1.0 / (1.0 + np.exp(-l1_logits_scaled)).reshape(-1)
        except Exception:
            p1 = 1.0 / (1.0 + np.exp(-l1_logits.reshape(-1)))

        go2 = p1 >= self.gate_cfg.th1
        X_l2 = np.hstack([l1_emb, X_tab_all_scaled[t_indices]])
        if self.l2_backend == "xgb":
            p2 = self.l2_model.predict_proba(X_l2)[:, 1]
        else:
            p2 = self._mlp_predict_proba(self.l2_model, X_l2)
        go3 = (p2 >= self.gate_cfg.th2) & go2

        p3 = np.zeros_like(p1)
        rhat = np.zeros_like(p1)
        if go3.any():
            X_l3 = X_l2[go3]
            l3_logits = self._l3_infer_logits(X_l3)
            try:
                l3_logits_scaled = self.l3_temp.transform(l3_logits)
                p3_vals = 1.0 / (1.0 + np.exp(-l3_logits_scaled)).reshape(-1)
            except Exception:
                p3_vals = 1.0 / (1.0 + np.exp(-l3_logits.reshape(-1)))
            p3[go3] = p3_vals
            rhat_vals = p3_vals - 0.5
            rhat[go3] = rhat_vals

        trade = (p3 >= self.gate_cfg.th3) & go3
        size = np.clip(rhat, 0, None) * trade.astype(float)

        out = pd.DataFrame({
            "t": t_indices,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "go2": go2.astype(int),
            "go3": go3.astype(int),
            "trade": trade.astype(int),
            "size": size
        })
        return out

# Chunk 7/8: breadth/sweep, supabase logger, helpers, session state init

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    out = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out
    # enforce exclusivity: we'll compute explicit scopes and ensure they don't overlap
    # Expect levels_config to have min/max for buy and sell (buy_min, buy_max, sell_min, sell_max)
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

            # enforce exclusivity by requiring signal within a min/max band (if provided)
            # here buy_th/sell_th are single values; treat them as central thresholds with default ±0.5 margin to create a band
            band_margin = cfg.get("band_margin", 0.5)
            buy_min = cfg.get("buy_min", buy_th - band_margin)
            buy_max = cfg.get("buy_max", buy_th + band_margin)
            sell_min = cfg.get("sell_min", sell_th - band_margin)
            sell_max = cfg.get("sell_max", sell_th + band_margin)

            df["pred_label"] = 0
            # label long only if signal within buy_min..buy_max
            df.loc[(df["signal"] >= buy_min) & (df["signal"] <= buy_max), "pred_label"] = 1
            # label short only if signal within  sell_min..sell_max
            df.loc[(df["signal"] >= sell_min) & (df["signal"] <= sell_max) & (df["pred_label"] == 0), "pred_label"] = -1

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

# Supabase logger
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

# Helpers
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

# session state containers
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

# Chunk 7/8: breadth/sweep, supabase logger, helpers, session state init

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    out = {"summary": [], "detailed_trades": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out
    # enforce exclusivity: we'll compute explicit scopes and ensure they don't overlap
    # Expect levels_config to have min/max for buy and sell (buy_min, buy_max, sell_min, sell_max)
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

            # enforce exclusivity by requiring signal within a min/max band (if provided)
            # here buy_th/sell_th are single values; treat them as central thresholds with default ±0.5 margin to create a band
            band_margin = cfg.get("band_margin", 0.5)
            buy_min = cfg.get("buy_min", buy_th - band_margin)
            buy_max = cfg.get("buy_max", buy_th + band_margin)
            sell_min = cfg.get("sell_min", sell_th - band_margin)
            sell_max = cfg.get("sell_max", sell_th + band_margin)

            df["pred_label"] = 0
            # label long only if signal within buy_min..buy_max
            df.loc[(df["signal"] >= buy_min) & (df["signal"] <= buy_max), "pred_label"] = 1
            # label short only if signal within  sell_min..sell_max
            df.loc[(df["signal"] >= sell_min) & (df["signal"] <= sell_max) & (df["pred_label"] == 0), "pred_label"] = -1

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

# Supabase logger
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

# Helpers
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

# session state containers
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

# Chunk 8/8: Streamlit actions including the one-button run_full_pipeline (your replacement chunk)

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
        bars["rvol"] = (bars.get("volume", pd.Series(1.0, index=bars.index)) / bars.get("volume", pd.Series(1.0, index=bars.index)).rolling(20, min_periods=1).mean()).fillna(1.0)
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
    trader = CascadeTrader()  # use defaults
    try:
        trader.fit(bars, events)
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
        # prepare levels config with explicit scopes (set buy_min/buy_max, sell_min/sell_max to ensure exclusivity)
        levels = {
            "L1": {"buy_th": lvl1_buy, "sell_th": lvl1_sell, "rr_min": lvl1_rr_min, "rr_max": lvl1_rr_max, "sl_min": lvl1_sl_min, "sl_max": lvl1_sl_max,
                   "buy_min": lvl1_buy - 0.5, "buy_max": lvl1_buy + 0.5, "sell_min": lvl1_sell - 0.5, "sell_max": lvl1_sell + 0.5},
            "L2": {"buy_th": lvl2_buy, "sell_th": lvl2_sell, "rr_min": lvl2_rr_min, "rr_max": lvl2_rr_max, "sl_min": lvl2_sl_min, "sl_max": lvl2_sl_max,
                   "buy_min": lvl2_buy - 0.4, "buy_max": lvl2_buy + 0.4, "sell_min": lvl2_sell - 0.4, "sell_max": lvl2_sell + 0.4},
            "L3": {"buy_th": lvl3_buy, "sell_th": lvl3_sell, "rr_min": lvl3_rr_min, "rr_max": lvl3_rr_max, "sl_min": lvl3_sl_min, "sl_max": lvl3_sl_max,
                   "buy_min": lvl3_buy - 0.3, "buy_max": lvl3_buy + 0.3, "sell_min": lvl3_sell - 0.3, "sell_max": lvl3_sell + 0.3}
        }
        res = run_breadth_backtest(clean=cands, bars=bars, levels_config=levels, feature_cols=["atr","rvol","duration","hg"], model_train_kwargs={"max_bars": 60})
        st.session_state.last_breadth = res
        if res["summary"]:
            st.subheader("Breadth summary")
            st.dataframe(pd.DataFrame(res["summary"]))
        else:
            st.warning("Breadth returned no summary rows.")
    # sweep
    if run_sweep:
        st.info("Running grid sweep (light)")
        rr_vals = [lvl1_rr_min, lvl1_rr_max, lvl2_rr_min, lvl2_rr_max, lvl3_rr_min, lvl3_rr_max]
        rr_vals = sorted(list(set([float(round(x, 2)) for x in rr_vals if x is not None])))
        sl_ranges = [(float(lvl1_sl_min), float(lvl1_sl_max)), (float(lvl2_sl_min), float(lvl2_sl_max)), (float(lvl3_sl_min), float(lvl3_sl_max))]
        mpt_list = [float(p_fast), float(p_slow), float(p_deep)]
        sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list, feature_cols=["atr","rvol","duration","hg"], model_train_kwargs={"max_bars":60})
        st.session_state.last_sweep = sweep_results
        st.success("Sweep completed.")
    # 6) export artifacts & models (per-level)
    st.info("Exporting artifacts and .pt bundles")
    out_dir = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        trader_path = os.path.join(out_dir, "cascade_trader")
        os.makedirs(trader_path, exist_ok=True)
        torch.save(trader.l1.state_dict(), os.path.join(trader_path, "l1_state.pt"))
        torch.save(trader.l3.state_dict(), os.path.join(trader_path, "l3_state.pt"))
        if trader.l2_backend == "xgb":
            try:
                trader.l2_model.save_model(os.path.join(trader_path, "l2_xgb.json"))
            except Exception:
                joblib.dump(trader.l2_model, os.path.join(trader_path, "l2_xgb.joblib"))
        else:
            torch.save(trader.l2_model.state_dict(), os.path.join(trader_path, "l2_mlp_state.pt"))
        with open(os.path.join(trader_path, "scaler_seq.pkl"), "wb") as f:
            pickle.dump(trader.scaler_seq, f)
        with open(os.path.join(trader_path, "scaler_tab.pkl"), "wb") as f:
            pickle.dump(trader.scaler_tab, f)
        with open(os.path.join(trader_path, "metadata.json"), "w") as f:
            json.dump(trader.metadata, f, default=str, indent=2)
    except Exception as e:
        logger.exception("Saving artifacts failed: %s", e)
        st.error(f"Saving artifacts failed: {e}")

    # minimal wrapper classes for export use
    class L2Wrapper:
        def __init__(self, model=None, backend=None):
            self.booster = model
            self.backend = backend
        def feature_importance(self):
            try:
                if hasattr(self.booster, "get_booster"):
                    return pd.DataFrame(self.booster.get_booster().get_score(importance_type="gain").items(), columns=["feature","gain"])
            except Exception:
                pass
            return pd.DataFrame([{"feature": "dummy", "gain": 1.0}])
        def save_model(self, path):
            if hasattr(self.booster, "save_model"):
                self.booster.save_model(path)
            else:
                joblib.dump(self.booster, path)

    class L3Wrapper:
        def __init__(self, model=None):
            self.model = model
        def feature_importance(self):
            return pd.DataFrame([{"feature": "l3_emb", "gain": 1.0}])

    export_paths = {}
    try:
        l2_wrapper = L2Wrapper(model=trader.l2_model, backend=trader.l2_backend)
        paths_l2 = export_model_and_metadata(l2_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2_model"), save_fi=True)
        export_paths["l2"] = paths_l2

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

# Trigger full pipeline button
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
                for lvl, df in br["detailed_trades"].items():
                    st.write(lvl, len(df))
                    if not df.empty:
                        st.dataframe(df.head(20))