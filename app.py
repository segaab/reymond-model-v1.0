# app.py — Entry-Range Triangulation Demo (full script, fixed)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
import uuid
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any

# Internal imports
from fetch_data import fetch_price, fetch_cot, init_socrata_client
from features import compute_rvol, calculate_health_gauge, ensure_no_duplicate_index
from utils import build_or_fetch_candidates, generate_candidates_and_labels
from utils import predict_confirm_prob, export_model_and_metadata
from utils import simulate_limits, run_breadth_backtest, summarize_sweep
from utils.supabase_logger import SupabaseLogger

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Entry-Range Triangulation Demo", layout="wide")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

# ---------------------------
# User Inputs / Config
# ---------------------------
symbol = st.text_input("Symbol", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)

buy_threshold = st.number_input("Buy threshold (HealthGauge)", min_value=0.0, max_value=1.0, value=0.55)
sell_threshold = st.number_input("Sell threshold (HealthGauge)", min_value=0.0, max_value=1.0, value=0.45)

num_boost = st.number_input("XGBoost num_boost_round", min_value=1, value=50)
early_stop = st.number_input("XGBoost early_stopping_rounds", min_value=1, value=10)
test_size = st.number_input("Test set fraction", min_value=0.0, max_value=1.0, value=0.2)

p_fast = st.number_input("Threshold fast", min_value=0.0, max_value=1.0, value=0.5)
p_slow = st.number_input("Threshold slow", min_value=0.0, max_value=1.0, value=0.55)
p_deep = st.number_input("Threshold deep", min_value=0.0, max_value=1.0, value=0.6)

force_run = st.checkbox("Force run even outside buy/sell band", value=False)
show_confusion = st.checkbox("Show confusion matrix / classification report", value=True)
overlay_entries_on_price = st.checkbox("Overlay entries on price chart", value=True)
include_health_as_feature = st.checkbox("Include HealthGauge as feature", value=True)
save_feature_importance = st.checkbox("Save feature importance on export", value=True)

# Breadth / sweep buttons
run_breadth = st.button("Run breadth modes (Low → Mid → High)")
run_sweep_btn = st.button("Run grid sweep")

# ---------------------------
# Asset object placeholder
# ---------------------------
from dataclasses import dataclass
@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

asset_obj = Asset(name="Gold", cot_name="GOLD - COMMODITY EXCHANGE INC.", symbol=symbol)
max_bars = 100  # Example

# RR, SL, MPT configs
rr_vals = [1.0, 1.5, 2.0]
sl_ranges = [(0.5, 1.0), (1.0, 2.0)]
mpt_list = [0.5, 0.6, 0.7]
session_modes = ["Low", "Mid", "High"]

# Initialize empty placeholders
bars = pd.DataFrame()
clean = pd.DataFrame()
health_df = pd.DataFrame()
trades = pd.DataFrame()

# ---------------------------
# BoosterWrapper for saving models
# ---------------------------
class BoosterWrapper:
    def __init__(self, booster: xgb.Booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.booster.predict(dtest)

    def save_model(self, path: str):
        self.booster.save_model(path)

# ---------------------------
# Train XGBoost confirm model
# ---------------------------
def train_xgb_confirm(clean: pd.DataFrame,
                      feature_cols: List[str],
                      label_col: str = "label",
                      num_boost_round: int = 200,
                      early_stopping_rounds: int = 20,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[BoosterWrapper, Dict[str, Any]]:

    if xgb is None:
        raise RuntimeError("xgboost not installed. Install with `pip install xgboost`.")

    # Prepare data
    X = clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(clean[label_col], errors="coerce")
    mask = X.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y.notnull()
    X, y = X.loc[mask], y.loc[mask]

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("No valid rows after cleaning features/labels for training.")

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least two classes to train; found {unique_labels}")

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

    bst = xgb.train(params,
                    dtr,
                    num_boost_round=int(num_boost_round),
                    evals=[(dva, "validation")],
                    early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
                    verbose_eval=False)

    wrap = BoosterWrapper(bst, feature_cols)

# Validation metrics
    y_proba_val = wrap.predict_proba(Xva)
    y_pred_val = (y_proba_val >= 0.5).astype(int)
    metrics: Dict[str, Any] = {
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "accuracy": float(accuracy_score(yva, y_pred_val)),
        "f1": float(f1_score(yva, y_pred_val, zero_division=0)),
        "val_proba_mean": float(np.nanmean(y_proba_val)) if len(y_proba_val) else 0.0,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(yva, y_proba_val))
    except Exception:
        metrics["roc_auc"] = None

    return wrap, metrics

# ---------------------------
# Main pipeline
# ---------------------------
def run_main_pipeline():
    try:
        st.info("Fetching price data…")
        bars = fetch_price(symbol, start_date=start_date, end_date=end_date, interval=interval)

        if bars.empty:
            st.error("No data returned from fetch_price.")
            return

        bars = ensure_no_duplicate_index(bars)

        # Add RVOL + HealthGauge
        bars["rvol"] = compute_rvol(bars, lookback=asset_obj.rvol_lookback)
        bars["health"] = calculate_health_gauge(bars, rvol_col="rvol", threshold=buy_threshold)

        # Build candidates + labels
        st.info("Building candidates + labels…")
        cands = build_or_fetch_candidates(bars, symbol=symbol)
        clean = generate_candidates_and_labels(bars, cands, buy_threshold, sell_threshold)

        if clean.empty:
            st.error("No candidates generated.")
            return

# Save model + metadata
        export_model_and_metadata(model_wrap.booster, metrics, feature_names=feat_cols,
                                  save_feature_importance=save_feature_importance)

        # Breadth backtest
        if run_breadth:
            st.info("Running breadth backtest…")
            breadth_res = run_breadth_backtest(clean, model_wrap, feat_cols,
                                               rr_vals, sl_ranges, mpt_list, session_modes)
            st.write(breadth_res)

        # Sweep
        if run_sweep_btn:
            st.info("Running sweep…")
            sweep_res = summarize_sweep(clean, model_wrap, feat_cols,
                                        rr_vals, sl_ranges, mpt_list, session_modes)
            st.write(sweep_res)

    except Exception as e:
        st.error("Training failed")
        logger.error("Training failed", exc_info=True)
        traceback.print_exc()

# Features
        feat_cols = ["rvol"]
        if include_health_as_feature:
            feat_cols.append("health")

        # Train model
        st.info("Training confirm model…")
        model_wrap, metrics = train_xgb_confirm(clean, feat_cols, "label",
                                                num_boost_round=num_boost,
                                                early_stopping_rounds=early_stop,
                                                test_size=test_size)

        st.success(f"Training done. Accuracy {metrics['accuracy']:.2f}, F1 {metrics['f1']:.2f}")

        if show_confusion:
            y_true = clean["label"]
            y_pred = (model_wrap.predict_proba(clean[feat_cols]) >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            st.write("Confusion Matrix:", cm)
            st.text(classification_report(y_true, y_pred, zero_division=0))

        # Overlay entries
        if overlay_entries_on_price:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(bars.index, bars["close"], label="Price")
            entries = clean[clean["label"] == 1]
            exits = clean[clean["label"] == 0]
            ax.scatter(entries.index, bars.loc[entries.index, "close"], marker="^", color="g", label="Buy")
            ax.scatter(exits.index, bars.loc[exits.index, "close"], marker="v", color="r", label="Sell")
            ax.legend()
            st.pyplot(fig)

        # Log to Supabase
        supa = SupabaseLogger()
        supa.log_training(uuid.uuid4().hex, metrics)

# ---------------------------
# Streamlit main entry
# ---------------------------
def main():
    st.title("Entry-Range Triangulation Demo")
    run_main_pipeline()

if __name__ == "__main__":
    main()