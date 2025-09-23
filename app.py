def _compute_explicit_windows(levels_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Given levels_config with per-level buy_th and sell_th, return explicit non-overlapping windows:
      return { level: {"buy": (buy_min, buy_max), "sell": (sell_min, sell_max)} }
    buy windows: [buy_min, buy_max)  (upper exclusive); highest level => upper = +inf
    sell windows: (sell_min, sell_max] (lower exclusive); lowest level => lower = -inf
    """
    out = {}
    # Build buy windows (descending by buy_th)
    buy_list = sorted([(lvl, cfg.get("buy_th", 0.0)) for lvl, cfg in levels_config.items()], key=lambda x: x[1], reverse=True)
    # For descending order, the first is highest buy
    for i, (lvl, buy_th) in enumerate(buy_list):
        upper = buy_list[i - 1][1] if i - 1 >= 0 else np.inf
        lower = buy_th
        out.setdefault(lvl, {})["buy"] = (float(lower), float(upper))

    # Build sell windows (ascending by sell_th)
    sell_list = sorted([(lvl, cfg.get("sell_th", 0.0)) for lvl, cfg in levels_config.items()], key=lambda x: x[1])
    for i, (lvl, sell_th) in enumerate(sell_list):
        lower = sell_list[i - 1][1] if i - 1 >= 0 else -np.inf
        upper = sell_th
        out.setdefault(lvl, {})["sell"] = (float(lower), float(upper))

    return out

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
    """
    For each level, apply explicit buy/sell windows (non-overlapping) and simulate trades.
    Returns standardized dict with per-level trades and a combined summary list.
    """
    out = {"summary": [], "detailed_trades": {}, "diagnostics": [], "metrics_by_level": {}}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out

    windows = _compute_explicit_windows(levels_config)

    # create numeric 'signal' if missing
    df_base = clean.copy()
    if "signal" not in df_base.columns:
        if "pred_prob" in df_base.columns:
            df_base["signal"] = (df_base["pred_prob"] * 10).round().astype(int)
        else:
            df_base["signal"] = 0

    # For each level, filter using explicit windows (buy and sell) and simulate
    for lvl_name, cfg in levels_config.items():
        try:
            w = windows.get(lvl_name, {})
            buy_min, buy_max = w.get("buy", (cfg.get("buy_th", 0.0), np.inf))
            sell_min, sell_max = w.get("sell", (-np.inf, cfg.get("sell_th", 0.0)))

            df = df_base.copy()
            df["assigned"] = False
            df["pred_label"] = 0
            df["level"] = lvl_name

            # Buy mask: signal in [buy_min, buy_max)
            buy_mask = (df["signal"] >= buy_min) & (df["signal"] < buy_max)
            # Sell mask: signal in (sell_min, sell_max]  => (df["signal"] > sell_min) & (df["signal"] <= sell_max)
            sell_mask = (df["signal"] > sell_min) & (df["signal"] <= sell_max)

            # Assign buys first, then sells (but masks are disjoint across levels by construction)
            df.loc[buy_mask & (~df["assigned"]), "pred_label"] = 1
            df.loc[buy_mask, "assigned"] = True
            df.loc[sell_mask & (~df["assigned"]), "pred_label"] = -1
            df.loc[sell_mask, "assigned"] = True

            # Count entries made on each level
            buy_entries = buy_mask.sum()
            sell_entries = sell_mask.sum()
            total_entries = buy_entries + sell_entries

            # Representative SL and TP from level config (mean)
            sl_pct = float((cfg.get("sl_min", 0.01) + cfg.get("sl_max", 0.02)) / 2.0)
            rr = float((cfg.get("rr_min", 1.0) + cfg.get("rr_max", 2.0)) / 2.0)
            tp_pct = rr * sl_pct

            trades = simulate_limits(df, bars, label_col="pred_label",
                                     sl=sl_pct, tp=tp_pct,
                                     max_holding=int(model_train_kwargs.get("max_bars", 60)))
            out["detailed_trades"][lvl_name] = trades
            
            # Enhanced metrics for this level
            level_metrics = {
                "total_entries": total_entries,
                "buy_entries": buy_entries,
                "sell_entries": sell_entries,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "rr_target": rr
            }
            
            summary = summarize_trades(trades)
            if not summary.empty:
                srow = summary.iloc[0].to_dict()
                srow.update({"mode": lvl_name, "sl_pct": sl_pct, "tp_pct": tp_pct})
                # Add the enhanced metrics
                level_metrics.update({
                    "win_rate": srow.get("win_rate", 0.0),
                    "total_trades": srow.get("total_trades", 0),
                    "total_pnl": srow.get("total_pnl", 0.0),
                    "avg_pnl": srow.get("avg_pnl", 0.0),
                    "rr_mean": srow.get("rr_mean", 0.0),
                    "rr_min": srow.get("rr_min", 0.0),
                    "rr_80th_percentile": srow.get("rr_80th_percentile", 0.0),
                    "annual_return": srow.get("annual_return", 0.0)
                })
                out["summary"].append(srow)
            
            out["metrics_by_level"][lvl_name] = level_metrics
            out["diagnostics"].append(f"{lvl_name}: buy_range=[{buy_min},{buy_max}), sell_range=({sell_min},{sell_max}], sl={sl_pct:.4f}, tp={tp_pct:.4f}, trades={len(trades)}, win_rate={level_metrics.get('win_rate', 0.0):.2f}")
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
    Sweep RR x SL ranges x model probability thresholds (mpt_list).
    Returns dict keyed by config string with overlay and summary.
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
                    summary = summarize_trades(trades)
                    
                    results[key] = {
                        "trades_count": len(trades), 
                        "overlay": trades,
                        "win_rate": float(summary["win_rate"].iloc[0]) if not summary.empty else 0.0,
                        "rr_mean": float(summary["rr_mean"].iloc[0]) if not summary.empty else 0.0,
                        "rr_80th": float(summary["rr_80th_percentile"].iloc[0]) if not summary.empty else 0.0,
                        "total_pnl": float(summary["total_pnl"].iloc[0]) if not summary.empty else 0.0
                    }
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results


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
        self.level_metrics_tbl = "level_metrics"  # New table for level-specific metrics

    def log_run(self, metrics: Dict, metadata: Dict, trades: Optional[List[Dict]] = None, level_metrics: Optional[Dict[str, Dict]] = None) -> str:
        run_id = metadata.get("run_id") or str(uuid.uuid4())
        metadata["run_id"] = run_id
        payload = {**metadata, "metrics": metrics}
        resp = self.client.table(self.runs_tbl).insert(payload).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to insert run: {resp.error}")
        
        # Log trades if provided
        if trades:
            for t in trades:
                t["run_id"] = run_id
            tr_resp = self.client.table(self.trades_tbl).insert(trades).execute()
            if getattr(tr_resp, "error", None):
                raise RuntimeError(f"Failed to insert trades: {tr_resp.error}")
        
        # Log level-specific metrics if provided
        if level_metrics:
            level_records = []
            for level_name, level_data in level_metrics.items():
                record = {
                    "run_id": run_id,
                    "level": level_name,
                    **level_data
                }
                level_records.append(record)
            
            if level_records:
                level_resp = self.client.table(self.level_metrics_tbl).insert(level_records).execute()
                if getattr(level_resp, "error", None):
                    raise RuntimeError(f"Failed to insert level metrics: {level_resp.error}")
        
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
        
    def fetch_level_metrics(self, run_id: str):
        resp = self.client.table(self.level_metrics_tbl).select("*").eq("run_id", run_id).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch level metrics: {resp.error}")
        return getattr(resp, "data", [])

# Helpers
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def pick_top_runs_by_metrics(runs: List[Dict], top_n: int = 3):
    """
    Ranks runs using weightings:
      1) total_pnl (primary importance)
      2) win_rate (secondary)
    """
    scored = []
    for r in runs:
        metrics = r.get("metrics") if isinstance(r, dict) else r
        total_pnl = metrics.get("total_pnl", 0.0) if metrics else 0.0
        win_rate = metrics.get("win_rate", 0.0) if metrics else 0.0
        score = float(total_pnl) + float(win_rate) * 0.01
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_n]]


# small placeholders / defaults
feat_cols_default = ["atr", "rvol", "duration", "hg"]

def run_main_pipeline():
    st.info(f"Fetching price for {symbol} …")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars is None or bars.empty:
        st.error("No price data returned.")
        return

    bars = ensure_unique_index(bars)
    # daily for health
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

    # compute rvol and candidates
    bars["rvol"] = compute_rvol(bars, lookback=20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates generated.")
        return

    # add health-gauge to candidates
    if not health.empty:
        hg_map = health["health_gauge"].reindex(pd.to_datetime(health.index)).ffill().to_dict()
        cands["hg"] = cands["candidate_time"].dt.normalize().map(lambda t: hg_map.get(pd.Timestamp(t).normalize(), 0.0))
    else:
        cands["hg"] = 0.0

    feat_cols = feat_cols_default.copy()
    if "hg" not in feat_cols:
        feat_cols.append("hg")

    # training confirm-stage (one model for pipeline)
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

    # predict and simulate
    cands["pred_prob"] = predict_confirm_prob(model_wrap, cands, feat_cols)
    cands["pred_label"] = (cands["pred_prob"] >= p_fast).astype(int)

    st.info("Simulating trades using predicted labels…")
    trades = simulate_limits(cands, bars, label_col="pred_label", max_holding=60)
    st.write("Simulated trades:", len(trades))
    if not trades.empty:
        st.dataframe(trades.head())
        s = summarize_trades(trades)
        st.dataframe(s)

    # Auto-log to Supabase (if configured) and then save model
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

            # After logging, fetch recent runs and pick top3 by weighting and offer saving
            runs = supa.fetch_runs(symbol=symbol, limit=20)
            top_runs = pick_top_runs_by_metrics(runs, top_n=3)
            st.subheader("Top 3 recent runs (by total_pnl then win_rate)")
            for rr in top_runs:
                st.write(rr)

            # Save model locally
            model_basename = f"confirm_model_{symbol.replace('=','_')}"
            saved_paths = export_model_and_metadata(model_wrap, feat_cols, combined, model_basename, save_fi=True)
            st.success(f"Saved model files: {saved_paths}")
        except Exception as exc:
            logger.exception("Logging/saving failed: %s", exc)
            st.error(f"Logging/saving failed: {exc}")

# Breadth handler
if run_breadth:
    st.info("Running breadth backtest across 3 levels...")
    levels = {
        "L1": {"buy_th": float(lvl1_buy), "sell_th": float(lvl1_sell), "rr_min": float(lvl1_rr_min), "rr_max": float(lvl1_rr_max), "sl_min": float(lvl1_sl_min), "sl_max": float(lvl1_sl_max)},
        "L2": {"buy_th": float(lvl2_buy), "sell_th": float(lvl2_sell), "rr_min": float(lvl2_rr_min), "rr_max": float(lvl2_rr_max), "sl_min": float(lvl2_sl_min), "sl_max": float(lvl2_sl_max)},
        "L3": {"buy_th": float(lvl3_buy), "sell_th": float(lvl3_sell), "rr_min": float(lvl3_rr_min), "rr_max": float(lvl3_rr_max), "sl_min": float(lvl3_sl_min), "sl_max": float(lvl3_sl_max)}
    }
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    bars["rvol"] = compute_rvol(bars, 20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates available for breadth run.")
    else:
        cands["pred_prob"] = cands.get("pred_prob", 0.0)
        try:
            res = run_breadth_backtest(clean=cands, bars=bars, levels_config=levels, feature_cols=feat_cols_default, model_train_kwargs={"max_bars": 60})
            
            # Display the enhanced metrics for each level
            st.subheader("Breadth Summary")
            summary_df = pd.DataFrame(res.get("summary", []))
            if not summary_df.empty:
                st.dataframe(summary_df)
            else:
                st.warning("Breadth run returned no summary rows.")
                
            # Display level-specific metrics in a more structured format
            st.subheader("Level-specific Metrics")
            for lvl, metrics in res.get("metrics_by_level", {}).items():
                with st.expander(f"{lvl} Metrics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2f}")
                        st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
                    with col2:
                        st.metric("RR Mean", f"{metrics.get('rr_mean', 0.0):.2f}")
                        st.metric("RR 80th Percentile", f"{metrics.get('rr_80th_percentile', 0.0):.2f}")
                    with col3:
                        st.metric("Total PnL", f"{metrics.get('total_pnl', 0.0):.2f}")
                        st.metric("Annual Return", f"{metrics.get('annual_return', 0.0):.2f}")
                    
                    st.write(f"Entries: Buy={metrics.get('buy_entries', 0)}, Sell={metrics.get('sell_entries', 0)}")
                    st.write(f"Target RR: {metrics.get('rr_target', 0.0):.2f} (SL: {metrics.get('sl_pct', 0.0):.4f}, TP: {metrics.get('tp_pct', 0.0):.4f})")
            
            # Display trades for each level
            detailed = res.get("detailed_trades", {})
            for lvl, df in detailed.items():
                with st.expander(f"{lvl} -- {len(df)} trades"):
                    st.dataframe(df.head(50))
            
            # Add a download button for the full results
            for lvl, df in detailed.items():
                if not df.empty:
                    st.download_button(f"Download {lvl} trades CSV", df_to_csv_bytes(df), f"{symbol}_{lvl}_trades.csv", "text/csv")
            
            # Log the breadth test results to Supabase if button is clicked
            if create_client is not None and st.button("Log breadth test to Supabase"):
                try:
                    supa = SupabaseLogger()
                    run_id = str(uuid.uuid4())
                    metadata = {
                        "run_id": run_id, 
                        "symbol": symbol, 
                        "start_date": str(start_date),
                        "end_date": str(end_date), 
                        "interval": interval,
                        "run_type": "breadth_backtest",
                        "levels": list(levels.keys())
                    }
                    
                    # Prepare the combined metrics
                    combined_metrics = {
                        "total_levels": len(res.get("metrics_by_level", {})),
                        "total_trades": sum(m.get("total_trades", 0) for m in res.get("metrics_by_level", {}).values()),
                        "avg_win_rate": np.mean([m.get("win_rate", 0.0) for m in res.get("metrics_by_level", {}).values()]) if res.get("metrics_by_level") else 0.0,
                        "total_pnl": sum(m.get("total_pnl", 0.0) for m in res.get("metrics_by_level", {}).values()),
                    }
                    
                    # Log with level-specific metrics
                    supa.log_run(metrics=combined_metrics, metadata=metadata, level_metrics=res.get("metrics_by_level", {}))
                    st.success(f"Logged breadth test with run_id {run_id}")
                except Exception as exc:
                    logger.exception("Logging breadth test failed: %s", exc)
                    st.error(f"Logging breadth test failed: {exc}")
            
            st.write("Diagnostics:")
            st.write(res.get("diagnostics", []))
        except Exception as exc:
            logger.exception("Breadth backtest failed: %s", exc)
            st.error(f"Breadth backtest failed: {exc}")

# Grid sweep handler
if run_sweep_btn:
    st.info("Running grid sweep (RR x SL x MPT)...")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    bars = ensure_unique_index(bars)
    bars["rvol"] = compute_rvol(bars, 20)
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.error("No candidates for sweep.")
    else:
        rr_vals = [lvl1_rr_min, lvl1_rr_max, lvl2_rr_min, lvl2_rr_max, lvl3_rr_min, lvl3_rr_max]
        rr_vals = sorted(list(set([float(round(x, 2)) for x in rr_vals if x is not None])))
        sl_ranges = [(float(lvl1_sl_min), float(lvl1_sl_max)), (float(lvl2_sl_min), float(lvl2_sl_max)), (float(lvl3_sl_min), float(lvl3_sl_max))]
        mpt_list = [float(p_fast), float(p_slow), float(p_deep)]
        try:
            sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list, feature_cols=feat_cols_default, model_train_kwargs={"max_bars":60})
            # summarize best runs
            summary_rows = []
            for k, v in sweep_results.items():
                if isinstance(v, dict) and "overlay" in v and not v["overlay"].empty:
                    # Add enhanced metrics to the summary
                    summary_rows.append({
                        "config": k, 
                        "trades_count": v.get("trades_count", 0),
                        "win_rate": v.get("win_rate", 0.0),
                        "rr_mean": v.get("rr_mean", 0.0),
                        "rr_80th": v.get("rr_80th", 0.0),
                        "total_pnl": v.get("total_pnl", 0.0)
                    })
            if summary_rows:
                sdf = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
                st.subheader("Sweep summary (top configs)")
                
                # Create more informative display with columns for key metrics
                st.dataframe(sdf.head(20))
                
                # Visualization of top configs
                if len(sdf) > 5:
                    st.subheader("Top 5 Configurations Comparison")
                    top5 = sdf.head(5)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Win rate vs Total PnL
                    ax1.scatter(top5["win_rate"], top5["total_pnl"], s=100)
                    for i, row in top5.iterrows():
                        ax1.annotate(row["config"], (row["win_rate"], row["total_pnl"]))
                    ax1.set_xlabel("Win Rate")
                    ax1.set_ylabel("Total PnL")
                    ax1.set_title("Win Rate vs Total PnL")
                    
                    # RR Mean vs Trade Count
                    ax2.scatter(top5["rr_mean"], top5["trades_count"], s=100)
                    for i, row in top5.iterrows():
                        ax2.annotate(row["config"], (row["rr_mean"], row["trades_count"]))
                    ax2.set_xlabel("RR Mean")
                    ax2.set_ylabel("Trade Count")
                    ax2.set_title("RR Mean vs Trade Count")
                    
                    st.pyplot(fig)
                
                st.download_button("Download sweep summary CSV", df_to_csv_bytes(sdf), "sweep_summary.csv", "text/csv")
            else:
                st.warning("Sweep returned no runs with trades.")
        except Exception as exc:
            logger.exception("Sweep failed: %s", exc)
            st.error(f"Sweep failed: {exc}")

# Help text
st.sidebar.markdown("### Notes\n- Level thresholds now produce explicit non-overlapping buy/sell windows per level.\n- L3 is the narrowest/highest-confidence scope; L1 is the widest/scope level.\n- Use the Breadth run to simulate trades per level and ensure exclusivity.")
