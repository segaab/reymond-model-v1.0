# Chunk 6/8: export_model_and_metadata (user-provided) + breadth and sweep helpers

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
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Starting export of .pt bundle")
            if 'torch' in globals() and torch is not None:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Using torch.save for .pt export")
                torch.save(payload, pt_path)
            else:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] torch not available, using joblib for .pt export")
                joblib.dump(payload, pt_path)
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Successfully saved .pt bundle")
            paths["pt"] = pt_path
            logger.info(f"Successfully saved .pt bundle to {pt_path}")
        except Exception as e:
            error_msg = f"Failed to save .pt payload: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            paths["pt_error"] = error_msg
            # fallback
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

def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame) -> Dict[str,Any]:
    """Apply exclusive level ranges to preds -> cands and simulate trades per level."""
    out = {"detailed": {}, "summary": []}
    df = cands.copy().reset_index(drop=True)
    # Map preds.t -> df rows (assume aligned as used earlier)
    preds_indexed = preds.set_index('t')
    df['signal'] = (preds_indexed.reindex(df.index)['p3'].fillna(0.0) * 10).values
    def run_level(name, buy_min, buy_max, sl_pct, rr):
        sel = df[(df['signal'] >= buy_min) & (df['signal'] <= buy_max)].copy()
        tp_pct = rr * sl_pct
        if sel.empty:
            out['detailed'][name] = pd.DataFrame()
            return
        sel['pred_label'] = (sel['signal'] >= buy_min).astype(int)  # simple mapping
        trades = simulate_limits(sel, bars, label_col='pred_label', sl=sl_pct, tp=tp_pct, max_holding=60)
        out['detailed'][name] = trades
        s = summarize_trades(trades)
        if not s.empty:
            row = s.iloc[0].to_dict(); row['mode'] = name; out['summary'].append(row)
    # apply L1/L2/L3 with sl/rr defaults from UI if present
    # default sl & rr values if the UI didn't include them (safe fallback)
    l1_sl = float((globals().get("lvl1_sl_min",0.02) + globals().get("lvl1_sl_max",0.04))/2.0)
    l1_rr = float((globals().get("lvl1_rr_min",2.0) + globals().get("lvl1_rr_max",2.5))/2.0)
    l2_sl = float((globals().get("lvl2_sl_min",0.01) + globals().get("lvl2_sl_max",0.03))/2.0)
    l2_rr = float((globals().get("lvl2_rr_min",2.0) + globals().get("lvl2_rr_max",3.5))/2.0)
    l3_sl = float((globals().get("lvl3_sl_min",0.005) + globals().get("lvl3_sl_max",0.02))/2.0)
    l3_rr = float((globals().get("lvl3_rr_min",3.0) + globals().get("lvl3_rr_max",5.0))/2.0)
    run_level("L1", lvl1_buy_min, lvl1_buy_max, l1_sl, l1_rr)
    run_level("L2", lvl2_buy_min, lvl2_buy_max, l2_sl, l2_rr)
    run_level("L3", lvl3_buy_min, lvl3_buy_max, l3_sl, l3_rr)
    return out

def run_grid_sweep(clean: pd.DataFrame, bars: pd.DataFrame, rr_vals: List[float], sl_ranges: List[Tuple[float,float]], mpt_list: List[float]) -> Dict[str,Any]:
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
                        df["pred_prob"] = 0.0
                    df["pred_label"] = (df["pred_prob"] >= mpt).astype(int)
                    trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
                    results[key] = {"trades_count": len(trades), "trades": trades}
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results

# Chunk 7/8: Streamlit actions including one-button run_full_pipeline

# session state containers
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

def fetch_and_prepare():
    st.info(f"Fetching {symbol} {interval} from YahooQuery …")
    bars = pd.DataFrame()
    if YahooTicker is None:
        st.error("yahooquery not installed — cannot fetch price data.")
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
        bars["rvol"] = (bars["volume"] / bars["volume"].rolling(20, min_periods=1).mean()).fillna(1.0)
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
        st.error("Torch not installed — cannot train cascade.")
        return {"error": "no_torch"}
    st.info("Training cascade (L1/L2/L3). This may take a while.")
    trader = CascadeTrader(seq_len=seq_len, feat_windows=(5,10,20), device=device_choice)
    try:
        trader.fit(bars, events, l2_use_xgb=(xgb is not None), epochs_l1=8, epochs_l23=8)
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
        res = run_breadth_levels(preds, cands, bars)
        st.session_state.last_breadth = res
        if res["summary"]:
            st.subheader("Breadth summary")
            st.dataframe(pd.DataFrame(res["summary"]))
        else:
            st.warning("Breadth returned no summary rows.")
    # sweep
    if run_sweep:
        st.info("Running grid sweep (light)")
        rr_vals = [2.0, 2.5, 3.0]  # small set for speed
        sl_ranges = [(0.02, 0.04), (0.01, 0.03), (0.005, 0.02)]
        mpt_list = [p_fast, p_slow, p_deep]
        sweep_results = run_grid_sweep(cands, bars, rr_vals, sl_ranges, mpt_list)
        st.session_state.last_sweep = sweep_results
        st.success("Sweep completed.")
    # 6) export artifacts & models (per-level)
    st.info("Exporting artifacts and .pt bundles")
    out_dir = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(out_dir, exist_ok=True)
    # Save L1/L3 Torch state_dicts + scalers + meta
    try:
        trader_path = os.path.join(out_dir, "cascade_trader")
        os.makedirs(trader_path, exist_ok=True)
        # L1
        torch.save(trader.l1.state_dict(), os.path.join(trader_path, "l1_state.pt"))
        torch.save(trader.l3.state_dict(), os.path.join(trader_path, "l3_state.pt"))
        # If L2 is xgb, save model; if mlp, save state_dict
        if trader.l2_backend == "xgb":
            # xgboost wrapper saving
            try:
                trader.l2_model.save_model(os.path.join(trader_path, "l2_xgb.json"))
            except Exception:
                joblib.dump(trader.l2_model, os.path.join(trader_path, "l2_xgb.joblib"))
        else:
            torch.save(trader.l2_model.state_dict(), os.path.join(trader_path, "l2_mlp_state.pt"))
        # scalers and metadata
        with open(os.path.join(trader_path, "scaler_seq.pkl"), "wb") as f:
            pickle.dump(trader.scaler_seq, f)
        with open(os.path.join(trader_path, "scaler_tab.pkl"), "wb") as f:
            pickle.dump(trader.scaler_tab, f)
        with open(os.path.join(trader_path, "metadata.json"), "w") as f:
            json.dump(trader.metadata, f, default=str, indent=2)
    except Exception as e:
        logger.exception("Saving artifacts failed: %s", e)
        st.error(f"Saving artifacts failed: {e}")
    # use export_model_and_metadata to save a .pt bundle for L2 (as an example) and L3 wrapper
    export_paths = {}
    try:
        # create a wrapper-like object for export (minimal)
        l2_wrapper = type("L2W", (), {})()
        if trader.l2_backend == "xgb":
            l2_wrapper.booster = trader.l2_model
        else:
            l2_wrapper.booster = None
            # provide feature_importance stub if desired
            def fi_stub(): return pd.DataFrame([{"feature":"dummy","gain":1.0}])
            l2_wrapper.feature_importance = fi_stub
        paths_l2 = export_model_and_metadata(l2_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2_model"), save_fi=True)
        export_paths["l2"] = paths_l2
        # export L3 as .pt bundle
        l3_wrapper = type("L3W", (), {})()
        def l3_fi(): return pd.DataFrame([{"feature":"l3_emb","gain":1.0}])
        l3_wrapper.feature_importance = l3_fi
        paths_l3 = export_model_and_metadata(l3_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l3_model"), save_fi=True)
        export_paths["l3"] = paths_l3
        st.success(f"Exported models to {out_dir}")
        st.write(export_paths)
        st.session_state.export_paths = export_paths
    except Exception as e:
        logger.exception("Export_model_and_metadata failed: %s", e)
        st.error(f"Export failed: {e}")
    return {"bars": bars, "cands": cands, "trader": trader, "export": export_paths}

# If user clicked the full pipeline button:
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
                for lvl, df in br["detailed"].items():
                    st.write(lvl, len(df))
                    if not df.empty:
                        st.dataframe(df.head(20))

# Chunk 8/8: final notes / help text

st.sidebar.markdown("### Notes\n- Level thresholds adjust the breadth backtest behaviors.\n- Breadth returns per-level trade lists and summaries.\n- Save/Log model after a successful training run.\n\nIf you encounter pickling/export issues for L2 wrappers, consider defining module-level wrapper classes (L2Wrapper/L3Wrapper) instead of dynamic types.")
