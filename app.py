# Chunk 7/7: Streamlit actions including one-button run_full_pipeline

# session state containers
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

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
        st.error("Torch not installed -- cannot train cascade.")
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
    
    # Define proper picklable wrapper classes instead of using type()
    class L2Wrapper:
        def __init__(self, model=None, backend=None):
            self.booster = model
            self.backend = backend
            
        def feature_importance(self):
            return pd.DataFrame([{"feature": "dummy", "gain": 1.0}])
            
        def save_model(self, path):
            if hasattr(self.booster, "save_model"):
                self.booster.save_model(path)
            else:
                raise AttributeError("Model doesn't have save_model method")

    class L3Wrapper:
        def __init__(self, model=None):
            self.model = model
            
        def feature_importance(self):
            return pd.DataFrame([{"feature": "l3_emb", "gain": 1.0}])
    
    # use export_model_and_metadata to save a .pt bundle for L2 (as an example) and L3 wrapper
    export_paths = {}
    try:
        # Create proper wrapper instances with the models
        l2_wrapper = L2Wrapper(model=trader.l2_model, backend=trader.l2_backend)
        paths_l2 = export_model_and_metadata(l2_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2_model"), save_fi=True)
        export_paths["l2"] = paths_l2
        
        # export L3 as .pt bundle
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
