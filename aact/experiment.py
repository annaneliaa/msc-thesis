import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from build_features import build_dyn_features, build_static_features, build_sym_features
from schema import FeatureSchema
from memory import SymbolicMemory
from train import *
from metrics import eval_subset_metrics


def make_time_windows(df, window_days=7, step_days=7, ts_col="timestamp"):
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    start = d[ts_col].min()
    end = d[ts_col].max()

    windows = []
    cur = start
    while cur + pd.Timedelta(days=window_days) <= end:
        w_start = cur
        w_end = cur + pd.Timedelta(days=window_days)
        mask = (d[ts_col] >= w_start) & (d[ts_col] < w_end)
        windows.append((w_start, w_end, d.loc[mask].reset_index(drop=True)))
        cur = cur + pd.Timedelta(days=step_days)
    return windows

def select_symbolic_features(
    ablation_results,      # dict feat -> res_feat
    X_full, y,
    res_base,
    sym_feats,
    threshold=0.5,
    min_support=200,
    max_fp_increase=50,
    min_auc_gain=0.0,
    memory=None,
    prefer_memory=None
):
    """
    Returns selected features to keep active in next window.
    """
    chosen = []

    # baseline metrics
    base_stats = eval_subset_metrics(X_full, y, res_base, threshold=threshold)

    for f in sym_feats:
        res_f = ablation_results.get(f)
        if res_f is None:
            continue

        # support in test subset
        split = res_base["test_idx_start"]
        X_test = X_full.iloc[split:].reset_index(drop=True)
        support = int((X_test[f].fillna(0).astype(int) == 1).sum())
        if support < min_support:
            continue

        stats_f = eval_subset_metrics(X_full, y, res_f, threshold=threshold)

        auc_gain = stats_f["auc"] - base_stats["auc"]
        fp_increase = stats_f["fp"] - base_stats["fp"]

        # selection rule: must not explode FP, and must not hurt AUC
        # too strict?
        if (auc_gain >= min_auc_gain) and (fp_increase <= max_fp_increase):
            chosen.append(f)

    if memory is not None and prefer_memory:
        mem_active = set(memory.active())
        # union, but keep order stable
        chosen = list(dict.fromkeys([*chosen, *[f for f in sym_feats if f in mem_active]]))

    return chosen

def run_ablation_holdout(
    X_full,
    y,
    base_feats,
    sym_feats,
    test_frac=0.3,
):
    schema_base = FeatureSchema("base", base_feats)

    print("\nTraining BASELINE model...")
    res_base = train_eval_holdout(X_full, y, schema_base, test_frac=test_frac)
    diag_base = burst_diagnostics(X_full, res_base)

    ablation_results = {}
    ablation_diags = {}

    for feat in sym_feats:
        print(f"\nTraining base + '{feat}' ...")
        schema_feat = FeatureSchema(f"base+{feat}", base_feats + [feat])
        res_feat = train_eval_holdout(X_full, y, schema_feat, test_frac=test_frac)

        ablation_results[feat] = res_feat
        ablation_diags[feat] = burst_diagnostics(X_full, res_feat)

    return res_base, diag_base, ablation_results, ablation_diags

def burst_diagnostics(
    X_full, res, burst_col="is_suspicious_auth_burst", auth_col="is_auth_event"
):
    split = res["test_idx_start"]
    X_test_full = X_full.iloc[split:].reset_index(drop=True)
    y_test = res["y_test"]
    s = res["proba_test"]

    out = {"auc": res["auc"]}

    if burst_col in X_test_full.columns:
        burst = X_test_full[burst_col].fillna(0).astype(int).values == 1
        out["burst_count_test"] = int(burst.sum())
        out["mean_score_burst_test"] = float(s[burst].mean()) if burst.any() else np.nan
        out["mean_score_nonburst_test"] = float(s[~burst].mean())
        if burst.any() and len(np.unique(y_test[burst])) > 1:
            out["auc_burst_test"] = float(roc_auc_score(y_test[burst], s[burst]))
        else:
            out["auc_burst_test"] = np.nan

    if auth_col in X_test_full.columns:
        auth = X_test_full[auth_col].fillna(0).astype(int).values == 1
        out["auth_count_test"] = int(auth.sum())
        if auth.any() and len(np.unique(y_test[auth])) > 1:
            out["auc_auth_test"] = float(roc_auc_score(y_test[auth], s[auth]))
        else:
            out["auc_auth_test"] = np.nan

    return out

def run_ablation_holdout(
    X_full,
    y,
    base_feats,
    sym_feats,
    test_frac=0.3,
):
    schema_base = FeatureSchema("base", base_feats)

    res_base = train_eval_holdout(X_full, y, schema_base, test_frac=test_frac)
    diag_base = burst_diagnostics(X_full, res_base)

    ablation_results = {}
    ablation_diags = {}

    for feat in sym_feats:
        schema_feat = FeatureSchema(f"base+{feat}", base_feats + [feat])
        res_feat = train_eval_holdout(X_full, y, schema_feat, test_frac=test_frac)

        ablation_results[feat] = res_feat
        ablation_diags[feat] = burst_diagnostics(X_full, res_feat)

    return res_base, diag_base, ablation_results, ablation_diags


def run_experiment(df, window_size, scenario_name=None):
    """
    Train baseline vs baseline+burst for one window size.
    Optionally restrict to a single scenario.
    """

    if scenario_name is not None:
        df = df[df["scenario"] == scenario_name].copy()
        print(f"\nRunning scenario = {scenario_name} ({len(df)} alerts)")

    # guard: need both classes
    if df["y"].nunique() < 2:
        raise ValueError(
            f"Scenario '{scenario_name}' has only one class in y: {df['y'].unique()}"
        )

    # --- build features ---
    X_dyn, y, df_used = build_dyn_features(df, window_size)

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"After preprocessing, scenario '{scenario_name}' has only one class in y."
        )

    X_static = build_static_features(df_used)
    X_symbolic = symbolic_features.build_symbolic_features(df_used, X_dyn=X_dyn)

    print("\nSymbolic features generated:")
    print(sorted(X_symbolic.columns))

    # derive active features from what the builder emitted
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]
    sym_miss = [c for c in X_symbolic.columns if c.startswith("m_")]

    print("Active symbolic:", sym_feats)

    X_full = pd.concat([X_dyn, X_static, X_symbolic], axis=1).reset_index(drop=True)
    y = np.asarray(y)

    assert len(X_full) == len(y)

    # --- schemas ---
    base_feats = list(X_dyn.columns) + list(X_static.columns)

    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]

    schema_base = FeatureSchema("base", base_feats)
    schema_symbolic = FeatureSchema("base+symbolic", base_feats + sym_feats)

    print("\nSymbolic feature positives (auto):")
    for f in sorted([c for c in X_full.columns if c.startswith("is_")]):
        print(f"  {f}: {int(X_full[f].sum())}")

    print(
        f"Total features: {X_full.shape[1]} (static: {len(X_static.columns)}, dynamic: {len(X_dyn.columns)}, symbolic: {len(X_symbolic.columns)})"
    )

    # --- train ---

    print("\nTraining BASELINE model...")
    res_base = train_eval_holdout(X_full, y, schema_base, test_frac=0.3)
    diag_base = burst_diagnostics(X_full, res_base)

    # store per-feature results
    ablation_results = {}  # feat -> res
    ablation_diags = {}  # feat -> diag

    for feat in sym_feats:
        print(f"\nTraining base + '{feat}' ...")
        schema_feat = FeatureSchema(f"base+{feat}", base_feats + [feat])
        res_feat = train_eval_holdout(X_full, y, schema_feat, test_frac=0.3)

        ablation_results[feat] = res_feat
        ablation_diags[feat] = burst_diagnostics(X_full, res_feat)

    # also train the full bundle once (optional but useful)
    print("\nTraining BASE + ALL symbolic features...")
    schema_all = FeatureSchema("base+symbolic_all", base_feats + sym_feats)
    res_all = train_eval_holdout(X_full, y, schema_all, test_frac=0.3)
    diag_all = burst_diagnostics(X_full, res_all)

    return {
        "base": res_base,
        "diag_base": diag_base,
        # one-feature-at-a-time ablation
        "ablation": ablation_results,
        "diag_ablation": ablation_diags,
        # full symbolic bundle
        "sym_all": res_all,
        "diag_sym_all": diag_all,
        "X_full": X_full,
        "y": y,
        "sym_feats": sym_feats,
        "base_feats": base_feats,
    }

def run_windowed_symbolic_pipeline(
    df,
    window_days=7,
    step_days=7,
    lookback_days=1,
    test_frac=0.3,
    threshold=0.5,
    out_dir="../plots/windowed",
    use_memory=True,
):
    """
    Windowed experiment:
    - Split df into rolling time windows.
    - For each window k (train window), build features and run baseline + symbolic ablations.
    - Select symbolic features to activate (optionally with memory).
    - Train final model on window k using (baseline + selected symbolic).
    - Evaluate that SAME frozen feature set on the NEXT window k+1 (future generalization).
    - Append per-window results to history.
    """

    # 1) Split full dataframe into windows (each window is a training period)
    windows = make_time_windows(df, window_days=window_days, step_days=step_days)

    active_syms = []  # symbolic features currently "active" (intended to carry to next window)
    history = []      # per-window results

    # Optional "memory" of symbolic features across windows (stability / carry-over)
    mem = (SymbolicMemory(decay=0.85, reward=1.0, min_score=0.9) if use_memory else None)
    
    print(f"[pipeline] Symbolic memory enabled: {use_memory}")

    # Iterate over window pairs: current window k for training, next window k+1 for evaluation
    for k in range(len(windows) - 1):
        w_start, w_end, df_k = windows[k]
        _, _, df_next = windows[k + 1]

        sc = df_k["scenario"].dropna().unique()
        print(f"[window {k}] scenarios_in_window={sc}")
        assert len(sc) == 1, f"Mixed scenarios in window {k}: {sc}"


        print(
            f"\n[window {k}] raw df_k rows={len(df_k)} "
            f"y_counts={df_k['y'].value_counts(dropna=False).to_dict()}"
        )

        # 2) Define lookback horizon used by dynamic features (e.g., counters/recency/rarity)
        window_size = pd.Timedelta(days=lookback_days)

        # 3) Build features ON CURRENT WINDOW (training window)
        # 3a) Dynamic features (computed using the lookback window)
        X_dyn_k, y_k, df_used_k = build_dyn_features(df_k, window_size)
        print(
            f"[window {k}] after build_dyn_features df_used_k rows={len(df_used_k)} "
            f"y_counts={pd.Series(y_k).value_counts(dropna=False).to_dict()}"
        )

        # 3b) Static features (non-temporal metadata features)
        X_static_k = build_static_features(df_used_k)

        # 3c) Symbolic features (rule/ontology-derived indicators like is_*)
        X_sym_all_k = build_sym_features(df_used_k, X_dyn=X_dyn_k)

        # 3d) Full design matrix for this window
        X_full_k = pd.concat([X_dyn_k, X_static_k, X_sym_all_k], axis=1).reset_index(drop=True)
        y_k = np.asarray(y_k)

        # Skip windows where the label is single-class (AUC etc becomes undefined)
        if len(np.unique(y_k)) < 2:
            print(f"[window {k}] Skipping (single-class in window).")
            continue

        # Baseline schema: dynamic + static only
        base_feats = list(X_dyn_k.columns) + list(X_static_k.columns)

        # Candidate symbolic features: only boolean indicators starting with "is_"
        sym_feats = [c for c in X_sym_all_k.columns if c.startswith("is_")]

        # 4) Train baseline + run ablations (one symbolic feature at a time)
        #    Purpose: measure marginal effect of each symbolic feature on AUC/FP/alerts etc.
        try:
            res_base, diag_base, ablation_results, ablation_diags = run_ablation_holdout(
                X_full_k, y_k, base_feats, sym_feats, test_frac=test_frac
            )
        except ValueError as e:
            print(f"[window {k}] Skipping (holdout split single-class): {e}")
            continue

        # 5) Update memory (decay old scores), then select symbols for NEXT stage
        if use_memory:
            mem.step_decay()


        # 6) Choose symbolic features to activate based on ablation metrics + constraints
        selected = select_symbolic_features(
            ablation_results=ablation_results,
            X_full=X_full_k,
            y=y_k,
            res_base=res_base,
            sym_feats=sym_feats,
            threshold=threshold,
        )

        # Reward features that were selected this window
        if use_memory:
            mem.reward_feats(selected)
            active_syms = list(dict.fromkeys(selected + mem.active()))
        else:
            active_syms = selected


        # 7) Train a "final" model on current window using baseline + active symbols
        final_feats = base_feats + active_syms
        res_final = train_eval_holdout(
            X_full_k, y_k, FeatureSchema("final", final_feats), test_frac=test_frac
        )

        # 8) Build features ON NEXT WINDOW (future evaluation window)
        X_dyn_n, y_n, df_used_n = build_dyn_features(df_next, window_size)
        X_static_n = build_static_features(df_used_n)

        X_sym_all_n = build_sym_features(df_used_n, X_dyn=X_dyn_n)

        X_full_n = pd.concat([X_dyn_n, X_static_n, X_sym_all_n], axis=1).reset_index(drop=True)

        # 9) Freeze the feature set (same columns), train on window k, test on window k+1
        X_train = X_full_k[final_feats].fillna(0)
        y_train = y_k
        X_test  = X_full_n[final_feats].fillna(0)
        y_test  = np.asarray(y_n)

        # Train (L1 logistic regression) and evaluate next-window metrics
        model = train_lr_l1(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc_next = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
        alerts_next = int((proba >= threshold).sum())
        fp_next = int(((proba >= threshold) & (y_test == 0)).sum())
        fn_next = int(((proba < threshold) & (y_test == 1)).sum())

        # 10) Store per-window diagnostics
        history.append(
            {
                "k": k,
                "train_window": f"{w_start.date()}→{w_end.date()}",
                "active_syms": active_syms,
                "mem_scores": dict(mem.scores),
                "auc_train_base": res_base["auc"],   # baseline performance on current window
                "auc_next": auc_next,                # generalization to next window
                "alerts_next": alerts_next,
                "fp_next": fp_next,
                "fn_next": fn_next,
            }
        )

    print(
        f"\nFinished. Produced {len(history)} history entries out of {len(windows)-1} window pairs."
    )

    return history
