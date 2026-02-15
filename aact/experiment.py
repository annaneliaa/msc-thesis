import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from build_features import build_dyn_features, build_static_features, build_sym_features
from classes import *
from train import *
from metrics import eval_subset_metrics
import os
import json

def log_row(history, out_path, row: dict):
    """
    Appends row to history and writes to output file.
    """
    history.append(row)
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")


def ensure_single_scenario(df_k: pd.DataFrame, k: int) -> str:
    """
    A check to make sure each time window contains alerts from exactly one scenario.
    The windowed protocol assumes not mixing scenarios when training/evaluating.
    - If df_k contains multiple scenario labels, results become meaningless.
    - We assert (hard fail) to catch bugs in window slicing / filtering early.

    Returns
    - The single scenario name (string) present in df_k.
    """

    sc = df_k["scenario"].dropna().unique()
    assert len(sc) == 1, f"Mixed scenarios in window {k}: {sc}"
    return sc[0]


def build_all_features(df_k: pd.DataFrame, lookback_days: int):
    """
    Feature construction wrapper for one window dataframe df_k

    - Builds:
        1) Dynamic features (time-dependent, computed using `lookback_days`)
        2) Static features (time-independent metadata)
        3) Symbolic features (rule flags, e.g. is_*)
    - Concatenates them into one design matrix X_full aligned with y.

    Returns
    - X_full:   DataFrame with [dynamic + static + symbolic] columns
    - X_sym_all: DataFrame with only the symbolic feature columns (and possible extra symbolic metadata columns)
    - y:        np.ndarray labels aligned with X_full rows
    - df_used:  dataframe after preprocessing/filters inside build_dyn_features
    - base_feats: list of feature names used in the baseline model (dyn + static)
    - sym_feats:  list of symbolic rule feature names (is_*)
    """
    window_size = pd.Timedelta(days=lookback_days)
    X_dyn, y, df_used = build_dyn_features(df_k, window_size)
    X_static = build_static_features(df_used)
    X_sym_all = build_sym_features(df_used, X_dyn=X_dyn)

    X_full = pd.concat([X_dyn, X_static, X_sym_all], axis=1).reset_index(drop=True)
    y = np.asarray(y)

    base_feats = list(X_dyn.columns) + list(X_static.columns)
    sym_feats = [c for c in X_sym_all.columns if c.startswith("is_")]
    return X_full, X_sym_all, y, df_used, base_feats, sym_feats


def count_sym_feat_fires(X_sym_all: pd.DataFrame, sym_feats: list[str]) -> dict:
    """
    Count how often each symbolic rule fires in the CURRENT window.

    Interpretation
    - support[f] = number of rows in this window where rule f == 1
    - Used to avoid selecting symbolic features that almost never trigger

    Returns
    - dict: feature -> integer support count
    """
    return {f: int((X_sym_all[f].fillna(0).astype(int) == 1).sum()) for f in sym_feats}


def fp_only_select_topk_by_support(
    supports: dict, min_support: int, max_k: int
) -> list[str]:
    """
    Filter: keep only rules that fire at least `min_support` times in the window.
    Then rank to sort remaining rules by support in descending order.
    Then select the top max_k features.

    Returns
    - list of selected symbolic feature names
    """
    chosen = [f for f, s in supports.items() if s >= min_support]
    chosen = sorted(chosen, key=lambda f: supports[f], reverse=True)[:max_k]
    return chosen


def fp_only_select_by_mem_threshold(
    mem, sym_feats: list[str], score_threshold: float
) -> list[str]:
    """
    Select only symbolic features whose memory score >= score_threshold (defined in the declaration of the SymbolicMemory class)
    Features can enter/leave the active set as scores change.

    Returns
    - list of selected symbolic feature names
    """
    # take any feature whose memory score crosses threshold
    return [f for f in sym_feats if mem.scores.get(f, 0.0) >= score_threshold]


def union_preserve_order(a: list[str], b: list[str]) -> list[str]:
    """
    Combine two feature lists without duplicates, preserving first-seen order.
    Used for merging a list of selected symbolic features with ones that are active in memory.
    """
    return list(dict.fromkeys(list(a) + list(b)))


def compute_suppression_stats_from_mask(
    y_test: np.ndarray, suppressed_mask: np.ndarray
):
    """
    Convert a suppression mask into FP-suppression metrics on a labeled dataset.

    Inputs
    - y_test: labels for the evaluated window (0=benign/FP, 1=attack/TP)
    - suppressed_mask: boolean array; True means "this alert would be suppressed"

    Outputs (counts)
    - suppressed_next_total / benign / attack
    - total_next / benign / attack

    Outputs (rates)
    - supp_rate_total  = suppressed_total / total
    - supp_rate_benign = suppressed_benign / total_benign   (this is your FP suppression rate)
    - supp_rate_attack = suppressed_attack / total_attack   (danger: suppressing attacks)

    Returns
    - dict with counts + rates (NaN-safe for divide-by-zero)
    """
    suppressed_mask = suppressed_mask.astype(bool)
    suppressed_next_total = int(suppressed_mask.sum())
    suppressed_next_attack = int(((y_test == 1) & suppressed_mask).sum())
    suppressed_next_benign = int(((y_test == 0) & suppressed_mask).sum())

    total_next = int(len(y_test))
    total_attack_next = int((y_test == 1).sum())
    total_benign_next = int((y_test == 0).sum())

    benign_supp_rate = (
        (suppressed_next_benign / total_benign_next)
        if total_benign_next > 0
        else np.nan
    )
    attack_supp_rate = (
        (suppressed_next_attack / total_attack_next)
        if total_attack_next > 0
        else np.nan
    )
    total_supp_rate = (suppressed_next_total / total_next) if total_next > 0 else np.nan

    return {
        "suppressed_next_total": suppressed_next_total,
        "suppressed_next_benign": suppressed_next_benign,
        "suppressed_next_attack": suppressed_next_attack,
        "total_next": total_next,
        "total_benign_next": total_benign_next,
        "total_attack_next": total_attack_next,
        "supp_rate_total": total_supp_rate,
        "supp_rate_benign": benign_supp_rate,
        "supp_rate_attack": attack_supp_rate,
    }


def suppression_mask_from_active_syms(
    X_sym_all_next: pd.DataFrame, active_syms: list[str]
) -> np.ndarray:
    """
    Build the suppression mask for a window given an active symbolic feature set.
    Suppress an alert if ANY active symbolic rule fires on that alert: suppressed = OR_{f in active_syms} (X_sym_all_next[f] == 1)

    Inputs
    - X_sym_all_next: symbolic feature matrix for the evaluated (next) window
    - active_syms: list of rule names to use as the suppression rule set

    Returns
    - boolean numpy array aligned with rows of X_sym_all_next
    """
    if not active_syms:
        return np.zeros(len(X_sym_all_next), dtype=bool)
    return (
        X_sym_all_next[active_syms].fillna(0).astype(int).sum(axis=1) > 0
    ).to_numpy()


def loo_ablation_fp_only(
    X_sym_all_next: pd.DataFrame, y_test: np.ndarray, active_syms: list[str]
):
    """
    Returns:
      base_stats: suppression stats with full active set
      deltas: dict feature -> dict of deltas (base - without_feature) for benign/attack/total suppressed
    """
    if not active_syms:
        base_mask = np.zeros(len(X_sym_all_next), dtype=bool)
        base_stats = compute_suppression_stats_from_mask(y_test, base_mask)
        return base_stats, {}

    M = X_sym_all_next[active_syms].fillna(0).astype(int).to_numpy()
    row_sum = M.sum(axis=1)

    base_mask = row_sum > 0
    base_stats = compute_suppression_stats_from_mask(y_test, base_mask)

    deltas = {}
    for j, f in enumerate(active_syms):
        mask_wo = (row_sum - M[:, j]) > 0
        wo_stats = compute_suppression_stats_from_mask(y_test, mask_wo)

        deltas[f] = {
            "delta_suppressed_benign": base_stats["suppressed_next_benign"]
            - wo_stats["suppressed_next_benign"],
            "delta_suppressed_attack": base_stats["suppressed_next_attack"]
            - wo_stats["suppressed_next_attack"],
            "delta_suppressed_total": base_stats["suppressed_next_total"]
            - wo_stats["suppressed_next_total"],
        }

    return base_stats, deltas

def make_time_windows(df, window_days=7, step_days=7, ts_col="timestamp", cover_tail=True):
    """
    Returns: list of (w_start, w_end, w_df)
    Optionally covers the tail so the last partial window is included
    """
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col]).sort_values(ts_col)  # <-- no reset_index

    start = d[ts_col].min()
    end = d[ts_col].max()

    window = pd.Timedelta(days=float(window_days))
    step   = pd.Timedelta(days=float(step_days))

    windows = []
    cur = start

    if cover_tail:
        # generate windows until cur passes end; clamp last window end
        while cur <= end:
            w_start = cur
            w_end = min(cur + window, end + pd.Timedelta(microseconds=1))  # include max timestamp
            w = d[(d[ts_col] >= w_start) & (d[ts_col] < w_end)]
            windows.append((w_start, w_end, w))
            cur = cur + step
    else:
        # old behavior (drops tail)
        while cur + window <= end:
            w_start = cur
            w_end = cur + window
            w = d[(d[ts_col] >= w_start) & (d[ts_col] < w_end)]
            windows.append((w_start, w_end, w))
            cur = cur + step

    return windows



def select_symbolic_features(
    ablation_results,  # dict feat -> res_feat
    X_full,
    y,
    res_base,
    sym_feats,
    threshold=0.5,
    min_support=200,
    max_fp_increase=50,
    min_auc_gain=0.0,
    min_fp_reduction=10,  # require at least this many fewer FPs vs baseline
    max_fn_increase=0,  # in case positives exist; keep strict by default
    mode="auto",  # "auto" | "fp_only" | "classification"
    memory=None,
    prefer_memory=None,
):
    """
    Returns selected features to keep active in next window based on 1-feature ablation results from current window.

    In FP-only windows (no attacks in the test split),
    the objective is:  "reduce false positives / alerts" because AUC is undefined and you don't want any FN increase.

    In classification windows (both classes in test split), objective becomes:
        "do not hurt discrimination (AUC) and do not explode false positives"

    Inputs
    - ablation_results:
        dict mapping feature name -> fitted holdout result for model(base + feature).
        Each `res_feat` should contain what `eval_subset_metrics` needs
        (e.g., predictions/probas or indices + model info).
    - X_full, y:
        the feature matrix and labels used for training and evaluation.
    - res_base:
        holdout result for the baseline model (must contain `test_idx_start` so we know
        which rows are the test subset).
    - sym_feats:
        list of candidate symbolic rule features (typically is_* columns).
    """
    chosen = []
    split = res_base["test_idx_start"]
    y_test = np.asarray(y)[split:]

    # 1) Determine whether we're in "fp_only" vs "classification" mode:
    # - mode="auto": fp_only if test split has only one class
    # - mode="fp_only": force fp-only criteria
    # - mode="classification": force classification criteria
    if mode == "auto":
        fp_only = len(np.unique(y_test)) < 2  # single-class test
    else:
        fp_only = mode == "fp_only"

    # 2) Compute baseline metrics on the test split via eval_subset_metrics.
    base_stats = eval_subset_metrics(X_full, y, res_base, threshold=threshold)

    # 3) For each symbolic feature f:
    for f in sym_feats:
        res_f = ablation_results.get(f)

        # skip eif abalation result is missing
        if res_f is None:
            continue

        # Check support in test subset
        # Require f fires at least min_support times in the test split (prevents selecting noisy/meaningless sym feats)
        split = res_base["test_idx_start"]
        X_test = X_full.iloc[split:].reset_index(drop=True)
        support = int((X_test[f].fillna(0).astype(int) == 1).sum())
        if support < min_support:
            continue

        # Compute metrics for base+f
        stats_f = eval_subset_metrics(X_full, y, res_f, threshold=threshold)

        # Apply mode-specific acceptance criteria:
        # - FP-only:
        #   keep f if it reduces FP by at least `min_fp_reduction` AND
        #   does not increase FN by more than `max_fn_increase` (default 0).
        # - Classification:
        #   keep f if AUC gain >= `min_auc_gain` AND FP increase <= `max_fp_increase`.
        if fp_only:
            # objective: reduce false positives (equivalently reduce alerts when y_test all 0)
            fp_reduction = base_stats["fp"] - stats_f["fp"]
            fn_increase = stats_f["fn"] - base_stats["fn"]

            if (fp_reduction >= min_fp_reduction) and (fn_increase <= max_fn_increase):
                chosen.append(f)
        else:
            # objective: must not explode FP, and must not hurt AUC
            auc_gain = stats_f["auc"] - base_stats["auc"]
            fp_increase = stats_f["fp"] - base_stats["fp"]

            if (auc_gain >= min_auc_gain) and (fp_increase <= max_fp_increase):
                chosen.append(f)

    # 4) Optional memory bias:
    #     if `memory` is provided and `prefer_memory` is true, append memory.active()
    #     features (that are in sym_feats) to the chosen list, preserving order
    if memory is not None and prefer_memory:
        mem_active = set(memory.active())
        # union, but keep order stable
        chosen = list(
            dict.fromkeys([*chosen, *[f for f in sym_feats if f in mem_active]])
        )

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


# remove/change this function later
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


def simple_ablation_experiment(df, window_size, scenario_name=None):
    """
    One shot, single-scenario ablation experiment (no time windows, no next window eval).
    For one scenario's full dataset, test whether adding each symbolic feature helps vs baseline

    Function filters dataset to one scenario. Then build features once:
    - dynamic features (from window_size lookback)
    - "static" features (taken from alert)
    - symbolic features (all rules emitted by build_sym_features)

    Train on a random holdout split:
    1) baseline = dynamic + static
    2) baseline + one symbolic feature
    3) baseline + all (active) symbolic features

    Returns per-feature performance. Shows if adding feature f helps on the scenario data.
    """

    if scenario_name is not None:
        df = df[df["scenario"] == scenario_name].copy()
        print(f"\nRunning scenario = {scenario_name} ({len(df)} alerts)")

    # guard: need both classes to compute AUC
    if df["y"].nunique() < 2:
        raise ValueError(
            f"Scenario '{scenario_name}' has only one class in y: {df['y'].unique()}"
        )

    # --- Build features ONCE for the whole dataset ---
    # window_size here is a lookback used inside build_dyn_features
    X_dyn, y, df_used = build_dyn_features(df, window_size)

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"After preprocessing, scenario '{scenario_name}' has only one class in y."
        )

    X_static = build_static_features(df_used)
    X_symbolic = build_sym_features(df_used, X_dyn=X_dyn)

    # derive active features from what the builder emitted
    # Symbolic features = all "is_*" rules that fire at least somewhere in this dataset
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]
    # sym_miss = [c for c in X_symbolic.columns if c.startswith("m_")]

    X_full = pd.concat([X_dyn, X_static, X_symbolic], axis=1).reset_index(drop=True)
    y = np.asarray(y)

    assert len(X_full) == len(y)

    # ----- Baseline model -----
    base_feats = list(X_dyn.columns) + list(X_static.columns)
    schema_base = FeatureSchema("base", base_feats)
    print("\nTraining BASELINE model...")
    res_base = train_eval_holdout(X_full, y, schema_base, test_frac=0.3)
    diag_base = burst_diagnostics(X_full, res_base)

    # ------ Ablation study (baseline + feat) -------
    # store per-feature results
    ablation_results = {}
    ablation_diags = {}

    for feat in sym_feats:
        print(f"\nTraining base + '{feat}' ...")
        schema_feat = FeatureSchema(f"base+{feat}", base_feats + [feat])
        res_feat = train_eval_holdout(X_full, y, schema_feat, test_frac=0.3)

        ablation_results[feat] = res_feat
        ablation_diags[feat] = burst_diagnostics(X_full, res_feat)

    # --- Full bundle (baseline + all symbolic) ---
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


def windowed_ablation_experiment(
    df,
    window_days=7,
    step_days=7,
    lookback_days=1,
    test_frac=0.3,
    threshold=0.5,
    out_dir=None,
    use_memory=True,
    tau=0.9,
    force_fp_only=False,
    # FP-only selection knobs
    fp_min_support=200,
    fp_max_k=6,
    # # Option 2: thresholded activation
    # fp_activation="topk",  # "topk" | "mem_threshold" | "hybrid"
    # fp_score_threshold=1.5,  # used when fp_activation includes mem_threshold
    # fp_support_gate=True,  # if True, also require support>=fp_min_support
    # # Option 1: leave-one-out ablation (on NEXT window)
    # fp_loo_ablation=False,
):
    """
    For each time window k, make decisions using data in k, then evaluate on the next window k+1.

    Hybrid windowed experiment:
    - If current window has both classes -> supervised ablation + ML evaluation
    - If current window is single-class or holdout fails -> FP-only symbolic selection (no training)
        + evaluate as suppression mask on the next window

    Modes:
    - supervised: if window k has both classes, train models + pick symbolic features
    - fp_only: if window k is a single-cass (all benign alerts) or holdout fails
        - pick symbolic features and measure FP suppression on benign alerts labelled as attack (in window k+1)
    """

    # 1) Make time windows
    windows = make_time_windows(df, window_days=window_days, step_days=step_days)
    # we need at least two windows to evaluate on "next" window
    if len(windows) < 2:
        return []

    # 2) Setup output directories
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "all_history.jsonl")
    else:
        out_path = None

    # 3) Initialize memory to track active symbolic feature set (carried across windows)
    mem = SymbolicMemory(decay=0.85, reward=1.0, min_score=tau) if use_memory else None
    active_syms = []
    history = []

    # 4) Loop over all window pairs (k -> k+1)
    for k in range(len(windows) - 1):
        w_start, w_end, df_k = windows[k]
        _, _, df_next = windows[k + 1]

        # Enforce one scenario per window
        scenario = ensure_single_scenario(df_k, k)

        # Decay memory (if enabled)
        if use_memory:
            mem.step_decay()

        # Build features on current window
        X_full_k, X_sym_all_k, y_k, df_used_k, base_feats, sym_feats = (
            build_all_features(df_k, lookback_days)
        )

        # Determine if supervised training is possible
        # If only one class in y_k, supervised AUC training can’t work reliably -> measure feature effects using FP suppression rates
        supervised_possible = len(np.unique(y_k)) >= 2

        # FORCE FP-ONLY MODE (for ablation study)
        if force_fp_only:
            supervised_possible = False
        # ----------------------------
        # SUPERVISED MODE
        # ----------------------------
        mode = "supervised"

        if supervised_possible:
            try:
                # Run ablation on window k
                # Trains baseline model + one model for each symbolic feature
                res_base, diag_base, ablation_results, ablation_diags = (
                    run_ablation_holdout(
                        X_full_k, y_k, base_feats, sym_feats, test_frac=test_frac
                    )
                )
            except ValueError:
                # Throws if train/test split ends up with a single class
                supervised_possible = False

            # Select symbolic features important NOW based on ablation deltas
            selected = select_symbolic_features(
                ablation_results=ablation_results,
                X_full=X_full_k,
                y=y_k,
                res_base=res_base,
                sym_feats=sym_feats,
                threshold=threshold,
            )

            # Update memory + define active symbolic features by merging what was
            # important BEFORE (mem.active) with what is important NOW (selected)
            if use_memory:
                mem.reward_feats(selected)
                active_syms = union_preserve_order(selected, mem.active())
            else:
                active_syms = selected

            # Train final model for window k
            final_feats = base_feats + active_syms
            res_final = train_eval_holdout(
                X_full_k, y_k, FeatureSchema("final", final_feats), test_frac=test_frac
            )

            # Evaluate on next window k+1
            # Build features on df_next (dataframe of window k+1)
            X_full_n, X_sym_all_n, y_n, df_used_n, _, _ = build_all_features(
                df_next, lookback_days
            )
            X_train = X_full_k[final_feats].fillna(0)
            y_train = y_k
            X_test = X_full_n[final_feats].fillna(0)
            y_test = np.asarray(y_n)

            # Train on window k
            model = train_lr_l1(X_train, y_train)

            # Predict probabilities on next window
            proba = model.predict_proba(X_test)[:, 1]

            # Compute next window metrics: AUC, alerts, FP, FN
            auc_next = (
                roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
            )
            alerts_next = int((proba >= threshold).sum())
            fp_next = int(((proba >= threshold) & (y_test == 0)).sum())
            fn_next = int(((proba < threshold) & (y_test == 1)).sum())

            # Log results
            log_row(
                history,
                out_path,
                {
                    "k": k,
                    "mode": mode,
                    "scenario": scenario,
                    "train_window": f"{w_start.date()}→{w_end.date()}",
                    "active_syms": active_syms,
                    "mem_scores": dict(mem.scores) if use_memory else None,
                    "auc_train_base": res_base.get("auc", np.nan),
                    "auc_train_final": res_final.get("auc", np.nan),
                    "auc_next": auc_next,
                    "alerts_next": alerts_next,
                    "fp_next": fp_next,
                    "fn_next": fn_next,
                    # fp-only fields
                    "suppressed_next_total": np.nan,
                    "suppressed_next_benign": np.nan,
                    "suppressed_next_attack": np.nan,
                },
            )
            continue  # Continue to next k

        # ----------------------------
        # FP-ONLY MODE (no training)
        # ----------------------------
        mode = "fp_only"

        # Compute supports on window k
        # Counts how often each symbolic feature fires in window k
        counts = count_sym_feat_fires(X_sym_all_k, sym_feats)

        # Pick a candidate symbolic feature set using counts
        # Filters out features that don't fire enough in the current window
        topk = fp_only_select_topk_by_support(
            counts, fp_min_support, fp_max_k
        )  # list of feature names

        # Update memory if enabled
        if use_memory:
            mem.reward_feats(topk)

    
        selected_syms_k = topk  # treat topk as "selected in window k"

        mem_scores_selected = (
            {f: float(mem.scores.get(f, 0.0)) for f in selected_syms_k}
            if use_memory else None
        )

        # Start with empty candidate set
        selected = []

        # Build features for next window
        # X_sym_all_n contains all symbolic features that fired in the next window
        _, X_sym_all_n, y_n, df_used_n, _, _ = build_all_features(
            df_next, lookback_days
        )
        y_test = np.asarray(y_n)

        # Filter out features that do not exist in the next window
        sym_feats_eval = [f for f in sym_feats if f in X_sym_all_n.columns]

        # Add here: filter out feats that dont fire enough

        # Baseline: no suppression
        base_mask = np.zeros(len(y_test), dtype=bool)  # all false, suppress nothing
        base_stats = compute_suppression_stats_from_mask(
            y_test, base_mask
        )  # should be all zeroes

        # Ablation: evaluate FP suppression of each feature alone on next window k+1
        per_feat = {}
        for f in sym_feats_eval:
            # For each feature f, suppress exactly the alerts where f fires
            mask_f = X_sym_all_n[f].fillna(0).astype(int).to_numpy() == 1

            # Compute stats to see how many benign alerts are supressed + how many attacks are suppressed (FP vs FN)
            stats_f = compute_suppression_stats_from_mask(y_test, mask_f)
            per_feat[f] = stats_f

        # Ceiling check: if we used all symbolic features together, how much FP supression can we get?

        # Compute mask as an OR of all candidate rules: if one fires, use all of them
        all_mask = (
            (
                X_sym_all_n[sym_feats_eval].fillna(0).astype(int).sum(axis=1) > 0
            ).to_numpy()
            if sym_feats_eval
            else base_mask
        )
        all_stats = compute_suppression_stats_from_mask(y_test, all_mask)

        # Log results to history
        log_row(
            history,
            out_path,
            {
                "k": k,
                "mode": mode,
                "scenario": scenario,
                "train_window": f"{w_start.date()}→{w_end.date()}",
                "selected_syms": selected_syms_k,
                "tau": tau,
                "mem_scores_selected": mem_scores_selected,
                "supports_k": counts,
                "suppression_base": base_stats,  # baseline suppression
                "suppression_per_feat": per_feat,  # per festure ablation suppression stats
                "suppression_all_or": all_stats,  # all OR suppression stats
            },
        )
        continue

    return history


def greedy_symbolic_search(
    df,
    window_size,
    train_fn,  # function(X_full, y, feature_list) -> res dict with model, proba_test, test_idx_start
    threshold=0.5,
    max_k=6,
    lambda_fp=0.000001,  # penalty per FP (tune)
    min_subset_size=100,  # require feature fires at least this many times in test (optional)
):
    """Greedy forward feature-selection loop that tries to pick up max_k symbolic (is_X) features
    to add on top of a baseline (dynamic + static features), keeping only additions that improve
    the objective (currently, AUC and FP).
    """
    # --- build features ---
    X_dyn, y, df_used = build_dyn_features(df, window_size)
    X_static = build_static_features(df_used)
    X_symbolic = build_sym_features(df_used, X_dyn=X_dyn)

    print("\nSymbolic features generated:")
    print(sorted(X_symbolic.columns))

    # derive active features from what the builder emitted
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]
    sym_miss = [c for c in X_symbolic.columns if c.startswith("m_")]

    print("Active symbolic:", sym_feats)

    # concatenate features
    X_full = pd.concat([X_dyn, X_static, X_symbolic], axis=1).reset_index(drop=True)
    y = np.asarray(y)

    assert len(X_full) == len(y)

    # --- schemas ---
    base_feats = list(X_dyn.columns) + list(X_static.columns)

    # only columns starting with is_ are treated as symbolic features now
    sym_feats = [c for c in X_symbolic.columns if c.startswith("is_")]

    # train baseline feature set (dyn + static)
    res_base = train_fn(X_full, y, base_feats)
    # ccompute metrics
    m_base = eval_subset_metrics(X_full, y, res_base, threshold=threshold)

    # Greedy subset selection
    chosen = []
    remaining = list(sym_feats)
    history = []

    def objective(m, m_base):
        if m["subset_size"] is not None and m["subset_size"] < min_subset_size:
            return -np.inf

        fp_reduction = m_base["fp"] - m["fp"]

        return (
            (m["auc"] if not np.isnan(m["auc"]) else -1.0)
            + 0.00001 * fp_reduction
            - lambda_fp * m["fp"]
        )


    best_score = objective(m_base, m_base)

    for step in range(max_k):

        best_candidate = None
        best_candidate_res = None
        best_candidate_metrics = None
        best_candidate_score = best_score

        # For each remaining symbolic feature f
        # train a model on base_feats + chosen + [f]
        for f in remaining:
            feats = base_feats + chosen + [f]
            res = train_fn(X_full, y, feats)
            m = eval_subset_metrics(X_full, y, res, threshold=threshold, subset_col=f)

            # compute score based on objective
            score = objective(m, m_base)

            # stops when no candidate improves the objective
            if score > best_candidate_score:
                best_candidate = f
                best_candidate_res = res
                best_candidate_metrics = m
                best_candidate_score = score

        if best_candidate is None:
            break

        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score

        history.append(
            {
                "step": step + 1,
                "added": best_candidate,
                "chosen": chosen.copy(),
                **best_candidate_metrics,
            }
        )

    return {
        "base": res_base,
        "base_metrics": m_base,
        "chosen": chosen,
        "history": history,
    }
