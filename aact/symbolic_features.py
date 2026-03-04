import os
import ast
import json
from glob import glob
import pandas as pd


def _parse_tokens_cell(x):
    """Parse a tokens cell from CSV (list[str] stored as json or python literal or delimited string)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(t) for t in x]

    s = str(x).strip()
    if not s:
        return []

    # JSON list
    if s.startswith("[") and s.endswith("]"):
        # try JSON then python literal
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t) for t in v]
            except Exception:
                pass

    # fallback: pipe-separated tokens
    if "|" in s:
        return [t for t in s.split("|") if t]

    # single token
    return [s]


def _candidate_to_tokens(candidate_str: str) -> list[str]:
    """Turn 'tokA&tokB' or 'tokA & tokB' into ['tokA','tokB']."""
    if candidate_str is None or (isinstance(candidate_str, float) and pd.isna(candidate_str)):
        return []
    s = str(candidate_str).strip()
    if not s:
        return []
    # normalize separators
    s = s.replace(" & ", "&")
    # if your formatter ever uses commas, OR, etc, extend here
    toks = [t.strip() for t in s.split("&") if t.strip()]
    return toks


def build_symbolic_features_from_tokens(
    df_used: pd.DataFrame,
    scenario: str,
    run_name: str,
    tokens_dir: str = "../out/ait_ads/tokens",
    rankings_dir: str = "../out/ait_ads/rankings",
    tokens_glob: str = "*.csv",  # your saved token files
    stable_filename: str | None = None,
    top_n: int | None = None,
    min_fires: int = 1,  # drop features that never fire
    id_col: str = "alert_id",
) -> pd.DataFrame:
    """
    Build mined symbolic features for ONE scenario from:
      1) saved token files containing columns: alert_id, tokens
      2) saved stable features CSV containing candidate_str

    Returns DataFrame aligned to df_used rows (same order), with:
      - is_mined__* feature columns (0/1)
      - m_is_mined__* computable flags (=1)
    """

    if id_col not in df_used.columns:
        raise ValueError(f"df_used is missing '{id_col}' (needed to join tokens)")

    # ---- load & combine token files for this scenario ----
    # expected layout: ../out/ait_ads/tokens/{run_name}/{scenario}/*.csv
    scen_tok_dir = os.path.join(tokens_dir, run_name, scenario)
    tok_files = sorted(glob(os.path.join(scen_tok_dir, tokens_glob)))
    if not tok_files:
        raise FileNotFoundError(f"No token files found in {scen_tok_dir} matching {tokens_glob}")

    tok_parts = []
    for f in tok_files:
        tdf = pd.read_csv(f)
        if id_col not in tdf.columns or "tokens" not in tdf.columns:
            raise ValueError(f"Token file {f} must have columns: '{id_col}', 'tokens'")
        tdf = tdf[[id_col, "tokens"]].copy()
        tdf["tokens"] = tdf["tokens"].apply(_parse_tokens_cell)
        tok_parts.append(tdf)

    tok_df = pd.concat(tok_parts, ignore_index=True)

    # combine duplicates (same alert_id across windows) by unioning token lists
    tok_df = (
        tok_df.groupby(id_col)["tokens"]
              .agg(lambda lists: sorted(set(t for L in lists for t in (L or []))))
              .reset_index()
    )

    # ---- align tokens to df_used row order ----
    tok_map = tok_df.set_index(id_col)["tokens"]
    tokens_aligned = df_used[id_col].map(tok_map)

    # alerts without tokens -> empty list (feature will be 0)
    tokens_aligned = tokens_aligned.apply(lambda x: x if isinstance(x, list) else [])

    # ---- build multi-hot token matrix (fast ANDs) ----
    # join tokens into string then get_dummies -> columns are token strings
    X_tokens = tokens_aligned.apply(lambda L: "|".join(L)).str.get_dummies(sep="|")

    # ---- load stable candidates for this scenario ----
    scen_rank_dir = os.path.join(rankings_dir, run_name)
    if stable_filename is None:
        stable_filename = f"{scenario}_stable_features.csv"
    stable_path = os.path.join(scen_rank_dir, stable_filename)
    if not os.path.exists(stable_path):
        raise FileNotFoundError(f"Stable features file not found: {stable_path}")

    stable = pd.read_csv(stable_path)
    if "candidate_str" not in stable.columns:
        raise ValueError(f"{stable_path} must contain a 'candidate_str' column")

    if top_n is not None:
        stable = stable.head(top_n)

    candidates = stable["candidate_str"].dropna().astype(str).tolist()

    # ---- compute mined symbolic features ----
    X_sym = pd.DataFrame(index=df_used.index)

    for cand in candidates:
        toks = _candidate_to_tokens(cand)
        if not toks:
            continue

        # AND over token columns; missing token => always false
        mask = pd.Series(True, index=df_used.index)
        for t in toks:
            if t in X_tokens.columns:
                mask &= (X_tokens[t] == 1)
            else:
                mask &= False

        col = "is_mined__" + cand.replace(" ", "").replace("&", "__AND__")[:200]
        X_sym[col] = mask.astype(int)
        X_sym["m_" + col] = 1

    # optionally keep only features that fire at least once
    is_cols = [c for c in X_sym.columns if c.startswith("is_mined__")]
    if is_cols:
        fires = X_sym[is_cols].sum(axis=0)
        keep = fires[fires >= min_fires].index.tolist()
        keep_cols = keep + ["m_" + c for c in keep if ("m_" + c) in X_sym.columns]
        X_sym = X_sym[keep_cols]

    return X_sym

def normalize_groups(x):
    if isinstance(x, list):
        return [str(g).lower() for g in x]
    if isinstance(x, str):
        return [g.strip().lower() for g in x.split("|") if g.strip()]
    return []

def build_symbolic_features(df, X_dyn=None):
    """
    Build symbolic (knowledge-driven) features.
    is_X: concept holds
    m_is_X: concept computable
    """
    df = df.copy()
    X_sym = pd.DataFrame(index=df.index)

    if X_dyn is not None:
        X_dyn = X_dyn.reset_index(drop=True)
        df = df.reset_index(drop=True)
    else:
        raise ValueError("build_symbolic_features needs X_dyn for dyn-based rules.")

    # ---------- helpers / base fields ----------
    src = df["source"].fillna("")
    wazuh = (src == "wazuh")
    aminer = (src == "aminer")
    ids = (df.get("is_ids_alert", 0).fillna(0).astype(int) == 1)

    raw = df.get("raw_log", "").fillna("").astype(str)

    # Make sure these exist as ints (your df already has many of these cols)
    is_auth_event = df.get("is_auth_event", 0).fillna(0).astype(int)
    is_success = df.get("is_success", 0).fillna(0).astype(int)
    is_uid0 = df.get("is_uid0", 0).fillna(0).astype(int)

    username = df.get("username", "").fillna("").astype(str).str.lower()
    procname = df.get("procname", "").fillna("").astype(str)

    wazuh_level = pd.to_numeric(df.get("wazuh_level"), errors="coerce").fillna(0).astype(int)
    wazuh_update = df.get("wazuh_update", 0).fillna(0).astype(int)
    wazuh_av = df.get("wazuh_antivirus", 0).fillna(0).astype(int)

    exploit_class = df.get("exploit_class", "").fillna("").astype(str).str.lower()
    mitre_tactic = df.get("mitre_tactic", "").fillna("").astype(str).str.lower()

    # ---------- derived static context (computed locally) ----------
    # Presence cues
    has_username = (username != "").astype(int)
    has_procname = (procname != "").astype(int)

    # Category scan hint (use category if present; fall back to raw log)
    cat = df.get("category", "").fillna("").astype(str)
    cat_scan = cat.str.contains("scan", case=False, na=False).astype(int)

    # Internal IP heuristic (string prefix; good enough for now)
    srcip = df.get("srcip", "").fillna("").astype(str)
    dstip = df.get("dstip", "").fillna("").astype(str)
    is_internal_ip = srcip.str.startswith(("10.", "192.168.", "172.16."), na=False).astype(int)
    src_eq_dst = (srcip != "") & (srcip == dstip)
    src_eq_dst = src_eq_dst.astype(int)

    # Wazuh low level
    wazuh_low_level = (wazuh & (wazuh_level <= 3)).astype(int)

    # IDS low severity
    ids_sev = pd.to_numeric(df.get("ids_severity"), errors="coerce")
    ids_low_severity = (ids_sev.fillna(0) <= 2).astype(int)

    # MITRE presence
    mitre_ids = df.get("mitre_ids", "").fillna("").astype(str)
    has_mitre = ((mitre_ids != "") & (mitre_ids != "null") & (mitre_ids != "[]")).astype(int)

    # Cron hint
    is_cron = raw.str.contains("cron", case=False, na=False).astype(int)

    # Rule 1: SuspiciousAuthBurst
    X_sym["m_is_suspicious_auth_burst"] = 1
    X_sym["is_suspicious_auth_burst"] = (
        (is_auth_event == 1) &
        (is_success == 0) &
        (X_dyn["ent_count_1d"] >= 3)
    ).astype(int)

    # Rule 2: RootLoginAttempt
    X_sym["m_is_root_login_attempt"] = (has_username == 1).astype(int)
    X_sym["is_root_login_attempt"] = (
        (is_auth_event == 1) &
        (is_success == 0) &
        (username == "root")
    ).astype(int)

    # Rule 3: BehavioralNovelty (AMiner)
    X_sym["m_is_behavioral_novelty"] = aminer.astype(int)
    X_sym["is_behavioral_novelty"] = (
        aminer & (df.get("aminer_new_event", 0).fillna(0).astype(int) == 1)
    ).astype(int)

    # Rule 4: HighSeverityWazuh
    X_sym["m_is_high_severity_wazuh"] = wazuh.astype(int)
    X_sym["is_high_severity_wazuh"] = (wazuh & (wazuh_level >= 7)).astype(int)

    # Rule 5: Wazuh critical >= 10
    X_sym["m_is_wazuh_critical"] = wazuh.astype(int)
    X_sym["is_wazuh_critical"] = (wazuh & (wazuh_level >= 10)).astype(int)

    # Rule 6: Mitre mapping
    X_sym["m_is_wazuh_mitre_mapped"] = wazuh.astype(int)
    X_sym["is_wazuh_mitre_mapped"] = (wazuh & (has_mitre == 1)).astype(int)

    # Rule 7: IDS high priority (note: your threshold looked low; keep as-is)
    X_sym["m_is_ids_high_priority"] = ids_sev.notna().astype(int)
    X_sym["is_ids_high_priority"] = (ids_sev.notna() & (ids_sev >= 2)).astype(int)

    # Rule 8: IDS concept
    X_sym["m_is_ids_concept"] = 1
    X_sym["is_ids_concept"] = ids.astype(int)

    # Rule 9: High severity and novelty (missing m_ in your original; add it)
    X_sym["m_is_high_sev_and_novel"] = (wazuh & aminer).astype(int)
    X_sym["is_high_sev_and_novel"] = (
        (X_sym["is_high_severity_wazuh"] == 1) &
        (X_sym["is_behavioral_novelty"] == 1)
    ).astype(int)

    # (A1) rare entity then burst
    X_sym["m_is_rare_entity_then_burst"] = 1
    X_sym["is_rare_entity_then_burst"] = (
        (X_dyn["days_since_ent_seen"] >= 7) &
        (X_dyn["ent_count_1d"] >= 5)
    ).astype(int)

    # (A2) auth failure then uid0
    X_sym["m_is_auth_failure_then_uid0"] = 1
    X_sym["is_auth_failure_then_uid0"] = (
        (is_auth_event == 1) &
        (is_success == 0) &
        (is_uid0 == 1)
    ).astype(int)

    # (A4) high sev but not update / AV
    X_sym["m_is_high_sev_non_update_non_av"] = wazuh.astype(int)
    X_sym["is_high_sev_non_update_non_av"] = (
        wazuh & (wazuh_level >= 7) & (wazuh_update == 0) & (wazuh_av == 0)
    ).astype(int)

    # (B6) IDS class malware/c2/tls/dns suspicious
    X_sym["m_is_ids_category_malware_or_c2"] = (exploit_class != "").astype(int)
    X_sym["is_ids_category_malware_or_c2"] = exploit_class.isin(
        ["malware", "c2_activity", "tls_fingerprint", "dns_suspicious"]
    ).astype(int)

    # (B5) MITRE execution/persistence
    X_sym["m_is_mitre_execution_or_persistence"] = (mitre_tactic != "").astype(int)
    X_sym["is_mitre_execution_or_persistence"] = mitre_tactic.str.contains(
        "execution|persistence", regex=True
    ).astype(int)

    # Combined features
    X_sym["m_is_novel_and_ids"] = 1
    X_sym["is_novel_and_ids"] = (
        (X_sym["is_behavioral_novelty"] == 1) &
        (X_sym["is_ids_concept"] == 1)
    ).astype(int)

    X_sym["m_is_auth_burst_and_ids"] = 1
    X_sym["is_auth_burst_and_ids"] = (
        (X_sym["is_suspicious_auth_burst"] == 1) &
        (X_sym["is_ids_concept"] == 1)
    ).astype(int)

    X_sym["m_is_auth_burst_high_ent_rate"] = 1
    X_sym["is_auth_burst_high_ent_rate"] = (
        (X_sym["is_suspicious_auth_burst"] == 1) &
        (X_dyn["ent_rate_1d"] >= 0.2)
    ).astype(int)

    # composite FP-revealing symbolic rules
    # (C1) Benign auth noise:
    # successful auth + user known + low-sev wazuh
    X_sym["m_is_benign_auth_noise"] = wazuh.astype(int)
    X_sym["is_benign_auth_noise"] = (
        (is_auth_event == 1) &
        (is_success == 1) &
        (has_username == 1) &
        (wazuh_low_level == 1)
    ).astype(int)

    # (C2) AMiner novelty but stable (likely benign):
    # aminer new event + process present + internal IP
    X_sym["m_is_aminer_novel_but_stable"] = aminer.astype(int)
    X_sym["is_aminer_novel_but_stable"] = (
        aminer &
        (df.get("aminer_new_event", 0).fillna(0).astype(int) == 1) &
        (has_procname == 1) &
        (is_internal_ip == 1)
    ).astype(int)

    # (C3) Scanner-looking but internal:
    X_sym["m_is_internal_scan_like"] = 1
    X_sym["is_internal_scan_like"] = (
        (cat_scan == 1) &
        (is_internal_ip == 1) &
        (src_eq_dst == 1)
    ).astype(int)

    # (C4) Low-severity IDS background noise:
    X_sym["m_is_low_sev_ids_background"] = 1
    X_sym["is_low_sev_ids_background"] = (
        (ids.astype(int) == 1) &
        (ids_low_severity == 1) &
        (has_mitre == 1)
    ).astype(int)

    # (C5) Periodic maintenance (cron + high recent recurrence)
    # Requires you to have something like ent_count_1d; tune threshold as needed.
    X_sym["m_is_periodic_maintenance"] = 1
    X_sym["is_periodic_maintenance"] = (
        (is_cron == 1) &
        (X_dyn["ent_count_1d"] >= 3)
    ).astype(int)

    # (D1) Benign-prone entity burst (FP-ish):
    # entity is very active now, but historically almost never attacks
    X_sym["m_is_benign_prone_entity_burst"] = 1
    X_sym["is_benign_prone_entity_burst"] = (
        (X_dyn["ent_count_1d"] >= 5) &
        (X_dyn["ent_rate_1d"] <= 0.01)
    ).astype(int)

    # (D2) Benign-prone category burst:
    X_sym["m_is_benign_prone_category_burst"] = 1
    X_sym["is_benign_prone_category_burst"] = (
        (X_dyn["cat_count_1d"] >= 10) &
        (X_dyn["cat_rate_1d"] <= 0.01)
    ).astype(int)

    # (D3) Auth failure on benign-prone entity:
    X_sym["m_is_auth_fail_on_benign_entity"] = 1
    X_sym["is_auth_fail_on_benign_entity"] = (
        (df.get("is_auth_event", 0).fillna(0).astype(int) == 1) &
        (df.get("is_success", 0).fillna(0).astype(int) == 0) &
        (X_dyn["ent_rate_1d"] <= 0.01) &
        (X_dyn["ent_count_1d"] >= 3)
    ).astype(int)

    # (D4) IDS high-severity but entity historically benign (possible FP):
    sev = pd.to_numeric(df.get("ids_severity"), errors="coerce").fillna(0)
    X_sym["m_is_ids_high_sev_on_benign_entity"] = 1
    X_sym["is_ids_high_sev_on_benign_entity"] = (
        (df.get("is_ids_alert", 0).fillna(0).astype(int) == 1) &
        (sev >= 3) &
        (X_dyn["ent_rate_1d"] <= 0.01)
    ).astype(int)

    # (D5) Wazuh high severity but category historically benign (possible FP):
    X_sym["m_is_wazuh_high_sev_on_benign_category"] = wazuh.astype(int)
    X_sym["is_wazuh_high_sev_on_benign_category"] = (
        wazuh &
        (wazuh_level >= 7) &
        (X_dyn["cat_rate_1d"] <= 0.01)
    ).astype(int)

    # Attack indicators
    # (D6) Novel entity spike:
    # host not seen for a long time, then suddenly active (good attack heuristic)
    X_sym["m_is_novel_entity_spike"] = 1
    X_sym["is_novel_entity_spike"] = (
        (X_dyn["days_since_ent_seen"] >= 7) &
        (X_dyn["ent_count_1d"] >= 5)
    ).astype(int)

    # (D7) Novel category burst:
    X_sym["m_is_novel_category_burst"] = 1
    X_sym["is_novel_category_burst"] = (
        (X_dyn["days_since_cat_seen"] >= 7) &
        (X_dyn["cat_count_1d"] >= 10)
    ).astype(int)

    # (D8) "Chronic attacker" entity (good attack indicator; sanity check rule)
    X_sym["m_is_chronic_attacker_entity"] = 1
    X_sym["is_chronic_attacker_entity"] = (
        (X_dyn["ent_rate_1d"] >= 0.2) &
        (X_dyn["ent_count_1d"] >= 5)
    ).astype(int)

    # (D9) "Chronic attacker" category (good attack indicator)
    X_sym["m_is_chronic_attacker_category"] = 1
    X_sym["is_chronic_attacker_category"] = (
        (X_dyn["cat_rate_1d"] >= 0.2) &
        (X_dyn["cat_count_1d"] >= 10)
    ).astype(int)


    return X_sym
