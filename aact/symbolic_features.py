import pandas as pd
import numpy as np

def normalize_groups(x):
    if isinstance(x, list):
        return [str(g).lower() for g in x]
    if isinstance(x, str):
        return [g.strip().lower() for g in x.split("|") if g.strip()]
    return []

def build_symbolic_features(df, X_dyn=None):
    """
    Build symbolic (knowledge-driven) features.
    These features represent high-level SOC concepts,
    not raw alert properties.
    """
    df = df.copy()
    X_sym = pd.DataFrame(index=df.index)
    
    if X_dyn is not None:
        # align indices
        X_dyn = X_dyn.reset_index(drop=True)
        df = df.reset_index(drop=True)

    # ==================================================
    # Rule 0 (optional / may be sparse)
    # SuspiciousRemoteAccess
    # ==================================================
    df["groups_norm"] = df["groups"].apply(normalize_groups)
    dstport = pd.to_numeric(df["dstport"], errors="coerce")

    is_auth_group = df["groups_norm"].apply(
        lambda gs: "authentication" in gs
    ).astype(int)
    is_auth = (is_auth_group == 1) | (df["is_auth_event"].fillna(0).astype(int) == 1)

    # X_sym["m_is_suspicious_remote_access"] = dstport.notna().astype(int)
    # X_sym["is_suspicious_remote_access"] = (
    #     is_auth & dstport.notna() & (dstport.astype("Int64") != 22)
    # ).astype(int)

    # ==================================================
    # Rule 1: SuspiciousAuthBurst
    # IF auth failure AND many alerts from same entity
    # ==================================================
    if X_dyn is None:
        raise ValueError("SuspiciousAuthBurst needs X_dyn (ent_count_1d).")

    X_sym["m_is_suspicious_auth_burst"] = 1  # always computable

    X_sym["is_suspicious_auth_burst"] = (
        (df["is_auth_event"] == 1) &
        (df["is_success"] == 0) &
        (X_dyn["ent_count_1d"] >= 3)
    ).astype(int)

    # ==================================================
    # Rule 2: RootLoginAttempt
    # IF auth event AND username == root AND failure
    # ==================================================
    username = df["username"].fillna("").str.lower()

    X_sym["m_is_root_login_attempt"] = (username != "").astype(int)
    X_sym["is_root_login_attempt"] = (
        (df["is_auth_event"] == 1) &
        (df["is_success"] == 0) &
        (username == "root")
    ).astype(int)

    # ==================================================
    # Rule 3: BehavioralNovelty (AMiner)
    # IF AMiner flags new behavior
    # ==================================================
    X_sym["m_is_behavioral_novelty"] = (df["source"] == "aminer").astype(int)
    X_sym["is_behavioral_novelty"] = (
        (df["source"] == "aminer") &
        (df["aminer_new_event"] == 1)
    ).astype(int)

    # -------------------------
    # Rule 4: HighSeverityWazuh (simple expert rule)
    # "high rule level is more suspicious"
    # -------------------------
    # High severity Wazuh alert (expert threshold)
    X_sym["m_is_high_severity_wazuh"] = (df["source"] == "wazuh").astype(int)

    X_sym["is_high_severity_wazuh"] = (
        (df["source"] == "wazuh") &
        (df["wazuh_level"].fillna(0).astype(int) >= 7) # set threshold, add mid, high later
    ).astype(int)

    # Rule 5: Wazuh critical >= 10
    X_sym["m_is_wazuh_critical"] = (df["source"]=="wazuh").astype(int)
    X_sym["is_wazuh_critical"] = ((df["source"]=="wazuh") & (df["wazuh_level"].fillna(0).astype(int) >= 10)).astype(int)

    # Rule 6: Mitre mapping
    m = df["mitre_ids"].fillna("").astype(str)
    X_sym["m_is_wazuh_mitre_mapped"] = (df["source"]=="wazuh").astype(int)
    X_sym["is_wazuh_mitre_mapped"] = ((df["source"]=="wazuh") & (m != "") & (m != "null") & (m != "[]")).astype(int)

    # Rule 7: Is IDS alert
    X_sym["m_is_ids_concept"] = 1
    X_sym["is_ids_concept"] = (df["is_ids_alert"].fillna(0).astype(int) == 1).astype(int)
    sev = pd.to_numeric(df["ids_severity"], errors="coerce")
    X_sym["m_is_ids_high_priority"] = sev.notna().astype(int)
    X_sym["is_ids_high_priority"] = (sev.notna() & (sev >= 2)).astype(int)  # tune based on distribution

    # Rule 8: High severity and novelty
    X_sym["is_high_sev_and_novel"] = (
        (X_sym["is_high_severity_wazuh"] == 1) &
        (X_sym["is_behavioral_novelty"] == 1)
    ).astype(int)


    print(X_sym.sum().sort_values(ascending=False))
    print(X_sym.mean().sort_values(ascending=False))

    return X_sym
