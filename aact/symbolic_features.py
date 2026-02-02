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

    is_X says whether the symbolic concept X holds for an alert (1 = yes, 0 = no)
    m_is_X says whether this concept was computable for the alert (1 = yes, 0 = no)

    State	    Meaning
    m=0, is=0	concept not applicable
    m=1, is=0	applicable but false
    m=1, is=1	applicable and true
    """
    df = df.copy()
    X_sym = pd.DataFrame(index=df.index)
    
    if X_dyn is not None:
        # align indices
        X_dyn = X_dyn.reset_index(drop=True)
        df = df.reset_index(drop=True)
    else:
        raise ValueError("build_symbolic_features needs X_dyn for dyn-based rules.")
        X_dyn = X_dyn.reset_index(drop=True)

    src = df["source"].fillna("")
    wazuh = (src == "wazuh")
    aminer = (src == "aminer")
    ids = (df.get("is_ids_alert", 0).fillna(0).astype(int) == 1)

    wazuh_level = pd.to_numeric(df.get("wazuh_level"), errors="coerce").fillna(0).astype(int)
    wazuh_update = df.get("wazuh_update", 0).fillna(0).astype(int)
    wazuh_av = df.get("wazuh_antivirus", 0).fillna(0).astype(int)

    exploit_class = df.get("exploit_class", "").fillna("").astype(str).str.lower()
    mitre_tactic = df.get("mitre_tactic", "").fillna("").astype(str).str.lower()

    # ==================================================
    # Rule 0 (optional / may be sparse)
    # SuspiciousRemoteAccess
    # ==================================================
    # df["groups_norm"] = df["groups"].apply(normalize_groups)
    # dstport = pd.to_numeric(df["dstport"], errors="coerce")

    # is_auth_group = df["groups_norm"].apply(
    #     lambda gs: "authentication" in gs
    # ).astype(int)
    # is_auth = (is_auth_group == 1) | (df["is_auth_event"].fillna(0).astype(int) == 1)

    # X_sym["m_is_suspicious_remote_access"] = dstport.notna().astype(int)
    # X_sym["is_suspicious_remote_access"] = (
    #     is_auth & dstport.notna() & (dstport.astype("Int64") != 22)
    # ).astype(int)

    # Rule 1: SuspiciousAuthBurst
    # IF auth failure AND many alerts from same entity
    X_sym["m_is_suspicious_auth_burst"] = 1 
    X_sym["is_suspicious_auth_burst"] = (
        (df["is_auth_event"] == 1) &
        (df["is_success"] == 0) &
        (X_dyn["ent_count_1d"] >= 3)
    ).astype(int)

    # Rule 2: RootLoginAttempt
    # IF auth event AND username == root AND failure
    username = df["username"].fillna("").str.lower()

    X_sym["m_is_root_login_attempt"] = (username != "").astype(int)
    X_sym["is_root_login_attempt"] = (
        (df["is_auth_event"] == 1) &
        (df["is_success"] == 0) &
        (username == "root")
    ).astype(int)

    # Rule 3: BehavioralNovelty (AMiner)
    # IF AMiner flags new behavior
    X_sym["m_is_behavioral_novelty"] = aminer.astype(int)
    X_sym["is_behavioral_novelty"] = (aminer & (df["aminer_new_event"].fillna(0).astype(int) == 1)).astype(int)

    # Rule 4: HighSeverityWazuh (expert threshold)
    # "high rule level is more suspicious"
    X_sym["m_is_high_severity_wazuh"] = wazuh.astype(int)
    X_sym["is_high_severity_wazuh"] = (wazuh & (wazuh_level >= 7)).astype(int)

    # Rule 5: Wazuh critical >= 10
    X_sym["m_is_wazuh_critical"] = wazuh.astype(int)
    X_sym["is_wazuh_critical"] = (wazuh & (wazuh_level >= 10)).astype(int)

    # Rule 6: Mitre mapping
    m_mitre = df.get("mitre_ids", "").fillna("").astype(str)
    X_sym["m_is_wazuh_mitre_mapped"] = wazuh.astype(int)
    X_sym["is_wazuh_mitre_mapped"] = (wazuh & (m_mitre != "") & (m_mitre != "null") & (m_mitre != "[]")).astype(int)

    # Rule 7: Is IDS high priority
    sev = pd.to_numeric(df["ids_severity"], errors="coerce")
    X_sym["m_is_ids_high_priority"] = sev.notna().astype(int)
    X_sym["is_ids_high_priority"] = (sev.notna() & (sev >= 2)).astype(int)  # tune based on distribution

    # Rule 8: Is IDS concept
    X_sym["m_is_ids_concept"] = 1
    X_sym["is_ids_concept"] = ids.astype(int)
    
    # Rule 8: High severity and novelty
    X_sym["is_high_sev_and_novel"] = (
        (X_sym["is_high_severity_wazuh"] == 1) &
        (X_sym["is_behavioral_novelty"] == 1)
    ).astype(int)


    # -------------------------------------------------
    # Combined features

    # (A1) rare entity then burst (temporal)
    X_sym["m_is_rare_entity_then_burst"] = 1
    X_sym["is_rare_entity_then_burst"] = (
        (X_dyn["days_since_ent_seen"] >= 7) &
        (X_dyn["ent_count_1d"] >= 5)
    ).astype(int)

    # (A2) auth failure then uid0
    X_sym["m_is_auth_failure_then_uid0"] = 1
    X_sym["is_auth_failure_then_uid0"] = (
        (df["is_auth_event"].fillna(0).astype(int) == 1) &
        (df["is_success"].fillna(0).astype(int) == 0) &
        (df["is_uid0"].fillna(0).astype(int) == 1)
    ).astype(int)

    # (A4) high sev but not update / AV (reduce benign noise)
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


    # print(X_sym.sum().sort_values(ascending=False))
    # print(X_sym.mean().sort_values(ascending=False))

    return X_sym
