import pandas as pd
import numpy as np
from collections import deque, defaultdict
from symbolic_features import build_symbolic_features_from_tokens


def normalize_groups(x):
    if isinstance(x, list):
        return [str(g).lower() for g in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):  # list-string case
            s = s.strip("[]")
        return [g.strip().strip("'\"").lower() for g in s.split(",") if g.strip()]
    return []

# written to mimic AACT baseline
def build_static_features(df):
    df = df.copy()
    X = pd.DataFrame(index=df.index)

    X["source_aminer"] = (df.get("source") == "aminer").astype(int)
    X["source_wazuh"]  = (df.get("source") == "wazuh").astype(int)

    # severity-like (robust to missing cols)
    X["wazuh_level"] = pd.to_numeric(df.get("wazuh_level", 0), errors="coerce").fillna(0)
    X["ids_severity"] = pd.to_numeric(df.get("ids_severity", 0), errors="coerce").fillna(0)

    # presence
    X["has_username"] = df.get("username").notna().astype(int) if "username" in df else 0
    X["has_procname"] = df.get("procname").notna().astype(int) if "procname" in df else 0
    X["has_rule_id"]  = df.get("rule_id").notna().astype(int) if "rule_id" in df else 0
    X["has_mitre"]    = df.get("mitre_ids").notna().astype(int) if "mitre_ids" in df else 0

    # simple network structure
    srcip = df.get("srcip", "").fillna("").astype(str)
    dstip = df.get("dstip", "").fillna("").astype(str)
    X["is_internal_ip"] = srcip.str.startswith(("10.", "192.168.", "172.16."), na=False).astype(int)
    X["src_eq_dst"] = ((srcip != "") & (srcip == dstip)).astype(int)

    return X


# def build_static_features(df):
#     df = df.copy()
#     X_static = pd.DataFrame(index=df.index)

#     # --- alert semantics (category-based) ---
#     X_static["cat_scan"] = (
#         df["category"].str.contains("scan", case=False, na=False).astype(int)
#     )

#     X_static["cat_auth"] = (
#         df["category"]
#         .str.contains("auth|login|ssh|pam", case=False, na=False)
#         .astype(int)
#     )

#     X_static["cat_web"] = (
#         df["category"]
#         .str.contains("web|http|wp|apache|nginx", case=False, na=False)
#         .astype(int)
#     )

#     # --- source system ---
#     X_static["source_aminer"] = (df["source"] == "aminer").astype(int)
#     X_static["source_wazuh"] = (df["source"] == "wazuh").astype(int)

#     # --- protocol / service hints from raw log ---
#     X_static["proto_http"] = (
#         df["raw_log"].str.contains("http|get|post", case=False, na=False).astype(int)
#     )

#     X_static["proto_ssh"] = (
#         df["raw_log"].str.contains("ssh", case=False, na=False).astype(int)
#     )

#     X_static["is_cron"] = (
#         df["raw_log"].str.contains("cron", case=False, na=False).astype(int)
#     )
#     # --- authentication / credential context ---

#     X_static["is_auth_event"] = (
#         df["raw_log"].str.contains("auth|login|pam", case=False, na=False).astype(int)
#     )

#     X_static["is_cred_event"] = (
#         df["raw_log"].str.contains("cred", case=False, na=False).astype(int)
#     )

#     X_static["is_uid0"] = (
#         df["raw_log"].str.contains("uid=0", case=False, na=False).astype(int)
#     )

#     X_static["is_success"] = (
#         df["raw_log"].str.contains("res=success", case=False, na=False).astype(int)
#     )

#     # --- AMiner-specific ---
#     X_static["aminer_new_event"] = (df["source"] == "aminer") & df[
#         "aminer_new_event"
#     ].fillna(0).astype(int)

#     X_static["aminer_training_mode"] = (df["source"] == "aminer") & df[
#         "aminer_training_mode"
#     ].fillna(0).astype(int)

#     # --- Wazuh-specific ---
#     X_static["wazuh_low_level"] = (
#         (df["source"] == "wazuh") & (df["wazuh_level"].fillna(0).astype(int) <= 3)
#     ).astype(int)

#     X_static["wazuh_antivirus"] = (
#         (df["source"] == "wazuh") & (df["wazuh_antivirus"].fillna(0).astype(int) == 1)
#     ).astype(int)

#     X_static["wazuh_update"] = (
#         (df["source"] == "wazuh") & (df["wazuh_update"].fillna(0).astype(int) == 1)
#     ).astype(int)

#     # --- identity presence ---
#     X_static["has_username"] = df["username"].notna().astype(int)
#     X_static["has_procname"] = df["procname"].notna().astype(int)
#     X_static["has_rule_id"] = df["rule_id"].notna().astype(int)

#     # --- fixed infrastructure hints ---
#     X_static["is_internal_ip"] = (
#         df["srcip"].str.startswith(("10.", "192.168.", "172.16."), na=False).astype(int)
#     )

#     X_static["src_eq_dst"] = (df["srcip"] == df["dstip"]).astype(int)

#     # decoder / signature semantics
#     X_static["decoder_known"] = df["decoder"].notna().astype(int)
#     X_static["decoder_parent_known"] = df["decoder_parent"].notna().astype(int)

#     X_static["ids_low_severity"] = (
#         df["ids_severity"].fillna(0).astype(int) <= 2
#     ).astype(int)

#     # MITRE structure
#     X_static["has_mitre"] = df["mitre_ids"].notna().astype(int)
#     X_static["is_mitre_recon"] = (
#         df["mitre_tactic"].str.contains("recon", case=False, na=False)
#     ).astype(int)

#     return X_static


def build_dyn_features(df, window_size):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # stm
    window = deque()  # (timestamp, category, entity, y)

    # counters
    cat_tot = defaultdict(int)
    cat_pos = defaultdict(int)
    ent_tot = defaultdict(int)
    ent_pos = defaultdict(int)

    # rarity tracking
    last_cat_ts = {}
    last_ent_ts = {}

    feats = []

    for _, row in df.iterrows():
        t = row["timestamp"]
        c = row["category"]
        e = row["entity"]
        y = row["y"]

        # remove old alerts from window
        cutoff = t - window_size
        while window and window[0][0] < cutoff:
            t0, c0, e0, y0 = window.popleft()
            cat_tot[c0] -= 1
            cat_pos[c0] -= y0
            ent_tot[e0] -= 1
            ent_pos[e0] -= y0

        # compute features from history
        ct = cat_tot[c]
        et = ent_tot[e]

        feats.append(
            {
                "cat_count_1d": ct,  # how often the alert type appeard in the sliding window
                "cat_rate_1d": (
                    cat_pos[c] / ct if ct > 0 else 0.0
                ),  # attack rate, 0 = benign
                "ent_count_1d": et,  # how many alerts this host generated in the sliding window
                "ent_rate_1d": (
                    ent_pos[e] / et if et > 0 else 0.0
                ),  # fraction of host alerts that were attacks
                "days_since_cat_seen": (
                    (t - last_cat_ts[c]).days if c in last_cat_ts else 999
                ),  # newness of the alert type
                "days_since_ent_seen": (
                    (t - last_ent_ts[e]).days if e in last_ent_ts else 999
                ),  # newness of the host
            }
        )

        # update memory with current alert
        window.append((t, c, e, y))
        cat_tot[c] += 1
        cat_pos[c] += y
        ent_tot[e] += 1
        ent_pos[e] += y
        last_cat_ts[c] = t
        last_ent_ts[e] = t

    X = pd.DataFrame(feats)
    y = df["y"].values

    return X, y, df


def build_sym_features(df, run_name, scenario_name):
    return build_symbolic_features_from_tokens(
        df_used=df,
        scenario=scenario_name,
        run_name=run_name,
    )
