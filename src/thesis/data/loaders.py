import json
import pandas as pd
from glob import glob
import os
import re
import hashlib
import numpy as np

# --- fast.log (Suricata/Snort) parsing fallbacks ---
SIG_RE = re.compile(r"\[\*\*\]\s*\[\d+:\d+:\d+\]\s*(.*?)\s*\[\*\*\]")
CLASS_RE = re.compile(r"\[Classification:\s*(.*?)\]")
PRIO_RE = re.compile(r"\[Priority:\s*(\d+)\]")
SSH_PORT_RE = re.compile(r"\bport\s+(?P<port>\d{1,5})\b", re.IGNORECASE)
DPT_RE = re.compile(r"\bdpt[:=](?P<port>\d{1,5})\b", re.IGNORECASE)
PROTO_RE = re.compile(r"\{(\w+)\}")
FLOW_RE = re.compile(
    r"\}\s*(?P<srcip>\d{1,3}(?:\.\d{1,3}){3}):(?P<srcport>\d+)\s*->\s*(?P<dstip>\d{1,3}(?:\.\d{1,3}){3}):(?P<dstport>\d+)"
)
USER_RE = re.compile(r"\buser(?:name)?[=\s:]+(?P<user>[a-zA-Z0-9._-]+)")
SSHD_USER_RE = re.compile(r"\bfor\s+(invalid user\s+)?(?P<user>[a-zA-Z0-9._-]+)\b")
IP_RE = re.compile(r"\b(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\b")
PROC_RE = re.compile(r"\b(sshd|sudo|cron|nginx|apache2|php|python)\b", re.IGNORECASE)


def extract_port(raw_log):
    if not isinstance(raw_log, str):
        return None
    for rx in (DPT_RE, SSH_PORT_RE):
        m = rx.search(raw_log)
        if m:
            p = int(m.group("port"))
            if 0 < p <= 65535:
                return p
    return None


def extract_host_entities(raw_log):
    if not isinstance(raw_log, str):
        return None, None, None
    user = None
    m = SSHD_USER_RE.search(raw_log) or USER_RE.search(raw_log)
    if m:
        user = m.group("user")
    ip = None
    m = IP_RE.search(raw_log)
    if m:
        ip = m.group("ip")
    proc = None
    m = PROC_RE.search(raw_log)
    if m:
        proc = m.group(1).lower()
    return user, ip, proc


# parse username/proc/ip hints from raw_log (entities, not semantics)
def parse_entities(raw_log):
    u, ip, proc = extract_host_entities(raw_log)
    return u, proc, ip


def normalize_groups(x):
    if isinstance(x, list):
        return [str(g).lower() for g in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):  # list-string case
            s = s.strip("[]")
        return [g.strip().strip("'\"").lower() for g in s.split(",") if g.strip()]
    return []


def extract_scenario(filename: str) -> str:
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    return name.split("_", 1)[0] if "_" in name else name


def get_nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_exploit_class(category: str, ids_sig: str, ids_cat: str, groups) -> str:
    text = " ".join(
        [
            str(category or ""),
            str(ids_sig or ""),
            str(ids_cat or ""),
            " ".join(groups) if isinstance(groups, list) else str(groups or ""),
        ]
    ).lower()

    # very simple
    if any(k in text for k in ["et policy", "user-agent", "apt user-agent"]):
        return "policy_indicator"

    if any(k in text for k in ["c2", "command and control", "beacon"]):
        return "c2_activity"

    if any(k in text for k in ["tls", "ja3", "sni", "certificate"]):
        return "tls_fingerprint"

    if any(k in text for k in ["dns query", "tld", "potentially bad traffic"]):
        return "dns_suspicious"

    if any(k in text for k in ["et info", "et policy"]):
        return "ids_policy_info"

    if any(k in text for k in ["sql", "sqli"]):
        return "web_sqli"
    if any(k in text for k in ["xss", "cross site"]):
        return "web_xss"
    if any(k in text for k in ["path traversal", "directory traversal"]):
        return "web_traversal"
    if any(k in text for k in ["rce", "remote code", "command injection"]):
        return "rce"
    if any(k in text for k in ["shellcode", "exploit kit"]):
        return "exploit_generic"
    if any(k in text for k in ["brute force", "password guess", "ssh", "auth fail"]):
        return "bruteforce_auth"
    if any(k in text for k in ["scan", "recon", "nmap", "portscan"]):
        return "recon_scan"
    if any(k in text for k in ["malware", "trojan", "virus", "botnet"]):
        return "malware"
    if any(k in text for k in ["dos", "ddos", "denial of service"]):
        return "dos"
    return "unknown"


def load_alerts_raw(dir_path):
    """
    Ingest and normalize alerts from raw JSON files.
    Read JSON lines (one row per alert). Parse minimal, stable fields.
    No regex fallbacks, no heuristics.
    """
    files = glob(f"{dir_path}/*.json")
    rows = []

    for file in files:
        print(f"Opening file {file}...")

        # Extract scenario
        scenario = extract_scenario(file)

        with open(file, "r") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)

                # ---------- AMiner ----------
                if "AnalysisComponent" in obj:
                    ts = obj.get("LogData", {}).get("DetectionTimestamp", [None])[0]
                    category = obj.get("AnalysisComponent", {}).get(
                        "AnalysisComponentName", "UNKNOWN"
                    )
                    entity = obj.get("AMiner", {}).get("ID", "UNKNOWN")
                    raw_log = obj.get("LogData", {}).get("RawLogData", [""])[0]

                    rows.append(
                        {
                            "timestamp": ts,
                            "timestamp_raw": ts,
                            "scenario": scenario,
                            "source": "aminer",
                            "category": category,
                            "entity": entity,
                            "raw_log": raw_log,
                            # host
                            "host_ip": obj.get("AMiner", {}).get("ID"),
                            "host": None,
                            # rule
                            "rule_id": None,
                            "rule_desc": obj.get("AnalysisComponent", {}).get(
                                "Message"
                            ),
                            # groups (keep raw string + normalized)
                            "groups": "",
                            "groups_raw": "[]",
                            "groups_str": "",
                            # metadata
                            "alert_channel": "aminer",
                            "decoder": None,
                            "decoder_parent": None,
                            "location": None,
                            "mitre_ids": None,
                            "mitre_tactic": None,
                            "mitre_technique": None,
                            # declare entities (parse later)
                            "username": None,
                            "procname": None,
                            # AMiner-specific fields
                            "aminer_component_type": obj.get(
                                "AnalysisComponent", {}
                            ).get("AnalysisComponentType", "UNKNOWN"),
                            "aminer_training_mode": int(
                                obj.get("AnalysisComponent", {}).get(
                                    "TrainingMode", False
                                )
                            ),
                            "aminer_new_event": int(
                                "new event" in str(category).lower()
                            ),
                            "aminer_component_name": obj.get(
                                "AnalysisComponent", {}
                            ).get("AnalysisComponentName"),
                            "aminer_message": obj.get("AnalysisComponent", {}).get(
                                "Message"
                            ),
                            "aminer_persistence_file": obj.get(
                                "AnalysisComponent", {}
                            ).get("PersistenceFileName"),
                            "aminer_affected_paths": json.dumps(
                                obj.get("AnalysisComponent", {}).get(
                                    "AffectedLogAtomPaths", []
                                )
                                or []
                            ),
                            "aminer_affected_values": json.dumps(
                                obj.get("AnalysisComponent", {}).get(
                                    "AffectedLogAtomValues", []
                                )
                                or []
                            ),
                            "aminer_log_resources": json.dumps(
                                obj.get("LogData", {}).get("LogResources", []) or []
                            ),
                            "aminer_log_lines_count": obj.get("LogData", {}).get(
                                "LogLinesCount"
                            ),
                            # placeholders for parsed fields (filled later)
                            "srcip": None,
                            "dstip": None,
                            "srcport": None,
                            "dstport": None,
                            "proto": None,
                            "is_ids_alert": 0,
                            "ids_signature": None,
                            "ids_category": None,
                            "ids_severity": None,
                            # prevent data loss
                            "data_json": json.dumps(obj),
                        }
                    )

                # ---------- Wazuh ----------
                elif "@timestamp" in obj:
                    ts = obj.get("@timestamp")
                    category = obj.get("rule", {}).get("description", "UNKNOWN")
                    raw_log = obj.get("full_log", "")

                    entity = (
                        obj.get("agent", {}).get("ip")
                        or obj.get("predecoder", {}).get("hostname")
                        or "UNKNOWN"
                    )

                    host_ip = obj.get("agent", {}).get("ip")
                    host = obj.get("predecoder", {}).get("hostname") or obj.get(
                        "agent", {}
                    ).get("name")

                    rule_id = obj.get("rule", {}).get("id")
                    rule_desc = obj.get("rule", {}).get("description")

                    groups = obj.get("rule", {}).get("groups", []) or []
                    groups_list = normalize_groups(groups)

                    rows.append(
                        {
                            "timestamp": ts,
                            "timestamp_raw": ts,
                            "scenario": scenario,
                            "source": "wazuh",
                            "category": category,
                            "entity": entity,
                            "raw_log": raw_log,
                            # core host/entity info
                            "host_ip": host_ip,
                            "host": host,
                            "agent_id": get_nested(obj, "agent", "id"),
                            "agent_name": get_nested(obj, "agent", "name"),
                            "manager_name": get_nested(obj, "manager", "name"),
                            # rule info
                            "rule_id": rule_id,
                            "rule_desc": rule_desc,
                            "rule_firedtimes": get_nested(obj, "rule", "firedtimes"),
                            "rule_frequency": get_nested(obj, "rule", "frequency"),
                            "rule_mail": get_nested(obj, "rule", "mail"),
                            "wazuh_level": get_nested(obj, "rule", "level"),
                            # groups
                            "groups": (
                                "|".join(groups)
                                if isinstance(groups, list)
                                else str(groups or "")
                            ),
                            "groups_raw": json.dumps(groups_list),
                            "groups_str": "|".join(groups_list),
                            "rule_groups_json": json.dumps(
                                get_nested(obj, "rule", "groups", default=[]) or []
                            ),
                            # compliance / taxonomy fields
                            "rule_pci_dss": json.dumps(
                                get_nested(obj, "rule", "pci_dss", default=[]) or []
                            ),
                            "rule_gdpr": json.dumps(
                                get_nested(obj, "rule", "gdpr", default=[]) or []
                            ),
                            "rule_hipaa": json.dumps(
                                get_nested(obj, "rule", "hipaa", default=[]) or []
                            ),
                            "rule_nist_800_53": json.dumps(
                                get_nested(obj, "rule", "nist_800_53", default=[]) or []
                            ),
                            "rule_tsc": json.dumps(
                                get_nested(obj, "rule", "tsc", default=[]) or []
                            ),
                            # wazuh metadata
                            "decoder": get_nested(obj, "decoder", "name"),
                            "decoder_parent": get_nested(obj, "decoder", "parent"),
                            "location": obj.get("location"),
                            "log_source_path": obj.get("location"),
                            "input_type": get_nested(obj, "input", "type"),
                            "predecoder_timestamp": get_nested(
                                obj, "predecoder", "timestamp"
                            ),
                            # mitre
                            "mitre_ids": get_nested(obj, "rule", "mitre", "id"),
                            "mitre_tactic": get_nested(obj, "rule", "mitre", "tactic"),
                            "mitre_technique": get_nested(
                                obj, "rule", "mitre", "technique"
                            ),
                            # channel unknown until parsed
                            "alert_channel": "host",
                            # entities (filled later)
                            "username": None,
                            "procname": None,
                            # structured network / IDS fields
                            "ids_event_id": get_nested(obj, "data", "id"),
                            "data_srcip": get_nested(obj, "data", "srcip"),
                            "data_dstip": get_nested(obj, "data", "dstip"),
                            # placeholders for parsed fields
                            "srcip": None,
                            "dstip": None,
                            "srcport": None,
                            "dstport": None,
                            "proto": None,
                            "is_ids_alert": 0,
                            "ids_signature": None,
                            "ids_category": None,
                            "ids_severity": None,
                            # keep full original object to avoid data loss
                            "data_json": json.dumps(obj),
                        }
                    )
                else:
                    continue

    df = pd.DataFrame(rows)

    # assign IDs to each alert
    df = add_alert_id(df)

    # normalize timestamps
    if not df.empty:
        mask_aminer = df["source"] == "aminer"
        df.loc[mask_aminer, "timestamp"] = pd.to_datetime(
            df.loc[mask_aminer, "timestamp"], unit="s", utc=True, errors="coerce"
        )

        mask_wazuh = df["source"] == "wazuh"
        df.loc[mask_wazuh, "timestamp"] = pd.to_datetime(
            df.loc[mask_wazuh, "timestamp"], utc=True, format="mixed", errors="coerce"
        )

        # Ensure consistent datetime64 dtype
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        df = df.dropna(subset=["timestamp", "category", "entity"])

        # preserve unix timestamp in whole seconds for fast joins with authors' alerts_csv
        df["timestamp_unix"] = (df["timestamp"].astype("int64") // 10**9).astype(
            "int64"
        )

        # preserve millisecond precision too
        df["timestamp_unix_ms"] = (df["timestamp"].astype("int64") // 10**6).astype(
            "int64"
        )

        # store mitre_ids consistently as string/json to avoid mixed types in column
        df["mitre_ids"] = df["mitre_ids"].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
        )

    return df


def add_parsed_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive structured fields from raw log / nested JSON fields.

    Expects (recommended) df to contain a `data_json` column for Wazuh rows:
      - data_json = json.dumps(obj.get("data", {}) or {})
    If data_json is missing, it will fall back to regex-only parsing.
    """
    df = df.copy()

    wazuh_mask = df["source"].eq("wazuh")

    # --- parse username/proc/ip hints from raw_log ---
    parsed = df.loc[wazuh_mask, "raw_log"].apply(parse_entities)
    df.loc[wazuh_mask, "username"] = parsed.apply(lambda t: t[0])
    df.loc[wazuh_mask, "procname"] = parsed.apply(lambda t: t[1])
    srcip_text_series = parsed.apply(lambda t: t[2])

    # --- helper: prefer structured fields from data_json, then fall back to regex parsing ---
    def parse_structured_then_regex(row):
        raw_log = row.get("raw_log", "")

        # start with whatever is already there
        ids_sig = row.get("ids_signature")
        ids_cat = row.get("ids_category")
        ids_sev = row.get("ids_severity")
        proto = row.get("proto")
        srcip = row.get("srcip") or srcip_text_series.loc[row.name]
        dstip = row.get("dstip")
        srcport = row.get("srcport")
        dstport = row.get("dstport")

        # 1) Structured parse from Wazuh `data` (stored as JSON string in df["data_json"])
        data = {}
        if "data_json" in row and row.get("data_json"):
            try:
                data = json.loads(row.get("data_json"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}

        if isinstance(data, dict) and data:
            # Suricata can be under data.alert.* or data.suricata.alert.* or flat-ish
            ids_sig = (
                ids_sig
                or get_nested(data, "alert", "signature")
                or get_nested(data, "suricata", "alert", "signature")
                or data.get("signature")
            )
            ids_cat = (
                ids_cat
                or get_nested(data, "alert", "category")
                or get_nested(data, "suricata", "alert", "category")
                or data.get("category")
            )
            ids_sev = (
                ids_sev
                or get_nested(data, "alert", "severity")
                or get_nested(data, "suricata", "alert", "severity")
                or data.get("severity")
            )

            srcip = srcip or data.get("srcip") or data.get("src_ip")
            dstip = dstip or data.get("dstip") or data.get("dest_ip")
            srcport = srcport or data.get("srcport") or data.get("src_port")
            dstport = dstport or data.get("dstport") or data.get("dest_port")
            proto = proto or data.get("proto")

        # normalize numeric ports if strings
        if srcport is not None:
            try:
                srcport = int(srcport)
            except Exception:
                pass
        if dstport is not None:
            try:
                dstport = int(dstport)
            except Exception:
                pass

        # handle "172.28.255.254:53" style dstip
        if isinstance(dstip, str) and ":" in dstip and dstport is None:
            ip, port = dstip.rsplit(":", 1)
            if port.isdigit():
                dstip = ip
                dstport = int(port)

        # 2) Regex fallbacks from fast.log text if still missing
        if isinstance(raw_log, str):
            if ids_sig is None and "[**]" in raw_log:
                m = SIG_RE.search(raw_log)
                if m:
                    ids_sig = m.group(1).strip()

            if ids_cat is None:
                m = CLASS_RE.search(raw_log)
                if m:
                    ids_cat = m.group(1).strip()

            if ids_sev is None:
                m = PRIO_RE.search(raw_log)
                if m:
                    try:
                        ids_sev = float(m.group(1))
                    except Exception:
                        pass

            if proto is None:
                m = PROTO_RE.search(raw_log)
                if m:
                    proto = m.group(1).upper()

            # flow "a:b -> c:d" (use captured dstport!)
            if (
                srcip is None or dstip is None or srcport is None or dstport is None
            ) and "->" in raw_log:
                m = FLOW_RE.search(raw_log)
                if m:
                    srcip = srcip or m.group("srcip")
                    dstip = dstip or m.group("dstip")
                    if srcport is None:
                        srcport = int(m.group("srcport"))
                    if dstport is None:
                        dstport = int(m.group("dstport"))

            if dstport is None:
                dstport = extract_port(raw_log)

        is_ids_alert = int(
            (ids_sig is not None)
            or (ids_cat is not None)
            or (ids_sev is not None)
            or (isinstance(raw_log, str) and "[**]" in raw_log)
            or (row.get("decoder_parent") in ["snort", "suricata"])
        )
        alert_channel = "ids" if is_ids_alert else "host"

        return pd.Series(
            {
                "ids_signature": ids_sig,
                "ids_category": ids_cat,
                "ids_severity": ids_sev,
                "proto": proto,
                "srcip": srcip,
                "dstip": dstip,
                "srcport": srcport,
                "dstport": dstport,
                "is_ids_alert": is_ids_alert,
                "alert_channel": alert_channel,
            }
        )

    df.loc[
        wazuh_mask,
        [
            "ids_signature",
            "ids_category",
            "ids_severity",
            "proto",
            "srcip",
            "dstip",
            "srcport",
            "dstport",
            "is_ids_alert",
            "alert_channel",
        ],
    ] = df.loc[wazuh_mask].apply(parse_structured_then_regex, axis=1)

    return df


def add_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristics/semantics that we can mine (later). Move this function to feature extraction part of pipeline later?
    """
    df = df.copy()
    raw_lower = df["raw_log"].fillna("").astype(str).str.lower()

    df["is_auth_event"] = raw_lower.apply(
        lambda s: int(any(k in s for k in ["auth", "login", "pam"]))
    )
    df["is_cred_event"] = raw_lower.apply(lambda s: int("cred" in s))
    df["is_web_event"] = raw_lower.apply(
        lambda s: int(any(k in s for k in ["http", "wp", "apache", "nginx"]))
    )
    df["is_cron"] = raw_lower.apply(lambda s: int("cron" in s))
    df["is_success"] = raw_lower.apply(lambda s: int("res=success" in s))
    df["is_uid0"] = raw_lower.apply(lambda s: int("uid=0" in s))

    # Wazuh-only heuristics (keep separate from parsing)
    df["wazuh_antivirus"] = 0
    df["wazuh_update"] = 0

    wazuh_mask = df["source"].eq("wazuh")
    # note: groups_str is normalized list joined by "|"
    groups_str = df.loc[wazuh_mask, "groups_str"].fillna("").astype(str)

    df.loc[wazuh_mask, "wazuh_antivirus"] = groups_str.apply(
        lambda s: int(any(g in s.split("|") for g in ["clamd", "freshclam", "virus"]))
    )
    df.loc[wazuh_mask, "wazuh_update"] = (
        df.loc[wazuh_mask, "category"]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(lambda s: int("update" in s))
    )

    # exploit_class (heuristic classifier) — easy to disable later
    def _exploit_class(row):
        return infer_exploit_class(
            row.get("category"),
            row.get("ids_signature"),
            row.get("ids_category"),
            (
                (row.get("groups_str") or "").split("|")
                if isinstance(row.get("groups_str"), str)
                else []
            ),
        )

    df["exploit_class"] = df.apply(_exploit_class, axis=1)

    return df


def add_alert_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.copy()

    required = ["timestamp", "scenario", "source", "entity", "category", "raw_log"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_alert_id() missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # make timestamp stable string (so hashing is consistent)
    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .astype("int64")
        .astype(str)
    )

    key = (
        df["scenario"].fillna("").astype(str)
        + "|"
        + df["source"].fillna("").astype(str)
        + "|"
        + ts
        + "|"
        + df["entity"].fillna("").astype(str)
        + "|"
        + df["category"].fillna("").astype(str)
        + "|"
        + df["raw_log"].fillna("").astype(str)
    )

    # fix to avoid duplicate ids
    df["_row_id"] = np.arange(len(df))

    key = key + "|" + df["_row_id"].astype(str)

    df["alert_id"] = key.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    return df


def load_alerts_from_json(output_file: str, in_path: str, out_path: str):
    os.makedirs(os.path.dirname(os.path.join(out_path)), exist_ok=True)

    file_name = os.path.join(out_path, f"{output_file}.parquet")

    print(f"Reading raw alerts (JSON) from {in_path}.")
    print(f"Writing data to {file_name}...\n")

    df = load_alerts_raw(in_path)

    df = add_parsed_fields(df)
    # df = add_semantic_features(df)
    if "timestamp_raw" in df.columns:
        df["timestamp_raw"] = pd.to_datetime(
            df["timestamp_raw"], utc=True, errors="coerce"
        )

    df.to_parquet(file_name, index=False)
    print("Done.")
