import json
import pandas as pd
from glob import glob
import os
import re

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

def infer_port_from_text(raw_log):
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
    if m: user = m.group("user")
    ip = None
    m = IP_RE.search(raw_log)
    if m: ip = m.group("ip")
    proc = None
    m = PROC_RE.search(raw_log)
    if m: proc = m.group(1).lower()
    return user, ip, proc

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
    text = " ".join([
        str(category or ""),
        str(ids_sig or ""),
        str(ids_cat or ""),
        " ".join(groups) if isinstance(groups, list) else str(groups or "")
    ]).lower()

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

    if any(k in text for k in ["sql", "sqli"]): return "web_sqli"
    if any(k in text for k in ["xss", "cross site"]): return "web_xss"
    if any(k in text for k in ["path traversal", "directory traversal"]): return "web_traversal"
    if any(k in text for k in ["rce", "remote code", "command injection"]): return "rce"
    if any(k in text for k in ["shellcode", "exploit kit"]): return "exploit_generic"
    if any(k in text for k in ["brute force", "password guess", "ssh", "auth fail"]): return "bruteforce_auth"
    if any(k in text for k in ["scan", "recon", "nmap", "portscan"]): return "recon_scan"
    if any(k in text for k in ["malware", "trojan", "virus", "botnet"]): return "malware"
    if any(k in text for k in ["dos", "ddos", "denial of service"]): return "dos"
    return "unknown"

def load_alerts_from_json(output_file, dir_path):
    files = glob(f"{dir_path}/*.json")
    rows = []

    for file in files:
        print(f"Opening file {file}...")
        with open(file, "r") as f:
            for line in f:
                scenario = extract_scenario(file)
                obj = json.loads(line)

                # ---------- AMiner ----------
                if "AnalysisComponent" in obj:
                    ts = obj.get("LogData", {}).get("DetectionTimestamp", [None])[0]
                    category = obj.get("AnalysisComponent", {}).get(
                        "AnalysisComponentName", "UNKNOWN"
                    )
                    entity = obj.get("AMiner", {}).get("ID", "UNKNOWN")

                    host_ip = obj.get("AMiner", {}).get("ID")
                    host = None
                    rule_id = None
                    rule_desc = obj.get("AnalysisComponent", {}).get("Message")
                    groups = []
                    groups_list = []
                    alert_channel = "aminer"
                    decoder = decoder_parent = location = None
                    mitre_ids = mitre_tactic = mitre_tech = None
                    username = procname = None

                    srcip = dstip = srcport = dstport = proto = None

                    raw_log = obj.get("LogData", {}).get("RawLogData", [""])[0]
                    source = "aminer"

                    aminer_component_type = obj.get("AnalysisComponent", {}).get(
                        "AnalysisComponentType", "UNKNOWN"
                    )
                    aminer_training_mode = int(
                        obj.get("AnalysisComponent", {}).get("TrainingMode", False)
                    )
                    aminer_new_event = int("new event" in category.lower())

                    wazuh_level = None
                    wazuh_antivirus = 0
                    wazuh_update = 0

                    ids_sig = None
                    ids_cat = None
                    ids_sev = None
                    is_ids_alert = 0
                    exploit_class = "unknown"

                # ---------- Wazuh ----------
                elif "@timestamp" in obj:
                    ts = obj.get("@timestamp")
                    category = obj.get("rule", {}).get("description", "UNKNOWN")

                    entity = (
                        obj.get("agent", {}).get("ip")
                        or obj.get("predecoder", {}).get("hostname")
                        or "UNKNOWN"
                    )

                    host_ip = obj.get("agent", {}).get("ip")
                    host = obj.get("predecoder", {}).get("hostname") or obj.get("agent", {}).get("name")
                    rule_id = obj.get("rule", {}).get("id")
                    rule_desc = obj.get("rule", {}).get("description")
                    groups = obj.get("rule", {}).get("groups", []) or []
                    groups_list = normalize_groups(groups)  # will return list

                    alert_channel = "host"
                    decoder = decoder_parent = location = None
                    mitre_ids = mitre_tactic = mitre_tech = None
                    username = procname = None
                    srcip_text = None

                    data = obj.get("data", {}) or {}

                    raw_log = obj.get("full_log", "")
                    username, srcip_text, procname = extract_host_entities(raw_log)


                    # --- Suricata can be structured (data.alert.*) OR just in fast.log text ---
                    ids_sig = (
                        get_nested(data, "alert", "signature")
                        or get_nested(data, "suricata", "alert", "signature")
                        or data.get("signature")
                    )
                    ids_cat = (
                        get_nested(data, "alert", "category")
                        or get_nested(data, "suricata", "alert", "category")
                        or data.get("category")
                    )
                    ids_sev = (
                        get_nested(data, "alert", "severity")
                        or get_nested(data, "suricata", "alert", "severity")
                        or data.get("severity")
                    )

                    # network fields (may be in data or only in fast.log)
                    srcip = data.get("srcip") or data.get("src_ip") or srcip_text
                    dstip = data.get("dstip") or data.get("dest_ip")
                    srcport = data.get("srcport") or data.get("src_port")
                    dstport = data.get("dstport") or data.get("dest_port")
                    proto = data.get("proto")

                    # normalize types
                    try:
                        srcport = int(srcport) if srcport is not None else None
                    except:
                        pass
                    try:
                        dstport = int(dstport) if dstport is not None else None
                    except:
                        pass

                    # handle "172.28.255.254:53" style dstip
                    if isinstance(dstip, str) and ":" in dstip and dstport is None:
                        ip, port = dstip.rsplit(":", 1)
                        if port.isdigit():
                            dstip = ip
                            dstport = int(port)

                    # fallbacks: parse from fast.log line
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
                                ids_sev = float(m.group(1))

                        if proto is None:
                            m = PROTO_RE.search(raw_log)
                            if m:
                                proto = m.group(1).upper()

                        # if src/dst ports missing, parse flow "a:b -> c:d"
                        if (srcip is None or dstip is None or srcport is None or dstport is None) and "->" in raw_log:
                            m = FLOW_RE.search(raw_log)
                            if m:
                                srcip = srcip or m.group("srcip")
                                dstip = dstip or m.group("dstip")
                                if srcport is None:
                                    srcport = int(m.group("srcport"))
                                if dstport is None:
                                    dstport = infer_port_from_text(raw_log)

                        if dstport is None:
                            dstport = infer_port_from_text(raw_log)


                    # stronger IDS detection (covers "IDS event." cases)
                    decoder = get_nested(obj, "decoder", "name")
                    decoder_parent = get_nested(obj, "decoder", "parent")
                    location = obj.get("location")
                    mitre_ids = get_nested(obj, "rule", "mitre", "id")
                    mitre_tactic = get_nested(obj, "rule", "mitre", "tactic")
                    mitre_tech = get_nested(obj, "rule", "mitre", "technique")

                    is_ids_alert = int(
                        (ids_sig is not None) or (ids_cat is not None) or (ids_sev is not None) or
                        (isinstance(raw_log, str) and "[**]" in raw_log) or
                        (decoder_parent in ["snort", "suricata"])
                    )
                    alert_channel = "ids" if is_ids_alert else "host"

                    exploit_class = infer_exploit_class(category, ids_sig, ids_cat, groups)

                    source = "wazuh"

                    wazuh_level = obj.get("rule", {}).get("level")
                    wazuh_antivirus = int(
                        any(
                            g in ["clamd", "freshclam", "virus"]
                            for g in obj.get("rule", {}).get("groups", [])
                        )
                    )
                    wazuh_update = int("update" in category.lower())

                    aminer_component_type = None
                    aminer_training_mode = 0
                    aminer_new_event = 0

                else:
                    continue

                raw_lower = (raw_log or "").lower()

                rows.append({
                    "timestamp": ts,
                    "category": category,
                    "entity": entity,
                    "raw_log": raw_log,
                    "scenario": scenario,
                    "source": source,

                    # host fields
                    "host_ip": host_ip,
                    "host": host,

                    # rule metadata
                    "rule_id": rule_id,
                    "rule_desc": rule_desc,

                    # groups
                    "groups": "|".join(groups) if isinstance(groups, list) else groups,
                    "groups_raw": json.dumps(groups_list),
                    "groups_str": "|".join(groups_list),

                    # network fields
                    "srcip": srcip,
                    "dstip": dstip,
                    "srcport": srcport,
                    "dstport": dstport,
                    "proto": proto,

                    # static semantic features
                    "is_auth_event": int(
                        any(k in raw_lower for k in ["auth", "login", "pam"])
                    ),
                    "is_cred_event": int("cred" in raw_lower),
                    "is_web_event": int(
                        any(k in raw_lower for k in ["http", "wp", "apache", "nginx"])
                    ),
                    "is_cron": int("cron" in raw_lower),
                    "is_success": int("res=success" in raw_lower),
                    "is_uid0": int("uid=0" in raw_lower),

                    # AMiner specific
                    "aminer_component_type": aminer_component_type,
                    "aminer_training_mode": aminer_training_mode,
                    "aminer_new_event": aminer_new_event,

                    # Wazuh specific
                    "wazuh_level": wazuh_level,
                    "wazuh_antivirus": wazuh_antivirus,
                    "wazuh_update": wazuh_update,

                    # IDS / exploit
                    "is_ids_alert": is_ids_alert,
                    "ids_signature": ids_sig,
                    "ids_category": ids_cat,
                    "ids_severity": ids_sev,
                    "exploit_class": exploit_class,

                    "alert_channel": alert_channel,
                    "decoder": decoder,
                    "decoder_parent": decoder_parent,
                    "location": location,
                    "mitre_ids": json.dumps(mitre_ids) if isinstance(mitre_ids, (list, dict)) else mitre_ids,
                    "mitre_tactic": mitre_tactic,
                    "mitre_technique": mitre_tech,
                    "username": username,
                    "procname": procname,

                })

    df = pd.DataFrame(rows)

    # --- normalize timestamps ---
    mask_aminer = df["source"] == "aminer"
    df.loc[mask_aminer, "timestamp"] = pd.to_datetime(
        df.loc[mask_aminer, "timestamp"], unit="s", utc=True, errors="coerce"
    )

    mask_wazuh = df["source"] == "wazuh"
    df.loc[mask_wazuh, "timestamp"] = pd.to_datetime(
        df.loc[mask_wazuh, "timestamp"], utc=True, format="mixed", errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "category", "entity"])

    print(f"Writing data to {output_file}...\n")
    df.to_csv(f"../data/ait_ads/{output_file}", index=False)
    print("...")

    # # Parquet-safe: avoid pandas "string[python]" / extension dtypes
    # for c in df.columns:
    #     if df[c].dtype.name.startswith(("string", "boolean", "Int", "UInt")):
    #         df[c] = df[c].astype(object)

    # df.to_parquet("../data/ait_ads/combined.parquet", index=False)

    print("Done.")
