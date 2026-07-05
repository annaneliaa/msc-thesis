import pandas as pd
import re
import os
import json

ALWAYS_BENIGN = [
    "aminer: new event type",
    "user_acct",
    "cron",
    "clamav",
    "freshclam",
    "systemd",
]

# Per-attack semantics: structured fields first, then raw-text fallback
ATTACK_SEMANTICS = {
    "network_scans": {
        "structured": {
            "ids_signature": ["nmap", "masscan", "zmap", "scan"],
            "ids_category": ["attempted-recon", "network scan", "recon", "portscan"],
            "decoder_parent": ["snort", "suricata"],
            "rule_groups_json": ["ids"],
            "log_source_path": ["suricata", "fast.log"],
        },
        "strong": ["nmap", "masscan", "zmap"],
        "weak": ["scan", "syn", "portscan", "icmp"],
        "negative": ["wordpress", "wp-", "sudo", "sshd", "login"],
    },
    "service_scans": {
        "structured": {
            "ids_signature": ["banner", "service probe", "version scan"],
            "ids_category": ["attempted-recon", "recon"],
            "decoder_parent": ["snort", "suricata"],
            "rule_groups_json": ["ids"],
        },
        "strong": ["banner", "service probe", "version scan"],
        "weak": ["scan", "port", "open port", "tcp"],
        "negative": ["wordpress", "wp-", "sudo", "uid=0"],
    },
    "wpscan": {
        "structured": {
            "ids_signature": ["wpscan", "wp-login.php", "xmlrpc.php", "wordpress"],
            "ids_category": ["web application attack", "attempted-recon"],
            "log_source_path": ["apache", "nginx"],
        },
        "strong": ["wpscan", "wp-login.php", "xmlrpc.php"],
        "weak": ["wordpress", "wp-", "wp-content", "wp-includes", "plugin", "theme"],
        "negative": ["nmap", "masscan", "sudo", "uid=0"],
    },
    "dirb": {
        "structured": {
            "ids_signature": ["dirb", "dirbuster", "gobuster"],
            "ids_category": ["web application attack", "attempted-recon"],
            "log_source_path": ["apache", "nginx"],
        },
        "strong": ["dirb", "dirbuster", "gobuster"],
        "weak": ["/admin", "/uploads", "/backup", "/.git", "/wp-admin"],
        "negative": ["sudo", "uid=0", "reverse shell"],
    },
    "webshell": {
        "structured": {
            "ids_signature": [
                "c99",
                "r57",
                "webshell",
                "cmd=",
                "eval(",
                "base64_decode",
            ],
            "ids_category": ["web application attack", "trojan activity"],
            "log_source_path": ["apache", "nginx"],
        },
        "strong": ["c99", "r57", "webshell", "cmd=", "eval(", "base64_decode"],
        "weak": ["php", "upload", ".php", "shell", "wso"],
        "negative": ["nmap", "masscan", "wpscan"],
    },
    "cracking": {
        "structured": {
            "rule_desc": ["authentication failure", "failed password", "invalid user"],
            "decoder": ["sshd", "pam"],
            "groups_str": ["authentication", "auth", "sshd"],
            "aminer_persistence_file": ["login"],
            "aminer_affected_paths": ["/model/type/login/"],
        },
        "strong": [
            "failed password",
            "authentication failure",
            "invalid user",
            "bruteforce",
        ],
        "weak": ["login", "auth", "sshd", "pam", "password"],
        "negative": ["wp-content", "xmlrpc.php"],
    },
    "reverse_shell": {
        "structured": {
            "ids_signature": ["reverse shell", "meterpreter"],
            "procname": ["bash", "python"],
        },
        "strong": ["reverse shell", "meterpreter", "shell connected", "connect back"],
        "weak": ["nc -e", "netcat", "bash -i", "tcp connection"],
        "negative": ["wpscan", "dirb"],
    },
    "privilege_escalation": {
        "structured": {
            "rule_desc": ["sudo", "permission denied"],
            "procname": ["sudo", "su"],
        },
        "strong": ["uid=0", "euid=0", "sudo:", "permission denied", "setuid"],
        "weak": ["sudo", "su ", "root", "elevation"],
        "negative": ["nmap", "wpscan", "dirb"],
    },
    "dnsteal": {
        "structured": {
            "ids_signature": ["dns exfil", "dnsteal", "iodine", "dns query"],
            "ids_category": ["potentially bad traffic", "trojan activity"],
            "proto": ["udp"],
            "dstport": ["53"],
        },
        "strong": ["dns exfil", "dnsteal", "tunnel", "iodine"],
        "weak": ["dns", "txt query", "long subdomain", "query length"],
        "negative": ["sudo", "sshd", "wpscan"],
    },
    "service_stop": {
        "structured": {
            "rule_desc": ["systemctl stop", "service stopped", "shutdown"],
            "procname": ["systemd"],
        },
        "strong": ["systemctl stop", "service stopped", "shutdown", "killed process"],
        "weak": ["stopping", "shutdown", "terminated"],
        "negative": ["scan", "wpscan", "dirb"],
    },
}


def contains_any(text, keywords):
    text = str(text).lower()
    return any(str(k).lower() in text for k in keywords)


def _safe_json_list(x):
    if isinstance(x, list):
        return [str(v).lower() for v in x]
    if pd.isna(x) or x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return [str(v).lower() for v in val]
        except Exception:
            pass
        return [p.strip().lower() for p in re.split(r"[|,]", s) if p.strip()]
    return [str(x).lower()]


def _field_text(row, field):
    val = row.get(field)

    json_list_fields = {
        "rule_groups_json",
        "aminer_affected_paths",
        "aminer_affected_values",
        "aminer_log_resources",
        "rule_pci_dss",
        "rule_gdpr",
        "rule_hipaa",
        "rule_nist_800_53",
        "rule_tsc",
    }

    if field in json_list_fields:
        return " ".join(_safe_json_list(val))

    if pd.isna(val) or val is None:
        return ""

    return str(val).lower()


def build_search_text(row):
    parts = [
        row.get("category", ""),
        row.get("rule_desc", ""),
        row.get("raw_log", ""),
        row.get("ids_signature", ""),
        row.get("ids_category", ""),
        row.get("decoder", ""),
        row.get("decoder_parent", ""),
        row.get("groups_str", ""),
        row.get("aminer_component_type", ""),
        row.get("aminer_component_name", ""),
        row.get("aminer_message", ""),
        row.get("aminer_persistence_file", ""),
        _field_text(row, "aminer_affected_paths"),
        _field_text(row, "aminer_log_resources"),
        row.get("log_source_path", ""),
        row.get("procname", ""),
        row.get("proto", ""),
    ]
    return " | ".join(str(p) for p in parts if p is not None).lower()


def score_match(row, sem: dict) -> int:
    """
    Structured fields first, then text fallback.
    structured match: +3 per field hit
    strong match: +3
    weak match: +1
    negative match: -2
    """
    score = 0
    text = build_search_text(row)

    # structured fields
    for field, kws in sem.get("structured", {}).items():
        field_text = _field_text(row, field)
        if contains_any(field_text, kws):
            score += 3

    # text fallback
    if contains_any(text, sem.get("strong", [])):
        score += 3
    if contains_any(text, sem.get("weak", [])):
        score += 1
    if contains_any(text, sem.get("negative", [])):
        score -= 2

    return score


def build_windows_index(labels_df: pd.DataFrame):
    # dict: scenario -> list of (start, end, attack)
    idx = {}
    for scenario, g in labels_df.groupby("scenario"):
        idx[scenario] = list(
            g[["start", "end", "attack"]].itertuples(index=False, name=None)
        )
    return idx


def assign_label(row, windows_index, min_score=1):
    """
    Returns only:
      1 = attack
      0 = benign
    """
    ts = row["timestamp"]
    scenario = row["scenario"]

    combined = build_search_text(row)

    # hard benign filter
    if contains_any(combined, ALWAYS_BENIGN):
        return 0

    windows = windows_index.get(scenario, [])
    for start, end, attack_type in windows:
        if start <= ts <= end:
            sem = ATTACK_SEMANTICS.get(
                attack_type,
                {"structured": {}, "strong": [], "weak": [], "negative": []},
            )
            s = score_match(row, sem)
            return int(s >= min_score)

    return 0


def assign_labels(
    df,
    dir_path: str,
    labels_file="labels.csv",
    min_score=1,
    out_file="labeled_combined_ait_ads.parquet",
):
    """
    Assign binary labels to alerts based on attack windows and semantic matching.

    Returns:
        pd.DataFrame with:
        - y (int, 0/1)
        - attack_type (str or NaN)
    """
    print("Labeling dataset...")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, format="mixed", errors="coerce"
    )

    labels_path = os.path.join(dir_path, labels_file)

    print(f"Reading labels from: {labels_path}")
    labels = pd.read_csv(labels_path)
    labels["start"] = pd.to_datetime(labels["start"], unit="s", utc=True)
    labels["end"] = pd.to_datetime(labels["end"], unit="s", utc=True)

    windows_index = build_windows_index(labels)

    # binary label only
    df["y"] = df.apply(
        assign_label,
        axis=1,
        windows_index=windows_index,
        min_score=min_score,
    )

    # keep attack_type only for positive rows, based on window membership
    def _get_attack_type(row):
        if row["y"] != 1:
            return pd.NA
        scenario = row["scenario"]
        ts = row["timestamp"]
        for start, end, attack_type in windows_index.get(scenario, []):
            if start <= ts <= end:
                return attack_type
        return pd.NA

    df["attack_type"] = df.apply(_get_attack_type, axis=1)

    out_path = os.path.join(dir_path, out_file)
    df.to_parquet(out_path, index=False)
    print(f"Done. Wrote labeled dataset to: {out_path}")
    return df
