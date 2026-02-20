import pandas as pd

ALWAYS_BENIGN = [
    "aminer: new event type",
    "user_acct",
    "cron",
    "clamav",
    "freshclam",
    "systemd",
]

ALWAYS_BENIGN = [
    "aminer: new event type",
    "user_acct",
    "cron",
    "clamav",
    "freshclam",
    "systemd",
]

# Per-attack semantics: strong/weak/negative indicators
ATTACK_SEMANTICS = {
    "network_scans": {
        "strong": ["nmap", "masscan", "zmap"],
        "weak": ["scan", "syn", "portscan", "icmp"],
        "negative": ["wordpress", "wp-", "sudo", "sshd", "login"],
    },
    "service_scans": {
        "strong": ["banner", "service probe", "version scan"],
        "weak": ["scan", "port", "open port", "tcp"],
        "negative": ["wordpress", "wp-", "sudo", "uid=0"],
    },
    "wpscan": {
        "strong": ["wpscan", "wp-login.php", "xmlrpc.php"],
        "weak": ["wordpress", "wp-", "wp-content", "wp-includes", "plugin", "theme"],
        "negative": ["nmap", "masscan", "sudo", "uid=0"],
    },
    "dirb": {
        "strong": ["dirb", "dirbuster", "gobuster"],
        "weak": ["/admin", "/uploads", "/backup", "/.git", "/wp-admin"],
        "negative": ["sudo", "uid=0", "reverse shell"],
    },
    "webshell": {
        "strong": ["c99", "r57", "webshell", "cmd=", "eval(", "base64_decode"],
        "weak": ["php", "upload", ".php", "shell", "wso"],
        "negative": ["nmap", "masscan", "wpscan"],
    },
    "cracking": {
        "strong": ["failed password", "authentication failure", "invalid user", "bruteforce"],
        "weak": ["login", "auth", "sshd", "pam", "password"],
        "negative": ["wp-content", "xmlrpc.php"],
    },
    "reverse_shell": {
        "strong": ["reverse shell", "meterpreter", "shell connected", "connect back"],
        "weak": ["nc -e", "netcat", "bash -i", "tcp connection", "connect"],
        "negative": ["wpscan", "dirb"],
    },
    "privilege_escalation": {
        "strong": ["uid=0", "euid=0", "sudo:", "permission denied", "setuid"],
        "weak": ["sudo", "su ", "root", "elevation"],
        "negative": ["nmap", "wpscan", "dirb"],
    },
    "dnsteal": {
        "strong": ["dns exfil", "dnsteal", "tunnel", "iodine"],
        "weak": ["dns", "txt query", "long subdomain", "query length"],
        "negative": ["sudo", "sshd", "wpscan"],
    },
    "service_stop": {
        "strong": ["systemctl stop", "service stopped", "shutdown", "killed process"],
        "weak": ["stop", "terminated", "stopping", "shutdown"],
        "negative": ["scan", "wpscan", "dirb"],
    },
}

def contains_any(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)

def score_match(text: str, sem: dict) -> int:
    """Simple scoring mechanism: strong=+3, weak=+1, negative=-2."""
    score = 0
    if contains_any(text, sem.get("strong", [])):
        score += 3
    if contains_any(text, sem.get("weak", [])):
        score += 1
    if contains_any(text, sem.get("negative", [])):
        score -= 2
    return score

def build_windows_index(labels_df: pd.DataFrame):
    # dict: scenario -> list of (start,end,attack)
    idx = {}
    for scenario, g in labels_df.groupby("scenario"):
        idx[scenario] = list(g[["start", "end", "attack"]].itertuples(index=False, name=None))
    return idx

def assign_label(row, windows_index, min_score=1):
    ts = row["timestamp"]
    scenario = row["scenario"]
    category = (row.get("category") or "").lower()
    raw = str(row.get("raw_log", "")).lower()
    combined = f"{category} {raw}"

    # hard benign filter
    if contains_any(combined, ALWAYS_BENIGN):
        return "benign"

    windows = windows_index.get(scenario, [])
    for start, end, attack_type in windows:
        if start <= ts <= end:
            sem = ATTACK_SEMANTICS.get(attack_type, {"strong": [], "weak": [], "negative": []})
            s = score_match(combined, sem)

            if s >= min_score:
                return f"attack:{attack_type}"

            # # optionally: we can label everything inside an attack window as that attack type
            # if in_window_fallback:
            #     return f"attack:{attack_type}"

            return "benign"  # in-window but not semantically matching

    return "benign"


def assign_labels(df, labels_path="../data/ait_ads/labels.csv", min_score=1,
                     out_path="../data/ait_ads/labeled_combined_ait.csv"):
    """
    Assign ground-truth labels to alerts based on attack windows and semantic matching.

    Args:
        df (pd.DataFrame):
            Alert dataframe containing at least:
            - "timestamp" (datetime or parseable string)
            - "scenario"
            - fields used by assign_label (e.g., raw_log, category)

        labels_path (str, optional):
            Path to labels.csv containing attack windows.
            Must include columns: scenario, attack, start, end.

        min_score (int, optional):
            Minimum semantic matching score required inside an attack window
            to label an alert as an attack.

        out_path (str, optional):
            Path where the labeled dataframe will be written as CSV.

    Returns:
        pd.DataFrame:
            A copy of the input dataframe with added columns:
            - event_label (str)
            - y (int, 0/1)
            - attack_type (str or NaN)
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")

    # Load attack windows from labels.csv (scenario, attack, start, end).
    labels = pd.read_csv(labels_path)
    labels["start"] = pd.to_datetime(labels["start"], unit="s", utc=True)
    labels["end"] = pd.to_datetime(labels["end"], unit="s", utc=True)

    windows_index = build_windows_index(labels)

    # Assign full semantic label per alert
    # event label = full semantic label
    # benign, attack:webshell, etc.
    df["event_label"] = df.apply(
        assign_label, axis=1,
        windows_index=windows_index,
        min_score=min_score
    )

    # Derive binary label and attack type
    # ground truth label (0/1) derived from event label
    df["y"] = df["event_label"].str.startswith("attack").astype(int)

    # attack type is one of the values from the attack column in labels.csv (network scans, reverse shell, etc.)
    df["attack_type"] = df["event_label"].where(df["y"].eq(1)).str.split(":", n=1).str[-1]

    print(df["event_label"].value_counts())
    print(df.groupby("scenario")["event_label"].value_counts())

    df.to_csv(out_path, index=False)
    print(f"Done. Wrote: {out_path}")