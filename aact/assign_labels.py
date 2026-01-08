import pandas as pd

ALWAYS_BENIGN = [
    "aminer: new event type",
    "user_acct",
    "cron",
    "clamav",
    "freshclam",
    "systemd",
]

ATTACK_RELEVANCE = {
    "network_scans": ["scan", "nmap"],
    "service_scans": ["scan", "port"],
    "wpscan": ["wp-", "wordpress", "wp-includes"],
    "dirb": ["dirb", "/admin", "/uploads"],
    "webshell": ["php", "shell", "upload"],
    "cracking": ["login", "auth", "ssh"],
    "reverse_shell": ["reverse", "connect"],
    "privilege_escalation": ["sudo", "uid=0"],
    "dnsteal": ["dns", "exfil"],
    "service_stop": ["stop", "shutdown"],
}


def contains_any(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


def contains_any(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


def assign_label(row, labels_df):
    ts = row["timestamp"]
    scenario = row["scenario"]
    category = (row["category"] or "").lower()
    raw = str(row.get("raw_log", "")).lower()
    combined = category + " " + raw

    # hard filter
    if contains_any(combined, ALWAYS_BENIGN):
        return "benign"

    # check attack windows
    windows = labels_df[labels_df["scenario"] == scenario]

    for _, w in windows.iterrows():
        if w["start"] <= ts <= w["end"]:
            attack_type = w["attack"]
            keywords = ATTACK_RELEVANCE.get(attack_type, [])

            if contains_any(combined, keywords):
                return "attack"

    return "benign"


def add_labels_to_df(df):
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, format="mixed", errors="coerce"
    )

    # load attack windows
    labels = pd.read_csv("../data/ait_ads/labels.csv")

    # convert unix timestamps to datetime
    labels["start"] = pd.to_datetime(labels["start"], unit="s", utc=True)
    labels["end"] = pd.to_datetime(labels["end"], unit="s", utc=True)

    df["event_label"] = df.apply(assign_label, axis=1, labels_df=labels)

    # add binary labels
    df["y"] = (df["event_label"] == "attack").astype(int)

    print(df["event_label"].value_counts())
    print(df.groupby("scenario")["event_label"].value_counts())

    print("Writing to output file...")

    df.to_csv("../data/ait_ads/combined_ait_labeled.csv", index=False)

    print("Done.")
