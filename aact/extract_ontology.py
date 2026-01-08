import pandas as pd
from pathlib import Path

PORT2SVC = {
    21:"ftp", 22:"ssh", 23:"telnet", 25:"smtp", 53:"dns",
    80:"http", 110:"pop3", 123:"ntp",
    139:"netbios", 143:"imap", 389:"ldap", 443:"https",
    445:"smb", 587:"smtp-submission",
    3306:"mysql", 3389:"rdp", 5432:"postgres", 6379:"redis",
    9200:"elasticsearch"
}

KEYWORD2SVC = [
    ("sshd", "ssh"),
    ("nginx", "http"),
    ("apache", "http"),
    ("postfix", "smtp"),
    ("dovecot", "imap"),
    ("mysql", "mysql"),
    ("postgres", "postgres"),
    ("rdp", "rdp"),
    ("smb", "smb"),
    ("dns", "dns"),
]

def _to_num(series):
    return pd.to_numeric(series, errors="coerce")

def infer_services(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ports -> service
    df["dstport"] = _to_num(df.get("dstport"))
    df["srcport"] = _to_num(df.get("srcport"))
    df["dst_service"] = df["dstport"].map(lambda p: PORT2SVC.get(int(p)) if pd.notna(p) else None)

    # text hints
    text = (
        df.get("raw_log", "").fillna("").astype(str) + " " +
        df.get("ids_signature", "").fillna("").astype(str) + " " +
        df.get("ids_category", "").fillna("").astype(str)
    ).str.lower()

    text_services = []
    for t in text:
        s = set()
        for kw, name in KEYWORD2SVC:
            if kw in t:
                s.add(name)
        text_services.append(sorted(s))
    df["text_services"] = text_services

    # final guess: dst_service else text_services exploded
    df["service_guess"] = df["dst_service"]
    df_text = df[df["service_guess"].isna()].explode("text_services")
    df_text["service_guess"] = df_text["text_services"]

    df_svc = pd.concat([df[df["service_guess"].notna()], df_text], ignore_index=True)
    df_svc = df_svc.dropna(subset=["scenario", "dstip", "service_guess"])
    df_svc["dstip"] = df_svc["dstip"].astype(str).str.replace(r":\d+$", "", regex=True)

    services_table = (
        df_svc.groupby(["scenario", "dstip"])["service_guess"]
        .agg(lambda x: sorted(set([v for v in x if isinstance(v, str) and v])))
        .reset_index()
        .rename(columns={"dstip": "host_ip", "service_guess": "services"})
    )
    return services_table

def topology_edges(df, min_weight=1):
    edges = df.dropna(subset=["scenario", "srcip", "dstip"]).copy()

    # normalize if any dstip still has ":port"
    edges["dstip"] = edges["dstip"].astype(str).str.replace(r":\d+$", "", regex=True)

    edges["proto"] = edges["proto"].fillna("UNK")
    edges["dstport"] = pd.to_numeric(edges["dstport"], errors="coerce")

    agg = (edges.groupby(["scenario", "srcip", "dstip", "proto", "dstport"])
           .size()
           .reset_index(name="weight"))

    return agg[agg["weight"] >= min_weight]

def host_sets(df, edge_table):
    active = (df.dropna(subset=["scenario","host_ip"])
                .groupby("scenario")["host_ip"]
                .agg(lambda x: set(x.astype(str)))
                .to_dict())

    in_edges = (edge_table.groupby("scenario")
                .apply(lambda g: set(g["srcip"]).union(set(g["dstip"])))
                .to_dict())

    out = []
    for sc in sorted(set(df["scenario"])):
        a = active.get(sc, set())
        e = in_edges.get(sc, set())
        out.append({
            "scenario": sc,
            "n_active_hosts": len(a),
            "n_hosts_in_edges": len(e),
            "isolated_but_active": len(a - e),  # host-only alerts (AMiner/Wazuh host logs)
        })
    return pd.DataFrame(out)

def scenario_hosts(df: pd.DataFrame) -> pd.DataFrame:
    # "hosts we observed in any way" (host alerts OR topology nodes)
    host_alert_hosts = df.dropna(subset=["scenario", "host_ip"]).copy()
    host_alert_hosts["host_ip"] = host_alert_hosts["host_ip"].astype(str)

    # include hosts that appear only in edges
    edge_hosts = df.dropna(subset=["scenario", "srcip", "dstip"])[["scenario", "srcip", "dstip"]].copy()
    edge_hosts["srcip"] = edge_hosts["srcip"].astype(str)
    edge_hosts["dstip"] = edge_hosts["dstip"].astype(str).str.replace(r":\d+$", "", regex=True)

    src_nodes = edge_hosts.rename(columns={"srcip": "host_ip"})[["scenario", "host_ip"]]
    dst_nodes = edge_hosts.rename(columns={"dstip": "host_ip"})[["scenario", "host_ip"]]

    hosts = pd.concat([host_alert_hosts[["scenario", "host_ip"]], src_nodes, dst_nodes], ignore_index=True)
    hosts = hosts.dropna().drop_duplicates().sort_values(["scenario", "host_ip"]).reset_index(drop=True)
    return hosts

def run(input_csv: str, out_dir: str, min_edge_weight: int = 3):
    df = pd.read_csv(input_csv)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Failed to load df. Check input_csv path.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hosts = scenario_hosts(df)
    services = infer_services(df)
    edges = topology_edges(df, min_weight=min_edge_weight)

    hosts.to_csv(out / "scenario_hosts.csv", index=False)
    services.to_csv(out / "scenario_services.csv", index=False)
    edges.to_csv(out / "scenario_edges.csv", index=False)

    print("Wrote:")
    print(out / "scenario_hosts.csv")
    print(out / "scenario_services.csv")
    print(out / "scenario_edges.csv")

if __name__ == "__main__":
    # example:
    # python extract_ontology_inputs.py ../data/ait_ads/all_alerts.csv ./ontology_inputs 3
    import sys
    input_csv = sys.argv[1]
    out_dir = sys.argv[2]
    min_w = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    run(input_csv, out_dir, min_edge_weight=min_w)