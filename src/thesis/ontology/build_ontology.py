import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, Literal
import os

EX = Namespace("http://example.org/aitads#")

os.makedirs("../out/ontology_out", exist_ok=True)


def host_uri(scenario, ip):
    return EX[f"{scenario}_host_{str(ip).replace('.', '_')}"]


def subnet_uri(scenario, subnet):
    return EX[f"{scenario}_subnet_{subnet.replace('.', '_')}"]


def service_uri(name):
    return EX[f"service_{name}"]


def exploit_class_uri(name):
    return EX[f"exploitClass_{str(name).strip().lower().replace(' ', '_')}"]


def alert_type_uri(name):
    # signature strings can be messy; keep a short-ish safe ID
    s = str(name).strip().lower()
    s = s.replace(" ", "_").replace("/", "_").replace(":", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    return EX[f"alertType_{s[:80] or 'unknown'}"]


def subnet_of_ip(ip, k=3):
    parts = str(ip).split(".")
    if len(parts) < 4:
        return str(ip)
    return ".".join(parts[:k]) + ".0"


def build_ontology_for_scenario(
    scenario, hosts_df, services_df, edges_df, alerts_df, out_path, k=3
):
    g = Graph()
    g.bind("ex", EX)

    # Classes
    g.add((EX.Host, RDF.type, RDFS.Class))
    g.add((EX.Subnet, RDF.type, RDFS.Class))
    g.add((EX.Service, RDF.type, RDFS.Class))
    g.add((EX.ExploitClass, RDF.type, RDFS.Class))
    g.add((EX.AlertType, RDF.type, RDFS.Class))

    # Properties
    g.add((EX.inSubnet, RDF.type, RDF.Property))
    g.add((EX.runsService, RDF.type, RDF.Property))
    g.add((EX.connectsTo, RDF.type, RDF.Property))
    g.add((EX.ip, RDF.type, RDF.Property))
    g.add((EX.edgeWeight, RDF.type, RDF.Property))
    g.add((EX.hasAlertType, RDF.type, RDF.Property))  # observed on host
    g.add((EX.edgeHasAlertType, RDF.type, RDF.Property))
    g.add((EX.hasExploitClass, RDF.type, RDF.Property))

    # Hosts + subnets
    scen_hosts = hosts_df[hosts_df["scenario"] == scenario].copy()
    for ip in scen_hosts["host_ip"].astype(str).unique():
        h = host_uri(scenario, ip)
        g.add((h, RDF.type, EX.Host))
        g.add((h, EX.ip, Literal(ip)))

        sn = subnet_of_ip(ip, k=k)
        s = subnet_uri(scenario, sn)
        g.add((s, RDF.type, EX.Subnet))
        g.add((h, EX.inSubnet, s))

    # Services
    scen_services = services_df[services_df["scenario"] == scenario].copy()
    for r in scen_services.itertuples(index=False):
        h = host_uri(scenario, str(r.host_ip))
        # r.services is a list-like string; parse robustly
        svcs = r.services
        if isinstance(svcs, str):
            cleaned = svcs.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            parts = [
                p.strip().strip("'").strip('"') for p in cleaned.split(",") if p.strip()
            ]
        else:
            parts = []
        for sname in parts:
            su = service_uri(sname)
            g.add((su, RDF.type, EX.Service))
            g.add((h, EX.runsService, su))

    # Edges
    scen_edges = edges_df[edges_df["scenario"] == scenario].copy()
    for r in scen_edges.itertuples(index=False):
        src = host_uri(scenario, str(r.srcip))
        dst = host_uri(scenario, str(r.dstip))
        g.add((src, EX.connectsTo, dst))
        g.add((src, EX.edgeWeight, Literal(int(r.weight))))

        scen_alerts = alerts_df[alerts_df["scenario"] == scenario].copy()

    # hasExploitClass(Host, ExploitClass)
    if "exploit_class" in scen_alerts.columns:
        for host_ip, grp in scen_alerts.dropna(subset=["host_ip"]).groupby(
            scen_alerts["host_ip"].astype(str)
        ):
            h = host_uri(scenario, str(host_ip))
            classes = sorted(
                set(
                    [
                        c
                        for c in grp["exploit_class"].dropna().astype(str)
                        if c and c != "unknown"
                    ]
                )
            )
            for c in classes:
                cu = exploit_class_uri(c)
                g.add((cu, RDF.type, EX.ExploitClass))
                g.add((h, EX.hasExploitClass, cu))

    # AlertType(signature/exploit_class)
    # We'll treat "ids_signature" as AlertType when available; otherwise fall back to exploit_class
    # Then attach to host (observed on that host) and also approximate edgeHasAlertType by linking src host to the type
    def _alert_type_row(row):
        sig = row.get("ids_signature")
        if isinstance(sig, str) and sig.strip():
            return sig.strip()
        ec = row.get("exploit_class")
        if isinstance(ec, str) and ec.strip() and ec.strip() != "unknown":
            return f"exploit_class::{ec.strip()}"
        return None

    # host-level alert types
    for r in scen_alerts.itertuples(index=False):
        host_ip = getattr(r, "host_ip", None)
        if host_ip is None or (isinstance(host_ip, float) and pd.isna(host_ip)):
            continue
        at_name = _alert_type_row(pd.Series(r._asdict()))
        if at_name == "unknown":
            continue
        h = host_uri(scenario, str(host_ip))
        atu = alert_type_uri(at_name)
        g.add((atu, RDF.type, EX.AlertType))
        g.add((h, EX.hasAlertType, atu))

    # edge-level (approx) alert types: if alert has srcip/dstip, attach to src host as "edgeHasAlertType"
    if "srcip" in scen_alerts.columns and "dstip" in scen_alerts.columns:
        net_alerts = scen_alerts.dropna(subset=["srcip", "dstip"]).copy()
        for r in net_alerts.itertuples(index=False):
            srcip = getattr(r, "srcip", None)
            dstip = getattr(r, "dstip", None)
            if srcip is None or dstip is None:
                continue
            at_name = _alert_type_row(pd.Series(r._asdict()))
            if at_name == "unknown":
                continue
            src = host_uri(scenario, str(srcip))
            atu = alert_type_uri(at_name)
            g.add((atu, RDF.type, EX.AlertType))
            g.add((src, EX.edgeHasAlertType, atu))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=out_path, format="turtle")


def build_all_scenarios(
    hosts_csv, services_csv, edges_csv, alerts_csv, out_dir="../out/ontology_out", k=3
):
    hosts_df = pd.read_csv(hosts_csv)
    services_df = pd.read_csv(services_csv)
    edges_df = pd.read_csv(edges_csv)
    alerts_df = pd.read_csv(alerts_csv)

    scenarios = sorted(set(hosts_df["scenario"]).union(set(edges_df["scenario"])))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        out_path = Path(out_dir) / f"ontology_{sc}.ttl"
        build_ontology_for_scenario(
            sc, hosts_df, services_df, edges_df, alerts_df, out_path, k=k
        )
        print("Wrote", out_path)
