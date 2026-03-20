import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import math


def _ensure_outdir(path="../out/topologies"):
    os.makedirs(path, exist_ok=True)
    return path


def collapse_edges(edge_table: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple edges per (src,dst) (e.g., different ports/protos) into a single weighted edge.
    """
    cols = ["scenario", "srcip", "dstip"]
    out = edge_table.dropna(subset=cols).groupby(cols)["weight"].sum().reset_index()
    return out


def remove_isolates(G):
    G2 = G.copy()
    G2.remove_nodes_from(list(nx.isolates(G2)))
    return G2


def add_communities(G):
    Gu = G.to_undirected()
    comms = nx.algorithms.community.greedy_modularity_communities(Gu)
    node2comm = {}
    for i, c in enumerate(comms):
        for n in c:
            node2comm[n] = i
    return node2comm


def prune_hub_edges(G, hub_topk=2, keep_per_hub=25):
    G2 = G.copy()
    deg = dict(G2.degree())
    hubs = sorted(deg, key=deg.get, reverse=True)[:hub_topk]
    for h in hubs:
        nbrs = list(G2.neighbors(h))
        nbrs_sorted = sorted(
            nbrs, key=lambda n: G2[h][n].get("weight", 1), reverse=True
        )
        for n in nbrs_sorted[keep_per_hub:]:
            if G2.has_edge(h, n):
                G2.remove_edge(h, n)
    return G2


def filter_edges(
    edge_table: pd.DataFrame,
    min_weight: int | None = None,
    quantile: float | None = None,
) -> pd.DataFrame:
    """
    Filter edges by absolute threshold and/or top-quantile threshold.
    """
    df = edge_table.copy()
    if quantile is not None:
        q = df["weight"].quantile(quantile)
        df = df[df["weight"] >= q]
    if min_weight is not None:
        df = df[df["weight"] >= min_weight]
    return df


def ip_to_subnet(ip: str, k: int = 3) -> str:
    """
    k=3 -> /24-ish aggregation for IPv4: a.b.c.0
    """
    parts = str(ip).split(".")
    if len(parts) < 4:
        return str(ip)
    return ".".join(parts[:k]) + ".0"


def subnet_edge_table(edge_table: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Aggregate edges to subnet-level graph.
    """
    df = edge_table.dropna(subset=["scenario", "srcip", "dstip"]).copy()
    df["src_subnet"] = df["srcip"].map(lambda x: ip_to_subnet(x, k=k))
    df["dst_subnet"] = df["dstip"].map(lambda x: ip_to_subnet(x, k=k))
    out = (
        df.groupby(["scenario", "src_subnet", "dst_subnet"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"src_subnet": "srcip", "dst_subnet": "dstip"})
    )
    return out


def edge_df_to_graph(edge_table: pd.DataFrame, scenario: str, directed: bool = False):
    """
    Build (Di)Graph from an edge table for one scenario.
    Expects columns: scenario, srcip, dstip, weight
    Optionally supports proto, dstport (kept if present).
    """
    df = edge_table[edge_table["scenario"] == scenario].copy()
    G = nx.DiGraph() if directed else nx.Graph()

    has_proto = "proto" in df.columns
    has_dstport = "dstport" in df.columns

    for r in df.itertuples(index=False):
        src = str(r.srcip)
        dst = str(r.dstip)

        attrs = {"weight": int(getattr(r, "weight", 1))}
        if has_proto:
            attrs["proto"] = str(getattr(r, "proto", "UNK"))
        if has_dstport:
            dp = getattr(r, "dstport", None)
            attrs["dstport"] = None if pd.isna(dp) else int(dp)

        # sum weights if edge repeats
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += attrs["weight"]
        else:
            G.add_edge(src, dst, **attrs)

    return G


def draw_graph(
    G,
    title: str | None = None,
    out_dir: str = "../out/topologies",
    seed: int = 42,
    node_base: int = 40,
    node_scale: float = 8.0,
    edge_alpha: float = 0.25,
    show_labels: bool = True,
):
    _ensure_outdir(out_dir)

    # # reduce clutter
    # G = prune_hub_edges(G, hub_topk=2, keep_per_hub=25)
    # G = remove_isolates(G)

    plt.figure(figsize=(12, 10))

    # deterministic layout
    pos = nx.spring_layout(G, seed=seed, k=0.35)

    deg = dict(G.degree())
    node_sizes = [node_base + node_scale * math.log1p(deg.get(n, 0)) for n in G.nodes()]

    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    widths = [max(0.3, min(6.0, math.log1p(w))) for w in weights]

    node2comm = add_communities(G)
    node_colors = [node2comm.get(n, 0) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.tab20
    )
    nx.draw_networkx_edges(G, pos, width=widths, alpha=edge_alpha)

    if show_labels:
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:5]
        labels = {n: n for n in top_nodes}
        hub_sizes = [node_base + node_scale * 6 for _ in top_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_size=hub_sizes)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    if title:
        plt.title(title)
        fname = title.replace(" ", "_").replace("/", "_")
    else:
        fname = "topology"

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=200)
    plt.close()


def plot_scenario_topologies(
    edge_table: pd.DataFrame,
    scenario: str,
    out_dir: str = "../out/topologies",
    # filtering
    collapse: bool = True,
    min_weight: int | None = 5,
    quantile: float | None = None,  # e.g. 0.95
    # simplify
    directed: bool = False,
    remove_hubs_k: int = 3,
    # subnet view
    make_subnet_view: bool = True,
    subnet_k: int = 3,
):
    """
    Produces up to 3 PNGs per scenario:
    1) host-level filtered
    3) subnet-level aggregated
    """
    df = edge_table.copy()

    # 1) optionally collapse (src,dst) across ports/protos
    if collapse:
        df = collapse_edges(df)

    # 2) filter edges
    df = filter_edges(df, min_weight=min_weight, quantile=quantile)

    # 3) host-level graph
    # this graph shows which hosts interact frequently in security-relevant ways
    G = edge_df_to_graph(df, scenario=scenario, directed=directed)
    draw_graph(G, title=f"topology_{scenario}_host_filtered", out_dir=out_dir)

    # 5) subnet view
    if make_subnet_view:
        subnet_df = subnet_edge_table(edge_table, k=subnet_k)
        subnet_df = filter_edges(subnet_df, min_weight=min_weight, quantile=quantile)
        Gs = edge_df_to_graph(subnet_df, scenario=scenario, directed=False)
        draw_graph(Gs, title=f"topology_{scenario}_subnet_k{subnet_k}", out_dir=out_dir)
