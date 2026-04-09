import os
import pickle
from itertools import combinations
import pandas as pd
from collections import defaultdict


def _cache_path(cache_dir: str, cache_key: str, level: int) -> str:
    return os.path.join(cache_dir, f"{cache_key}_k{level}.pkl")


def _save_level_cache(path: str, c0: pd.Series, c1: pd.Series, n0: int, n1: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "c0": c0,
                "c1": c1,
                "n0": n0,
                "n1": n1,
            },
            f,
        )


def _load_level_cache(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["c0"], obj["c1"], obj["n0"], obj["n1"]


# -----------------------------------
# Counters
# -----------------------------------
def count_tokens(
    tokens: pd.Series,
    y: pd.Series,
) -> tuple[pd.Series, pd.Series, int, int]:
    """
    Counts token presence per alert (transaction) separately for benign and attack alerts.

    - Treats each alert as one transaction containing a set of tokens.
    - c0[token] = number of benign alerts that contain token (presence, not occurrences)
    - c1[token] = number of attack alerts that contain token
    - n0/n1 = number of benign/attack alerts (transactions)
    """
    y = y.loc[tokens.index]

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    counts0: dict[str, int] = {}
    counts1: dict[str, int] = {}

    for idx, row_tokens in tokens.items():
        if not row_tokens:
            continue
        present = set(row_tokens)  # presence per alert
        target = counts0 if y.loc[idx] == 0 else counts1
        for tok in present:
            target[tok] = target.get(tok, 0) + 1

    c0 = pd.Series(counts0, dtype=int)
    c1 = pd.Series(counts1, dtype=int)

    all_tokens = c0.index.union(c1.index)
    c0 = c0.reindex(all_tokens, fill_value=0)
    c1 = c1.reindex(all_tokens, fill_value=0)

    return c0, c1, n0, n1


def count_itemsets(
    tokens: pd.Series,
    y: pd.Series,
    k: int = 2,
):
    """
    Fixed-size k-itemset mining with transactional support.
    Returns (c0, c1, n0, n1) where c0/c1 count alerts containing the itemset.
    """
    y = y.loc[tokens.index]

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    itemset_counts_0 = {}
    itemset_counts_1 = {}

    for idx, row_tokens in tokens.items():
        if not row_tokens or len(row_tokens) < k:
            continue

        unique_tokens = sorted(set(row_tokens))
        for itemset in combinations(unique_tokens, k):
            if y.loc[idx] == 0:
                itemset_counts_0[itemset] = itemset_counts_0.get(itemset, 0) + 1
            else:
                itemset_counts_1[itemset] = itemset_counts_1.get(itemset, 0) + 1

    c0 = pd.Series(itemset_counts_0, dtype=int)
    c1 = pd.Series(itemset_counts_1, dtype=int)

    all_itemsets = c0.index.union(c1.index)
    c0 = c0.reindex(all_itemsets, fill_value=0)
    c1 = c1.reindex(all_itemsets, fill_value=0)

    return c0, c1, n0, n1


def count_itemsets_apriori(
    tokens: pd.Series,
    y: pd.Series,
    k: int = 2,
    min_support: int = 1,  # set to 1 because we filter later
):
    """
    Apriori-style frequent itemset mining (up to fixed size k), with transaction-level support.

    - Each row is a transaction (set of tokens).
    - Finds frequent 1-itemsets, then iteratively builds candidates of size 2..k.
    - Uses Apriori pruning: a candidate is kept only if all its (m-1)-subsets are frequent.
    - Counts support as: number of transactions containing the itemset (not total occurrences).
    - Also returns per-class transaction counts (benign vs attack) for size-k itemsets only.

    Returns:
        c0, c1: pd.Series indexed by itemset tuples (length k) with class-specific supports
        n0, n1: number of benign / attack transactions
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    # Align labels to the same rows as tokens
    y = y.loc[tokens.index]

    # Count benign and true alerts
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    # Preprocess transactions
    # remove duplicate tokens, sort for stable ordering, and store as tuple
    transactions = {
        idx: tuple(sorted(set(row_tokens))) if row_tokens else tuple()
        for idx, row_tokens in tokens.items()
    }

    # Build frequent 1-item sets
    # Counts in how many alerts each single token appears
    counts_1 = {}
    for t in transactions.values():
        for item in t:
            counts_1[(item,)] = counts_1.get((item,), 0) + 1

    # Create the set of frequent item sets of the previous size
    previous_set = {itemset for itemset, cnt in counts_1.items() if cnt >= min_support}

    if k == 1:
        # Easiest case, stop early and compute class counts
        # also compute per-class supports for size-1
        c0 = {}
        c1 = {}
        for idx, t in transactions.items():
            present = set((i,) for i in t)
            present &= previous_set
            target = c0 if y.loc[idx] == 0 else c1
            for it in present:
                target[it] = target.get(it, 0) + 1
        c0 = pd.Series(c0, dtype=int).reindex(sorted(previous_set), fill_value=0)
        c1 = pd.Series(c1, dtype=int).reindex(sorted(previous_set), fill_value=0)
        return c0, c1, n0, n1

    # Build frequent item set up untill size k
    for m in range(2, k + 1):
        # Generate candidates
        previous_set_sorted = sorted(previous_set)
        C_m = set()

        # Combine two (m-1)-itemsets if they share the same prefix of sixe m-2
        # Ex: for m=3, join (a,b) and (a,c) to (a,b,c)
        for i in range(len(previous_set_sorted)):
            for j in range(i + 1, len(previous_set_sorted)):
                a = previous_set_sorted[i]
                b = previous_set_sorted[j]
                if a[:-1] != b[:-1]:
                    break  # because sorted => prefixes stop matching
                cand = tuple(sorted(set(a) | set(b)))

                # ensure candidate size is equal to m
                if len(cand) != m:
                    continue

                # Apriori pruning: if the m-subset is frequent, then all (m-1)-subsets must be frequent
                ok = True
                for sub in combinations(cand, m - 1):
                    if sub not in previous_set:
                        ok = False
                        break
                if ok:
                    C_m.add(cand)

        if not C_m:
            # no candidates survive -> stop early
            return (
                pd.Series(dtype=int),
                pd.Series(dtype=int),
                n0,
                n1,
            )

        # Count candidate suports
        # For each alert check which candidates it contains, then increment supprt once per alert
        counts_m = {c: 0 for c in C_m}
        for t in transactions.values():
            if len(t) < m:
                continue
            tset = set(t)
            for c in C_m:
                # subset test
                if set(c).issubset(tset):
                    counts_m[c] += 1

        previous_set = {it for it, cnt in counts_m.items() if cnt >= min_support}
        if not previous_set:
            return (
                pd.Series(dtype=int),
                pd.Series(dtype=int),
                n0,
                n1,
            )

        # If we've reached size k, compute per-class supports for previous set (size-k)
        if m == k:
            # Count for each frequent k-itemset how many benign and true alerts contain it
            c0 = {it: 0 for it in previous_set}
            c1 = {it: 0 for it in previous_set}

            for idx, t in transactions.items():
                if len(t) < k:
                    continue
                tset = set(t)
                present = [it for it in previous_set if set(it).issubset(tset)]
                if not present:
                    continue
                target = c0 if y.loc[idx] == 0 else c1
                for it in present:
                    target[it] += 1

            c0 = pd.Series(c0, dtype=int).reindex(sorted(previous_set), fill_value=0)
            c1 = pd.Series(c1, dtype=int).reindex(sorted(previous_set), fill_value=0)
            return c0, c1, n0, n1

    # Code should never reach this point
    return pd.Series(dtype=int), pd.Series(dtype=int), n0, n1


def count_itemsets_apriori_cached(
    tokens: pd.Series,
    y: pd.Series,
    k: int = 2,
    min_support: int = 50,
    cache_dir: str = "../out/apriori_cache",
    cache_key: str = "default",
):
    """
    Apriori miner with per-level caching.

    Cache behavior:
    - looks for cached levels 1..k
    - computes and saves missing levels
    - if possible, continues from the largest cached level instead of restarting

    Returns:
        c0, c1, n0, n1   for size-k itemsets
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    y = y.loc[tokens.index]
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    # -------------------------------------------------
    # 1) fill missing cache files up to k-1 if needed
    # -------------------------------------------------
    for m in range(1, k):
        path_m = _cache_path(cache_dir, cache_key, m)
        if not os.path.exists(path_m):
            print(f"[cache miss] computing level k={m} for window {cache_key}")
            c0_m, c1_m, n0_m, n1_m = count_itemsets_apriori(
                tokens=tokens,
                y=y,
                k=m,
                min_support=min_support,
            )
            _save_level_cache(path_m, c0_m, c1_m, n0_m, n1_m)

    # -------------------------------------------------
    # 2) if requested level k is already cached, return it
    # -------------------------------------------------
    path_k = _cache_path(cache_dir, cache_key, k)
    if os.path.exists(path_k):
        print(f"[cache hit] loading level k={k}")
        return _load_level_cache(path_k)

    # -------------------------------------------------
    # 3) find largest cached level < k
    # -------------------------------------------------
    start_level = 0
    previous_set = None

    for m in range(k - 1, 0, -1):
        path_m = _cache_path(cache_dir, cache_key, m)
        if os.path.exists(path_m):
            c0_m, c1_m, _, _ = _load_level_cache(path_m)
            previous_set = set(c0_m.index).union(set(c1_m.index))
            start_level = m
            print(f"[resume] starting from cached level k={m}")
            break

    # -------------------------------------------------
    # 4) preprocess transactions
    # -------------------------------------------------
    transactions = {
        idx: tuple(sorted(set(row_tokens))) if row_tokens else tuple()
        for idx, row_tokens in tokens.items()
    }

    # if nothing cached, build level 1
    if start_level == 0:
        counts_1 = {}
        for t in transactions.values():
            for item in t:
                counts_1[(item,)] = counts_1.get((item,), 0) + 1

        previous_set = {it for it, cnt in counts_1.items() if cnt >= min_support}
        start_level = 1

        c0_1 = {}
        c1_1 = {}
        for idx, t in transactions.items():
            present = set((i,) for i in t) & previous_set
            target = c0_1 if y.loc[idx] == 0 else c1_1
            for it in present:
                target[it] = target.get(it, 0) + 1

        c0_1 = pd.Series(c0_1, dtype=int).reindex(sorted(previous_set), fill_value=0)
        c1_1 = pd.Series(c1_1, dtype=int).reindex(sorted(previous_set), fill_value=0)

        _save_level_cache(_cache_path(cache_dir, cache_key, 1), c0_1, c1_1, n0, n1)

        if k == 1:
            return c0_1, c1_1, n0, n1

    # -------------------------------------------------
    # 5) continue Apriori from cached level up to k
    # -------------------------------------------------
    for m in range(start_level + 1, k + 1):
        previous_set_sorted = sorted(previous_set)
        C_m = set()

        for i in range(len(previous_set_sorted)):
            for j in range(i + 1, len(previous_set_sorted)):
                a = previous_set_sorted[i]
                b = previous_set_sorted[j]

                if a[:-1] != b[:-1]:
                    break

                cand = tuple(sorted(set(a) | set(b)))
                if len(cand) != m:
                    continue

                ok = True
                for sub in combinations(cand, m - 1):
                    if sub not in previous_set:
                        ok = False
                        break
                if ok:
                    C_m.add(cand)

        if not C_m:
            empty = pd.Series(dtype=int)
            _save_level_cache(
                _cache_path(cache_dir, cache_key, m), empty, empty, n0, n1
            )
            return empty, empty, n0, n1

        counts_m = {c: 0 for c in C_m}
        for t in transactions.values():
            if len(t) < m:
                continue
            tset = set(t)
            for c in C_m:
                if set(c).issubset(tset):
                    counts_m[c] += 1

        previous_set = {it for it, cnt in counts_m.items() if cnt >= min_support}
        if not previous_set:
            empty = pd.Series(dtype=int)
            _save_level_cache(
                _cache_path(cache_dir, cache_key, m), empty, empty, n0, n1
            )
            return empty, empty, n0, n1

        c0_m = {it: 0 for it in previous_set}
        c1_m = {it: 0 for it in previous_set}

        for idx, t in transactions.items():
            if len(t) < m:
                continue
            tset = set(t)
            present = [it for it in previous_set if set(it).issubset(tset)]
            target = c0_m if y.loc[idx] == 0 else c1_m
            for it in present:
                target[it] += 1

        c0_m = pd.Series(c0_m, dtype=int).reindex(sorted(previous_set), fill_value=0)
        c1_m = pd.Series(c1_m, dtype=int).reindex(sorted(previous_set), fill_value=0)

        _save_level_cache(_cache_path(cache_dir, cache_key, m), c0_m, c1_m, n0, n1)

    return c0_m, c1_m, n0, n1


def count_itemsets_eclat(
    tokens: pd.Series,
    y: pd.Series,
    k: int = 1,
    min_support: int = 1,
) -> tuple[pd.Series, pd.Series, int, int]:
    """
    Eclat-style itemset counting using vertical tidsets.

    Builds token -> set of alert ids
    Then uses set intersections:
    - support of (A, B) = alerts containing A intersect alerts containing B
    - support of (A, B, C) = A ∩ B ∩ C

    Tracks class-specific counts:
    - one tidset for all alerts
    - one for benign alerts
    - one for attack alerts
    Interface:
        counter(tokens, y, **kwargs) -> (c0, c1, n0, n1)

    Args:
        tokens:
            pd.Series of list-of-tokens per alert.
        y:
            pd.Series of binary labels aligned to tokens index.
        k:
            Exact itemset size to return.
        min_support:
            Minimum total support (c0 + c1) for an itemset to be kept.

    Returns:
        c0, c1, n0, n1
        - c0/c1 are pd.Series indexed by candidate
        - candidates are tuples, e.g. ('tokA', 'tokB')
        - n0/n1 are total benign/attack alerts
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    y = y.reindex(tokens.index)

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    # build vertical tidsets
    # token -> set(transaction_ids)
    tid_all = defaultdict(set)
    tid_0 = defaultdict(set)
    tid_1 = defaultdict(set)

    # use integer transaction ids for speed
    for tid, (idx, row_tokens) in enumerate(tokens.items()):
        if not row_tokens:
            continue

        uniq = set(row_tokens)  # transactional presence
        label = y.loc[idx]

        for tok in uniq:
            tid_all[tok].add(tid)
            if label == 0:
                tid_0[tok].add(tid)
            else:
                tid_1[tok].add(tid)

    # keep only frequent singletons
    items = []
    for tok in sorted(tid_all.keys()):
        supp = len(tid_all[tok])
        if supp >= min_support:
            items.append((tok, tid_all[tok], tid_0[tok], tid_1[tok]))

    # k=1 shortcut: count is exactly the postings length
    if k == 1:
        c0 = pd.Series(
            {(tok,): len(t0) for tok, _, t0, _ in items},
            dtype=int,
        )
        c1 = pd.Series(
            {(tok,): len(t1) for tok, _, _, t1 in items},
            dtype=int,
        )
        all_idx = c0.index.union(c1.index)
        c0 = c0.reindex(all_idx, fill_value=0)
        c1 = c1.reindex(all_idx, fill_value=0)
        return c0, c1, n0, n1

    # recursive Eclat search
    out_c0 = {}
    out_c1 = {}

    def recurse(
        prefix: tuple[str, ...],
        prefix_all: set[int],
        prefix_0: set[int],
        prefix_1: set[int],
        suffix_items: list[tuple[str, set[int], set[int], set[int]]],
    ):
        current_size = len(prefix)

        # if we reached exact size k, store counts
        if current_size == k:
            supp = len(prefix_all)
            if supp >= min_support:
                out_c0[prefix] = len(prefix_0)
                out_c1[prefix] = len(prefix_1)
            return

        # try extending with later items only
        for i, (tok, tok_all, tok_0, tok_1) in enumerate(suffix_items):
            if current_size == 0:
                new_all = tok_all
                new_0 = tok_0
                new_1 = tok_1
            else:
                new_all = prefix_all & tok_all
                supp = len(new_all)
                if supp < min_support:
                    continue

                new_0 = prefix_0 & tok_0
                new_1 = prefix_1 & tok_1

            new_prefix = prefix + (tok,)

            # build next suffix by intersecting with later items
            next_suffix = []
            if len(new_prefix) < k:
                for tok2, tok2_all, tok2_0, tok2_1 in suffix_items[i + 1 :]:
                    inter_all = new_all & tok2_all
                    if len(inter_all) >= min_support:
                        next_suffix.append(
                            (
                                tok2,
                                inter_all,
                                new_0 & tok2_0,
                                new_1 & tok2_1,
                            )
                        )

            recurse(new_prefix, new_all, new_0, new_1, next_suffix)

    recurse(tuple(), set(), set(), set(), items)

    c0 = pd.Series(out_c0, dtype=int)
    c1 = pd.Series(out_c1, dtype=int)

    all_idx = c0.index.union(c1.index)
    c0 = c0.reindex(all_idx, fill_value=0)
    c1 = c1.reindex(all_idx, fill_value=0)

    return c0, c1, n0, n1


def count_itemsets_matmul(
    tokens: pd.Series,
    y: pd.Series,
    min_support: int = 1,
    X_tokens=None,
    vocab=None,
) -> tuple[pd.Series, pd.Series, int, int]:
    """
    Count 2-token itemsets using a precached binary token matrix + matrix multiplication.

    Interface:
        counter(tokens, y, **kwargs) -> (c0, c1, n0, n1)

    Args:
        tokens:
            Kept for interface compatibility. Not used when X_tokens and vocab are provided.
        y:
            Binary labels aligned to the rows of X_tokens.
        min_support:
            Minimum total support (c0 + c1) required to keep a pair.
        X_tokens:
            Sparse or dense binary alert-token matrix of shape (n_alerts, n_tokens).
        vocab:
            Column names of X_tokens; token string at position j corresponds to column j.

    Returns:
        c0, c1, n0, n1
    """
    if X_tokens is None or vocab is None:
        raise ValueError("count_itemsets_matmul requires X_tokens and vocab")

    y = pd.Series(y).reset_index(drop=True)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    if X_tokens.shape[0] != len(y):
        raise ValueError(
            f"Row mismatch: X_tokens has {X_tokens.shape[0]} rows but y has {len(y)} rows"
        )

    y_arr = y.to_numpy()
    X0 = X_tokens[y_arr == 0]
    X1 = X_tokens[y_arr == 1]

    C0 = X0.T @ X0
    C1 = X1.T @ X1
    C = C0 + C1

    # convert to array if sparse result
    if hasattr(C0, "toarray"):
        C0 = C0.toarray()
    if hasattr(C1, "toarray"):
        C1 = C1.toarray()
    if hasattr(C, "toarray"):
        C = C.toarray()

    out_c0 = {}
    out_c1 = {}

    m = len(vocab)
    for i in range(m):
        for j in range(i + 1, m):
            total = int(C[i, j])
            if total < min_support:
                continue

            pair = (vocab[i], vocab[j])
            out_c0[pair] = int(C0[i, j])
            out_c1[pair] = int(C1[i, j])

    c0 = pd.Series(out_c0, dtype=int)
    c1 = pd.Series(out_c1, dtype=int)

    all_idx = c0.index.union(c1.index)
    c0 = c0.reindex(all_idx, fill_value=0)
    c1 = c1.reindex(all_idx, fill_value=0)

    return c0, c1, n0, n1
