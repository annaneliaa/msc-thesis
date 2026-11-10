# Mining methods: preparation plan

Preparation for the mining comparison experiment described in
`Paper/Mining/Mining comparison plan.md` (Google Drive).

Status: draft. The library choices (Phase 3) and the open questions at the end still need confirming.

## Phase 1: Restructure the mining module (done, not committed)

Result: 268 tests pass, the same as before the move (the 3 `lime` failures in
`tests/training/test_explain.py` predate it). Every `thesis` module imports
cleanly, and cached CSCAS schemas still resolve. Fingerprints hash only config
and file stats, and only sklearn models are pickled, so the move doesn't touch
the caches. One change from the proposal: the two schema caches went to
`contrast_tree/`, because they fingerprint `AttributeMiningConfig` and only
that method uses them.

Layout:

```
src/thesis/mining/
  common/        load_mining_alert_groups, util (filters, supports), token_abstraction,
                 attribute_features, mining_dummy_job
                 + new (Phase 2): item_table.py (plan 7.3), budget.py (top-K), gate.py (g=2)
  atemporal/
    eclat/          eclat_mining, itemset_mining_job, or_mining
    contrast_tree/  attribute_contrast_mining, decision_tree_rule_mining, attribute_mining_job,
                    attribute_schema_cache, window_schema_cache
    ripper/         (new)
  temporal/
    prefixspan/     prefixspan_mining, sequence_mining_job, repeat_encoding
    mctsextent/     (new)
  cooccurrence_schema.py   stays at the top as the Eclat+PrefixSpan orchestrator
```

- Move files with `git mv` so their history is kept.
- Update every import in `src/`, `tests/`, the shell scripts and the notebooks (about 100 places), with no compatibility shims.
- **Main risk: the caches.** If any fingerprint, or any pickled artifact, stores a module path, the move would invalidate it or break loading it. Check this before moving anything.
- **Done when:** the full test suite passes and one cached CSCAS window loads and gives the same schema as before.

## Phase 2: Audit Eclat and PrefixSpan against the current thesis setup

### 2a. Bugs: done (not committed)

Each bug was confirmed with a brute-force repro before fixing. 295 tests pass
(27 new). The same 3 `lime` failures as before remain.

| # | Bug | Fix |
|---|---|---|
| B1 | Sequence features were checked as **unordered sets** at encode time: `a → b` fired on `[b],[a]`, so PrefixSpan's order never reached the model. | The encoder now checks `item_sequence` features as an ordered subsequence of `sorted_items`, with the same rule used to count support (`SEQUENCE_MINING_TYPES` in `schemas/features.py`). |
| B2 | Mined patterns were renamed with repeat tokens (`a, a → a__repeat_2`). That token only exists for *consecutive identical* alerts, so the feature didn't fire on the groups it was counted on. | The rename is removed; patterns are kept as mined. |
| B3 | A sequence and an itemset over the same items both got the name `sym__a__b`, and one was dropped as a "duplicate". | Sequences are now named `seq__…`. |
| B4 | Itemset-PrefixSpan only tracked the first place the prefix matched, so it missed or undercounted patterns (822 missing in the repro). | The projection now keeps every position where the last step can end. It matches brute force exactly. |
| B5 | `min_count = int(min_support·n)` rounded down: 1 of 30 groups (3.3%) passed a 5% threshold. | `min_support_count()` rounds up (all three miners). |
| B6 | `run_eclat` / `build_tidsets` crashed when `run_dir=None`. | The debug dumps are now optional. |
| B7 | `confidence_*` is support_c / Σ support, i.e. balanced precision, not the class rate inside the pattern. At π≈1.5%, a confidence_attack of 0.97 can mean 31% attacks inside the pattern. | Documented. `precision_<label>` (class rate inside the pattern) is added next to it. The filters still use `confidence_*`, since the existing configs are tuned on it. |

Eclat and item-PrefixSpan themselves already matched brute force; only their
thresholds (B5) and the encoder side (B1–B3) were wrong. Results from earlier
co-occurrence runs that used sequence features are stale. No final-thesis
numbers depend on them.

### 2b. Shared infrastructure: done (not committed)

307 tests pass (12 new). The same 3 `lime` failures remain.

1. **Shared item table** (`common/item_table.py`).
   - Built from the contrast-tree's own `build_categorical_predicate_matrix`, so all methods share exactly the thesis method's candidate space.
   - `items()` / `baskets()`: one item per field value, plus numerics as quantile-bin items (default 5 bins). Edges are fitted on the mining window only.
   - Each item maps to `AttributePredicate`s. A bin `lo < x <= hi` is two predicates, so patterns are encoded through the existing predicate path.
   - On a 20k-group CSCAS slice (oracle fields excluded): 41 items, with no signature words.
2. **Per-host look-back sequences** (`temporal/lookback.py`).
   - For each group: the alerts on its host from `start − w` to `end`, in 1-minute buckets of `short:` items. Default w = 10 min.
   - Stored on the new `AlertGroup.context_steps`. The encoder checks sequence features on it when it's set, so mining and encoding see the same sequence.
   - On fox (cscas_grouping): 6,092 groups built in 5 s, median 3–4 buckets, none empty.
   - Not done: benign subsampling, which belongs in the comparison runner.
3. **Shared pattern format, budget K and schema writer** (`common/patterns.py`).
   - Pattern frame: `items, mining_type, source_label, n_attack, n_benign`.
   - `add_pattern_stats` derives support, per-class support, precision and score (= lean precision × support).
   - `select_budget(k, per_class=False|True)`: in per-class mode, if one class runs short the other fills the gap.
   - `patterns_to_schema` writes the frame into the existing schema.
   - A round-trip test checks that each feature fires on exactly the groups the miner counted.
   - Note: a pooled budget is all benign-leaning on CSCAS-like data (benign support dominates the score), which bears on the open K/2 + K/2 question.
4. **Not done yet:**
   - Replacing `top_k_per_pass` with K in the old co-occurrence job.
   - An adapter that brings the contrast-tree output (survivors + leaves) into the pattern frame, so K counts both.
   - Wiring the gate into the comparison runner.

   All three belong to the comparison runner (plan 7.4–7.6), not to the miners.
5. **Design choices to state, not bugs:**
   - `remove_subset_subsumed` drops a subset when a superset survives, even though the subset may cover more.
   - The direction filter is applied before the config filters.
6. **Found, not fixed:** the old `schema_builder` names `proto=1` and `proto=-1` both `sym__proto_1`, so one would be dropped as a duplicate. It doesn't occur in any of the 7 cached schemas. Fixing it would rename every existing contrast feature. The new writer spells out operators, so it doesn't have this problem.

Original audit list:

1. **Eclat input.** It mines `tx.items`, the tokenised signature words. That is where the "Eclat doesn't work on CSCAS" result came from (plan 6.7). It should mine the shared item table instead: one item per field, with numerics binned into quantiles.
2. **Eclat mines one class.** It mines only `target_label` (benign by default), and attack itemsets come from a separate pass. Check that both passes go through the same filters.
3. **`confidence_*` might be recall, not precision.** The tree leaves had exactly this bug (fixed in `44e5410`). Check `add_confidence_scores` the same way.
4. **Small bugs:**
   - `run_eclat` crashes when `run_dir=None`.
   - It writes the full tidset JSON on every run.
   - `min_count = int(min_support * n)` rounds down, which can let in an itemset just below the threshold.
5. **PrefixSpan input.** It currently mines `sorted_items` inside each group, which is empty for CSCAS. For AIT-ADS the plan needs a per-host look-back stream instead (w minutes, 1-minute buckets as itemsets). That needs a new sequence builder that PrefixSpan and MCTSExtent share.
6. **Matching rule.** The support count in `contains_subsequence` must use the same matching rule as the miner (whether two items can match in the same alert), or the supports won't line up.
7. **AIT-ADS-specific steps run everywhere.** The OR-disjunction step (hard-coded top-100, conf 0.7) and the mail-host abstraction run unconditionally. Proposal: turn both off for the comparison, since they aren't in the plan.
8. **Alignment with the thesis setup:**
   - Every method goes through the shared gate (g = 2), budget K, schema writer, class-weighted RF and walk-forward.
   - The thesis method's features include the Stage-1 survivors, so the "number of features" for K has to count those too.
9. **No unit tests** cover the Eclat or PrefixSpan core. Add brute-force equivalence tests on small random data.

## Phase 3: New methods (to confirm)

| Method | Proposed source | Notes |
|---|---|---|
| Eclat | Keep the in-house code, and test it against `mlxtend.fpgrowth` (gives the same frequent itemsets) | Borgelt's `pyfim` is faster C code, but it might not build on the DGX (aarch64). Only switch if speed turns out to be a problem. |
| PrefixSpan | Keep the in-house itemset-PrefixSpan, and test it against SPMF (Java) on a sample | The `prefixspan` package on PyPI only handles sequences of single items, and AIT-ADS needs sequences of itemsets (the 1-minute buckets). |
| RIPPER | [`wittgenstein`](https://github.com/imoscovitz/wittgenstein), run once per class | It can be slow on big windows. If so, subsample benign groups during fitting only and document it. |
| MCTSExtent | [Romathonat/MCTSExtent](https://github.com/Romathonat/MCTSExtent) ([KAIS 2021 paper](https://perso.liris.cnrs.fr/jean-francois.boulicaut/kais_2021.pdf)), added as a pinned git dependency like DeepCASE, not copied in | Check its license and Python version, and write a converter from our sequences to its `.dat` format. Its `get_patterns(data, target_class, time, top_k, theta)` call fits what we need. |

All five write their output through the same function: clause, leaning class, support and precision, then the gate, then top-K.

## Phase 4: Mini experiment per method

Each method runs on one CSCAS window (W₀) and one AIT-ADS scenario window, including the combinations expected to fail (temporal methods on CSCAS). Those failures back up the paper's claim that CSCAS has no order to mine.

For each run, the report gives:

- the number of patterns before filtering, after the gate and after the budget
- the length distribution of the patterns
- the top 10 patterns per class, with support and precision
- **the share of features that fire at least once in the test window**, to catch dead columns like the contrast-feature bug
- attack coverage
- the share of features that just restate a single signature, to catch the earlier Eclat failure
- runtime

**Pass criterion:** non-empty features for both classes that aren't constant on the test window and aren't single-signature tautologies.

**Format:** one script plus a notebook under `notebooks/paper/`. Mini runs are small enough to run locally.

## Order

Phase 1 → 2 → 3 (RIPPER before MCTSExtent) → 4. Run each method's mini experiment as soon as that method is ready, rather than leaving them all to the end.

## Open questions

- Confirm the library choices in the Phase 3 table.
- Can the OR-disjunctions and the mail-host abstraction be turned off for the comparison?
- Who launches the mini experiments: Claude locally, or the user?
