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

Problems already seen, to confirm and fix:

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
