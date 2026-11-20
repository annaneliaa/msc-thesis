"""Settings-sensitivity check: the grid, the cache keys (defaults keep the
existing cache, every other grid point gets its own), RIPPER's prune size
reaching wittgenstein, and the spread / stress-test summary."""

import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from thesis.config import load_mining_settings
from thesis.system_eval import mining_comparison as mc
from thesis.system_eval.mining_mini_experiment import MiniConfig
from thesis.system_eval.mining_settings_sweep import (
    KNOB,
    SETTINGS_GRID,
    SETTINGS_GRID_FILE,
    apply_variant,
    knob_value,
    settings_spread,
    stress_contrasts,
)

from test_mining_comparison import _cscas_spec_and_step


def test_variants_move_each_methods_knob():
    base = MiniConfig()
    assert apply_variant(base, "default") == base
    lo, hi = apply_variant(base, "low"), apply_variant(base, "high")
    for method, field in KNOB.items():
        d = getattr(base, field)
        assert getattr(lo, field) != d and getattr(hi, field) != d, method
    assert lo.eclat_min_support < base.eclat_min_support < hi.eclat_min_support
    assert (
        lo.mctsextent_iterations < base.mctsextent_iterations < hi.mctsextent_iterations
    )
    assert lo.ripper_prune_size < base.ripper_prune_size < hi.ripper_prune_size
    # the support difference moves with the min support (it floors it)
    for m in ("eclat", "prefixspan"):
        for v in (lo, hi):
            assert getattr(v, f"{m}_min_support_diff") == getattr(v, f"{m}_min_support")
    assert set(mc._LATE_FIELDS) <= set(asdict(base))
    assert all(getattr(base, k) == d for k, d in mc._LATE_FIELDS.items())
    assert knob_value("ripper", "default") == 0.33
    with pytest.raises(ValueError, match="unknown settings variant"):
        apply_variant(base, "medium")


def test_contrast_tree_grid_differs_only_in_rounds():
    specs = {
        s.name: s.to_attribute_mining_config()
        for s in load_mining_settings(SETTINGS_GRID_FILE)
    }
    rounds = {name: specs[name].tree.max_diverse_rounds for name in specs}
    assert rounds == {
        "gr3_md1_mda4": 1,
        "gr3_md1_mda4_rounds2": 2,
        "gr3_md1_mda4_rounds3": 3,
    }

    def norm(c):
        return {
            k: v for k, v in c.tree.model_dump().items() if k != "max_diverse_rounds"
        }

    assert norm(specs["gr3_md1_mda4"]) == norm(specs["gr3_md1_mda4_rounds3"])
    for v in ("low", "high"):
        assert SETTINGS_GRID[v]["contrast_tree_setting"] in specs


def _old_key(method, step, spec, mini, token_arm):
    """The cache key as computed before ripper_prune_size existed."""
    settings = {
        k: v
        for k, v in asdict(mini).items()
        if k not in mc._BUDGET_FIELDS and k not in mc._LATE_FIELDS
    }
    f = Path(mini.mining_settings_path)
    return mc._sha1(
        {
            "version": mc.MINING_CACHE_VERSION,
            "dataset": spec.dataset,
            "train": mc._sha1([g.group_id for g in step.train]),
            "n_train": len(step.train),
            "method": method,
            "settings": settings,
            "settings_file": hashlib.sha1(f.read_bytes()).hexdigest()
            if f.exists()
            else None,
            "exclude": sorted(spec.exclude_fields),
            "include": sorted(spec.include_fields),
            "benign_gate_g": spec.benign_gate_g,
            "token_arm": token_arm if spec.dataset == "ait_ads" else None,
        }
    )


def test_default_keys_unchanged_and_grid_points_get_their_own():
    spec, step = _cscas_spec_and_step()
    base = MiniConfig()
    for method in KNOB:
        k_default = mc.mining_cache_key(method, step, spec, base, "T2")
        # adding ripper_prune_size didn't change existing keys
        assert k_default == _old_key(method, step, spec, base, "T2"), method
        for v in ("low", "high"):
            k = mc.mining_cache_key(method, step, spec, apply_variant(base, v), "T2")
            assert k != k_default, (method, v)
    assert mc.mining_cache_key(
        "ripper", step, spec, replace(base, ripper_prune_size=0.2), "T2"
    ) != (mc.mining_cache_key("ripper", step, spec, base, "T2"))


def test_ripper_prune_size_reaches_wittgenstein(monkeypatch):
    from thesis.mining.atemporal.ripper import miner as rm
    from thesis.system_eval import mining_mini_experiment as mme

    seen = {}

    def fake_mine_ripper(table, random_state=0, **params):
        seen.update(params, random_state=random_state)
        return pd.DataFrame()

    monkeypatch.setattr(mme, "mine_ripper", fake_mine_ripper)
    mme.mine("ripper", None, None, replace(MiniConfig(), ripper_prune_size=0.5, seed=3))
    assert seen == {"prune_size": 0.5, "random_state": 3}
    assert rm.mine_ripper  # the real one is untouched


def _run(tmp, label, vals):
    """per_step_results.csv: vals[arm] = per-fold PR-AUC (one seed)."""
    rows = [
        {"arm": arm, "seed": 0, "step_i": i, "pr_auc": v, "f1": v}
        for arm, vs in vals.items()
        for i, v in enumerate(vs)
    ]
    d = tmp / label
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "per_step_results.csv", index=False)
    return d


def test_spread_and_stress(tmp_path):
    none = [0.1, 0.1, 0.1, 0.1]
    runs = {
        "default": _run(
            tmp_path,
            "default",
            {"none": none, "eclat": [0.3] * 4, "contrast_tree": [0.5] * 4},
        ),
        "low": _run(
            tmp_path,
            "low",
            {"none": none, "eclat": [0.45] * 4, "contrast_tree": [0.6] * 4},
        ),
        "high": _run(
            tmp_path,
            "high",
            {"none": none, "eclat": [0.2] * 4, "contrast_tree": [0.4] * 4},
        ),
    }
    spread = settings_spread(runs).set_index("arm")
    assert spread.loc["eclat", "best_variant"] == "low" and spread.loc[
        "eclat", "best"
    ] == pytest.approx(0.45)
    assert spread.loc["contrast_tree", "worst_variant"] == "high"
    assert spread.loc["eclat", "range"] == pytest.approx(0.25)
    assert spread.loc["none", "best_variant"] == ""

    stress = stress_contrasts(
        runs, contrasts=(("after → during", "eclat", "contrast_tree"),)
    )
    by_case = stress.set_index("case")
    # contrast tree at its worst (0.4) vs Eclat at its best (0.45): the claim breaks
    worst_vs_best = by_case.loc["contrast_tree worst vs eclat best"]
    assert (
        worst_vs_best["x"] == "contrast_tree@high" and worst_vs_best["y"] == "eclat@low"
    )
    assert worst_vs_best["mean Δ (x−y)"] == pytest.approx(-0.05)
    assert worst_vs_best["x wins"] == "0/4"
    assert by_case.loc["default vs default", "mean Δ (x−y)"] == pytest.approx(0.2)
