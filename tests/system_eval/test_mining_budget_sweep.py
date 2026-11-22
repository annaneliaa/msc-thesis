"""Budget sweep: the variants, their configs (same mining, only the budget
changes), the mine-once-then-train flow, and the rank-agreement summary."""

from pathlib import Path

import pandas as pd
import pytest

from thesis.system_eval import mining_budget_sweep as mbs
from thesis.system_eval.mining_budget_sweep import (
    BudgetVariant,
    budget_variants,
    summarize_budget_sweep,
    variant_config,
)
from thesis.system_eval.mining_comparison import ComparisonConfig, mining_cache_key
from thesis.system_eval.mining_mini_experiment import MiniConfig


def test_default_variants():
    labels = [(v.label, v.k, v.per_class) for v in budget_variants()]
    assert labels == [
        ("k10", 10, True),
        ("k25", 25, True),
        ("k50", 50, True),
        ("k60", 60, True),
        ("k60_pooled", 60, False),
    ]
    assert [v.label for v in budget_variants((10,), ())] == ["k10"]
    with pytest.raises(ValueError, match="duplicate"):
        budget_variants((10, 10), ())


def test_variant_config_only_changes_the_budget(tmp_path):
    base = ComparisonConfig(mining=MiniConfig(k=60), mining_cache_dir=tmp_path)
    cfg = variant_config(base, BudgetVariant("k10_pooled", 10, False), tmp_path / "x")
    assert cfg.stage == "train" and cfg.out_dir == tmp_path / "x"
    assert cfg.mining.k == 10 and cfg.mining.per_class_budget is False
    # everything the mining cache key looks at is unchanged, so every variant
    # reads the same cached candidates
    from test_mining_comparison import _cscas_spec_and_step

    spec, step = _cscas_spec_and_step()
    assert mining_cache_key("eclat", step, spec, base.mining, "T2") == mining_cache_key(
        "eclat", step, spec, cfg.mining, "T2"
    )


def _fake_run(out_dir: Path, pr: dict[str, list[float]]):
    """per_step_results.csv with one seed per (arm, step)."""
    rows = []
    for arm, vals in pr.items():
        for step, v in enumerate(vals):
            rows.append(
                {
                    "arm": arm,
                    "seed": 0,
                    "step_i": step,
                    "f1": v,
                    "pr_auc": v,
                    "n_symbolic": 0 if arm == "none" else 10,
                    "tp_seen": 1,
                    "n_attack_seen": 2,
                    "tp_unseen": 0,
                    "n_attack_unseen": 0,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "per_step_results.csv", index=False)
    return out_dir


def test_summary_agreement(tmp_path):
    ref = _fake_run(
        tmp_path / "k60",
        {"none": [0.1, 0.2, 0.1], "a": [0.5, 0.6, 0.5], "b": [0.3, 0.4, 0.3]},
    )
    same = _fake_run(
        tmp_path / "k10",
        {"none": [0.1, 0.2, 0.1], "a": [0.4, 0.5, 0.4], "b": [0.2, 0.3, 0.2]},
    )
    flipped = _fake_run(
        tmp_path / "k60_pooled",
        {"none": [0.1, 0.2, 0.1], "a": [0.2, 0.3, 0.2], "b": [0.5, 0.6, 0.5]},
    )
    summary, ranks, agree = summarize_budget_sweep(
        {"k60": ref, "k10": same, "k60_pooled": flipped}
    )
    assert set(summary["variant"]) == {"k60", "k10", "k60_pooled"}
    a = agree[agree["metric"] == "pr_auc"].set_index("variant")
    assert a.loc["k60", "kendall_tau_vs_reference"] == pytest.approx(1.0)
    assert a.loc["k10", "same_top_arm"] and a.loc["k10", "top_arm"] == "a"
    assert not a.loc["k60_pooled", "same_top_arm"]
    assert a.loc["k60_pooled", "kendall_tau_vs_reference"] < 1.0
    r = ranks[(ranks["variant"] == "k60") & (ranks["metric"] == "pr_auc")]
    assert r.set_index("arm")["avg_rank"].to_dict() == {"a": 1.0, "b": 2.0, "none": 3.0}


def test_sweep_mines_once_then_trains_each_variant(tmp_path, monkeypatch):
    calls = []

    def fake_run_comparison(config):
        calls.append((config.stage, config.mining.k, config.mining.per_class_budget))
        if config.stage == "train":
            _fake_run(Path(config.out_dir), {"none": [0.1, 0.2], "a": [0.5, 0.4]})
        return Path(config.out_dir)

    monkeypatch.setattr(mbs, "run_comparison", fake_run_comparison)
    base = ComparisonConfig(mining_cache_dir=tmp_path / "cache")
    dirs = mbs.run_budget_sweep(
        base, budget_variants((10, 60), (60,)), tmp_path / "out"
    )
    assert calls == [
        ("mine", 60, True),
        ("train", 10, True),
        ("train", 60, True),
        ("train", 60, False),
    ]
    assert set(dirs) == {"k10", "k60", "k60_pooled"}
    for f in (
        "budget_summary.csv",
        "budget_ranks.csv",
        "budget_agreement.csv",
        "sweep.json",
    ):
        assert (tmp_path / "out" / f).exists()

    calls.clear()
    mbs.run_budget_sweep(
        base, budget_variants((60,), ()), tmp_path / "out2", mine=False
    )
    assert calls == [("train", 60, True)]


def test_sweep_needs_a_mining_cache(tmp_path):
    with pytest.raises(ValueError, match="mining_cache_dir"):
        mbs.run_budget_sweep(ComparisonConfig(), budget_variants(), tmp_path)


def test_budget_bootstrap_vs_none_from_variant_bootstraps(tmp_path):
    def boot(d: Path, rows):
        pd.DataFrame(
            [
                {
                    "scenario": s,
                    "metric": "pr_auc",
                    "a": a,
                    "b": b,
                    "delta": dl,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": lo > 0 or hi < 0,
                }
                for s, a, b, dl, lo, hi in rows
            ]
        ).to_csv(d / "bootstrap.csv", index=False)

    ref = _fake_run(tmp_path / "k60", {"none": [0.1, 0.2], "m1": [0.3, 0.4]})
    k10 = _fake_run(tmp_path / "k10", {"none": [0.1, 0.2], "m1": [0.15, 0.2]})
    # stored as a - b with "none" first in one and last in the other
    boot(
        ref,
        [
            ("all", "none", "m1", -0.2, -0.3, -0.1),
            ("fox", "none", "m1", -0.2, -0.3, -0.1),
        ],
    )
    boot(k10, [("all", "m1", "none", 0.02, -0.01, 0.05)])
    out = mbs.bootstrap_vs_none({"k60": ref, "k10": k10}).set_index("variant")
    assert out.loc["k60", "arm"] == "m1"
    assert out.loc["k60", "delta_vs_none"] == pytest.approx(0.2)
    assert (out.loc["k60", "ci_low"], out.loc["k60", "ci_high"]) == pytest.approx(
        (0.1, 0.3)
    )
    assert out.loc["k60", "significant"] and not out.loc["k10", "significant"]
    assert len(out) == 2  # only the "all" rows

    mbs.write_budget_summary(tmp_path, {"k60": ref, "k10": k10}, reference="k60")
    assert (tmp_path / "budget_bootstrap.csv").exists()
    assert "+0.200★" in (tmp_path / "budget_summary.txt").read_text()
