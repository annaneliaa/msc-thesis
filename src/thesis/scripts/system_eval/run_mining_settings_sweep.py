"""
Settings-sensitivity check for the mining comparison: every method at the
low / default / high value of its main knob (thesis.system_eval.
mining_settings_sweep.SETTINGS_GRID), then the spread per method and the
best-vs-worst stress test per paper contrast.

CSCAS (runs every variant in this process, mining cached per setting):
    python src/thesis/scripts/system_eval/run_mining_settings_sweep.py --dataset cscas

AIT-ADS: run the variants with the parallel launcher
(SETTINGS_VARIANT=low|high run_mining_comparison_ait_ads.sh), then summarise
them together with the main run as the default:
    python src/thesis/scripts/system_eval/run_mining_settings_sweep.py --summarize \\
        --runs default=<main run> low=<low run> high=<high run> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import atexit
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.mining_comparison import (  # noqa: E402
    ATEMPORAL_METHODS,
    ComparisonConfig,
    run_comparison,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig  # noqa: E402
from thesis.system_eval.mining_settings_sweep import (  # noqa: E402
    VARIANTS,
    apply_variant,
    write_settings_summary,
)

sys.path.insert(0, str(_HERE.parent))
from run_mining_comparison import _report_peak_memory  # noqa: E402


def _parse_runs(items: list[str]) -> dict[str, Path]:
    runs = {}
    for item in items:
        variant, _, path = item.partition("=")
        if variant not in VARIANTS or not path:
            raise SystemExit(f"--runs takes variant=DIR with variant in {VARIANTS}")
        runs[variant] = Path(path)
    return runs


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=("cscas",), default="cscas")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--mining-cache-dir",
        type=Path,
        default=_REPO / "artifacts" / "cache" / "mining_comparison",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--summarize",
        action="store_true",
        help="only summarise finished runs given with --runs",
    )
    p.add_argument("--runs", nargs="+", default=[], metavar="VARIANT=DIR")
    a = p.parse_args()
    atexit.register(_report_peak_memory)

    if a.summarize:
        if not a.runs or a.output_dir is None:
            p.error("--summarize needs --runs and --output-dir")
        write_settings_summary(a.output_dir, _parse_runs(a.runs))
        print(f"Saved → {a.output_dir}")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = a.output_dir or (
        _REPO / "artifacts" / "experiments" / "mining_settings_sweep" / a.dataset / ts
    )
    runs: dict[str, Path] = {}
    for variant in a.variants:
        print(f"== {variant}", flush=True)
        config = ComparisonConfig(
            dataset=a.dataset,
            seeds=tuple(a.seeds),
            methods=ATEMPORAL_METHODS,
            mining=apply_variant(MiniConfig(), variant),
            max_steps=a.max_steps,
            mining_cache_dir=a.mining_cache_dir,
            out_dir=out_root / variant,
            experiment=f"mining_settings_sweep_{a.dataset}",
        )
        runs[variant] = run_comparison(config)
    write_settings_summary(out_root, runs)
    print(f"Saved → {out_root}")


if __name__ == "__main__":
    main()
